"""Tier 1: embed normalized RAW query, ANN over stored original + variations. No LLM on this path."""

import threading

import numpy as np

from app.cache import store
from app.cache.slot_guard import compatible
from app.config import settings
from app.models import CacheEntry, Slots
from app.pipeline.slots import wants_configuration
from app.retrieval import dense

# One row per stored phrasing; _keys[i] says which cache entry row i belongs to.
# The engine writes variations from a background thread, so index() can run while lookup() reads:
# both take _index_lock, and lookup() reads the pair under it before doing the maths.
_keys: list[str] = []
_matrix: np.ndarray | None = None
_index_lock = threading.Lock()


def index(entry: CacheEntry) -> None:
    """Add a solved plan's phrasings to the vector index.

    Every variation points at the same plan, so a paraphrase of a solved query hits without
    re-running the pipeline. That is the whole of the paraphrase hit rate in block A3.
    """
    global _matrix
    texts = [text for text in entry.query_texts if text.strip()]
    if not texts:
        return
    vectors = np.asarray(dense.embed(texts), dtype=np.float32)  # outside the lock: it is the slow part
    with _index_lock:
        _matrix = vectors if _matrix is None else np.vstack([_matrix, vectors])
        _keys.extend([entry.key] * len(texts))


def rebuild() -> None:
    """Re-embed everything in the store. Called at startup after the snapshot is loaded."""
    global _keys, _matrix
    with _index_lock:
        _keys, _matrix = [], None
    for entry in store.entries().values():
        index(entry)


def lookup_with_score(norm_query: str, slots: Slots, siis_hash: str | None) -> tuple[dict, str, float] | None:
    """The plan of the closest stored phrasing, when it is close enough and does not contradict.

    Three conditions, all required: cosine >= threshold, the lexicon slots agree, and the
    article hash matches. The last one stops the same question with a different article from
    being served a stale plan.
    """
    with _index_lock:  # one consistent snapshot; a concurrent index() may add rows after this
        matrix, keys = _matrix, list(_keys)
    if matrix is None or not keys:
        return None

    query_vector = np.asarray(dense.embed([norm_query])[0], dtype=np.float32)
    similarities = matrix @ query_vector  # rows are normalized, so this is cosine similarity

    entries = store.entries()
    for position in np.argsort(-similarities):
        score = float(similarities[position])
        if score < settings.cache_sim_threshold:
            return None  # sorted, so nothing further can qualify
        entry = entries.get(keys[position])
        if entry is None:
            continue
        if entry.siis_hash != siis_hash:
            continue
        if not compatible(slots, entry.slots):
            continue
        # A configuration request must not be served the fault plan that shares its words.
        stored_query = entry.query_texts[0] if entry.query_texts else ""
        if wants_configuration(norm_query) != wants_configuration(stored_query):
            continue
        store.record_hit(entry.key)
        return entry.plan, entry.key, score
    return None


def lookup(norm_query: str, slots: Slots, siis_hash: str | None) -> dict | None:
    """The plan alone, for callers that do not report the match."""
    found = lookup_with_score(norm_query, slots, siis_hash)
    return found[0] if found else None


def best_match(norm_query: str) -> tuple[str | None, float]:
    """Closest stored phrasing and its score, ignoring every guard.

    Debugging only. A lookup must use lookup_with_score(), which reports the entry it actually
    served: the nearest phrasing is often one the guards rejected.
    """
    with _index_lock:
        matrix, keys = _matrix, list(_keys)
    if matrix is None or not keys:
        return None, 0.0
    query_vector = np.asarray(dense.embed([norm_query])[0], dtype=np.float32)
    similarities = matrix @ query_vector
    position = int(np.argmax(similarities))
    return keys[position], float(similarities[position])
