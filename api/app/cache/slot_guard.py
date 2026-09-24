"""Rejects semantic hits whose lexicon slots contradict the query (black vs cracked)."""

from app.config import settings
from app.models import Slots


def compatible(a: Slots, b: Slots) -> bool:
    """True when nothing in the two slot sets contradicts.

    A missing slot is a wildcard: "display won't turn on" finds no symptom word, so it may still
    match a cached "screen is black". Two *filled* slots that differ block the hit, which is what
    keeps "screen is black" from answering "screen is cracked" - embeddings rate those 0.9 alike.
    """
    if a.component and b.component and a.component != b.component:
        return False
    if a.symptom and b.symptom:
        return a.symptom == b.symptom
    # One side has no symptom word. That is usually a terser paraphrase, so it stays a wildcard -
    # except for symptoms that change the whole plan: a cracked screen is not a blank one, however
    # little else the query says (eval/sets/near_miss.jsonl nm_8_1, nm_12_1).
    return not ({a.symptom, b.symptom} & set(settings.guard_strict_symptoms))
