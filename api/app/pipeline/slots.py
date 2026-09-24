"""Component 1: lexicon/regex slot extractor (component, symptom). No LLM. Lexicon in data/slot_lexicon.json."""

import json
import re
from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.models import Slots

# settings.data_dir, not a path relative to this file: the Docker image copies api/ alone.
_LEXICON_PATH = Path(settings.data_dir) / "slot_lexicon.json"

# "no physical damage", "doesn't overheat": the phrase is present but denied.
_NEGATIONS = ("no ", "not ", "never ", "without ", "isn't ", "isnt ", "aren't ", "arent ")


@lru_cache(maxsize=1)
def _lexicon() -> dict[str, dict[str, list[str]]]:
    """The word lists, loaded once. Keys: component, symptom."""
    raw = _raw_lexicon()
    return {field: values for field, values in raw.items() if not field.startswith("_")}


@lru_cache(maxsize=1)
def _raw_lexicon() -> dict:
    return json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _weak_components() -> frozenset[str]:
    """Labels that only win when nothing else matches."""
    return frozenset(_raw_lexicon().get("_weak_components", []))


def _negated(text: str, start: int) -> bool:
    """True when the words just before the match deny it."""
    window = text[max(0, start - 14) : start]
    return any(window.endswith(word) for word in _NEGATIONS)


def _best_label(text: str, labels: dict[str, list[str]]) -> str | None:
    """The label mentioned earliest in the text; the longer phrase breaks a tie.

    Earliest wins because a complaint names its subject first ("my SCREEN went black ... I
    cannot use Smart Switch"): a later phrase is usually context, not the problem itself.
    """
    best_label, best_position, best_length = None, len(text) + 1, 0
    for label, phrases in labels.items():
        for phrase in phrases:
            match = re.search(rf"\b{re.escape(phrase)}\b", text)
            if not match or _negated(text, match.start()):
                continue
            position = match.start()
            if position < best_position or (position == best_position and len(phrase) > best_length):
                best_label, best_position, best_length = label, position, len(phrase)
    return best_label


# "I want my screen to go black while Smart Switch runs" names the same component and symptom as
# the fault it resembles, so slots alone cannot separate them: the difference is that the user is
# asking for the behaviour, not reporting it. eval/sets/near_miss.jsonl calls these differs_in
# "intent" and they were 15 of our 17 false cache hits.
_WISH_PATTERNS = (
    r"\bi want\b",
    r"\bi'?d like\b",
    r"\bi wish\b",
    r"\bhow (?:do|can) i (?:set|make|get|add|turn|enable|disable|configure|apply|shrink|fit)\b",
    r"\bdeliberately\b",
    r"\bon purpose\b",
    r"\bintentionally\b",
)
_WISH_RE = re.compile("|".join(_WISH_PATTERNS), re.IGNORECASE)


def wants_configuration(text: str) -> bool:
    """True when the query asks for a behaviour rather than reporting a fault.

    The cache uses it as a guard: a configuration request must not be answered with the
    troubleshooting plan for the fault that shares its words.
    """
    return bool(_WISH_RE.search(text or ""))


def _best_component(text: str, labels: dict[str, list[str]]) -> str | None:
    """Prefer a real part over a weak one: an app name usually says where a fault was noticed.

    "Opening an email in Gmail makes the screen flash" is a screen problem; taking 'app' there
    blocks the cache match against the stored screen answer (7 of 200 paraphrases).
    """
    weak = _weak_components()
    strong = {label: phrases for label, phrases in labels.items() if label not in weak}
    return _best_label(text, strong) or _best_label(text, labels)


def extract_slots(norm_query: str) -> Slots:
    """Component and symptom for the cache guard. Unknown fields stay None, which acts as a wildcard."""
    text = norm_query.lower()
    lexicon = _lexicon()
    return Slots(
        component=_best_component(text, lexicon.get("component", {})),
        symptom=_best_label(text, lexicon.get("symptom", {})),
    )
