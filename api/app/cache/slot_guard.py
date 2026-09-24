"""Rejects semantic hits whose lexicon slots contradict the query (black vs cracked)."""

from app.models import Slots


def compatible(query: Slots, cached: Slots) -> bool:
    """True when the cached entry's slots do not contradict the incoming query's.

    Order matters: the first argument is the incoming query, the second the cached entry.

    - component: two filled values that differ block the hit (battery is not screen).
    - intent: fault and configure never match ("I want my screen to go black" is not the black
      screen fault). None - entries cached before the field existed - is a wildcard.
    - symptom, one-sided: when the query names a symptom and the entry has none, the query is about
      something the entry never described, so it misses. The reverse stays a wildcard: a terser
      paraphrase often names no symptom at all, and treating that as a miss dropped paraphrase
      hits from 82% to 54%.
    """
    if query.component and cached.component and query.component != cached.component:
        return False
    if query.intent and cached.intent and query.intent != cached.intent:
        return False
    if query.symptom:
        return query.symptom == cached.symptom
    return True
