"""Catalog entries by id, exactly as the kit ships them (hard rule 8: copy values verbatim).

The compiler reads the raw kit file rather than the retrieval index, whose documents are cleaned
for search and must never reach the response.
"""

import json
from functools import lru_cache
from pathlib import Path

from app.config import settings

DUMMY_ENTRY_ID = "DL-DUMMY"
DUMMY_URI = "bixby://dummy_positive"
_ACTIONABLE_FIELDS = ("deeplink", "description", "message", "originalType")


@lru_cache(maxsize=1)
def _entries() -> dict[str, dict]:
    path = Path(settings.data_dir) / "kit" / "deeplinks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in data["deeplinks"]}


def entry(entry_id: str) -> dict | None:
    """The whole catalog row, or None for an id the catalog does not have."""
    return _entries().get(entry_id)


def actionable(entry_id: str) -> dict | None:
    """The `actionableDeeplink` object for a catalog entry, fields copied unchanged."""
    row = entry(entry_id)
    if row is None or entry_id == DUMMY_ENTRY_ID:
        return None
    return {field: row.get(field) for field in _ACTIONABLE_FIELDS}


def validation(entry_id: str) -> dict | None:
    """The entry's own validation object, unchanged (key-only for everything but onURL entries)."""
    row = entry(entry_id)
    value = row.get("validation") if row else None
    return dict(value) if isinstance(value, dict) else None
