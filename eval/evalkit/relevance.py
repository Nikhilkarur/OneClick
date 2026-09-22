"""Gold labels and the 0-2 deeplink relevance rubric (metrics.md section 2, ablation section 5).

The rubric follows the template's anchor, "exact target screen vs. parent menu":

    catalog gold  2  the expected entry, or one listed in acceptable_ids (true duplicates)
                  1  the right screen but the wrong control on it (same validation deeplink),
                     or a parent menu listed in parent_ids
                  0  any other entry, a dummy, or no link
    dummy gold    2  bixby://dummy_positive
                  1  a parent menu from parent_ids, or no link (safe, just less useful)
                  0  any other catalog entry: it opens the wrong screen
    manual gold   2  no link
                  0  any link: a physical step must not carry one (spec 4.1)

"Same screen" is read from the catalog itself: entries whose validation objects share a
validation deeplink belong to one Settings screen. Nothing here imports engine code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from evalkit.paths import CATALOG_PATH, GOLD_PATH
from evalkit.sets import read_jsonl

CATALOG, DUMMY, MANUAL = "catalog", "dummy", "manual"
TIERS = (CATALOG, DUMMY, MANUAL)


@dataclass(frozen=True)
class GoldCase:
    step: str
    screen: str
    verb: str | None
    tier: str
    expected_id: str | None
    owner: str
    acceptable_ids: frozenset[str] = frozenset()
    parent_ids: frozenset[str] = frozenset()

    @property
    def accepted(self) -> frozenset[str]:
        return self.acceptable_ids | ({self.expected_id} if self.expected_id else frozenset())


@dataclass(frozen=True)
class Prediction:
    """What a mapper decided for one step: a tier, and the entry id when the tier is catalog."""

    tier: str
    entry_id: str | None = None


@dataclass
class Screens:
    """Catalog entry id -> screen id, where a screen is the entry's validation deeplink."""

    by_entry: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = CATALOG_PATH) -> Screens:
        entries = json.loads(path.read_text())["deeplinks"]
        by_entry = {}
        for e in entries:
            screen = (e.get("validation") or {}).get("deeplink")
            if screen:
                by_entry[e["id"]] = screen
        return cls(by_entry)

    def same_screen(self, a: str | None, b: str | None) -> bool:
        if not a or not b:
            return False
        screen_a, screen_b = self.by_entry.get(a), self.by_entry.get(b)
        return screen_a is not None and screen_a == screen_b


def load_gold(path: Path = GOLD_PATH) -> list[GoldCase]:
    cases = []
    for row in read_jsonl(path):
        cases.append(
            GoldCase(
                step=row["step"],
                screen=row["screen"],
                verb=row.get("verb"),
                tier=row["tier"],
                expected_id=row.get("expected_id"),
                owner=row.get("owner", "unknown"),
                acceptable_ids=frozenset(row.get("acceptable_ids") or []),
                parent_ids=frozenset(row.get("parent_ids") or []),
            )
        )
    return cases


def relevance(case: GoldCase, pred: Prediction, screens: Screens) -> int:
    """0, 1 or 2 per the rubric in the module docstring."""
    linked = pred.tier == CATALOG and pred.entry_id is not None
    if case.tier == MANUAL:
        return 2 if pred.tier == MANUAL else 0
    if case.tier == DUMMY:
        if pred.tier == DUMMY:
            return 2
        if pred.tier == MANUAL or (linked and pred.entry_id in case.parent_ids):
            return 1
        return 0
    # catalog gold
    if not linked:
        return 0
    if pred.entry_id in case.accepted:
        return 2
    if pred.entry_id in case.parent_ids:
        return 1
    if any(screens.same_screen(pred.entry_id, ok) for ok in case.accepted):
        return 1
    return 0


def exact(case: GoldCase, pred: Prediction) -> bool:
    """Precision@1 on catalog gold: the top pick is an accepted entry."""
    return case.tier == CATALOG and pred.tier == CATALOG and pred.entry_id in case.accepted
