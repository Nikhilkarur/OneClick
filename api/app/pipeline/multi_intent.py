"""Component 11: dedupe actions across Goals; keep each in its most relevant Goal; drop empty Goals.

Two actions are the same when they act on the same screen with the same verb, or share a name, or
their steps are near-identical. The copy in the intent with the higher relevance stays (ties: the
lower intent index); the compiler then drops any Goal left without actions.
"""

from collections.abc import Callable

from app.config import settings
from app.models import DraftAction, Intent
from app.pipeline.text import jaccard


def _same(a: DraftAction, b: DraftAction) -> bool:
    same_screen = bool(a.screen_path and b.screen_path) and (
        a.screen_path.strip().lower() == b.screen_path.strip().lower()
    )
    if same_screen and a.intent_verb and a.intent_verb == b.intent_verb:
        return True
    if a.name and b.name and a.name.strip().lower() == b.name.strip().lower():
        return True
    steps_a = " ".join(s.text for s in a.steps)
    steps_b = " ".join(s.text for s in b.steps)
    return jaccard(steps_a, steps_b) >= settings.action_dup_jaccard


def dedupe_with_report(
    intents: list[Intent],
    actions: list[DraftAction],
    *,
    relevance_of: Callable[[DraftAction], float] | None = None,
) -> tuple[list[DraftAction], list[dict]]:
    """(kept actions in input order, [{action, kept_in_intent, removed_from_intents}]).

    `relevance_of(action)` scores an action for its own intent (the pipeline passes the relevance of
    the sections its steps cite); without it the intent's overall relevance decides.
    """
    if len({a.intent_index for a in actions}) < 2:
        return list(actions), []

    def relevance(action: DraftAction) -> tuple[float, int]:
        index = action.intent_index
        if relevance_of is not None:
            rel = relevance_of(action)
        else:
            rel = intents[index].relevance if 0 <= index < len(intents) else 0.0
        return (-rel, index)

    removed: set[int] = set()
    report: dict[str, dict] = {}
    for i, a in enumerate(actions):
        if i in removed:
            continue
        group = [i] + [
            j
            for j in range(i + 1, len(actions))
            if j not in removed and actions[j].intent_index != a.intent_index and _same(a, actions[j])
        ]
        if len(group) < 2:
            continue
        keep = min(group, key=lambda n: relevance(actions[n]))
        dropped = [n for n in group if n != keep]
        removed.update(dropped)
        name = actions[keep].name or actions[keep].screen_path or ""
        report[name] = {
            "action": name,
            "kept_in_intent": actions[keep].intent_index,
            "removed_from_intents": sorted({actions[n].intent_index for n in dropped}),
        }
    return [a for i, a in enumerate(actions) if i not in removed], list(report.values())


def dedupe_across_goals(intents: list[Intent], actions: list[DraftAction]) -> list[DraftAction]:
    kept, _ = dedupe_with_report(intents, actions)
    return kept
