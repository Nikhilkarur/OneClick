"""Component 9: disruption rank + dependency topological sort (data/dependencies.json), SIIS order tiebreak.

Least disruptive first; critical actions last in the order restart < safe mode < software update <
factory reset, unless a dependency needs something after them ("uninstall in safe mode" follows the
safe mode restart).
"""

from app.models import DraftAction
from app.pipeline.categorize import _headline, _words, critical_rank, dependency_edges

_NO_SOURCE = 10**6


def _siis_position(action: DraftAction) -> int:
    ids = [int(i[1:]) for s in action.steps for i in s.src_ids if i[1:].isdigit()]
    return min(ids, default=_NO_SOURCE)


def _provides(action: DraftAction, kind: str) -> bool:
    for before, before_words, _ in dependency_edges():
        if before == kind and before_words and before_words <= _words(_headline(action)):
            return True
    return False


def order(actions: list[DraftAction]) -> list[DraftAction]:
    ranked = sorted(
        enumerate(actions),
        key=lambda pair: (
            pair[1].disruption_rank,
            critical_rank(pair[1]) if pair[1].disruption_rank == 5 else -1,
            _siis_position(pair[1]),
            pair[0],
        ),
    )
    result = [action for _, action in ranked]
    # Dependency fix-up: a dependent that sits before its prerequisite moves to just after it.
    for _ in range(len(result)):
        moved = False
        for index, action in enumerate(result):
            providers = [
                i
                for i, other in enumerate(result)
                if other is not action and any(_provides(other, k) for k in action.depends_on)
            ]
            if providers and max(providers) > index:
                target = max(providers)
                result.insert(target, result.pop(index))  # lands right after the provider
                moved = True
                break
        if not moved:
            break
    return result
