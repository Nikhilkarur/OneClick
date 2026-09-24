"""Final pass: URL scrub + official schema validation + one repair cycle, then drop offending action.

Runs on every response before it leaves the API, cache hits included (hard rules 1, 3, 4). It never
raises: whatever cannot be repaired is dropped, and the worst case is an empty, valid `contexts`.
"""

import copy
import math
import re

from pydantic import ValidationError

from app.compiler.compile import clean_step
from app.compiler.scrub import has_leak, scrub
from app.compiler.templates import build_goal
from app.compiler.trimmer import trim_description, trim_title
from app.schema import ContextDeeplinkResponse

_CATALOG_URI = re.compile(r"^bixby://[A-Za-z0-9_./\-]+$")
_LINK_FIELDS = ("actionableDeeplink", "validationDeeplink")
_GOAL = re.compile(
    r"^Follow these steps to perform this (?P<topic>.+?) (?P<kind>Troubleshooting|Configuration)\.$"
)
_GOAL_LOOSE = re.compile(
    r"^Follow these steps to perform this (?P<topic>.+?) (?P<kind>Troubleshooting|Configuration)"
)
_MAX_PASSES = 3


def _scrub_strings(obj, report: dict, in_link: bool = False):
    """Scrub every string; a deeplink URI inside a link object is kept only if it is a catalog URI."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if in_link and key == "deeplink":
                out[key] = value if isinstance(value, str) and _CATALOG_URI.match(value) else None
            else:
                out[key] = _scrub_strings(value, report, in_link or key in _LINK_FIELDS)
        return out
    if isinstance(obj, list):
        return [_scrub_strings(value, report, in_link) for value in obj]
    if isinstance(obj, str) and has_leak(obj):
        report["url_leaks"] += 1
        return scrub(obj)
    return obj


def _repair_link(group: dict, field: str) -> None:
    link = group.get(field)
    if link is not None and (not isinstance(link, dict) or not link.get("deeplink")):
        group[field] = None


def _repair_action(action: dict) -> dict | None:
    """Enforce the string rules on one action; None when nothing usable is left."""
    groups = []
    for group in action.get("stepGroups") or []:
        if not isinstance(group, dict):
            continue
        steps = [clean_step(s) for s in group.get("steps") or [] if isinstance(s, str)]
        group = {**group, "steps": [s for s in steps if s]}
        for field in _LINK_FIELDS:
            _repair_link(group, field)
        if group["steps"]:
            groups.append(group)
    if not groups:
        return None
    category = action.get("category") or "manual"
    linked = any(g.get("actionableDeeplink") for g in groups)
    if category == "auto" and not linked:
        category = "manual"
    if category != "auto":  # only auto actions carry links (spec 4.1)
        groups = [{**g, "actionableDeeplink": None, "validationDeeplink": None} for g in groups]
    name = " ".join(str(action.get("actionName") or "").split()) or "Follow These Steps"
    return {
        **action,
        "actionName": name,
        "description": trim_description(str(action.get("description") or "")),
        "stepGroups": groups,
        "category": category,
    }


def _repair_goal(goal: dict) -> dict | None:
    actions = [a for a in (_repair_action(a) for a in goal.get("actions") or [] if isinstance(a, dict)) if a]
    if not actions:
        return None
    title = trim_title(str(goal.get("title") or ""))
    text = str(goal.get("goal") or "")
    if not _GOAL.match(text):
        loose = _GOAL_LOOSE.match(text)
        text = build_goal(loose.group("topic"), loose.group("kind")) if loose else build_goal(title)
    score = goal.get("score")
    score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0
    score = round(min(1.0, max(0.0, score)), 2) if math.isfinite(score) else 0.0
    return {**goal, "goal": text, "title": title, "actions": actions, "score": score}


def _drop_at(contexts: list, loc: tuple, report: dict) -> bool:
    """Drop the action (or goal) a validation error points at. False when it cannot be located."""
    if len(loc) >= 2 and loc[0] == "contexts" and isinstance(loc[1], int) and loc[1] < len(contexts):
        goal = contexts[loc[1]]
        if (
            len(loc) >= 4
            and loc[2] == "actions"
            and isinstance(loc[3], int)
            and loc[3] < len(goal["actions"])
        ):
            del goal["actions"][loc[3]]
            report["dropped_actions"] += 1
            if not goal["actions"]:
                del contexts[loc[1]]
        else:
            report["dropped_actions"] += len(goal.get("actions") or [])
            del contexts[loc[1]]
        return True
    return False


def validate_with_report(body: dict) -> tuple[dict, dict]:
    """(valid body, {url_leaks, schema_valid, repairs, dropped_actions})."""
    report = {"url_leaks": 0, "schema_valid": True, "repairs": 0, "dropped_actions": 0}
    body = body if isinstance(body, dict) else {}
    contexts = _scrub_strings(copy.deepcopy(body.get("contexts") or []), report)
    contexts = [c for c in contexts if isinstance(c, dict)]
    try:
        ContextDeeplinkResponse.model_validate({"contexts": contexts})
        first_pass_ok = True
    except ValidationError:
        first_pass_ok = False
    repaired = [g for g in (_repair_goal(g) for g in contexts) if g]
    if repaired != contexts or not first_pass_ok:
        report["repairs"] = 1
    contexts = repaired
    for _ in range(_MAX_PASSES + sum(len(g["actions"]) for g in contexts)):
        try:
            ContextDeeplinkResponse.model_validate({"contexts": contexts})
            break
        except ValidationError as exc:
            report["schema_valid"] = False
            if not _drop_at(contexts, tuple(exc.errors()[0]["loc"]), report):
                contexts = []
                break
    else:
        contexts = []
    leaks_left = sum(has_leak(s) for s in _strings(contexts))
    if leaks_left:  # scrub is idempotent, so this should never happen; never ship a leak regardless
        contexts = []
    out = {key: value for key, value in body.items() if key != "contexts"}
    return {"contexts": contexts, **out}, report


def _strings(obj, key: str = ""):
    if isinstance(obj, str):
        if key != "deeplink":
            yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v, key)


def validate_response(body: dict) -> dict:
    body, _ = validate_with_report(body)
    return body
