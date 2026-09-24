"""Component 10: DraftActions -> official Goal objects; score = 0.4 relevance + 0.3 grounding + 0.3 link coverage.

Every scored string is built here by rule; the LLM's drafts are only raw material. Only `auto` actions
carry an actionable deeplink (fixture assumption 1), catalog values are copied verbatim, and the dummy
placeholder gets our own 5-7 word text naming the screen.
"""

import re

from app.compiler import catalog
from app.compiler.scrub import scrub
from app.compiler.templates import build_goal, title_case_word
from app.compiler.trimmer import trim_description, trim_title
from app.models import DraftAction, Intent, LinkTier

SCORE_FORMULA = "0.4*relevance + 0.3*grounding_coverage + 0.3*link_coverage"
_LINK_VALUE = {LinkTier.catalog: 1.0, LinkTier.dummy: 0.5}

_STEP_NUMBERING = re.compile(r"^\s*(?:\d+\s*[.):-]|step\s*\d+\s*[:.)-]?|[-*•])\s*", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_VERB_WORD = {
    "enable": "Enable",
    "disable": "Disable",
    "open": "Open",
    "set": "Adjust",
    "adjust": "Adjust",
    "check": "Check",
    "restart": "Restart",
    "reset": "Reset",
    "visit": "Visit",
}
_VERB_DESCRIPTION = {
    "enable": "turn on",
    "disable": "turn off",
    "open": "open",
    "set": "adjust",
    "adjust": "adjust",
    "check": "check",
    "restart": "restart",
    "reset": "reset",
    "visit": "visit",
}


# ---- strings ----------------------------------------------------------------------------------------
def clean_step(text: str) -> str:
    """Imperative sentence, no numbering, trailing period."""
    step = _WHITESPACE.sub(" ", scrub(text or "")).strip()
    step = _STEP_NUMBERING.sub("", step).strip().rstrip(",;:").strip()
    if not step:
        return ""
    step = step[0].upper() + step[1:]
    if step[-1] in "!?":
        step = step[:-1].rstrip()
    return step if step.endswith(".") else f"{step}."


def action_name(name: str) -> str:
    """Title Case: every word capitalised, the rest of each word left as written ("Wi-Fi")."""
    words = [w for w in _WHITESPACE.split(scrub(name or "").strip(" .:;,")) if w]
    return " ".join(title_case_word(w) for w in words)


def screen_leaf(screen_path: str | None) -> tuple[str, str | None]:
    """("Storage", "Email app") from "Settings > Apps > Email app > Storage"."""
    parts = [p.strip() for p in re.split(r">|/", screen_path or "") if p.strip()]
    if not parts:
        return "", None
    parent = parts[-2] if len(parts) >= 2 and parts[-2].lower() != "settings" else None
    return parts[-1], parent


def _default_name(action: DraftAction) -> str:
    leaf, _ = screen_leaf(action.screen_path)
    verb = _VERB_WORD.get((action.intent_verb or "").lower())
    if leaf and verb:
        return f"{verb} {leaf}"
    first = action.steps[0].text if action.steps else "Follow Steps"
    return " ".join(first.rstrip(".").split()[:4])


def _default_description(action: DraftAction) -> str:
    leaf, _ = screen_leaf(action.screen_path)
    verb = _VERB_DESCRIPTION.get((action.intent_verb or "").lower(), "resolve")
    return f"It will {verb} {leaf.lower() or 'this issue'}"


def dummy_link(screen_path: str | None) -> dict:
    """`bixby://dummy_positive` with our own description and message, 5-7 words each, naming the screen."""
    leaf, parent = screen_leaf(screen_path)
    words = [w for w in f"{parent or ''} {leaf}".lower().split() if w != "settings"]
    if len(words) > 3:
        words = [w for w in leaf.lower().split() if w != "settings"] or words
    words = words[-3:] or ["relevant"]
    phrase = " ".join(words)
    message = f"Open {phrase} screen"
    if len(message.split()) < 5:
        message = f"Open the {phrase} settings screen"
    return {
        "deeplink": catalog.DUMMY_URI,
        "description": f"Opens the {phrase} settings page",
        "message": message,
        "originalType": "placeholder",
    }


def _links(action: DraftAction) -> tuple[dict | None, dict | None]:
    """(actionableDeeplink, validationDeeplink) for one action."""
    link = action.link
    if action.category != "auto" or link is None:
        return None, None
    if link.tier is LinkTier.catalog and link.entry_id:
        act = catalog.actionable(link.entry_id)
        return (act, catalog.validation(link.entry_id)) if act else (None, None)
    if link.tier is LinkTier.dummy:
        return dummy_link(action.screen_path), None
    return None, None


# ---- actions ----------------------------------------------------------------------------------------
def _merge_key(action: DraftAction) -> tuple | None:
    """Same screen and same entry: one action. Opposite toggles stay apart (fixture assumption 2)."""
    link = action.link
    if action.category != "auto" or not action.screen_path or link is None or link.tier is LinkTier.manual:
        return None
    return action.screen_path.strip().lower(), link.entry_id


def _build_action(action: DraftAction, steps: list[str]) -> dict:
    act, val = _links(action)
    category = action.category if (action.category != "auto" or act) else "manual"
    return {
        "actionName": action_name(action.name or _default_name(action)),
        "description": trim_description(action.description or _default_description(action)),
        "stepGroups": [{"steps": steps, "actionableDeeplink": act, "validationDeeplink": val}],
        "category": category,
    }


def _compile_actions(actions: list[DraftAction], report: dict) -> list[dict]:
    out: list[dict] = []
    by_merge_key: dict[tuple, dict] = {}
    by_name: dict[str, dict] = {}
    for action in actions:
        steps = []
        for step in action.steps:
            text = clean_step(step.text)
            if text and text not in steps:
                steps.append(text)
        if not steps:
            continue
        built = _build_action(action, steps)
        key = _merge_key(action)
        name_key = built["actionName"].lower()
        target = by_merge_key.get(key) if key else None
        if target is None and name_key in by_name and by_name[name_key]["category"] == built["category"]:
            target = by_name[name_key]
        if target is not None:
            group = target["stepGroups"][0]
            group["steps"] += [s for s in steps if s not in group["steps"]]
            report["merged"].append({"action": built["actionName"], "into": target["actionName"]})
            continue
        if name_key in by_name:  # same name, different kind of action: keep both, tell them apart
            n = 2
            while f"{name_key} {n}" in by_name:
                n += 1
            built["actionName"] = f"{built['actionName']} {n}"
            name_key = built["actionName"].lower()
        out.append(built)
        by_name[name_key] = built
        if key:
            by_merge_key[key] = built
    return out


# ---- goals ------------------------------------------------------------------------------------------
def goal_score(relevance: float, grounding: float, link: float, cap: float | None = None) -> float:
    score = 0.4 * relevance + 0.3 * grounding + 0.3 * link
    score = min(1.0, max(0.0, score))
    if cap is not None:
        score = min(score, cap)
    return round(score, 2)


def _link_coverage(actions: list[DraftAction]) -> float:
    values = [
        _LINK_VALUE[a.link.tier]
        for a in actions
        if a.category == "auto" and a.link is not None and a.link.tier in _LINK_VALUE
    ]
    return round(sum(values) / len(values), 2) if values else 1.0


def compile_with_report(
    intents: list[Intent],
    actions: list[DraftAction],
    *,
    topics: list[str] | None = None,
    coverage: dict[int, float] | None = None,
    score_cap: float | None = None,
    kind: str = "Troubleshooting",
) -> tuple[list[dict], dict]:
    """Goals (most relevant first) plus the numbers the stream's `compile` event shows."""
    report: dict = {"score_inputs": [], "score_formula": SCORE_FORMULA, "merged": [], "dropped_intents": []}
    goals: list[dict] = []
    for index, intent in enumerate(intents):
        mine = [a for a in actions if a.intent_index == index]
        compiled = _compile_actions(mine, report)
        if not compiled:
            report["dropped_intents"].append(index)
            continue
        title = trim_title(intent.title or intent.text)
        topic = (topics[index] if topics and index < len(topics) and topics[index] else None) or title
        grounding = (coverage or {}).get(index, 1.0)
        link = _link_coverage(mine)
        score = goal_score(intent.relevance, grounding, link, score_cap)
        goals.append({"goal": build_goal(topic, kind), "title": title, "actions": compiled, "score": score})
        report["score_inputs"].append(
            {
                "intent_index": index,
                "title": title,
                "score": score,
                "relevance": round(intent.relevance, 2),
                "grounding_coverage": round(grounding, 2),
                "link_coverage": link,
            }
        )
    order = sorted(range(len(goals)), key=lambda i: -goals[i]["score"])
    goals = [goals[i] for i in order]
    report["score_inputs"] = [report["score_inputs"][i] for i in order]
    report["goals"] = len(goals)
    report["actions"] = sum(len(g["actions"]) for g in goals)
    return goals, report


def compile_goals(intents: list[Intent], actions: list[DraftAction], **options) -> list[dict]:
    """Goal dicts in the official schema's shape; see compile_with_report for the options."""
    goals, _ = compile_with_report(intents, actions, **options)
    return goals
