"""Component 5: Stage 2 LLM call — actions with cited steps, screen_path, verb, draft description.

Two extractors with one output shape:
  llm    call B through the router (JSON-schema output); every step cites sentence ids.
  rules  no LLM: instruction sentences of the relevant sections become steps citing themselves.
         Grounded by construction. Used when the LLM path fails or runs out of budget, and in CI
         (no keys), so every SIIS request still gets a non-empty, honest answer.
"""

import re

from app.models import DraftAction, DraftStep, Intent, SiisSentence
from app.pipeline.text import word_block

# ---- rules-only extractor ---------------------------------------------------------------------------
_IMPERATIVE_VERBS = word_block(
    """
    tap touch go open press hold select swipe turn check restart reboot back update clear uninstall
    reinstall remove try make ensure connect disconnect disable enable charge contact visit navigate
    launch find choose enter reset install use wait clean inspect follow drag slide adjust set
    toggle sign delete verify confirm perform keep plug unplug insert take bring request switch move
    allow close force download log add power scroll long run start stop edit change look
    """
)
_LEAD_IN = re.compile(
    r"^(?:first|then|next|finally|also|now|alternatively|afterwards|additionally|lastly|once done|"
    r"to do (?:this|so)|if so|in that case|"
    r"if (?:this|that|it) (?:doesn't|does not|didn't|did not) (?:help|work)|if needed|if necessary)\s*,?\s*",
    re.IGNORECASE,
)
_CONDITION = re.compile(
    r"^(?:if|when|to|once|after|before|while)\b[^,]{3,120},\s*(?P<rest>.+)$", re.IGNORECASE
)
_HEADING_NUMBER = re.compile(r"^\s*(?:step\s*\d+\s*[:.)-]|\d+\s*[.):-])\s*", re.IGNORECASE)
_SETTINGS_ENTRY = re.compile(
    r"\b(?:go to|open|navigate to|launch|access)\s+(?:the\s+)?settings\b", re.IGNORECASE
)
_TAP_TARGET = re.compile(
    r"\b(?i:tap|touch)\s+(?i:on\s+)?(?i:the\s+)?(?P<target>[A-Z0-9][\w'’\-]*(?:\s+(?:[A-Za-z0-9][\w'’\-]*))*?)"
    r"(?=\s*(?:,|\.|;|$|\band then\b|\bthen\b|\bto\b|\bif\b|\bfrom\b|\bin\b|\bunder\b|\bon\b|\bor\b))"
)
# Last taps that press a button rather than open a screen: not part of the screen path.
_BUTTON_WORDS = re.compile(
    r"^(?:ok|done|save|apply|confirm|clear|delete|remove|reset|restart|uninstall|force stop|turn|"
    r"power off|allow|install|update|download|back up|restore|start|stop|sign|next|continue|"
    r"yes|no)\b",
    re.IGNORECASE,
)
_ENABLE = re.compile(r"\b(?:turn on|enable|switch on|activate)\b", re.IGNORECASE)
_DISABLE = re.compile(r"\b(?:turn off|disable|switch off|deactivate)\b", re.IGNORECASE)
_RESTART = re.compile(r"\b(?:restart|reboot)\b", re.IGNORECASE)
_RESET = re.compile(r"\bfactory (?:data )?reset\b|\breset\b", re.IGNORECASE)
_VISIT = re.compile(r"\bservice cent|\bcontact\b|\bvisit\b|samsung support", re.IGNORECASE)


def instruction(text: str) -> str | None:
    """The sentence as a step when it instructs the reader, else None ("Note: data is lost." is not)."""
    body = _LEAD_IN.sub("", text.strip())
    first = re.match(r"[A-Za-z']+", body)
    if first and first.group(0).lower() in _IMPERATIVE_VERBS:
        return body[0].upper() + body[1:]
    conditional = _CONDITION.match(body)
    if conditional:
        rest = _LEAD_IN.sub("", conditional.group("rest"))
        verb = re.match(r"[A-Za-z']+", rest)
        if verb and verb.group(0).lower() in _IMPERATIVE_VERBS:
            return body[0].upper() + body[1:]
    return None


_CLAUSE_SPLIT = re.compile(r",\s*(?:and\s+)?(?:then\s+)?|;\s*|\s+and then\s+", re.IGNORECASE)


def sentence_steps(text: str) -> list[str]:
    """Steps from one article sentence: "Go to Settings, tap Display, and then tap Navigation bar."
    becomes three steps. Split only when every part is itself an instruction, so a conditional
    ("If X, tap Y") or a purpose clause ("To confirm..., try...") stays whole. [] when the sentence
    does not instruct at all."""
    body = instruction(text)
    if body is None:
        return []
    parts = [p.strip(" .") for p in _CLAUSE_SPLIT.split(body.rstrip(".")) if p.strip(" .")]
    if len(parts) > 1 and all(_starts_with_verb(p) for p in parts):
        return [f"{p[0].upper()}{p[1:]}." for p in parts]
    return [body]


def _starts_with_verb(text: str) -> bool:
    first = re.match(r"[A-Za-z']+", _LEAD_IN.sub("", text))
    return bool(first) and first.group(0).lower() in _IMPERATIVE_VERBS


def screen_path(steps: list[str]) -> str | None:
    """`Settings > A > B` from "Go to Settings, tap A, then tap B." across an action's steps."""
    joined = " ".join(steps)
    entry = _SETTINGS_ENTRY.search(joined)
    if not entry:
        return None
    targets = []
    for match in _TAP_TARGET.finditer(joined[entry.end() :]):
        target = match.group("target").strip(" '’")
        if target and target.lower() != "settings" and target not in targets:
            targets.append(target)
    while targets and _BUTTON_WORDS.match(targets[-1]):
        targets.pop()
    return " > ".join(["Settings", *targets[:4]])


def intent_verb(steps: list[str], path: str | None) -> str | None:
    text = " ".join(steps)
    for verb, pattern in (("enable", _ENABLE), ("disable", _DISABLE)):
        if pattern.search(text) and path:
            return verb
    if _RESTART.search(text):
        return "restart"
    if _RESET.search(text):
        return "reset"
    if _VISIT.search(text) and not path:
        return "visit"
    return "open" if path else None


def heading_name(heading: str) -> str:
    return _HEADING_NUMBER.sub("", heading or "").strip(" :.-")


def rules_description(name: str) -> str:
    """ "It will ..." from a section heading: "Check Email Access" -> "It will check email access"."""
    words = name.split()
    if not words:
        return "It will help resolve this issue"
    first = words[0].lower()
    if first in _IMPERATIVE_VERBS:
        return f"It will {name[0].lower()}{name[1:]}"
    if first.endswith("ing"):
        return f"It will help with {name.lower()}"
    return f"It will address {name.lower()}"


def _topic(intent: Intent) -> str:
    return intent.title or " ".join(intent.text.split()[:3])


def extract_rules(
    intents: list[Intent], sentences: list[SiisSentence], sections: list[dict] | None = None
) -> tuple[list[DraftAction], list[str]]:
    """(actions, topics). One action per relevant section with at least one instruction sentence.

    When the relevant sections hold no instructions (a misspelt complaint can rank only the intro as
    relevant), the whole article is used: its instructions are still the only grounded answer.
    """
    table = sections or _sections_from_sentences(sentences, len(intents))
    relevant = [s for s in table if s.get("relevant")]
    actions = _rules_actions(intents, sentences, relevant) if relevant else []
    if not actions:
        actions = _rules_actions(intents, sentences, table)
    return actions, [_topic(i) for i in intents]


def _rules_actions(
    intents: list[Intent], sentences: list[SiisSentence], table: list[dict]
) -> list[DraftAction]:
    by_id = {s.id: s for s in sentences}
    actions: list[DraftAction] = []
    for section in table:
        scores = section.get("relevance") or [0.0]
        intent_index = max(range(len(scores)), key=lambda i: scores[i]) if intents else 0
        steps = []
        for sid in section["sentence_ids"]:
            steps += [DraftStep(text=t, src_ids=[sid]) for t in sentence_steps(by_id[sid].text)]
        if not steps:
            continue
        texts = [s.text for s in steps]
        path = screen_path(texts)
        verb = intent_verb(texts, path)
        name = heading_name(section["heading"]) or " ".join(texts[0].split()[:4])
        actions.append(
            DraftAction(
                steps=steps,
                screen_path=path,
                intent_verb=verb,
                name=name,
                description=rules_description(name),
                intent_index=min(intent_index, max(len(intents) - 1, 0)),
            )
        )
    return actions


def _sections_from_sentences(sentences: list[SiisSentence], n_intents: int) -> list[dict]:
    table: list[dict] = []
    for s in sentences:
        if not table or table[-1]["heading"] != s.section:
            table.append(
                {"heading": s.section, "sentence_ids": [], "relevance": [s.relevance] * max(n_intents, 1)}
            )
        table[-1]["sentence_ids"].append(s.id)
    for row in table:
        row["relevant"] = True
    return table


# ---- LLM path (call B) ------------------------------------------------------------------------------
VERBS = ["open", "enable", "disable", "set", "check", "restart", "reset", "visit", "none"]
SCHEMA = {
    "type": "object",
    "properties": {
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "intent_index": {"type": "integer"},
                    "topic": {"type": "string"},
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "screen_path": {"type": "string"},
                                "intent_verb": {"type": "string", "enum": VERBS},
                                "category_hint": {"type": "string", "enum": ["auto", "manual", "critical"]},
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "src_ids": {"type": "array", "items": {"type": "string"}},
                                        },
                                        "required": ["text", "src_ids"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": [
                                "name",
                                "description",
                                "screen_path",
                                "intent_verb",
                                "category_hint",
                                "steps",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["intent_index", "topic", "actions"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["goals"],
    "additionalProperties": False,
}
_SRC_ID = re.compile(r"^S\d+$")


def format_intents(intents: list[Intent]) -> str:
    return "\n".join(f"{n}. {i.text} (title: {i.title or '-'})" for n, i in enumerate(intents))


def format_sentences(sentences: list[SiisSentence], sections: list[dict] | None) -> str:
    """Sentences grouped under their section headings; sections off-topic for every intent are marked."""
    relevant = {s["heading"]: s.get("relevant", True) for s in sections or []}
    lines: list[str] = []
    current = None
    for sentence in sentences:
        if sentence.section != current:
            current = sentence.section
            mark = "" if relevant.get(current, True) else "  [probably unrelated to the complaint]"
            lines.append(f"\n### {current or 'Article'}{mark}")
        lines.append(f"[{sentence.id}] {sentence.text}")
    return "\n".join(lines).strip()


def actions_from_answer(
    answer: dict, intents: list[Intent], sentences: list[SiisSentence]
) -> tuple[list[DraftAction], list[str]]:
    """DraftActions from call B's JSON. Unknown sentence ids are dropped here; grounding judges the rest."""
    known = {s.id for s in sentences}
    topics = [_topic(i) for i in intents]
    actions: list[DraftAction] = []
    for goal in answer.get("goals") or []:
        index = goal.get("intent_index")
        index = index if isinstance(index, int) and 0 <= index < len(intents) else 0
        topic = " ".join(str(goal.get("topic") or "").split())
        if topic:
            topics[index] = topic
        for raw in goal.get("actions") or []:
            steps = []
            for step in raw.get("steps") or []:
                text = " ".join(str(step.get("text") or "").split())
                ids = [
                    i
                    for i in step.get("src_ids") or []
                    if isinstance(i, str) and _SRC_ID.match(i) and i in known
                ]
                if text:
                    steps.append(DraftStep(text=text, src_ids=ids))
            if not steps:
                continue
            verb = raw.get("intent_verb") if raw.get("intent_verb") in VERBS else "none"
            path = " ".join(str(raw.get("screen_path") or "").split()) or None
            actions.append(
                DraftAction(
                    steps=steps,
                    screen_path=path,
                    intent_verb=None if verb == "none" else verb,
                    name=" ".join(str(raw.get("name") or "").split()) or None,
                    description=" ".join(str(raw.get("description") or "").split()) or None,
                    intent_index=index,
                )
            )
    return actions, topics


def extract_llm(
    query: str, intents: list[Intent], sentences: list[SiisSentence], sections: list[dict] | None
) -> tuple[list[DraftAction], list[str], dict]:
    from app.llm.router import complete_json

    info: dict = {}
    variables = {
        "query": query,
        "intents": format_intents(intents),
        "sentences": format_sentences(sentences, sections),
    }
    answer = complete_json("extract", variables, SCHEMA, stage="extract", info=info)
    actions, topics = actions_from_answer(answer, intents, sentences)
    if not actions:
        raise ValueError("the extraction answer has no usable action")
    keys = ("model", "tokens_in", "tokens_out", "cost_usd", "attempts", "prompt")
    return actions, topics, {"source": "llm", **{k: info.get(k) for k in keys}}


# ---- LLM path, select mode (prompt extract.v2) --------------------------------------------------------
DOMAINS = ["Battery", "Display", "Camera", "Performance", "Other"]
SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "goals": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string"},
                    "title": {"type": "string"},
                    "topic": {"type": "string"},
                    "domain": {"type": "string", "enum": DOMAINS},
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "src_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "screen_path": {"type": "string"},
                                "intent_verb": {"type": "string", "enum": VERBS},
                            },
                            "required": ["src_ids", "name", "description", "screen_path", "intent_verb"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["problem", "title", "topic", "domain", "actions"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["goals"],
    "additionalProperties": False,
}


def usable_selection(answer: dict, known: set[str]) -> bool:
    """The router's accept check: at least one action that cites a real sentence."""
    return any(
        any(isinstance(i, str) and i in known for i in action.get("src_ids") or [])
        for goal in answer.get("goals") or []
        for action in goal.get("actions") or []
    )


def actions_from_selection(
    answer: dict, sentences: list[SiisSentence]
) -> tuple[list[DraftAction], list[str], list[Intent]]:
    """(actions, topics, intents) from a select-mode answer. Steps are the chosen sentences' own words
    (instruction sentences only, multi-instruction sentences split), each citing its sentence."""
    from app.compiler.trimmer import trim_title

    by_id = {s.id: s for s in sentences}
    actions: list[DraftAction] = []
    topics: list[str] = []
    intents: list[Intent] = []
    for goal in (answer.get("goals") or [])[:3]:
        goal_actions = []
        for raw in goal.get("actions") or []:
            ids = [i for i in raw.get("src_ids") or [] if isinstance(i, str) and i in by_id]
            steps: list[DraftStep] = []
            for sid in dict.fromkeys(ids):
                steps += [DraftStep(text=t, src_ids=[sid]) for t in sentence_steps(by_id[sid].text)]
            if not steps:  # the model chose only non-instruction sentences: keep them as written
                steps = [DraftStep(text=by_id[sid].text, src_ids=[sid]) for sid in dict.fromkeys(ids)]
            if not steps:
                continue
            verb = raw.get("intent_verb") if raw.get("intent_verb") in VERBS else "none"
            goal_actions.append(
                DraftAction(
                    steps=steps,
                    screen_path=" ".join(str(raw.get("screen_path") or "").split()) or None,
                    intent_verb=None if verb == "none" else verb,
                    name=" ".join(str(raw.get("name") or "").split()) or None,
                    description=" ".join(str(raw.get("description") or "").split()) or None,
                    intent_index=len(intents),
                )
            )
        if not goal_actions:
            continue
        problem = " ".join(str(goal.get("problem") or "").split())
        title = trim_title(str(goal.get("title") or problem))
        domain = goal.get("domain") if goal.get("domain") in DOMAINS else "Other"
        intents.append(Intent(text=problem or title, title=title, domain=domain))
        topics.append(" ".join(str(goal.get("topic") or "").split()) or title)
        actions += goal_actions
    return actions, topics, intents


def extract_select_llm(
    query: str, sentences: list[SiisSentence], sections: list[dict] | None
) -> tuple[list[DraftAction], list[str], dict]:
    from app.llm.router import complete_json

    known = {s.id for s in sentences}
    info: dict = {}
    variables = {"query": query, "sentences": format_sentences(sentences, sections)}
    answer = complete_json(
        "extract",
        variables,
        SELECT_SCHEMA,
        stage="extract",
        info=info,
        accept=lambda a: usable_selection(a, known),
    )
    actions, topics, intents = actions_from_selection(answer, sentences)
    if not actions:
        raise ValueError("the extraction answer has no usable action")
    keys = ("model", "tokens_in", "tokens_out", "cost_usd", "attempts", "prompt")
    detail = {"source": "llm", "mode": "select", "intents": [i.model_dump(mode="json") for i in intents]}
    return actions, topics, detail | {k: info.get(k) for k in keys}


# ---- entry point ------------------------------------------------------------------------------------
def extract_with_topics(
    intents: list[Intent],
    sentences: list[SiisSentence],
    *,
    sections: list[dict] | None = None,
    query: str | None = None,
) -> tuple[list[DraftAction], list[str], dict]:
    """(actions, topics, info); info says which extractor ran: {source, model, tokens_in, tokens_out, ...}.

    LLM first when a key is set; rules-only when it fails, returns nothing usable, or no key is set.
    Select mode also returns the intents it found in `info["intents"]` (the caller adopts them).
    """
    from app.config import settings
    from app.llm.router import available

    rules_info = {"source": "rules", "model": None, "tokens_in": 0, "tokens_out": 0}
    if available() and sentences:
        text = query or (intents[0].text if intents else "")
        try:
            if settings.extract_mode == "select":
                return extract_select_llm(text, sentences, sections)
            return extract_llm(text, intents, sentences, sections)
        except Exception as exc:  # noqa: BLE001 - LLMError or a malformed answer: degrade to rules
            actions, topics = extract_rules(intents, sentences, sections)
            rules_info["degraded"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            rules_info["attempts"] = getattr(exc, "attempts", [])
            return actions, topics, rules_info
    actions, topics = extract_rules(intents, sentences, sections)
    return actions, topics, rules_info


def extract(intents: list[Intent], sentences: list[SiisSentence]) -> list[DraftAction]:
    actions, _, _ = extract_with_topics(intents, sentences)
    return actions
