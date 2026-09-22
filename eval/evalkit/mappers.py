"""Deeplink mappers the ablation compares against the engine's Screen Graph resolver.

    RulesMapper  Variant B, pure rules: physical-step keywords, then an exact leaf-name match on
                 the catalog's own screen names, then the entry whose type fits the verb
    Bm25Mapper   keyword retrieval alone (evalkit/bm25.py), top hit, always links
    LlmMapper    Baseline, full LLM mapping: the whole catalog in the prompt, the model picks an
                 id or says DUMMY / MANUAL

None of these import engine code; the hybrid and Screen Graph variants come from engine.py.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import httpx

from evalkit.bm25 import Bm25Index, Entry, load_entries
from evalkit.relevance import CATALOG, DUMMY, MANUAL, Prediction

# Which catalog entry type a step verb asks for, best first. Mirrors the design doc's polarity
# table (enable -> onURL, disable -> offURL, set -> updateURL, open -> onClickURL).
VERB_TYPES = {
    "enable": ("onURL", "onClickURL"),
    "disable": ("offURL", "onClickURL"),
    "set": ("updateURL", "onClickURL"),
    "adjust": ("updateURL", "onClickURL"),
    "open": ("onClickURL",),
    "check": ("onClickURL",),
}
DEFAULT_TYPES = ("onClickURL", "onURL", "updateURL", "offURL")

# A step whose verb or wording is one of these never opens a Settings screen.
PHYSICAL_VERBS = frozenset(
    ["restart", "reboot", "reset", "visit", "contact", "clean", "replace", "charge", "insert", "remove"]
)
PHYSICAL_PHRASES = (
    "service cent",
    "power button",
    "side key",
    "volume down",
    "charger",
    "cable",
    "soft cloth",
    "brush",
    "sim tray",
    "screen protector",
    "quick settings",
    "quick panel",
    "notification panel",
    "safe mode",
    "power off",
)

_WORD = re.compile(r"[a-z0-9]+")
_LEAD = ("view", "open", "enable", "disable", "turn on", "turn off", "set", "adjust", "change", "manage")
_TRAIL = ("settings", "setting", "page", "menu", "options", "screen")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def screen_name(text: str) -> str:
    """'View Touch Sensitivity Settings' -> 'touch sensitivity'; 'Touch sensitivity' -> same."""
    name = _norm(text)
    for lead in _LEAD:
        if name.startswith(lead + " "):
            name = name[len(lead) + 1 :]
            break
    for trail in _TRAIL:
        if name.endswith(" " + trail):
            name = name[: -len(trail) - 1]
            break
    return name


def leaf(screen_path: str) -> str:
    return screen_name(screen_path.split(">")[-1])


def is_physical(step: str, screen: str, verb: str | None) -> bool:
    if (verb or "").lower() in PHYSICAL_VERBS:
        return True
    text = f"{screen} {step}".lower()
    return any(phrase in text for phrase in PHYSICAL_PHRASES)


def pick_by_verb(candidates: list[Entry], verb: str | None) -> Entry | None:
    """The candidate whose entry type fits the verb; ties broken by catalog order."""
    preference = VERB_TYPES.get((verb or "").lower(), DEFAULT_TYPES)
    for wanted in preference:
        for entry in candidates:
            if entry.original_type == wanted:
                return entry
    return candidates[0] if candidates else None


class RulesMapper:
    name = "rules"

    def __init__(self, entries: list[Entry] | None = None):
        self.entries = entries or load_entries()
        self.by_name: dict[str, list[Entry]] = {}
        for entry in self.entries:
            self.by_name.setdefault(screen_name(entry.message), []).append(entry)

    def __call__(self, step: str, screen: str, verb: str | None) -> Prediction:
        if is_physical(step, screen, verb):
            return Prediction(MANUAL)
        name = leaf(screen)
        candidates = list(self.by_name.get(name, []))
        if not candidates and name:
            phrase = f" {name} "
            candidates = [e for e in self.entries if phrase in f" {_norm(e.description)} "]
        chosen = pick_by_verb(candidates, verb)
        if chosen:
            return Prediction(CATALOG, chosen.id)
        return Prediction(DUMMY if _norm(screen).startswith("settings") else MANUAL)


class Bm25Mapper:
    name = "bm25"

    def __init__(self, entries: list[Entry] | None = None):
        self.index = Bm25Index.build(entries or load_entries())

    def __call__(self, step: str, screen: str, verb: str | None) -> Prediction:
        hits = self.index.search(screen, verb or "", top_k=1)
        return Prediction(CATALOG, hits[0][0].id) if hits else Prediction(MANUAL)


@dataclass
class LlmUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    failures: int = 0


LLM_SYSTEM = (
    "You map one smartphone troubleshooting step to a Settings deeplink catalog. Reply with JSON "
    'only: {"id": "<catalog id>"} for the entry that opens exactly the screen the step needs, '
    '{"id": "DUMMY"} if the step opens a Settings screen that no entry covers, or '
    '{"id": "MANUAL"} if the step is physical, external, or not a Settings screen at all. '
    "For toggles, pick the entry whose type matches the step: onURL turns on, offURL turns off, "
    "updateURL changes a value, onClickURL just opens the screen."
)


class LlmMapper:
    """Mistral chat completions over httpx, temperature 0, JSON mode. Needs MISTRAL_API_KEY."""

    name = "llm"
    url = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, model: str, entries: list[Entry] | None = None, timeout_s: float = 30.0):
        entries = entries or load_entries()
        self.key = os.environ.get("MISTRAL_API_KEY", "")
        self.model = model
        self.ids = {e.id for e in entries}
        self.catalog = "\n".join(
            f"{e.id} | {e.original_type or '-'} | {e.message} | {e.description}" for e in entries
        )
        self.usage = LlmUsage()
        self.client = httpx.Client(timeout=timeout_s)

    @property
    def available(self) -> bool:
        return bool(self.key)

    def __call__(self, step: str, screen: str, verb: str | None) -> Prediction:
        user = f"Catalog (id | type | name | description):\n{self.catalog}\n\nStep: {step}\nScreen: {screen}"
        if verb:
            user += f"\nVerb: {verb}"
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": user}],
        }
        body = self._post(payload)
        if body is None:
            self.usage.failures += 1
            return Prediction(MANUAL)
        usage = body.get("usage") or {}
        self.usage.calls += 1
        self.usage.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.usage.completion_tokens += int(usage.get("completion_tokens") or 0)
        return parse_llm_choice(body["choices"][0]["message"]["content"], self.ids)

    def _post(self, payload: dict) -> dict | None:
        headers = {"Authorization": f"Bearer {self.key}"}
        for attempt in range(3):
            try:
                response = self.client.post(self.url, json=payload, headers=headers)
            except httpx.HTTPError:
                response = None
            if response is not None and response.status_code == 200:
                return response.json()
            if response is not None and response.status_code not in (429, 500, 502, 503):
                return None
            time.sleep(2**attempt)
        return None


def parse_llm_choice(content: str, ids: set[str]) -> Prediction:
    """An id the catalog does not contain is a hallucination and counts as no link."""
    try:
        choice = str(json.loads(content).get("id", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        return Prediction(MANUAL)
    upper = choice.upper()
    if upper == "DUMMY":
        return Prediction(DUMMY)
    if upper == "MANUAL" or choice not in ids:
        return Prediction(MANUAL)
    return Prediction(CATALOG, choice)
