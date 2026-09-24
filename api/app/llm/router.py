"""ADR-002: Gemini primary, Mistral fallback on timeout/error; records tokens and cost.

    complete_json("extract", {"query": ..., "sentences": ...}, SCHEMA, stage="extract", info=info)

Per stage (settings): a primary model with its own timeout inside the stage budget, optionally a
fast model raced against it, then one fallback attempt while budget remains.

    race      primary and fast model start together; the primary's answer wins if it is back (and
              usable) by the prefer deadline, otherwise the first usable answer is taken
    accept    a caller check on the parsed JSON; a rejected answer counts as a failure
    cooldown  a model that answers 429/5xx is skipped for settings.llm_cooldown_s, so a busy free
              tier costs one wasted call a minute instead of one per request

If nothing usable comes back the caller gets LLMError and degrades (template variations, rules-only
extraction). Prompts are versioned files, prompts/<name>.<version>.md with {{placeholders}}; versions
come from settings.prompt_versions, and settings.prompt_version is part of the cache key.
"""

import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from app.config import settings
from app.llm import gemini, mistral
from app.llm.errors import LLMCallError

PROMPTS = Path(__file__).resolve().parent / "prompts"
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_MIN_FALLBACK_S = 0.5  # less than this left in the budget: not worth starting the fallback
_COOLDOWN_KINDS = ("http_429", "http_5")  # capacity errors; a slow answer (timeout) is not one
_cooldown_until: dict[str, float] = {}
_cooldown_lock = threading.Lock()

# Local runs read the repo-root .env; in Docker the variables come from the environment itself.
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)


class LLMError(Exception):
    """No provider produced a usable answer for this call; `attempts` says how each one failed."""

    def __init__(self, attempts: list[dict]):
        super().__init__("; ".join(f"{a['model']}: {a['error']}" for a in attempts) or "no provider")
        self.attempts = attempts


def available() -> bool:
    """True when at least one provider has a key: the LLM path is live, rules are only a fallback."""
    return gemini.available() or mistral.available()


@lru_cache(maxsize=16)
def load_prompt(name: str, version: str) -> str:
    return (PROMPTS / f"{name}.{version}.md").read_text(encoding="utf-8")


def prompt_version(name: str) -> str:
    return settings.prompt_versions.get(name, settings.prompt_version)


def render(template: str, variables: dict) -> str:
    return _PLACEHOLDER.sub(lambda m: str(variables.get(m.group(1), m.group(0))), template)


class _Stage:
    """What one call may use: models, thinking level, timeouts and output budget."""

    def __init__(self, stage: str):
        self.fast_model: str | None = None
        self.prefer_deadline: float | None = None
        if stage == "enrich":
            self.model, self.thinking = settings.enrich_model, settings.enrich_thinking
            self.primary_timeout, self.budget = settings.enrich_primary_timeout_s, settings.enrich_budget_s
            self.max_tokens = settings.enrich_max_tokens
        elif stage == "variations":
            self.model, self.thinking = settings.variations_model, None
            self.primary_timeout = self.budget = settings.variations_budget_s
            self.max_tokens = settings.variations_max_tokens
        else:
            self.model, self.thinking = settings.extract_model, settings.extract_thinking
            self.primary_timeout, self.budget = settings.extract_primary_timeout_s, settings.extract_budget_s
            self.max_tokens = settings.extract_max_tokens
            if settings.extract_fast_model and settings.extract_fast_model != self.model:
                self.fast_model = settings.extract_fast_model
                self.prefer_deadline = settings.extract_prefer_deadline_s


def provider(model: str) -> str:
    """gemini-* goes to Google; every other id (mistral-, ministral-, magistral-...) to Mistral."""
    return "gemini" if model.startswith("gemini") else "mistral"


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = settings.llm_prices.get(model, (0.0, 0.0))
    return round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 6)


def cooling_down(model: str) -> bool:
    with _cooldown_lock:
        return _cooldown_until.get(model, 0.0) > time.monotonic()


def _cool(model: str, kind: str) -> None:
    if kind.startswith(_COOLDOWN_KINDS):
        with _cooldown_lock:
            _cooldown_until[model] = time.monotonic() + settings.llm_cooldown_s


def reset_cooldowns() -> None:
    with _cooldown_lock:
        _cooldown_until.clear()


def complete_json(
    prompt_name: str,
    variables: dict,
    schema: dict,
    *,
    stage: str | None = None,
    info: dict | None = None,
    clients: dict | None = None,
    accept: Callable[[dict], bool] | None = None,
) -> dict:
    """The parsed JSON answer. Fills `info` with model, tokens, cost and the attempts made.

    `clients` ({"gemini": httpx.Client, "mistral": httpx.Client}) is for tests and pooling.
    """
    plan = _Stage(stage or prompt_name)
    version = prompt_version(prompt_name)
    prompt = render(load_prompt(prompt_name, version), variables)
    clients = clients or {}
    started = time.perf_counter()
    attempts: list[dict] = []
    lock = threading.Lock()

    def attempt(model: str, timeout: float) -> dict | None:
        t = time.perf_counter()
        error, result = None, None
        if cooling_down(model):
            error = "cooldown"
        else:
            try:
                if provider(model) == "gemini":
                    result = gemini.call(
                        prompt,
                        schema,
                        model=model,
                        thinking=plan.thinking,
                        timeout=timeout,
                        max_tokens=plan.max_tokens,
                        client=clients.get("gemini"),
                    )
                else:
                    result = mistral.call(
                        prompt,
                        schema,
                        model=model,
                        timeout=timeout,
                        max_tokens=plan.max_tokens,
                        client=clients.get("mistral"),
                    )
                if accept is not None and not accept(result["json"]):
                    error, result = "rejected", None
            except LLMCallError as exc:
                error = exc.kind
                _cool(model, exc.kind)
        with lock:
            attempts.append({"model": model, "ok": error is None, "error": error, "ms": _ms(t)})
        return result

    if plan.fast_model:
        result = _race(plan, attempt, started)
    else:
        result = attempt(plan.model, min(plan.primary_timeout, plan.budget))
    if result is None:
        remaining = plan.budget - (time.perf_counter() - started)
        if remaining >= _MIN_FALLBACK_S and settings.fallback_model not in (plan.model, plan.fast_model):
            result = attempt(settings.fallback_model, remaining)
    if info is not None:
        with lock:
            info["attempts"] = list(attempts)
        info["prompt"] = f"{prompt_name}.{version}"
    if result is None:
        with lock:
            raise LLMError(list(attempts))
    if info is not None:
        info.update(
            model=result["model"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            cost_usd=cost_usd(result["model"], result["tokens_in"], result["tokens_out"]),
            fallback_used=result["model"] not in (plan.model, plan.fast_model),
        )
    return result["json"]


def _race(plan: _Stage, attempt: Callable[[str, float], dict | None], started: float) -> dict | None:
    """Primary and fast model together; the primary wins if it is usable by the prefer deadline."""
    timeout = min(plan.primary_timeout, plan.budget)
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm-race")
    primary: Future = pool.submit(attempt, plan.model, timeout)
    fast: Future = pool.submit(attempt, plan.fast_model, timeout)
    pool.shutdown(wait=False)  # a losing call finishes on its own, bounded by its timeout
    prefer_until = started + (plan.prefer_deadline or plan.budget)
    give_up_at = started + plan.budget
    while True:
        if primary.done() and primary.result() is not None:
            return primary.result()
        fast_ok = fast.done() and fast.result() is not None
        primary_failed = primary.done() and primary.result() is None
        now = time.perf_counter()
        if fast_ok and (now >= prefer_until or primary_failed):
            return fast.result()
        if primary.done() and fast.done():
            return None
        if now >= give_up_at:
            return None  # stage budget spent; whatever is still running is ignored
        until = prefer_until if fast_ok else give_up_at
        pending = [f for f in (primary, fast) if not f.done()]
        wait(pending, timeout=max(0.01, until - now), return_when=FIRST_COMPLETED)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
