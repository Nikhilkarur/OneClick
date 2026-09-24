"""Mistral client (JSON output).

Plain REST over httpx (chat completions) with a strict JSON-schema response format. The fallback model
is open-weight (Mistral Small 4, Apache 2.0) served by Mistral's API. Raises LLMCallError on anything
that is not a usable JSON answer.
"""

import os

import httpx

from app.config import settings
from app.llm.errors import LLMCallError, parse_json

API = "https://api.mistral.ai/v1/chat/completions"
KEY_ENV = "MISTRAL_API_KEY"


def available() -> bool:
    return bool(os.getenv(KEY_ENV))


def _body(prompt: str, schema: dict, model: str, max_tokens: int, optional: bool) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": settings.llm_temperature_mistral,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema, "strict": True},
        },
    }
    if optional and settings.fallback_reasoning:
        body["reasoning_effort"] = settings.fallback_reasoning
    return body


def _text(content) -> str:
    """Message content is a string, or a list of chunks when the model reasons first."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


def call(
    prompt: str,
    schema: dict,
    *,
    model: str | None = None,
    timeout: float | None = None,
    max_tokens: int = 4096,
    client: httpx.Client | None = None,
) -> dict:
    """{"json": parsed answer, "model", "tokens_in", "tokens_out"}. Raises LLMCallError."""
    key = os.getenv(KEY_ENV)
    if not key:
        raise LLMCallError("no_key", f"{KEY_ENV} is not set")
    model = model or settings.fallback_model
    http = client or httpx.Client()
    try:
        response = None
        for optional in (True, False):  # retry once without reasoning_effort if the API rejects it
            response = http.post(
                API,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=_body(prompt, schema, model, max_tokens, optional),
                timeout=timeout or settings.llm_timeout_default_s,
            )
            if response.status_code != 400:
                break
    except httpx.TimeoutException as exc:
        raise LLMCallError("timeout", str(exc)) from exc
    except httpx.HTTPError as exc:
        raise LLMCallError("transport", str(exc)) from exc
    finally:
        if client is None:
            http.close()
    if response.status_code != 200:
        raise LLMCallError(
            f"http_{response.status_code}", response.text[:300], response.headers.get("retry-after")
        )
    data = response.json()
    choices = data.get("choices") or []
    text = _text((choices[0].get("message") or {}).get("content")) if choices else ""
    if not text:
        raise LLMCallError("empty", "no text in the answer")
    usage = data.get("usage") or {}
    return {
        "json": parse_json(text),
        "model": model,
        "tokens_in": int(usage.get("prompt_tokens") or 0),
        "tokens_out": int(usage.get("completion_tokens") or 0),
    }
