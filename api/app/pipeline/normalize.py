"""Component 1: normalize query, scrub URLs/emails from SIIS, compute siis_hash. No LLM."""

import hashlib
import re

from app.compiler.scrub import scrub

_LEADING_NUMBER = re.compile(r"^\d+\.\s*")
_QUOTES = "\"'“”‘’ "
_WHITESPACE = re.compile(r"\s+")
_HASH_LENGTH = 16  # matches the siis_hash values in data/fixtures/*/cache_events.json


def normalize_query(query: str) -> str:
    """Cache-key form of the complaint: kit numbering and quotes gone, whitespace collapsed, lowercase.

    Punctuation is kept: the exact tier should only match what really is the same question.
    """
    text = _LEADING_NUMBER.sub("", (query or "").strip()).strip(_QUOTES)
    return _WHITESPACE.sub(" ", text).strip().lower()


def siis_text(siis: dict | str | None) -> str:
    """The article body from whatever shape the request carries ({title, content} in the kit)."""
    if siis is None:
        return ""
    if isinstance(siis, str):
        return siis
    content = siis.get("content")
    if isinstance(content, str):
        return content
    # Unknown shape: every string value, in order, so nothing the article says is lost.
    return "\n".join(value for value in siis.values() if isinstance(value, str))


def siis_title(siis: dict | str | None) -> str | None:
    title = siis.get("title") if isinstance(siis, dict) else None
    return scrub(title) if isinstance(title, str) and title.strip() else None


def clean_siis(siis: dict | str | None) -> tuple[str, str | None]:
    """Return (siis_clean, siis_hash).

    The scrub runs here, before any LLM sees the article (the kit carries a real-looking address).
    The hash is over the cleaned text, so the same article always keys the same cache entries.
    """
    text = scrub(siis_text(siis).replace("\r\n", "\n"))
    if not text.strip():
        return "", None
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
