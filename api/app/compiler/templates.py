"""Component 10: goal template (strip trailing Troubleshooting/Configuration first)."""

import re

_KINDS = ("Troubleshooting", "Configuration")
_TRAILING_KIND = re.compile(r"[\s\-:,.]*\b(?:troubleshooting|configuration)\s*$", re.IGNORECASE)
# The goal is matched by a regex on the scorer's side: keep the topic to words, digits, spaces,
# hyphens and apostrophes ("Wi-Fi", "Samsung's").
_NOT_TOPIC = re.compile(r"[^\w\s\-']")
_DEFAULT_TOPIC = "Device"


def title_case_word(word: str) -> str:
    """Capitalise the first letter of each hyphen part, keep the rest ("wi-fi" -> "Wi-Fi", "S22")."""
    return "-".join(part[:1].upper() + part[1:] for part in word.split("-"))


def clean_topic(topic: str) -> str:
    text = _NOT_TOPIC.sub(" ", topic or "")
    previous = None
    while previous != text:  # "Screen Troubleshooting Configuration" loses both
        previous, text = text, _TRAILING_KIND.sub("", text).strip()
    words = [title_case_word(w) for w in text.split()]
    return " ".join(words) or _DEFAULT_TOPIC


def build_goal(topic: str, kind: str = "Troubleshooting") -> str:
    """`Follow these steps to perform this {Topic} {Kind}.` (FAQ Q6 form, trailing period included)."""
    kind = next((k for k in _KINDS if k.lower() == (kind or "").strip().lower()), _KINDS[0])
    return f"Follow these steps to perform this {clean_topic(topic)} {kind}."
