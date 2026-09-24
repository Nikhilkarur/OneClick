"""Zero-URL guard: removes http(s), www., domains, emails, markdown links from every string.

Runs twice: on the SIIS text before any LLM sees it (normalize) and on every output string before a
response leaves the API (validate), cache hits included. Catalog `bixby://` URIs in deeplink fields
are the only URLs allowed out; validate skips those fields, so scrub itself removes every scheme.
"""

import re

_TLDS = r"com|net|org|gov|edu|info|io|ly|biz|uk|html?|php|aspx?|jsp"

# Order matters: links that carry visible text keep the text, then bare URLs go.
_MD_IMAGE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")
_MD_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
# A tag starts right after "<" ("<a href>", "</b>"); "< 5 minutes >" in prose is not a tag.
_HTML_TAG = re.compile(r"</?[a-z][^<>]*>|<\s+(?:a|img|link|iframe)\b[^<>]*>", re.IGNORECASE)
_HTML_ATTR = re.compile(r"\b(?:href|src)\s*=\s*(?:\"[^\"]*\"|'[^']*'|\S+)", re.IGNORECASE)
_SCHEME_URL = re.compile(r"\b[a-z][a-z0-9+.\-]{1,20}://[^\s<>()\"']*", re.IGNORECASE)
_WWW = re.compile(r"\bwww\.[^\s<>()\"']*", re.IGNORECASE)
# Kit articles lose their spaces ("kidshome.pin@samsung.comusingyourregistered..."), so an address
# ends at its known TLD rather than at the next space; the generic form catches any other TLD.
_EMAIL_KNOWN_TLD = re.compile(
    r"[a-z0-9._%+\-]+@(?:[a-z0-9\-]+\.)+(?:com|net|org|gov|edu|info|io|biz|co|uk|in|de)", re.IGNORECASE
)
_EMAIL = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)+", re.IGNORECASE)
_DOMAIN = re.compile(
    rf"\b[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.(?:{_TLDS})\b(?:/[^\s<>()\"']*)?", re.IGNORECASE
)

_LEAK_PATTERNS = (_MD_IMAGE, _MD_LINK, _HTML_TAG, _HTML_ATTR, _SCHEME_URL, _WWW, _EMAIL, _DOMAIN)

_SPACES = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:!?])")
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]|<\s*>")


def scrub(text: str) -> str:
    """The same text with every link, address and markup link removed; line breaks are kept."""
    if not text:
        return text
    out = _MD_IMAGE.sub("", text)
    out = _MD_LINK.sub(r"\1", out)
    out = _HTML_ATTR.sub("", out)
    out = _HTML_TAG.sub("", out)
    out = _SCHEME_URL.sub("", out)
    out = _WWW.sub("", out)
    out = _EMAIL_KNOWN_TLD.sub("", out)
    out = _EMAIL.sub("", out)
    out = _DOMAIN.sub("", out)
    out = _EMPTY_BRACKETS.sub("", out)
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _SPACES.sub(" ", out)
    return "\n".join(line.strip() for line in out.split("\n")).strip()


def has_leak(text: str) -> bool:
    """True when the string still contains anything scrub() would remove."""
    return bool(text) and any(pattern.search(text) for pattern in _LEAK_PATTERNS)
