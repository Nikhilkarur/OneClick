"""Zero-URL guard: removes http(s), www., domains, emails, markdown links from every string.

Runs twice: on the SIIS text and the complaint before any LLM sees them (normalize), and on every
output string before a response leaves the API (validate), cache hits included. Catalog `bixby://`
URIs in deeplink fields are the only URLs allowed out; validate skips those fields, so scrub itself
removes every scheme.

Text is canonicalised first (Unicode TR #36): HTML entities decoded, NFKC normalisation (full-width
"ｈｔｔｐｓ：／／" becomes "https://"), invisible format characters removed (a zero-width space inside
"samsung.com" would otherwise hide it). Every pattern is length-bounded (RFC 5321/1035 limits: 64
characters before the @, 63 per domain label), so scrub runs in linear time on any input.
"""

import html
import re
import unicodedata

# Unambiguous endings match in any case. Endings that are also English words ("in", "me", "us") or
# common in glued kit text ("screen.In the...") only count in lowercase, as real domains are written.
_TLDS = r"com|net|org|gov|edu|info|io|ly|biz|uk|html?|php|aspx?|jsp"
_TLDS_LOWERCASE_ONLY = r"in|co|ai|app|dev|me|us"

_LABEL = r"[a-z0-9\-]{1,63}"
_LOCAL_PART = r"(?<![\w.%+\-])[\w.%+\-]{1,64}"  # starts only where a local part can start

# Order matters: links that carry visible text keep the text, then bare URLs go.
_MD_IMAGE = re.compile(r"!\[[^\]\n]{0,500}\]\([^)\n]{0,2000}\)")
_MD_LINK = re.compile(r"\[([^\]\n]{0,500})\]\([^)\n]{0,2000}\)")
_MD_REF_LINK = re.compile(r"\[([^\]\n]{1,500})\]\[[^\]\n]{0,100}\]")
_MD_REF_DEF = re.compile(r"^[ \t]{0,3}\[[^\]\n]{1,100}\]:[ \t]*\S+.*$", re.MULTILINE)
# A tag starts right after "<" ("<a href>", "</b>"); "< 5 minutes >" in prose is not a tag.
_HTML_TAG = re.compile(r"</?[a-z][^<>]*>|<\s+(?:a|img|link|iframe)\b[^<>]*>", re.IGNORECASE)
_HTML_ATTR = re.compile(r"\b(?:href|src)\s*=\s*(?:\"[^\"]*\"|'[^']*'|\S+)", re.IGNORECASE)
_SCHEME_URL = re.compile(r"\b[a-z][a-z0-9+.\-]{1,20}://[^\s<>()\"']*", re.IGNORECASE)
_MAILTO = re.compile(r"\bmailto:[^\s<>()\"']*", re.IGNORECASE)
_WWW = re.compile(r"\bwww\.[^\s<>()\"']*", re.IGNORECASE)
# Kit articles lose their spaces ("kidshome.pin@samsung.comusingyourregistered..."), so an address
# ends at its known TLD (plus an optional country code) rather than at the next space; the generic
# form catches any other TLD.
_EMAIL_KNOWN_TLD = re.compile(
    rf"{_LOCAL_PART}@(?:{_LABEL}\.){{1,8}}(?:com|net|org|gov|edu|info|io|biz|co|uk|in|de)(?:\.[a-z]{{2}}(?![a-z]))?",
    re.IGNORECASE,
)
_EMAIL = re.compile(rf"{_LOCAL_PART}@(?:{_LABEL}\.){{1,8}}{_LABEL}", re.IGNORECASE)
_DOMAIN = re.compile(
    rf"\b[a-z0-9][a-z0-9\-]{{0,62}}(?:\.{_LABEL}){{0,8}}\.(?:(?:{_TLDS})|(?-i:{_TLDS_LOWERCASE_ONLY}))\b"
    r"(?:/[^\s<>()\"']*)?",
    re.IGNORECASE,
)
_OCTET = r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
_IPV4 = re.compile(rf"\b(?:{_OCTET}\.){{3}}{_OCTET}\b(?::\d{{1,5}})?(?:/[^\s<>()\"']*)?")

_LEAK_PATTERNS = (
    _MD_IMAGE,
    _MD_LINK,
    _MD_REF_DEF,
    _HTML_TAG,
    _HTML_ATTR,
    _SCHEME_URL,
    _MAILTO,
    _WWW,
    _EMAIL,
    _DOMAIN,
    _IPV4,
)

_SPACES = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:!?])")
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]|<\s*>")


def canonical(text: str) -> str:
    """HTML entities decoded, NFKC-normalised, invisible format characters (Unicode Cf) removed."""
    text = unicodedata.normalize("NFKC", html.unescape(text))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def scrub(text: str) -> str:
    """The same text with every link, address and markup link removed; line breaks are kept."""
    if not text:
        return text
    out = canonical(text)
    out = _MD_REF_DEF.sub("", out)
    out = _MD_IMAGE.sub("", out)
    out = _MD_LINK.sub(r"\1", out)
    out = _MD_REF_LINK.sub(r"\1", out)
    out = _HTML_ATTR.sub("", out)
    out = _HTML_TAG.sub("", out)
    out = _SCHEME_URL.sub("", out)
    out = _MAILTO.sub("", out)
    out = _WWW.sub("", out)
    out = _EMAIL_KNOWN_TLD.sub("", out)
    out = _EMAIL.sub("", out)
    out = _DOMAIN.sub("", out)
    out = _IPV4.sub("", out)
    out = _EMPTY_BRACKETS.sub("", out)
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _SPACES.sub(" ", out)
    return "\n".join(line.strip() for line in out.split("\n")).strip()


def has_leak(text: str) -> bool:
    """True when the string still contains anything scrub() would remove."""
    if not text:
        return False
    text = canonical(text)
    return any(pattern.search(text) for pattern in _LEAK_PATTERNS)
