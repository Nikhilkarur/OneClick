"""Component 4: split SIIS into sections and numbered sentences; rank sections per intent.

Rule (data/fixtures/README.md): drop everything before the first `#`, split on `#` headers, split
each remaining line into sentences on `[.!?]` followed by whitespace and a capital, and number the
sentences `S1..Sn` across the whole article. Steps cite these ids, so the rule must stay stable.
"""

import re

import numpy as np

from app.config import settings
from app.models import Intent, SiisSentence
from app.retrieval import dense

_HEADER = re.compile(r"^(#{1,6})\s*(.*?)\s*#*\s*$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sections(siis_clean: str) -> list[dict]:
    """[{id, heading, level, sentences: [text]}] in article order. Text before the first header is
    the listing's breadcrumb, not the article, and is dropped."""
    start = siis_clean.find("#")
    if start < 0:
        body, sections = siis_clean, [{"id": "sec1", "heading": "", "level": 1, "sentences": []}]
    else:
        body, sections = siis_clean[start:], []
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        header = _HEADER.match(line)
        if header:
            sections.append(
                {
                    "id": f"sec{len(sections) + 1}",
                    "heading": header.group(2),
                    "level": len(header.group(1)),
                    "sentences": [],
                }
            )
            continue
        sections[-1]["sentences"].extend(s.strip() for s in _SENTENCE_BREAK.split(line) if s.strip())
    return [s for s in sections if s["sentences"] or s["heading"]]


def _section_text(section: dict) -> str:
    """Heading plus the opening of the body: what the section is about, and cheap to embed."""
    body = " ".join(section["sentences"])[: settings.section_embed_chars]
    return f"{section['heading']}. {body}".strip(". ")


def section_relevance(sections: list[dict], intents: list[Intent]) -> list[list[float]]:
    """Cosine between each intent and each section (heading + body): [section][intent], 0-1."""
    if not sections or not intents:
        return [[0.0] * len(intents) for _ in sections]
    section_texts = [_section_text(s) for s in sections]
    vectors = np.asarray(dense.embed(section_texts + [i.text for i in intents]), dtype=np.float32)
    sims = vectors[: len(sections)] @ vectors[len(sections) :].T
    return [[round(float(min(max(v, 0.0), 1.0)), 2) for v in row] for row in sims]


def segment(siis_clean: str, intents: list[Intent]) -> list[SiisSentence]:
    """Numbered sentences; each carries its section's best relevance over the intents."""
    sentences, _ = segment_with_sections(siis_clean, intents)
    return sentences


def segment_with_sections(siis_clean: str, intents: list[Intent]) -> tuple[list[SiisSentence], list[dict]]:
    """Sentences plus the section table the stream's `segment` event and the compiler use.

    Section rows: {id, heading, level, sentence_ids, relevance: [per intent], relevant}.
    """
    sections = split_sections(siis_clean)
    relevance = section_relevance(sections, intents)
    sentences: list[SiisSentence] = []
    table: list[dict] = []
    for section, scores in zip(sections, relevance):
        best = max(scores, default=0.0)
        ids = []
        for text in section["sentences"]:
            sid = f"S{len(sentences) + 1}"
            ids.append(sid)
            sentences.append(SiisSentence(id=sid, section=section["heading"], text=text, relevance=best))
        table.append(
            {
                "id": section["id"],
                "heading": section["heading"],
                "level": section["level"],
                "sentence_ids": ids,
                "relevance": scores,
                "relevant": best >= settings.section_relevance_floor,
            }
        )
    return sentences, table
