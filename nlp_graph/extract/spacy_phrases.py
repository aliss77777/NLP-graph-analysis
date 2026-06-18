"""Optional spaCy noun chunks and NER (graceful fallback if model missing)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        log.warning("spaCy model en_core_web_sm not installed; skipping spacy_phrases")
        _NLP = False
    return _NLP


def extract_spacy_phrases(posts: pd.DataFrame, *, max_per_doc: int = 10) -> pd.DataFrame:
    nlp = _get_nlp()
    if not nlp:
        return pd.DataFrame(columns=["post_id", "phrase", "score", "source_channel", "platform", "extractor"])

    records: list[dict[str, Any]] = []
    for row in posts.itertuples(index=False):
        doc = nlp(str(row.text)[:10000])
        seen: set[str] = set()
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            if len(phrase.split()) < 2 or phrase in seen:
                continue
            seen.add(phrase)
            records.append(
                {
                    "post_id": str(row.post_id),
                    "phrase": phrase,
                    "score": 0.8,
                    "source_channel": getattr(row, "source_channel", None),
                    "platform": getattr(row, "platform", None),
                    "extractor": "spacy_chunk",
                }
            )
            if len(seen) >= max_per_doc:
                break
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "GPE") and len(ent.text) >= 3:
                phrase = ent.text.lower().strip()
                if phrase not in seen:
                    records.append(
                        {
                            "post_id": str(row.post_id),
                            "phrase": phrase,
                            "score": 0.9,
                            "source_channel": getattr(row, "source_channel", None),
                            "platform": getattr(row, "platform", None),
                            "extractor": "spacy_ner",
                        }
                    )
    return pd.DataFrame(records)
