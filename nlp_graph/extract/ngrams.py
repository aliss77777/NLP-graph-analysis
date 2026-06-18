"""N-gram phrase extraction with channel-aware preprocessing."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_graph.extract.channel_stoplist import channel_stoplist, is_junk_phrase, strip_channel_phrases

DEFAULT_MIN_DF = 0.005  # ~87 docs for 17k corpus
DEFAULT_MAX_DF = 0.90
DEFAULT_NGRAM_RANGE = (2, 4)
DEFAULT_TOP_K_PER_DOC = 20


def preprocess_text(text: str, source_channel: str | None) -> str:
    cleaned = strip_channel_phrases(text, source_channel)
    cleaned = re.sub(r"http\S+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s\-\.]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def build_corpus_texts(posts: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    for row in posts.itertuples(index=False):
        texts.append(preprocess_text(str(row.text), getattr(row, "source_channel", None)))
    return texts


def extract_document_phrases(
    posts: pd.DataFrame,
    *,
    min_df: float = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    top_k_per_doc: int = DEFAULT_TOP_K_PER_DOC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (phrase_extractions, doc_term_matrix_metadata).

    phrase_extractions columns: post_id, phrase, score, source_channel, platform, extractor
    """
    post_ids = posts["post_id"].astype(str).tolist()
    corpus = build_corpus_texts(posts)

    # Union stoplist across channels for vectorizer (per-doc filter applied later)
    all_stops = set()
    for ch in posts["source_channel"].unique():
        all_stops.update(channel_stoplist(ch))

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=list(all_stops),
        lowercase=True,
        token_pattern=r"(?u)\b[a-z0-9][a-z0-9\-\.]{1,}\b",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    n_docs = len(post_ids)

    # Corpus-wide doc frequency for junk filter
    doc_freq: dict[str, int] = {}
    for col_idx, phrase in enumerate(feature_names):
        doc_freq[phrase] = int((matrix[:, col_idx] > 0).sum())

    records: list[dict[str, Any]] = []
    for doc_idx, post_id in enumerate(post_ids):
        row = posts.iloc[doc_idx]
        ch = row.get("source_channel")
        stops = channel_stoplist(ch)
        scores = matrix.getrow(doc_idx).toarray().ravel()
        ranked = np.argsort(scores)[::-1]
        kept = 0
        for col_idx in ranked:
            if scores[col_idx] <= 0:
                break
            phrase = feature_names[col_idx]
            if is_junk_phrase(phrase, stops, doc_freq=doc_freq.get(phrase, 0), n_docs=n_docs):
                continue
            records.append(
                {
                    "post_id": post_id,
                    "phrase": phrase,
                    "score": float(scores[col_idx]),
                    "source_channel": ch,
                    "platform": row.get("platform"),
                    "extractor": "tfidf_ngram",
                }
            )
            kept += 1
            if kept >= top_k_per_doc:
                break

    extractions = pd.DataFrame(records)
    meta = pd.DataFrame(
        {
            "n_docs": [len(post_ids)],
            "n_features": [len(feature_names)],
            "min_df": [min_df],
            "max_df": [max_df],
        }
    )
    return extractions, meta


def seed_phrases_from_patterns(text: str, patterns: dict[str, str]) -> list[str]:
    """Extract seed phrases from named regex patterns (e.g. clean_posts_features)."""
    found: list[str] = []
    for _name, pattern in patterns.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            span = text[m.start() : m.end()].strip()
            if len(span.split()) >= 2 or len(span) >= 6:
                found.append(span.lower())
    return found
