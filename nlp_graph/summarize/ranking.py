"""GEO-oriented phrase ranking (tfidf × b2b_boost × query_shape_boost)."""

from __future__ import annotations

import re

import pandas as pd

from nlp_graph.extract.channel_stoplist import is_filler_for_metrics
from nlp_graph.extract.entities import B2B_ALLOWLIST

QUERY_SHAPE_VERBS = re.compile(
    r"(?i)\b(migrate|migration|compare|cost|pricing|vs|versus|how|why|best|switch|"
    r"evaluate|choose|replace|avoid|reduce|optimize|scale|deploy|integrate|fine-tune)\b"
)

B2B_BOOST = 2.0
QUERY_SHAPE_BOOST = 1.5


def b2b_boost(phrase: str) -> float:
    pl = phrase.lower()
    if any(ent in pl for ent in B2B_ALLOWLIST):
        return B2B_BOOST
    return 1.0


def query_shape_boost(phrase: str) -> float:
    if QUERY_SHAPE_VERBS.search(phrase):
        return QUERY_SHAPE_BOOST
    return 1.0


def phrase_rank_score(phrase: str, tfidf_sum: float) -> float:
    return tfidf_sum * b2b_boost(phrase) * query_shape_boost(phrase)


def rank_phrases_in_community(
    community_id: int,
    partition: dict[str, int],
    extractions: pd.DataFrame,
    *,
    top_n: int = 25,
) -> list[str]:
    phrases = [p for p, c in partition.items() if c == community_id]
    if not phrases:
        return []
    sub = extractions[extractions.phrase.isin(phrases)]
    tfidf_sums = sub.groupby("phrase")["score"].sum()
    ranked = sorted(
        (p for p in phrases if not is_filler_for_metrics(p)),
        key=lambda p: phrase_rank_score(p, float(tfidf_sums.get(p, 0.0))),
        reverse=True,
    )
    return ranked[:top_n]
