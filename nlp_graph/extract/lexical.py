"""Lexical extraction — phrases (graph) vs entities (separate nodes)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from nlp_graph.extract.ngrams import extract_document_phrases, seed_phrases_from_patterns

from nlp_graph.extract.entities import KNOWN_ENTITIES

B2B_SEED_PATTERNS: dict[str, str] = {
    "lakehouse": r"(?i)lakehouse[\w\s\-]{0,30}(architecture|cost|pricing|platform)?",
    "consumption_pricing": r"(?i)consumption[\s\-]pricing",
    "cost_per_query": r"(?i)cost per query",
    "dbt_cloud": r"(?i)dbt\s+cloud",
    "unity_catalog": r"(?i)unity\s+catalog",
    "serverless_sql": r"(?i)serverless\s+sql",
    "serverless": r"(?i)serverless[\w\s\-]{0,20}(sql|compute|warehouse)?",
    "data_governance": r"(?i)data governance",
    "role_based_access": r"(?i)role[\s\-]based access",
    "pricing": r"(?i)(pricing|price per|cost per|tco|total cost)",
    "vs_comparison": r"(?i)(databricks|snowflake|dbt|bigquery|redshift)[\w\s]{0,20}(vs\.?|versus|compared to)[\w\s]{0,30}",
    "migration": r"(?i)(migrate|migration)[\w\s]{0,25}(snowflake|databricks|bigquery|redshift)?",
    "vendor_lock": r"(?i)vendor[\s\-]lock[\s\-]in",
}


def _extract_entity_mentions(row: Any, text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ent in KNOWN_ENTITIES:
        if re.search(rf"\b{re.escape(ent)}\b", text, re.IGNORECASE):
            records.append(
                {
                    "post_id": str(row.post_id),
                    "entity": ent,
                    "source_channel": getattr(row, "source_channel", None),
                    "platform": getattr(row, "platform", None),
                    "extractor": "entity_mention",
                }
            )
    return records


def extract_lexical_phrases(
    posts: pd.DataFrame,
    *,
    min_df: float = 0.005,
    max_df: float = 0.90,
    top_k_per_doc: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (phrase_extractions, entity_mentions) — entities excluded from phrase graph."""
    tfidf_df, _meta = extract_document_phrases(
        posts,
        min_df=min_df,
        max_df=max_df,
        top_k_per_doc=top_k_per_doc,
    )

    seed_records: list[dict[str, Any]] = []
    entity_records: list[dict[str, Any]] = []

    for row in posts.itertuples(index=False):
        text = str(row.text)
        for phrase in seed_phrases_from_patterns(text, B2B_SEED_PATTERNS):
            seed_records.append(
                {
                    "post_id": str(row.post_id),
                    "phrase": phrase,
                    "score": 1.0,
                    "source_channel": getattr(row, "source_channel", None),
                    "platform": getattr(row, "platform", None),
                    "extractor": "regex_seed",
                }
            )
        entity_records.extend(_extract_entity_mentions(row, text))

    phrase_df = tfidf_df
    if seed_records:
        seeds = pd.DataFrame(seed_records).drop_duplicates(subset=["post_id", "phrase"])
        phrase_df = (
            pd.concat([tfidf_df, seeds], ignore_index=True)
            .sort_values("score", ascending=False)
            .drop_duplicates(subset=["post_id", "phrase"], keep="first")
        )

    entity_df = (
        pd.DataFrame(entity_records).drop_duplicates(subset=["post_id", "entity"])
        if entity_records
        else pd.DataFrame(columns=["post_id", "entity", "source_channel", "platform", "extractor"])
    )
    return phrase_df, entity_df
