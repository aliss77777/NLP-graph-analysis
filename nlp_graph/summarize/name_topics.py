"""LLM community naming (stub for W3)."""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def name_communities(communities_df: pd.DataFrame, sample_posts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return communities_df with optional LLM labels — currently passthrough."""
    log.info("name_topics: LLM naming not configured; using cluster_<id> labels")
    return communities_df
