"""Optional LLM phrase extraction (stub — batch on cluster centroids post-Leiden)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def extract_llm_phrases(
    posts: pd.DataFrame,
    *,
    model: str | None = None,
    max_per_doc: int = 10,
) -> pd.DataFrame:
    """Placeholder for GenAI pass. Returns empty frame until LLM integration is wired."""
    log.info("llm_phrases: skipped (not configured); model=%s", model)
    return pd.DataFrame(columns=["post_id", "phrase", "score", "source_channel", "platform", "extractor"])
