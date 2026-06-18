"""Document–phrase adjacency list (ported from NLP-graph-analysis Step 2)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def phrase_extractions_to_adjacency(extractions: pd.DataFrame) -> pd.DataFrame:
    """Build post_id → phrase edge list with weights."""
    return (
        extractions.rename(columns={"post_id": "source_id", "phrase": "target"})
        .loc[:, ["source_id", "target", "score", "source_channel", "platform", "extractor"]]
        .copy()
    )


def adjacency_to_doc_term_matrix(adjacency: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pivot adjacency to doc×phrase matrix (for diagnostics / legacy export)."""
    pivot = adjacency.pivot_table(
        index="source_id",
        columns="target",
        values="score",
        aggfunc="max",
        fill_value=0.0,
    )
    phrases = list(pivot.columns)
    return pivot, phrases


def create_adjacency_list_from_matrix(adjacency_matrix: pd.DataFrame) -> pd.DataFrame:
    """Port of legacy create_adjacency_list — doc row to phrase columns."""
    rows: list[dict[str, Any]] = []
    for doc_id, row in adjacency_matrix.iterrows():
        for phrase, weight in row.items():
            if weight and float(weight) > 0:
                rows.append({"source_id": doc_id, "target": phrase, "score": float(weight)})
    return pd.DataFrame(rows)
