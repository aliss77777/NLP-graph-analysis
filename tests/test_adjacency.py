"""Tests for adjacency list construction."""

import pandas as pd

from nlp_graph.graph.adjacency import phrase_extractions_to_adjacency


def test_phrase_extractions_to_adjacency():
    df = pd.DataFrame(
        [
            {"post_id": "p1", "phrase": "dbt cloud pricing", "score": 0.9, "source_channel": "reddit", "platform": "dbt", "extractor": "tfidf"},
            {"post_id": "p1", "phrase": "lakehouse architecture", "score": 0.8, "source_channel": "reddit", "platform": "Databricks", "extractor": "tfidf"},
        ]
    )
    adj = phrase_extractions_to_adjacency(df)
    assert len(adj) == 2
    assert set(adj["target"]) == {"dbt cloud pricing", "lakehouse architecture"}
