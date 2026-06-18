"""Load post corpora from CSV, Parquet, JSONL, or BigQuery."""

from nlp_graph.ingest.posts import load_posts, load_posts_bigquery, load_posts_parquet

__all__ = ["load_posts", "load_posts_parquet", "load_posts_bigquery"]
