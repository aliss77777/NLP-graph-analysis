"""Post corpus loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ("post_id", "text")
OPTIONAL_COLUMNS = (
    "source_channel",
    "platform_mentioned",
    "carrier",
    "body",
    "title",
    "text_for_embedding",
)


def _normalize_posts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "text" not in out.columns:
        if "text_for_embedding" in out.columns:
            out["text"] = out["text_for_embedding"]
        elif "body" in out.columns:
            title = out["title"].fillna("") if "title" in out.columns else ""
            out["text"] = (title.astype(str) + " " + out["body"].fillna("").astype(str)).str.strip()
        else:
            raise ValueError("posts need text, text_for_embedding, or body column")

    if "platform" not in out.columns:
        if "platform_mentioned" in out.columns:
            out["platform"] = out["platform_mentioned"]
        elif "carrier" in out.columns:
            out["platform"] = out["carrier"]
        else:
            out["platform"] = None

    out["post_id"] = out["post_id"].astype(str)
    out["text"] = out["text"].fillna("").astype(str)
    if "source_channel" not in out.columns:
        out["source_channel"] = None
    out["source_channel"] = out["source_channel"].fillna("unknown")
    out["platform"] = out["platform"].fillna("unknown")
    return out


def load_posts_parquet(path: str | Path) -> pd.DataFrame:
    return _normalize_posts(pd.read_parquet(path))


def load_posts(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return load_posts_parquet(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return _normalize_posts(pd.DataFrame(rows))
    if path.suffix == ".csv":
        return _normalize_posts(pd.read_csv(path))
    raise ValueError(f"unsupported format: {path}")


def load_posts_bigquery(
    *,
    project_id: str = "insurance-intel-498502",
    dataset: str = "insurance_warehouse",
    table: str = "raw_posts",
    where: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    sql = f"""
    SELECT
      post_id,
      source_channel,
      platform_mentioned,
      carrier,
      body,
      title,
      text_for_embedding
    FROM `{project_id}.{dataset}.{table}`
  """
    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" LIMIT {limit}"
    rows = [dict(r) for r in client.query(sql).result()]
    return _normalize_posts(pd.DataFrame(rows))
