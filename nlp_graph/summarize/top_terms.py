"""Community summarization — GEO query-shape ranking + primary-post ownership."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from nlp_graph.summarize.post_community import platform_ownership_from_posts
from nlp_graph.summarize.ranking import rank_phrases_in_community


def top_terms_per_community(
    partition: dict[str, int],
    extractions: pd.DataFrame,
    _g: Any,
    *,
    top_n: int = 25,
) -> dict[int, list[str]]:
    communities = sorted({c for c in partition.values() if c >= 0})
    return {
        cid: rank_phrases_in_community(cid, partition, extractions, top_n=top_n) for cid in communities
    }


def platform_ownership_pivot(
    partition: dict[str, int],
    extractions: pd.DataFrame,
    *,
    top_n_platforms: int = 10,
) -> pd.DataFrame:
    """Deprecated path — use platform_ownership_from_posts via post_community."""
    from nlp_graph.summarize.post_community import assign_post_primary_community

    post_comm = assign_post_primary_community(extractions, partition)
    return platform_ownership_from_posts(post_comm)


def dominant_platform_per_community(pivot: pd.DataFrame) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if pivot.empty:
        return out
    for cid, grp in pivot.groupby("community_id"):
        total = grp["post_count"].sum()
        top = grp.iloc[0]
        out[int(cid)] = {
            "dominant_platform": top["platform"],
            "dominance_pct": round(100.0 * top["post_count"] / total, 1) if total else 0.0,
        }
    return out


def build_communities_table(
    partition: dict[str, int],
    extractions: pd.DataFrame,
    g: Any,
    platform_pivot: pd.DataFrame,
    post_community: pd.DataFrame,
    *,
    top_n: int = 25,
) -> pd.DataFrame:
    from datetime import datetime, timezone

    top_terms = top_terms_per_community(partition, extractions, g, top_n=top_n)
    dom = dominant_platform_per_community(platform_pivot)
    post_counts = post_community.groupby("community_id").size().to_dict()

    rows: list[dict[str, Any]] = []
    for cid, terms in top_terms.items():
        dominant_topic = terms[0] if terms else None
        d = dom.get(cid, {})
        rows.append(
            {
                "community_id": cid,
                "label": f"cluster_{cid}",
                "top_terms": json.dumps(terms),
                "post_count": int(post_counts.get(cid, 0)),
                "dominant_topic": dominant_topic,
                "dominant_platform": d.get("dominant_platform"),
                "platform_dominance_pct": d.get("dominance_pct"),
                "created_at": datetime.now(timezone.utc),
            }
        )
    return pd.DataFrame(rows).sort_values("post_count", ascending=False)
