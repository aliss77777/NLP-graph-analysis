"""End-to-end lexical KG pipeline (v3 — GEO post-primary communities)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from nlp_graph.export.kg_parquet import build_kg_edges, build_kg_nodes, write_kg_parquet
from nlp_graph.extract.lexical import extract_lexical_phrases
from nlp_graph.extract.spacy_phrases import extract_spacy_phrases
from nlp_graph.graph.communities import (
    community_stats,
    detect_communities_leiden,
    filter_small_communities,
)
from nlp_graph.graph.term_graph import build_phrase_cooccurrence_graph, phrase_post_map, prune_graph
from nlp_graph.summarize.post_community import assign_post_primary_community, platform_ownership_from_posts
from nlp_graph.summarize.top_terms import build_communities_table

log = logging.getLogger(__name__)


def _filter_b2b_posts(posts: pd.DataFrame) -> pd.DataFrame:
    if "source_channel" not in posts.columns:
        return posts
    return posts[
        posts["source_channel"].notna()
        & (posts["source_channel"] != "unknown")
        & (posts["source_channel"] != "")
    ].copy()


@dataclass
class PipelineConfig:
    min_df: float = 0.005
    max_df: float = 0.90
    top_k_per_doc: int = 20
    min_cooccur: int = 3
    leiden_resolution: float = 1.5
    min_posts_per_community: int = 25
    min_phrases_per_community: int = 2
    min_post_fraction: float = 0.002
    use_spacy: bool = False


@dataclass
class PipelineResult:
    stats: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)


def build_lexical_kg(
    posts: pd.DataFrame,
    output_dir: Path,
    *,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    config = config or PipelineConfig()
    t0 = time.perf_counter()

    posts = _filter_b2b_posts(posts)
    log.info("B2B synthetic corpus: %s posts", len(posts))

    extractions, entity_mentions = extract_lexical_phrases(
        posts,
        min_df=config.min_df,
        max_df=config.max_df,
        top_k_per_doc=config.top_k_per_doc,
    )

    if config.use_spacy:
        spacy_df = extract_spacy_phrases(posts)
        if not spacy_df.empty:
            extractions = (
                pd.concat([extractions, spacy_df], ignore_index=True)
                .sort_values("score", ascending=False)
                .drop_duplicates(subset=["post_id", "phrase"], keep="first")
            )

    log.info(
        "phrase extractions: %s rows, %s unique phrases; entities: %s",
        len(extractions),
        extractions["phrase"].nunique(),
        len(entity_mentions),
    )

    g = build_phrase_cooccurrence_graph(extractions, min_cooccur=config.min_cooccur)
    g = prune_graph(g)
    log.info("phrase graph (pruned): %s nodes, %s edges", g.number_of_nodes(), g.number_of_edges())

    partition = detect_communities_leiden(g, resolution=config.leiden_resolution)
    phrase_posts = phrase_post_map(extractions)
    partition = filter_small_communities(
        partition,
        phrase_posts,
        min_posts=config.min_posts_per_community,
        min_phrases=config.min_phrases_per_community,
        min_post_fraction=config.min_post_fraction,
    )

    n_communities = len({c for c in partition.values() if c >= 0})
    unassigned_phrases = sum(1 for c in partition.values() if c < 0)
    log.info("communities after filter: %s; unassigned phrases: %s", n_communities, unassigned_phrases)

    post_community = assign_post_primary_community(extractions, partition)
    platform_pivot = platform_ownership_from_posts(post_community)
    communities_df = build_communities_table(
        partition, extractions, g, platform_pivot, post_community
    )

    cooccur_edges = [
        (u, v, float(d.get("weight", 1.0)), int(d.get("count", 1)))
        for u, v, d in g.edges(data=True)
    ]
    nodes_df = build_kg_nodes(extractions, partition, posts["platform"], entity_mentions)
    edges_df = build_kg_edges(nodes_df, cooccur_edges, communities_df)

    paths = write_kg_parquet(
        output_dir, extractions, nodes_df, edges_df, communities_df, platform_pivot, post_community
    )

    elapsed = time.perf_counter() - t0
    post_partition_sum = int(post_community["community_id"].value_counts().sum())
    lakehouse = communities_df[communities_df.dominant_topic.str.contains("lakehouse", case=False, na=False)]
    lakehouse_dom = (
        float(lakehouse.iloc[0]["platform_dominance_pct"])
        if len(lakehouse) > 0 and lakehouse.iloc[0]["platform_dominance_pct"]
        else None
    )

    stats = {
        "build_version": "v3-beta3",
        "post_count": len(posts),
        "post_primary_assigned": len(post_community),
        "post_partition_sum": post_partition_sum,
        "extraction_rows": len(extractions),
        "unique_phrases": int(extractions["phrase"].nunique()),
        "entity_mentions": len(entity_mentions),
        "unassigned_phrases": unassigned_phrases,
        "graph_nodes": g.number_of_nodes(),
        "graph_edges": g.number_of_edges(),
        "node_count": len(nodes_df),
        "edge_count": len(edges_df),
        "community_count": n_communities,
        "community_stats": community_stats(partition, phrase_posts),
        "top_communities": communities_df.head(5).to_dict("records"),
        "lakehouse_platform_dominance_pct": lakehouse_dom,
        "elapsed_sec": round(elapsed, 2),
        "channel_distribution": posts["source_channel"].value_counts().to_dict(),
        "config": {
            "leiden_resolution": config.leiden_resolution,
            "min_posts_per_community": config.min_posts_per_community,
        },
    }
    return PipelineResult(stats=stats, paths=paths)
