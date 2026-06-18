"""Export lexical KG to parquet matching insurance-intel kg_* schema."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def build_kg_nodes(
    extractions: pd.DataFrame,
    partition: dict[str, int],
    platforms: pd.Series,
    entity_mentions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    node_ids: dict[tuple[str, str], int] = {}
    next_id = 1

    for phrase in sorted(set(extractions["phrase"])):
        node_ids[("phrase", phrase)] = next_id
        next_id += 1

    if entity_mentions is not None and not entity_mentions.empty:
        for entity in sorted(set(entity_mentions["entity"])):
            node_ids[("entity", entity)] = next_id
            next_id += 1

    for platform in sorted(set(platforms.dropna().astype(str))):
        if platform and platform != "unknown":
            node_ids[("platform", platform)] = next_id
            next_id += 1

    rows: list[dict[str, Any]] = []
    phrase_doc_count = extractions.groupby("phrase")["post_id"].nunique().to_dict()

    for (node_type, label), nid in sorted(node_ids.items(), key=lambda x: x[1]):
        props: dict[str, Any] = {}
        comm_id = None
        if node_type == "phrase":
            props["doc_count"] = phrase_doc_count.get(label, 0)
            comm_id = partition.get(label)
            if comm_id is not None and comm_id < 0:
                comm_id = None
        elif node_type == "entity":
            props["mention_type"] = "platform_product"
        rows.append(
            {
                "node_id": nid,
                "node_type": node_type,
                "node_label": label,
                "properties": json.dumps(props),
                "community_id": comm_id,
                "updated_at": now,
            }
        )
    return pd.DataFrame(rows)


def build_kg_edges(
    nodes_df: pd.DataFrame,
    cooccur_edges: list[tuple[str, str, float, int]],
    communities_df: pd.DataFrame,
) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    label_to_id = {
        (row.node_type, row.node_label): int(row.node_id) for row in nodes_df.itertuples(index=False)
    }
    edge_id = 1
    rows: list[dict[str, Any]] = []

    for comm in communities_df.itertuples(index=False):
        phrase = comm.dominant_topic
        platform = comm.dominant_platform
        if not phrase or not platform or platform == "unknown":
            continue
        phrase_id = label_to_id.get(("phrase", phrase))
        plat_id = label_to_id.get(("platform", str(platform)))
        if phrase_id and plat_id:
            rows.append(
                {
                    "edge_id": edge_id,
                    "from_node_id": phrase_id,
                    "to_node_id": plat_id,
                    "edge_type": "owned_by",
                    "weight": float(comm.post_count),
                    "properties": json.dumps(
                        {
                            "community_id": int(comm.community_id),
                            "dominance_pct": float(comm.platform_dominance_pct or 0),
                        }
                    ),
                    "updated_at": now,
                }
            )
            edge_id += 1

    for a, b, weight, count in cooccur_edges:
        a_id = label_to_id.get(("phrase", a))
        b_id = label_to_id.get(("phrase", b))
        if a_id and b_id:
            rows.append(
                {
                    "edge_id": edge_id,
                    "from_node_id": a_id,
                    "to_node_id": b_id,
                    "edge_type": "co_occurs",
                    "weight": weight,
                    "properties": json.dumps({"count": count}),
                    "updated_at": now,
                }
            )
            edge_id += 1

    return pd.DataFrame(rows)


def write_kg_parquet(
    output_dir: Path,
    phrase_extractions: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    communities_df: pd.DataFrame,
    platform_pivot: pd.DataFrame,
    post_community: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "phrase_extractions": output_dir / "phrase_extractions.parquet",
        "post_community": output_dir / "post_community.parquet",
        "nodes": output_dir / "kg_nodes.parquet",
        "edges": output_dir / "kg_edges.parquet",
        "communities": output_dir / "kg_communities.parquet",
        "platform_pivot": output_dir / "kg_platform_pivot.parquet",
    }
    phrase_extractions.to_parquet(paths["phrase_extractions"], index=False)
    post_community.to_parquet(paths["post_community"], index=False)
    nodes_df.to_parquet(paths["nodes"], index=False)
    edges_df.to_parquet(paths["edges"], index=False)
    communities_df.to_parquet(paths["communities"], index=False)
    platform_pivot.to_parquet(paths["platform_pivot"], index=False)
    return paths
