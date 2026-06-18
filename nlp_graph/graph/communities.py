"""Community detection — Leiden primary, Louvain fallback."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

log = logging.getLogger(__name__)


def detect_communities_leiden(
    g: nx.Graph,
    *,
    resolution: float = 1.0,
    seed: int = 42,
) -> dict[str, int]:
    if g.number_of_nodes() == 0:
        return {}
    try:
        import igraph as ig
        import leidenalg

        edges = list(g.edges(data=True))
        ig_g = ig.Graph()
        ig_g.add_vertices(list(g.nodes()))
        name_to_idx = {n: i for i, n in enumerate(g.nodes())}
        ig_g.add_edges([(name_to_idx[u], name_to_idx[v]) for u, v, _ in edges])
        weights = [d.get("weight", 1.0) for _, _, d in edges]
        if weights:
            ig_g.es["weight"] = weights
        partition = leidenalg.find_partition(
            ig_g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight" if weights else None,
            resolution_parameter=resolution,
            seed=seed,
        )
        node_list = list(g.nodes())
        return {node_list[i]: part for i, part in enumerate(partition.membership)}
    except ImportError:
        log.warning("leidenalg/igraph not installed; falling back to Louvain")
        return detect_communities_louvain(g)


def detect_communities_louvain(g: nx.Graph) -> dict[str, int]:
    if g.number_of_nodes() == 0:
        return {}
    import community as community_louvain

    return community_louvain.best_partition(g, weight="weight")


def filter_small_communities(
    partition: dict[str, int],
    phrase_posts: dict[str, set[str]],
    *,
    min_posts: int = 85,
    min_phrases: int = 5,
    min_post_fraction: float = 0.005,
) -> dict[str, int]:
    """Reassign noise community id -1 for communities below thresholds."""
    comm_posts: dict[int, set[str]] = {}
    comm_phrases: dict[int, set[str]] = {}

    for phrase, comm_id in partition.items():
        comm_phrases.setdefault(comm_id, set()).add(phrase)
        comm_posts.setdefault(comm_id, set()).update(phrase_posts.get(phrase, set()))

    total_posts = len({p for posts in phrase_posts.values() for p in posts})
    out = dict(partition)
    for comm_id, posts in comm_posts.items():
        if len(comm_phrases.get(comm_id, set())) < min_phrases:
            for phrase in comm_phrases[comm_id]:
                out[phrase] = -1
        elif len(posts) < min_posts or len(posts) / max(total_posts, 1) < min_post_fraction:
            for phrase in comm_phrases[comm_id]:
                out[phrase] = -1
    return out


def community_stats(partition: dict[str, int], phrase_posts: dict[str, set[str]]) -> list[dict[str, Any]]:
    comm_phrases: dict[int, list[str]] = {}
    for phrase, cid in partition.items():
        if cid < 0:
            continue
        comm_phrases.setdefault(cid, []).append(phrase)

    rows: list[dict[str, Any]] = []
    for cid, phrases in comm_phrases.items():
        posts: set[str] = set()
        for ph in phrases:
            posts.update(phrase_posts.get(ph, set()))
        rows.append(
            {
                "community_id": cid,
                "phrase_count": len(phrases),
                "post_count": len(posts),
            }
        )
    return sorted(rows, key=lambda r: r["post_count"], reverse=True)
