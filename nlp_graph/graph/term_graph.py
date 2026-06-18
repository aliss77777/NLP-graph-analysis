"""Phrase–phrase co-occurrence graph via PMI projection."""

from __future__ import annotations

from collections import Counter
from math import log

import networkx as nx
import pandas as pd

DEFAULT_MIN_COOCCUR = 3


def build_phrase_cooccurrence_graph(
    extractions: pd.DataFrame,
    *,
    min_cooccur: int = DEFAULT_MIN_COOCCUR,
) -> nx.Graph:
    """Project post↔phrase bipartite to phrase↔phrase weighted graph."""
    post_phrases: dict[str, set[str]] = {}
    phrase_doc_freq: Counter[str] = Counter()

    for row in extractions.itertuples(index=False):
        post_phrases.setdefault(row.post_id, set()).add(row.phrase)
        phrase_doc_freq[row.phrase] += 1

    pair_counts: Counter[tuple[str, str]] = Counter()
    n_docs = len(post_phrases)

    for phrases in post_phrases.values():
        sorted_p = sorted(phrases)
        for i, a in enumerate(sorted_p):
            for b in sorted_p[i + 1 :]:
                pair_counts[(a, b)] += 1

    g = nx.Graph()
    for (a, b), count in pair_counts.items():
        if count < min_cooccur:
            continue
        p_a = phrase_doc_freq[a] / n_docs
        p_b = phrase_doc_freq[b] / n_docs
        p_ab = count / n_docs
        pmi = log((p_ab / (p_a * p_b)) + 1e-12)
        weight = max(pmi, 0.0) * count
        g.add_edge(a, b, weight=weight, count=count)

    for phrase, df in phrase_doc_freq.items():
        if phrase not in g:
            g.add_node(phrase, doc_freq=df)

    return g


def prune_graph(g: nx.Graph, *, min_weight: float = 0.1, max_degree: int = 200) -> nx.Graph:
    """Reduce dense phrase graphs before community detection."""
    g2 = nx.Graph()
    for u, v, d in g.edges(data=True):
        if float(d.get("weight", 0)) >= min_weight:
            g2.add_edge(u, v, **d)
    if g2.number_of_nodes() == 0:
        return g
    # Cap hub degree
    for node in list(g2.nodes()):
        if g2.degree(node) > max_degree:
            edges = sorted(g2[node].items(), key=lambda x: x[1].get("weight", 0), reverse=True)
            for neighbor, _ in edges[max_degree:]:
                g2.remove_edge(node, neighbor)
    return g2


def phrase_post_map(extractions: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in extractions.itertuples(index=False):
        out.setdefault(row.phrase, set()).add(row.post_id)
    return out
