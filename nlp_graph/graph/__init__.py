"""Graph construction — adjacency, term graph, communities."""

from nlp_graph.graph.adjacency import (
    adjacency_to_doc_term_matrix,
    create_adjacency_list_from_matrix,
    phrase_extractions_to_adjacency,
)
from nlp_graph.graph.communities import (
    community_stats,
    detect_communities_leiden,
    detect_communities_louvain,
    filter_small_communities,
)
from nlp_graph.graph.term_graph import build_phrase_cooccurrence_graph, phrase_post_map

__all__ = [
    "phrase_extractions_to_adjacency",
    "adjacency_to_doc_term_matrix",
    "create_adjacency_list_from_matrix",
    "build_phrase_cooccurrence_graph",
    "phrase_post_map",
    "detect_communities_leiden",
    "detect_communities_louvain",
    "filter_small_communities",
    "community_stats",
]
