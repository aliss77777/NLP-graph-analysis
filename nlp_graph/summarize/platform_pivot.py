"""Platform × community pivot (alias)."""

from nlp_graph.summarize.top_terms import (
    build_communities_table,
    dominant_platform_per_community,
    platform_ownership_pivot,
    top_terms_per_community,
)

__all__ = [
    "platform_ownership_pivot",
    "dominant_platform_per_community",
    "top_terms_per_community",
    "build_communities_table",
]
