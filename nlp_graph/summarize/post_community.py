"""Post-primary community assignment (plurality of phrase memberships)."""

from __future__ import annotations

import pandas as pd


def assign_post_primary_community(
    extractions: pd.DataFrame,
    partition: dict[str, int],
) -> pd.DataFrame:
    """One primary community per post via plurality of assigned phrases."""
    mapped = extractions.copy()
    mapped["community_id"] = mapped["phrase"].map(partition)
    mapped = mapped[mapped["community_id"].notna() & (mapped["community_id"] >= 0)]

    rows: list[dict] = []
    for post_id, grp in mapped.groupby("post_id"):
        votes = grp.groupby("community_id").size()
        primary = int(votes.idxmax())
        platform = grp["platform"].mode().iloc[0] if not grp["platform"].mode().empty else grp["platform"].iloc[0]
        rows.append(
            {
                "post_id": str(post_id),
                "community_id": primary,
                "platform": platform,
                "phrase_votes": int(votes.max()),
            }
        )
    return pd.DataFrame(rows)


def platform_ownership_from_posts(post_community: pd.DataFrame) -> pd.DataFrame:
    """Platform × community counts from primary-assigned posts only."""
    if post_community.empty:
        return pd.DataFrame(columns=["community_id", "platform", "post_count"])
    pc = post_community[post_community["platform"].notna() & (post_community["platform"] != "unknown")]
    pivot = (
        pc.groupby(["community_id", "platform"])
        .size()
        .reset_index(name="post_count")
        .sort_values(["community_id", "post_count"], ascending=[True, False])
    )
    return pivot
