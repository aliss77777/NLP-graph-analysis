#!/usr/bin/env python3
"""CLI entry point for lexical KG build."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from nlp_graph.ingest.posts import load_posts
from nlp_graph.pipeline import PipelineConfig, build_lexical_kg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build lexical knowledge graph from post corpus.")
    parser.add_argument("--input", required=True, help="CSV, JSONL, or Parquet path")
    parser.add_argument("--output-dir", default="exports")
    parser.add_argument("--min-df", type=float, default=0.005, help="Min document frequency (use ~0.02 for small samples)")
    parser.add_argument("--resolution", type=float, default=1.5, help="Leiden resolution parameter")
    parser.add_argument("--min-posts", type=int, default=25, help="Min posts per community after filter")
    parser.add_argument("--use-spacy", action="store_true")
    parser.add_argument("--log-file", default="")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = output_dir / f"lexical_kg_build_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    posts = load_posts(args.input)
    log.info("loaded %s posts from %s", len(posts), args.input)

    config = PipelineConfig(
        min_df=args.min_df,
        leiden_resolution=args.resolution,
        min_posts_per_community=args.min_posts,
        use_spacy=args.use_spacy,
    )
    result = build_lexical_kg(posts, output_dir, config=config)

    summary_path = output_dir / "lexical_kg_build_summary.json"
    summary = {
        **result.stats,
        "paths": {k: str(v) for k, v in result.paths.items()},
        "log_file": str(log_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    log.info(
        "done: communities=%s phrases=%s post_partition_sum=%s elapsed=%ss",
        result.stats["community_count"],
        result.stats["unique_phrases"],
        result.stats.get("post_partition_sum"),
        result.stats["elapsed_sec"],
    )
    log.info("summary -> %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
