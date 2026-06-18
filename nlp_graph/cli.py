#!/usr/bin/env python3
"""CLI entry point for lexical KG build."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from nlp_graph.ingest.posts import load_posts, load_posts_bigquery, load_posts_parquet
from nlp_graph.pipeline import PipelineConfig, build_lexical_kg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build lexical knowledge graph from post corpus.")
    parser.add_argument("--input", help="CSV/JSONL/Parquet path")
    parser.add_argument("--bigquery", action="store_true", help="Load from insurance-intel BQ raw_posts")
    parser.add_argument("--synthetic-only", action="store_true", help="Exclude posts without source_channel")
    parser.add_argument("--output-dir", default="exports")
    parser.add_argument("--min-df", type=float, default=0.005)
    parser.add_argument("--resolution", type=float, default=0.75)
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

    if args.bigquery:
        posts = load_posts_bigquery()
    elif args.input:
        posts = load_posts(args.input)
    else:
        log.error("provide --input or --bigquery")
        return 1

    config = PipelineConfig(
        min_df=args.min_df,
        leiden_resolution=args.resolution,
        use_spacy=args.use_spacy,
        synthetic_only=args.synthetic_only,
    )
    result = build_lexical_kg(posts, output_dir, config=config)

    summary_path = output_dir / "lexical_kg_build_summary.json"
    summary = {**result.stats, "paths": {k: str(v) for k, v in result.paths.items()}, "log_file": str(log_path)}
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    log.info("done: communities=%s nodes=%s edges=%s elapsed=%ss", result.stats["community_count"], result.stats["node_count"], result.stats["edge_count"], result.stats["elapsed_sec"])
    log.info("summary -> %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
