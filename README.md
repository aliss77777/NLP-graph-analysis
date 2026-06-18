# NLP Graph Analysis v2

> **Release: `v2.0.0b1` (beta)** on branch **`v2_AI_update`** — verified offline lexical KG.  
> Not production-ready for DBX load until W4 (quota + Sprint 3 gate). See [docs/V2_TRANSFORMATION_WORKPLAN.md](docs/V2_TRANSFORMATION_WORKPLAN.md).

**Lexical knowledge graphs for AI search optimization (GEO).**

Extract the exact phrases B2B buyers use in discussions, cluster them into communities, and map platform ownership — so content can match verbatim AI search queries (ChatGPT, Perplexity, Gemini).

Evolution of the 2020–2024 [NLP-graph-analysis](https://github.com/aliss77777/NLP-graph-analysis) research notebooks (spaCy + CountVectorizer + Louvain). Step 1 (Twitter API ingestion) is retired. Steps 2–3 are ported to this package with a modern stack.

## What's new in v2

| v1 (notebooks) | v2 (package) |
|----------------|--------------|
| Unigram CountVectorizer | **2–4 gram TF-IDF** + B2B seed patterns |
| Generic co-occurrence topics | **Phrase graph** + Leiden communities |
| Louvain only | **Leiden** (Louvain fallback) |
| Manual topic names | LLM question-form naming (W3 stub) |
| Twitter CSV ingest | BQ / Parquet / CSV |
| Jupyter-only | `pip install -e .` + CLI |

## Transformation docs

**Start here:** [docs/V2_TRANSFORMATION_WORKPLAN.md](docs/V2_TRANSFORMATION_WORKPLAN.md)

Covers the initial plan, friction points (2-community failure, filler false pass, dominance bugs), fixes, verified beta metrics, and W3/W4 roadmap.

## Install

```bash
git clone -b v2_AI_update https://github.com/aliss77777/NLP-graph-analysis.git
cd NLP-graph-analysis
pip install -e ".[dev]"
pytest
```

## Quick start

```bash
# CLI (from parquet)
build-lexical-kg --input posts.parquet --output-dir exports

# Via insurance-intel-dbx consumer
cd ~/Documents/insurance-intel-dbx
pip install -e ~/Documents/NLP-graph-analysis[dev]
PYTHONPATH=. python scripts/build_lexical_kg_offline.py
```

## Outputs

| File | Description |
|------|-------------|
| `phrase_extractions.parquet` | post_id, phrase, score, channel, platform |
| `post_community.parquet` | primary community assignment per post |
| `kg_nodes.parquet` | phrase + entity + platform nodes |
| `kg_edges.parquet` | co_occurs + owned_by edges |
| `kg_communities.parquet` | Leiden clusters + top_terms |
| `kg_platform_pivot.parquet` | platform×community ownership |

## Beta metrics (v3-beta3, verified)

| Gate | Result |
|------|--------|
| Communities | 21 |
| Post partition | 11,753 / 11,770 |
| Surfaced top-25 filler | 0 / 342 (0.0%) |
| Runtime | ~16s on 11.7k posts |

## Package layout

```
nlp_graph/
├── ingest/       # posts from BQ, parquet, CSV
├── extract/      # TF-IDF phrases, stoplists, entities
├── graph/        # co-occurrence graph, Leiden
├── summarize/    # ranking, post-primary communities, pivot
├── export/       # kg_* parquet schema
└── pipeline.py   # end-to-end build
```

## Legacy notebooks

Preserved for reference (v1 research):

- `Social Content Analysis Step 1_DL tweets.ipynb` — deprecated
- `Social Content Analysis Step_2_adjacency list.ipynb`
- `Social Content Analysis Step_3_merge and analyze.ipynb`

## Integration

Used by **insurance-intel-dbx** DBX Phase 2:

- `semantic/lexical_kg.py` — adapter
- `scripts/build_lexical_kg_offline.py` — offline build

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | v1 notebooks (legacy) |
| **`v2_AI_update`** | **Canonical v2 lexical KG** — merge to main when stable |

## License

MIT
