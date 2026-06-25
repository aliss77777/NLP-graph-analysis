# Example data

## `posts_sample.parquet` / `posts_sample.csv`

A **1% stratified sample** (by `source_channel`) drawn from a larger synthetic B2B SaaS post corpus used during package development.

| Channel | Posts in sample |
|---------|-----------------|
| reddit | 50 |
| twitter | 31 |
| linkedin | 24 |
| g2 | 12 |
| **Total** | **117** |

**Disclaimer:** These posts are **synthetic** — generated for research and pipeline testing. They are not real user content or scraped social media. Use only to explore the `nlp_graph` pipeline.

### Columns

`post_id`, `source_channel`, `platform_mentioned`, `carrier`, `intent_category`, `title`, `body`, `text_for_embedding`

### Toy run

```bash
pip install -e ".[dev]"
build-lexical-kg \
  --input examples/posts_sample.parquet \
  --output-dir exports \
  --min-df 0.02 \
  --min-posts 5 \
  --resolution 1.0
```

On ~117 posts, use a higher `--min-df` and lower `--min-posts` than the defaults tuned for 10k+ post corpora.
