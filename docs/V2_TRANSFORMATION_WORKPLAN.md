# V2 Transformation Work Plan

**Project:** NLP-graph-analysis → lexical knowledge graph for GEO (AI search optimization)  
**Consumer:** insurance-intel-dbx (DBX Phase 2)  
**Branch:** `v2_AI_update` (canonical; merge to `main` when stable)  
**Release:** `v2.0.0b1` — beta, independently verified 2026-06-18  
**Domain:** B2B SaaS — data/AI platforms (Databricks, Snowflake, dbt, etc.)

---

## 1. Why this transformation exists

The original repo (2020–2024) was a **research notebook pipeline**: Twitter ingest → CountVectorizer unigrams → Louvain → manual topic labels. It answered “what words co-occur?”

DBX Phase 2 needs something different: **literal buyer phrases** that map to AI search queries, clustered into question-oriented communities, with **platform ownership** per cluster — so content can be optimized for verbatim GEO match (ChatGPT, Perplexity, Gemini).

> “You had a topical cluster of keywords. Now you're going to have a topical cluster of **questions**.” — Guy Yalif, Webflow GEO framework

The v2 package (`nlp_graph/`) is a full rewrite of Steps 2–3 from the legacy notebooks, not a refactor of Step 1 (Twitter API retired).

---

## 2. Initial plan (W1–W4)

### Architecture chosen (rejected vs accepted)

| Approach | Verdict |
|----------|---------|
| **Rejected:** KG from post embeddings + ML `predicted_topic` co-occurrence | Produced **2 communities from 17k posts** — structural failure |
| **Accepted:** Lexical phrase graph — `post → phrase → phrase` via TF-IDF 2–4 grams + Leiden | **21 communities, 3,934 phrases**, ~16s offline |

### Build phases (from architectural memo)

| Phase | Scope | Target |
|-------|--------|--------|
| **0** | Deprecate old `kg_*.parquet`; mark legacy modules DO NOT LOAD | Done |
| **1** | Scaffold `nlp_graph` package (ingest, extract, graph, summarize, export) | Done |
| **2** | Phrase extraction pipeline — TF-IDF, stoplists, Leiden, platform pivot | Done (beta) |
| **3** | Wire into insurance-intel-dbx (`lexical_kg.py`, build script, staging load) | Partial |
| **4** | Agentic flywheel — `query_intent` bridge, LLM naming, content variants | W3+ |
| **5** | DBX Delta load + Sprint 3 gate + stable `v2.0.0` | W4 (B10 quota blocked) |

### Week sequencing

| Week | Planned | Status |
|------|---------|--------|
| **W1** | Channel audit, stoplists, package scaffold | ✅ Complete |
| **W2** | Offline build, P0/P1 fixes, checkpoint gates, beta tag | ✅ Complete → **v2.0.0b1** |
| **W3** | LLM naming, corpus rebalance, intent bridge, empty-cluster cleanup | 🔜 Next |
| **W4** | DBX DDL, live load, `graph_builder_agent`, stable release | Blocked on quota + W3 |

### Release tiers

| Tag | Criteria |
|-----|----------|
| `v2.0.0a1` | First lexical build; 19 communities; P0 bugs documented |
| **`v2.0.0b1`** | P0 fixed; filler gate honest; 21 communities; dominance distribution documented |
| `v2.0.0` | W3 + W4 complete; Sprint 3 gate; DBX load verified |

---

## 3. What we built (v2.0.0b1)

### Package layout

```
nlp_graph/
├── ingest/posts.py           # BQ / parquet / CSV; text_for_embedding
├── extract/
│   ├── lexical.py            # TF-IDF 2–4 grams + B2B seed patterns
│   ├── ngrams.py             # channel-aware vectorizer
│   ├── channel_stoplist.py   # junk filters + is_filler_for_metrics()
│   ├── entities.py           # KNOWN_ENTITIES, B2B_ALLOWLIST
│   ├── spacy_phrases.py      # stub (W3)
│   └── llm_phrases.py        # stub (W3)
├── graph/
│   ├── term_graph.py         # phrase co-occurrence (PMI)
│   └── communities.py        # Leiden (+ Louvain fallback)
├── summarize/
│   ├── post_community.py     # primary post assignment (P0)
│   ├── ranking.py            # tfidf × b2b_boost × query_shape_boost
│   └── top_terms.py          # community table + platform pivot
├── export/kg_parquet.py
└── pipeline.py               # end-to-end orchestration
```

### Key outputs

| Artifact | Purpose |
|----------|---------|
| `phrase_extractions.parquet` | post ↔ phrase (TF-IDF weight) |
| `post_community.parquet` | **one primary community per post** |
| `kg_communities.parquet` | Leiden clusters + ranked top_terms |
| `kg_platform_pivot.parquet` | platform × community ownership (from primary posts) |

### Verified beta metrics (v3-beta3)

| Metric | Value |
|--------|-------|
| Posts | 11,770 (B2B synthetic; Reddit/G2/LinkedIn/Twitter) |
| Communities | 21 |
| Post partition | 11,753 / 11,770 |
| Surfaced top-25 filler | **0 / 342 (0.0%)** — strict B2B classifier |
| 50-phrase eyeball sample | 50/50 pass |
| Platform dominance ≥40% | 7 / 21 communities |
| Platform dominance <20% | 10 / 21 communities (**54% of post volume**) |
| Runtime | ~16s (laptop) |
| Pricing / governance phrases | 60 / 9 (corpus gap — not pipeline) |

---

## 4. Friction points, refinements, and how we overcame them

### 4.1 Structural failure — 2 communities from embedding co-occurrence

**Symptom:** Original offline KG: 54 nodes, **2 communities**, ~10% flat platform dominance.  
**Cause:** Graph built from carrier × ML-predicted topic labels, not literal phrases.  
**Fix:** Lexical phrase co-occurrence graph (TF-IDF 2–4 grams + Leiden).  
**Learning:** Embeddings remain valid for **semantic search only**, not community detection for GEO.

---

### 4.2 P0 — Post–community overlap (99.9%)

**Symptom:** `post_count` summed posts touching any top phrase — massive double-counting.  
**Cause:** Communities live on phrases; multi-phrase posts scattered across clusters.  
**Fix:** `assign_post_primary_community()` — plurality vote per post → `post_community.parquet`.  
**Fix:** Recompute platform pivot from primary-assigned posts only.  
**Learning:** All downstream metrics (dominance, briefs) must use **primary post assignment**, not phrase-level joins.

---

### 4.3 P0 — Flat platform dominance (~10% each)

**Symptom:** Every platform ~10% in every community.  
**Cause:** `owned_by` counted global (phrase, platform) pairs, not community-scoped posts.  
**Fix:** Same as 4.2 — pivot from `post_community` + community filter.  
**Outcome:** Strong clusters now show real skew (e.g. dbt Cloud 46%, Vertex AI 44%, Fabric 80% on governance).  
**Caveat:** **54% of post volume** still sits in communities with <20% dominance — property of broad migration-narrative synthetic posts, not a logic bug.

---

### 4.4 Entity seed unigrams polluting the graph

**Symptom:** `dbt`, `snowflake`, `bi` at 17–22% corpus frequency; hub nodes distort ranking.  
**Fix:** Entity mentions → separate `entity` node_type; excluded from phrase co-occurrence graph.  
**Learning:** Platform names are **entities**, not searchable phrases, unless part of a multi-word query (“dbt Cloud vs dbt Core”).

---

### 4.5 Filler rate false pass (the critical metrics bug)

| Build | Claimed filler | Independent measurement |
|-------|----------------|-------------------------|
| v3-beta | ~0.2% | **39.6%** |
| v3-beta2 | 3.9% | **39.6%** (same pool; claim used filter pass-through) |
| **v3-beta3** | **0.0% surfaced** | **0/342 confirmed** + 50-phrase eyeball |

**Root causes (layered):**

1. **`has_content_token()` length heuristic** — any token ≥5 chars counted as “content” (`"worth it"`, `"designed to"` passed).
2. **Substring false positive** — `"power"` (allowlist) matched inside `"powerful"`.
3. **Self-validation blind spot** — measuring `is_junk_phrase()` pass-through on already-filtered terms, not B2B relevance.

**Fixes:**

- `has_content_token()` requires `B2B_ALLOWLIST` / `DOMAIN_SIGNAL_TERMS` with **word boundaries** only (no length fallback).
- `is_filler_for_metrics()` — independent classifier for checkpoint measurement.
- `ranking.py` filters top-25 with the same strict bar so **what we ship = what we measure**.
- **50-phrase eyeball sample** as second line of defense (keep this for every gate).

**Learning:** Never report a quality metric using the same filter that produced the artifact without independent review. Always spot-check 30–50 surfaced phrases manually.

---

### 4.6 Community count vs filter aggressiveness

**Symptom:** Tightening junk filters dropped communities (23 → 19 → 12 at one point).  
**Fix:** Decouple strict content check (metrics + ranking) from extraction `is_junk_phrase()` (lighter).  
**Outcome:** 21 communities with strict surfaced top_terms.  
**Learning:** Extraction breadth and **ranking quality** are separate knobs; don’t collapse them.

---

### 4.7 Cherry-picked dominance examples

**Symptom:** README highlighted 5 strong clusters; hid that 48–54% of volume has weak dominance.  
**Fix:** Report **distribution buckets** (≥40% / 20–40% / <20%) + volume-weighted framing.  
**Learning:** Gate verdicts need full distributions, not best-case examples.

---

### 4.8 Pricing / governance phrase starvation

**Symptom:** Migration ~1,240 phrases; governance **9**, pricing **60** (unchanged across three builds).  
**Cause:** Synthetic corpus generation weights migration-heavy (`synthetic/config.py`).  
**Fix:** Not a pipeline fix — requires **corpus rebalance + regen** (W3).  
**Learning:** `query_intent` bridge cannot close for governance/pricing until corpus work lands.

---

### 4.9 Operational blockers

| Blocker | Impact | Status |
|---------|--------|--------|
| B10 DBX SQL quota | No Delta load | Deferred W4 |
| spaCy / LLM extractors | Stubs only | W3 |
| `--all-posts` / Trustpilot rows | Dead code / noise | Removed |

---

## 5. Learnings to carry forward

1. **GEO needs literal phrases, not embedding similarity** — TF-IDF ngrams + question clusters, not cosine proximity on topics.
2. **Post-primary community is the canonical assignment model** — plurality of phrase memberships; all ownership metrics derive from this.
3. **Quality gates need independent classifiers + human samples** — filter pass-through is not a metric.
4. **Word-boundary matching for allowlists** — substring matches create silent false negatives/positives (`power`/`powerful`).
5. **Platform dominance is often weak for broad narrative clusters** — design W3 intent bridge and content variants assuming **dirty attribution** on ~54% of volume.
6. **Sparse / empty communities** — strict ranking leaves 8 communities with <10 surfaced phrases (2 empty); merge or drop at filter threshold in W3.
7. **`_has_proper_noun_brand()`** — harmless today because upstream lowercases all text; documented assumption so future refactors don’t reintroduce leaks.

---

## 6. Open work (W3 → W4)

### W3 — Intelligence layer (next)

- [ ] Corpus rebalance in `synthetic/config.py` — pricing/governance intent weights; regen `raw_posts`
- [ ] Drop/merge empty communities (18, 20) in `filter_small_communities`
- [ ] LLM **question-form** community naming (GEO memo §4a)
- [ ] LLM phrase extraction on cluster centroids (question-form primary)
- [ ] `query_intent` bridge → `engine_visibility` (account for weak-dominance volume)
- [ ] FAQ content variant strategy in content generator
- [ ] spaCy NER pass for multi-word product names (`dbt Cloud` vs `dbt`)

### W4 — Production

- [ ] `kg_*` DDL: `phrase`, `entity` node_types; `question_label`; `is_canonical`
- [ ] `load_kg_staging.py` live run (after B10 quota reset)
- [ ] `graph_builder_agent.py` reads question-form labels
- [ ] Sprint 3 KG checkpoint gate sign-off → tag **`v2.0.0`**

---

## 7. Integration with insurance-intel-dbx

```bash
pip install -e ~/Documents/NLP-graph-analysis[dev]
cd ~/Documents/insurance-intel-dbx
PYTHONPATH=. python scripts/build_lexical_kg_offline.py
```

Adapter: `semantic/lexical_kg.py`  
Build CLI: `scripts/build_lexical_kg_offline.py` (defaults: `resolution=1.5`, `min_posts=25`)

Diagnostics mirror: `insurance-intel-dbx/docs/lexical_kg_pipeline_diagnostics.md`

---

## 8. Document history

| Date | Milestone |
|------|-----------|
| 2026-06-18 | Alpha build (`v2.0.0a1`) — 19 communities; P0 bugs identified |
| 2026-06-18 | v3-beta2 — P0 fixes; false filler pass caught in review |
| 2026-06-18 | **v3-beta3 / v2.0.0b1** — filler gate verified; dominance distribution honest |
| TBD | W3 complete → release candidate |
| TBD | W4 DBX load → **v2.0.0** stable |

---

*Prepared for the v2_AI_update branch — Insurance Intel DBX Phase 2 / NLP-graph-analysis transformation.*
