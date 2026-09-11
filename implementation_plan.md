# Hiver SDE Intern Assignment — Implementation Plan

## Current State

- **Workspace**: `c:\Users\adity\Desktop\Coding\Hiver`
- **Virtual Environment**: Python 3.8, `.venv/` — exists with basic packages (pandas, matplotlib, seaborn, numpy, kagglehub). Many packages need to be added.
- **Dataset**: `kagglehub/datasets/thoughtvector/customer-support-on-twitter/versions/10/twcs/twcs.csv` — confirmed present. `response_tweet_id` is object dtype (can contain comma-separated IDs).
- **Existing work**: `dataset.py` — a Phase 1 profiling script written for Colab (saves to `/content/`). `dataset_analysis/` — 18 EDA chart files (PNG/CSV). The existing script needs to be adapted for local use.

## Project Structure (to build)

```
Hiver/
├── .venv/                     # Existing
├── data/
│   ├── raw/                   # Symlink/reference to kagglehub CSV
│   └── artifacts/             # All cached artifacts (frozen once generated)
│       ├── brand_scores.json
│       ├── selected_brand.txt
│       ├── clean_threads_{brand}.pkl
│       ├── split_ids.json         # Thread IDs per split (frozen)
│       ├── taxonomy.json          # Frozen intent definitions
│       ├── golden_set.csv         # Read-only after labeling
│       ├── judge_validation.csv
│       ├── resolution_pairs.pkl   # Retrieval corpus
│       ├── embeddings.npy         # Cached corpus embeddings
│       ├── faiss_index.bin
│       ├── bm25_index.pkl
│       ├── classifier.pkl         # Fitted classifier
│       ├── thresholds.json        # Fitted escalation thresholds
│       └── risk_coverage.json
├── src/
│   ├── phase1_profiling.py        # Full-corpus brand profiling & scoring
│   ├── phase2_brand_select.py     # Apply scoring rubric, commit to brand
│   ├── phase3_cleaning.py         # Single-brand clean + thread reconstruction
│   ├── phase4_split.py            # Temporal conversation-level split
│   ├── phase5_taxonomy.py         # Embedding clustering + LLM-assisted labeling
│   ├── phase6_golden_set.py       # Stratified sampling tool + labeling interface
│   ├── phase7_baselines.py        # E1–E4: trivial, TF-IDF, embedding-kNN, LLM
│   ├── phase8_retrieval.py        # E5–E7: BM25, dense, hybrid, reranker ablation
│   ├── phase9_generation.py       # E9: structured LLM generation + groundedness
│   ├── phase10_escalation.py      # Risk tiers, risk-coverage curve, E10
│   ├── phase11_finetune.py        # E8: fine-tuned encoder experiment
│   ├── phase12_eval_harness.py    # Full scorecard, E11 judge validation, E12 ablation
│   ├── phase13_failure_analysis.py # Systematic failure taxonomy
│   ├── pipeline.py                # Online inference pipeline (classifier→retrieval→generate→escalate)
│   └── utils/
│       ├── thread_utils.py        # Thread reconstruction, branch handling
│       ├── dedup_utils.py         # Exact + near-dup detection
│       ├── lang_utils.py          # Language ID
│       ├── pii_utils.py           # PII scan/re-scan
│       ├── retrieval_utils.py     # BM25 + FAISS hybrid retrieval helpers
│       ├── generation_utils.py    # LLM call wrapper, structured output parsing
│       ├── escalation_utils.py    # Signal wiring + decision function
│       ├── eval_utils.py          # Metric computation helpers
│       └── judge_utils.py         # LLM-as-judge rubric + scoring
├── config/
│   ├── brand_scoring_weights.yaml # Pre-committed weights (set once, not re-tuned)
│   ├── intent_risk_tiers.yaml     # Manual risk tier assignment per intent
│   ├── llm_config.yaml            # Model versions, temperature, endpoint
│   └── experiment_config.yaml     # Hyperparameters for each experiment
├── experiments/
│   └── results.json               # Append-only log of all experiment results
├── tests/
│   ├── test_thread_utils.py
│   ├── test_dedup_utils.py
│   ├── test_pii_utils.py
│   ├── test_escalation_utils.py
│   └── test_pipeline_schema.py
├── notebooks/                     # Optional EDA/exploration notebooks
├── report/
│   └── report.md                  # ≤6-page final report
├── requirements.txt               # Pinned dependencies
├── Makefile                       # Entry points: `make profile`, `make reproduce`, etc.
└── README.md                      # Fast path (15-min) + slow path documentation
```

## Technology Choices (following blueprint Section 20)

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.8 (existing venv) | Already set up |
| Data processing | pandas (already installed) | Dev/golden scale work |
| Language ID | `langdetect` or `langid` | Compact, fast, matches blueprint's "fastText-style" intent |
| Near-dup | `datasketch` (MinHash LSH) | Standard, efficient |
| Clustering | `hdbscan` | Density-aware, no fixed-k needed (blueprint Section 6.1) |
| Embeddings | `sentence-transformers` with `all-MiniLM-L6-v2` | Quality/cost/speed balance; local, no API cost for corpus embedding |
| Classical ML | `scikit-learn` | TF-IDF+linear baseline, embedding+linear head, calibration |
| Vector index | `faiss-cpu` | Local, no server, sufficient at tens-of-thousands scale |
| BM25 | `rank_bm25` | Lightweight standard implementation |
| Fine-tuning (if E8 justifies) | `transformers` + `peft` (LoRA) | Reproducible, fast iteration |
| LLM (generation + judge) | OpenAI API (`gpt-4o-mini` for generation, `gpt-4o` for judging) | Version-pinnable, structured output support, cost-effective |
| Reranker (E7 candidate) | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` | Small, fast, free |
| Testing | `pytest` | Standard |

> **Note on LLM**: I will use `gpt-4o-mini` for generation (cost-effective, structured output) and `gpt-4o` for judging (better at nuanced evaluation). Both will be version-pinned and temperature=0. If no OpenAI API key is available, I will stop and ask before proceeding with Phase 9.

## Execution Sequence

### ✅ Phase 0 — Environment setup
- Install all required packages into `.venv`
- Verify dataset schema

### Phase 1 — Full-corpus brand profiling
- Rewrite `dataset.py` as `src/phase1_profiling.py` (local paths, proper outputs)
- Run language ID, near-dup detection, thread reconstruction, per-brand stats
- Compute brand scorecard per Section 4.2

### Phase 2 — Brand selection (**HUMAN DECISION REQUIRED**)
- After profiling, I will present the scored brand table and ask you to confirm the selection before proceeding

### Phase 3 — Single-brand cleaning + thread reconstruction
- Filter to selected brand, extract visible-resolution pairs

### Phase 4 — Temporal split
- Conversation-level temporal split; freeze split IDs

### Phase 5 — Intent taxonomy discovery (**HUMAN REVIEW REQUIRED**)
- Run HDBSCAN clustering, LLM-assisted labeling
- Present proposed taxonomy for your review/merge/split
- Freeze only after your approval

### Phase 6 — Golden set construction (**HUMAN LABELING REQUIRED**)
- Stratified sampling per Section 14.1
- Build a clear labeling interface (CSV + instructions)
- You label: primary_intent, secondary_intent, should_escalate, reference_resolution_summary, case_type_tags
- Freeze after labeling

### Phase 7 — Baselines (E1–E4)
### Phase 8 — Retrieval system (E5–E7)
### Phase 9 — Response generation (E9) — **needs OpenAI API key**
### Phase 10 — Escalation (E10)
### Phase 11 — Fine-tuning experiment (E8)
### Phase 12 — Full evaluation harness (E11, E12)
### Phase 13 — Failure analysis
### Phase 14 — Final polish
### Phase 15 — README + reproducibility packaging
### Phase 16 — Final report

## Human Decision Points

1. **Brand selection** (after Phase 1): I will present the scored table and recommend a brand
2. **Taxonomy review** (after Phase 5 clustering): You review and approve/modify proposed intents
3. **Golden set labeling** (Phase 6): You label 150-250 examples using a provided tool
4. **API key**: OpenAI key needed before Phase 9 (generation)

## Starting Now

I will begin immediately with Phase 0 (environment setup) and Phase 1 (profiling).
