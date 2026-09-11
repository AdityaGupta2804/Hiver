# Hiver SDE Internship Assignment: Twitter Customer Support AI System

An end-to-end, reproducible, and verifiable customer support response and escalation system built on the multi-million Twitter Customer Support dataset (`thoughtvector/customer-support-on-twitter`), adhering strictly to the **Technical Research Blueprint**.

---

## Architecture Overview

The system decomposes customer support into five decoupled, measurable stages:

```
[Customer Tweet]
       │
       ▼
[Stage 1: PII Redaction & Normalization]
       │
       ▼
[Stage 2: Calibrated Intent Classifier] (embedding + linear head / fine-tuned encoder)
       │
       ▼
[Stage 3: Intent-Conditioned Hybrid Retrieval] (BM25 + FAISS Dense + Reciprocal Rank Fusion)
       │
       ▼
[Stage 4: Structured Response Generator] (Evidence citations [doc_1] + Groundedness audit)
       │
       ▼
[Stage 5: Multi-Signal Risk-Tiered Escalation] (Threshold tau* from Dev Risk-Coverage curve)
       │
       ├─────────────────────────────────┬────────────────────────────────┐
       ▼                                 ▼                                ▼
[Autonomous Response]         [Escalated to Specialist]         [Safety Hard-Override]
(High confidence & evidence)   (Uncertainty / Missing info)    (Security / Billing dispute)
```

---

## Key Experimental Results (E1 – E12)

| Experiment | Component | Baseline | Proposed System | Delta / Significance |
|---|---|---|---|---|
| **E1 – E4** | Intent Classification | Majority Class (0.18 F1) | Embedding Classifier (0.88 Macro-F1) | **+0.70 F1** (Cleared all baselines) |
| **E5 – E7** | Retrieval System | BM25 Lexical (0.64 MRR) | Intent-Conditioned Hybrid (0.83 MRR) | **+0.19 MRR** (Intent filtering cuts false matches) |
| **E8** | Encoder Fine-Tuning | Embedding Classifier (0.881) | Fine-Tuned Head (0.892) | **+0.011 F1** (95% CI crosses threshold; retained baseline) |
| **E9** | Generation Groundedness | Zero-Shot (0.28 Groundedness) | Retrieval-Grounded (0.86 Groundedness) | **+0.58 Groundedness Lift** (Zero citation theater) |
| **E10** | Escalation Layer | Single-Threshold (0.46 Loss) | Multi-Signal Risk Tiering (0.19 Loss) | **58.7% Cost Reduction** (False auto rate: 4.2%) |
| **E11** | Judge Validation | Human Double-Labels | LLM Judge Rubric | **Cohen's Kappa 0.82**, **Spearman 0.86** |
| **E12** | Full System Ablation | Full System (4.52 Score) | Component Drops (2.65 - 4.18 Score) | Proves all components are load-bearing |

---

## 15-Minute Fast Reproduction Path

To verify headline results without re-indexing millions of tweets:

```bash
# 1. Activate virtual environment
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate

# 2. Run unit test suite
python -m pytest tests/ -v

# 3. Run full evaluation scorecard (Phase 12)
python src/phase12_eval_harness.py --brand SpotifyCares

# 4. Run interactive inference pipeline
python src/pipeline.py --interactive
```

Or run via `make`:
```bash
make reproduce
```

### Run With Docker (Single Command Deliverable)
```bash
# Build the Docker image
docker build -t hiver-customer-support .

# Run the complete deliverable (tests + scorecard + failure analysis + sample inference)
docker run --rm hiver-customer-support

# Or run interactively
docker run --rm -it hiver-customer-support python src/pipeline.py --interactive
```
Or via `docker compose`:
```bash
docker compose up
```

---

## Slow Path: Full Corpus Pipeline Regeneration

To regenerate all frozen artifacts from the raw CSV:

```bash
# Phase 1: Full-corpus profiling & brand scorecard
python src/phase1_profiling.py

# Phase 2: Brand selection (SpotifyCares selected with written justification)
python src/phase2_brand_select.py --brand SpotifyCares

# Phase 3: Single-brand cleaning & branch-aware thread reconstruction
python src/phase3_cleaning.py --brand SpotifyCares

# Phase 4: Temporal, conversation-level data split (retrieval/dev/golden)
python src/phase4_split.py --brand SpotifyCares

# Phase 5: Intent discovery & taxonomy freeze
python src/phase5_taxonomy.py --brand SpotifyCares --propose
python src/phase5_taxonomy.py --brand SpotifyCares --freeze

# Phase 6: Golden set construction & validation subset
python src/phase6_golden_set.py --brand SpotifyCares --sample
python src/phase6_golden_set.py --brand SpotifyCares --check

# Phase 7: Baselines E1-E4
python src/phase7_baselines.py --brand SpotifyCares --run all

# Phase 8: Hybrid retrieval index & E5-E7 ablation
python src/phase8_retrieval.py --brand SpotifyCares

# Phase 9: Structured response generation & E9 ablation
python src/phase9_generation.py --brand SpotifyCares

# Phase 10: Escalation layer risk-coverage curve fitting & E10
python src/phase10_escalation.py --brand SpotifyCares

# Phase 11: Fine-tuning bootstrap experiment E8
python src/phase11_finetune.py --brand SpotifyCares

# Phase 12: Full evaluation harness & scorecard
python src/phase12_eval_harness.py --brand SpotifyCares

# Phase 13: Systematic failure analysis
python src/phase13_failure_analysis.py --brand SpotifyCares
```

---

## Repository Structure

```
Hiver/
├── config/                        # Pre-committed weights and risk tiers
│   ├── brand_scoring_weights.yaml
│   ├── intent_risk_tiers.yaml
│   ├── llm_config.yaml
│   └── experiment_config.yaml
├── data/
│   └── artifacts/                 # Versioned, frozen artifacts (golden set, taxonomy, index)
├── experiments/                   # Experiment output logs (E1 through E12)
├── report/
│   └── report.md                  # Comprehensive final report
├── src/
│   ├── phase1_profiling.py        # Full corpus brand scorecard
│   ├── phase2_brand_select.py     # Data-grounded brand selection
│   ├── phase3_cleaning.py         # Thread reconstruction & visible resolution mining
│   ├── phase4_split.py            # Temporal conversation-level data splitting
│   ├── phase5_taxonomy.py         # Intent discovery & freezing
│   ├── phase6_golden_set.py       # Stratified evaluation set sampling
│   ├── phase7_baselines.py        # E1-E4 Baselines
│   ├── phase8_retrieval.py        # E5-E7 Hybrid retrieval system
│   ├── phase9_generation.py       # E9 Generation with citations
│   ├── phase10_escalation.py      # E10 Risk-coverage curve & escalation
│   ├── phase11_finetune.py        # E8 Fine-tuning ablation & bootstrap CI
│   ├── phase12_eval_harness.py    # E11 Judge validation & E12 Scorecard
│   ├── phase13_failure_analysis.py# Top-5 failure modes & mitigations
│   ├── pipeline.py                # Unified online inference engine
│   └── utils/                     # Modular utility libraries
├── tests/                         # Automated pytest suite
├── Makefile                       # Fast & slow path automation targets
├── requirements.txt               # Pinned dependencies
└── README.md
```
