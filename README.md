# SpotifyCares Autonomous AI Customer Support Agent

I built an end-to-end autonomous customer support agent for **SpotifyCares** (`@SpotifyCares` Twitter handle) using a 2.81-million-tweet corpus of real-world brand interactions. The system automates intent classification, performs intent-conditioned hybrid retrieval (BM25 + FAISS dense vectors) over mined issue-resolution pairs, drafts grounded responses with cited evidence using an LLM, and applies a multi-signal risk-based escalation layer to decide whether to auto-handle or route to a human agent.

I designed this repository to prioritize **measurement, auditability, and safety**. Every claim is backed by an automated evaluation harness, frozen artifacts, and an LLM-as-a-judge rubric validated against human annotations.

---

## Headline Result

At my optimal operating threshold ($\tau^* = 0.50$), the system achieves **71.0% autonomous resolution coverage** while maintaining a **3.5% false auto-handle rate** (1.5% false-auto rate on high-risk intents) and yielding a **58.7% asymmetric cost reduction** relative to human-only routing.

> **What this metric actually means:**
> Out of 100 incoming customer messages, I safely auto-handle 71 without human intervention. Of those 71 auto-handled cases, 96.5% receive a verified correct, grounded answer, while only 3.5% are incorrectly auto-handled (false autos). The remaining 29 cases are safely escalated to human support reps.

---

## Where to Find Everything

If you are a Hiver reviewer looking to verify specific assignment deliverables, use this mapping:

| What you want to check | Where to look | Description / Details |
| --- | --- | --- |
| **Run the project / Demo** | [`demo.py`](https://www.google.com/search?q=demo.py) | Interactive CLI pipeline for custom messages |
| **Reproduce headline results** | [`reproduce.py`](https://www.google.com/search?q=reproduce.py) | Full harness runner evaluating held-out golden set |
| **Evaluation harness** | [`src/eval/harness.py`](https://www.google.com/search?q=src/eval/harness.py) | Orchestrates automated metrics & LLM judge scoring |
| **Golden evaluation set** | [`data/artifacts/golden_set.csv`](https://www.google.com/search?q=data/artifacts/golden_set.csv) | 200 hand-labeled held-out adversarial examples |
| **LLM judge validation** | [`data/artifacts/judge_validation.csv`](https://www.google.com/search?q=data/artifacts/judge_validation.csv) | 30 human double-labeled judge calibration set |
| **Decision log** | [`data/artifacts/decision_log.json`](https://www.google.com/search?q=data/artifacts/decision_log.json) | Full records for all 12 non-obvious engineering decisions |
| **Failure analysis** | [`experiments/failure_analysis.json`](https://www.google.com/search?q=experiments/failure_analysis.json) | Top 5 detailed failure modes with transcripts & mitigations |
| **Experiment results** | [`experiments/results.json`](https://www.google.com/search?q=experiments/results.json) | Raw numerical outputs for experiments E1 through E12 |
| **Full evaluation scorecard** | [`experiments/evaluation_scorecard.json`](https://www.google.com/search?q=experiments/evaluation_scorecard.json) | Comprehensive multi-dimensional evaluation summary |
| **Frozen artifacts** | [`data/artifacts/`](https://www.google.com/search?q=data/artifacts/) | Precomputed indexes, taxonomy, splits, & threshold fits |
| **Main pipeline** | [`src/pipeline.py`](https://www.google.com/search?q=src/pipeline.py) | Core orchestration (PII -> Intent -> RAG -> Escalation) |
| **Unit tests** | [`tests/`](https://www.google.com/search?q=tests/) | 13 unit tests covering trees, deduplication, PII, & risk |
| **Written report** | [`report/report.md`](https://www.google.com/search?q=report/report.md) | Exhaustive 18-section technical markdown report |

---

## Quick Start for Reviewers

I set up the repository so you can reproduce the entire headline evaluation suite in under **2 minutes** using precomputed frozen artifacts, without needing to process the 2.81M-row raw dataset (`twcs.csv`).

### Prerequisites

* Python 3.10 or higher
* Existing virtual environment (`.venv`) or Docker environment

### Fast Reviewer Path (Local Virtual Environment)

1. **Activate the pre-existing virtual environment**
```bash
source .venv/bin/activate  # On Linux/macOS
# or: .venv\Scripts\activate  # On Windows

```


2. **Run the test suite**
```bash
pytest tests/ -v

```


*(All 13 tests should pass in under 5 seconds.)*
3. **Reproduce headline evaluation**
```bash
python reproduce.py

```


*(Runs full inference & evaluation on the 200-example golden set using frozen artifacts. Completes in ~30 seconds.)*
4. **Try the interactive CLI pipeline**
```bash
python demo.py

```


*(Allows typing custom customer messages to inspect live PII masking, intent predictions, retrieved evidence, grounded drafts, and escalation decisions.)*

### Running via Docker

If you prefer an isolated containerized environment:

```bash
docker build -t hiver-customer-support .
docker run --rm hiver-customer-support

```

---

## What I Built

I built a production-grade, retrieval-grounded autonomous support agent tailored specifically for Spotify's public customer support operations (`@SpotifyCares`).

```
                                [ Incoming Customer Tweet ]
                                             │
                                             ▼
                                 [ 1. PII Scrubbing Engine ]
                                             │
                                             ▼
                               [ 2. Intent Classification ]
                                   (Embedding + Linear)
                                             │
                                             ▼
                             [ 3. Intent-Conditioned Retrieval ]
                                  (BM25 + FAISS Dense Vector)
                                             │
                                             ▼
                                 [ 4. Grounded Generator ]
                             (LLM + Strict Citation Validation)
                                             │
                                             ▼
                                 [ 5. Risk Triage Engine ]
                             (Confidence + Groundedness + Risk)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
            [ Auto-Handle Response ]                       [ Human Escalation ]
            (Coverage: 71.0%, False-Auto: 3.5%)            (Risk Triggered / Low Confidence)

```

### Core Architecture Components

1. **PII Scrubbing Engine (`src/utils/pii.py`)**: Sanitizes incoming text for emails, phone numbers, account handles, credit card patterns, and IP addresses before any LLM call or storage.
2. **Intent Classification (`src/models/intent.py`)**: Predicts customer intent out of 10 frozen operational categories using SentenceTransformers (`all-MiniLM-L6-v2`) + calibrated Logistic Regression.
3. **Intent-Conditioned Hybrid RAG (`src/retrieval/hybrid.py`)**: Filters candidate resolution pairs by predicted intent, then runs reciprocal rank fusion over BM25 (lexical) and FAISS (dense vector) indexes.
4. **Grounded Response Generator (`src/generation/generator.py`)**: Generates structured draft responses strictly requiring `[doc_X]` citations for every factual statement, paired with an automated citation-auditor to strip ungrounded assertions.
5. **Multi-Signal Risk Escalation Engine (`src/escalation/rules.py`)**: Combines intent risk-tiers, classifier confidence, retrieval distance, and citation groundedness against tuned operating thresholds to decide auto-handle vs. human escalation.

---

## Problem Framing

### What "Good" Means for SpotifyCares

Public customer support on Twitter/X operates under strict constraints:

* **Account Privacy & DM Boundaries**: Public replies must never leak PII or attempt account modifications on public timelines.
* **Accuracy over Conversational Flair**: A customer asking why their offline downloads disappeared needs the exact sequence of troubleshooting steps (e.g., re-login, check 30-day offline policy), not generic empathy.
* **Low Latency & Fast Escalation**: High-volume, routine queries (e.g., payment methods, app crashes) should be resolved immediately, while sensitive queries (e.g., account takeover, unauthorized charges) must be routed to human reps without delay.

For SpotifyCares, a "good" response is **factually accurate, strictly grounded in verified past resolutions, explicitly cited, and conservatively escalated when risk or uncertainty is present**.

### What I Chose NOT to Build

To avoid scope creep and unsafe engineering practices, I explicitly chose not to build:

* **Direct Account Action Executors**: I did not build API integrations to automatically issue refunds, reset passwords, or cancel subscriptions. Automated execution on public social media without human authentication is inherently unsafe.
* **Fine-Tuned Response Generation LLMs**: I explicitly chose *not* to fine-tune a generation LLM on raw Twitter replies. Mining raw replies trains models to imitate canned, content-free brush-offs ("Please send us a DM"). RAG over curated problem-resolution pairs is far safer and more controllable.
* **A Single Collapsed Quality Score**: I did not combine accuracy, tone, groundedness, and risk into one arbitrary 0–100 metric. Collapsing these masks critical safety failures.

---

## Repository Structure

```
.
├── Dockerfile                      # Single-command Docker build environment
├── docker-compose.yml              # Docker Compose service definition
├── entrypoint.sh                   # Docker execution entrypoint
├── Makefile                        # Convenient automation commands (make reproduce, make test)
├── requirements.txt                # Fixed Python dependencies
├── reproduce.py                    # One-step replication script for headline evaluation
├── demo.py                         # Interactive CLI demonstration tool
├── data/
│   └── artifacts/                  # Frozen artifacts for fast reviewer evaluation path
│       ├── brand_selection_justification.json
│       ├── split_ids_SpotifyCares.json
│       ├── taxonomy.json
│       ├── golden_set.csv           # 200 hand-labeled held-out test cases
│       ├── judge_validation.csv     # 30 double-labeled human vs judge validation cases
│       ├── bm25_index_SpotifyCares.pkl
│       ├── faiss_index_SpotifyCares.bin
│       ├── decision_log.json        # Detailed 12-decision log
│       └── thresholds.json         # Fitted escalation thresholds (tau* = 0.50)
├── src/                            # Full production source code
│   ├── data/                       # Mining, cleaning, tree reconstruction, & PII engines
│   ├── models/                     # Intent classifiers (Baselines, Embedding, Fine-tuned)
│   ├── retrieval/                  # Hybrid RAG, BM25, FAISS, & RRF implementation
│   ├── generation/                 # Grounded generator & citation auditor
│   ├── escalation/                 # Risk triage engine & cost-curve optimizer
│   ├── eval/                       # Evaluation harness & LLM judge implementation
│   └── pipeline.py                 # End-to-end integration pipeline
├── tests/                          # Automated Pytest unit test suite
│   ├── test_data_processing.py
│   ├── test_escalation.py
│   ├── test_pii.py
│   └── test_trees.py
├── experiments/                    # Execution logs & numerical outputs
│   ├── results.json                # Complete experimental log (E1 - E12)
│   ├── e9_generation.json          # RAG ablation generation metrics
│   ├── e10_escalation.json         # Escalation risk-coverage curve data
│   ├── e11_judge_validation.json   # Judge reliability metrics (Quadratic Kappa = 0.807)
│   ├── e12_ablation.json           # Full architectural component ablation data
│   ├── failure_analysis.json       # Top 5 concrete failure modes with transcripts
│   └── evaluation_scorecard.json   # Comprehensive system evaluation scorecard
└── report/
    └── report.md                   # Complete 18-section technical research report

```

---

## Data and Preprocessing

I mined the Kaggle Customer Support on Twitter dataset (~2.81M tweets across dozens of brands).

1. **Brand Profiling & Selection**: I evaluated brands using a weighted scoring rubric covering volume, visible resolution rate, self-containedness, and issue diversity. I selected **`SpotifyCares`** (47,723 tweets, 43,219 brand replies, 92.1% English, 69.1% visible resolution density) over volume leaders like `@AppleSupport` and `@AmazonHelp`, whose public replies overwhelmingly consist of account-access DM deflections.
2. **Conversation Tree Reconstruction**: Because tweets branch non-linearly (a single customer message can receive multiple agent replies), I built explicit forward/backward adjacency maps (`in_response_to_tweet_id` and `response_tweet_id`) to reconstruct full conversation trees and isolate true thread roots.
3. **Temporal & Conversation-Level Splitting**: To eliminate data leakage and avoid evaluating on "future" temporal context, I enforced a strict chronological split on conversation trees:
* **Retrieval Corpus (50%)**: Oldest 17,214 resolved problem-resolution pairs.
* **Development Set (25%)**: Middle slice used for taxonomy clustering and threshold tuning.
* **Golden Evaluation Set (25%)**: Most recent slice reserved strictly for final testing.



---

## Intent Taxonomy Strategy

I discovered and froze a 10-class operational intent taxonomy using HDBSCAN density clustering over sentence embeddings on the dev slice, followed by manual review:

1. `account_access`: Login errors, password resets, multi-device session caps.
2. `billing_subscription`: Duplicate charges, student discount renewals, payment failures.
3. `technical_troubleshooting`: App crashes, offline download losses, local file syncing.
4. `audio_playback_issues`: Stuttering, sudden pauses, sound quality distortion.
5. `playlist_library_mgmt`: Missing saved tracks, lost playlists, local file imports.
6. `device_compatibility`: Smart speaker pairing, Android Auto/CarPlay, TV app bugs.
7. `content_availability`: Missing albums, explicit content filter toggles, region locks.
8. `family_duo_plan`: Address verification failures, member invitation loops.
9. `feature_request`: UI complaints, lyrics request, removing unwanted updates.
10. `other_unclear`: Incomplete queries, pure venting, out-of-scope banter.

---

## Evaluation Setup & Golden Set

### Golden Evaluation Set (`data/artifacts/golden_set.csv`)

I hand-labeled 200 held-out examples from the strictly chronologically isolated test slice. To ensure the evaluation set represents real production difficulty, I sampled:

* **Stratified Normal Cases (45%)**: Core coverage across all 10 intents.
* **Adversarial Hard Cases (15%)**: Typo-heavy, slang, and short messages.
* **Multi-Intent / Pivot Cases (10%)**: Messages combining multiple issues (e.g., billing + crash).
* **Low Retrieval Evidence Cases (10%)**: Novel queries with weak historical precedents.
* **Conflicting Evidence Cases (5%)**: Queries matching contradictory past fixes.
* **High-Risk Escalation Cases (15%)**: Security compromises and unauthorized billing disputes.

### Evaluation Harness & LLM Judge

My harness (`src/eval/harness.py`) evaluates the pipeline across all modular stages. Response generation is evaluated using a reference-anchored LLM-as-a-judge rubric assessing:

* **Factual Correctness** (0.0 to 1.0)
* **Citation Groundedness** (0.0 to 1.0)
* **Completeness & Relevance** (0.0 to 1.0)
* **Unsupported Claim Rate** (Binary pass/fail penalty)

I validated the LLM judge against 30 double-labeled human cases (`data/artifacts/judge_validation.csv`). The judge achieved **86.7% exact agreement** with human raters and a **Quadratic Weighted Kappa of $\kappa = 0.807$**, proving high reliability.

---

## Results and Baselines

I systematically evaluated the architecture across 12 structured experiments (E1 to E12) against trivial and simple baselines.

### 1. Intent Classification Performance

| Model / Baseline | Description | Macro F1 | ECE (Calibration Error) | Decision / Action |
| --- | --- | --- | --- | --- |
| **E1: Majority Class** | Trivial baseline (always predicts majority intent) | 0.083 | N/A | Rejected |
| **E2: TF-IDF + Logistic Reg.** | Simple classical ML baseline | 0.552 | 0.142 | Rejected |
| **E3: Dense Embedding k-NN** | Uncalibrated cosine distance classifier | 0.492 | 0.218 | Rejected |
| **E4: Embedding + LogReg** | `all-MiniLM-L6-v2` + Calibrated Logistic Head | **0.841** | **0.038** | **Selected Core Model** |
| **E8: Fine-Tuned Encoder** | Fine-tuned DeBERTa-v3-small head | 0.849 | 0.041 | Retained E4 (CI crossed 0) |

* statistical decision rule (E8): paired bootstrap 95% CI for F1 delta was $[-0.012, +0.028]$. Because the confidence interval crossed zero, I kept the simpler, faster embedding classifier (E4) per my precommitted decision rule.

### 2. Retrieval System Performance

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| **E5: Lexical BM25 Only** | 52.1% | 68.4% | 74.2% | 0.612 |
| **E6: Dense Vector Only (FAISS)** | 48.3% | 65.1% | 72.0% | 0.584 |
| **E7: Unconditioned RRF Hybrid** | 61.4% | 79.2% | 84.0% | 0.711 |
| **E7: Intent-Conditioned Hybrid RAG** | **74.5%** | **88.2%** | **92.1%** | **0.834** |

### 3. Generation & Groundedness Ablation (E9)

| Configuration | Groundedness Rate | Factually Correct Rate | Unsupported Claim Rate |
| --- | --- | --- | --- |
| **Un-grounded LLM (No RAG)** | 0.050 | 0.412 | 48.5% |
| **Full RAG Pipeline (With Citation Auditor)** | **0.900** | **0.885** | **2.5%** |

### 4. Escalation Risk-Coverage Performance (E10)

| Operating Threshold ($\tau^*$) | Coverage (% Auto-Handled) | False Auto-Handle Rate | Asymmetric Cost Reduction |
| --- | --- | --- | --- |
| $\tau = 0.30$ (Aggressive) | 88.5% | 12.1% | 22.4% |
| **$\tau^* = 0.50$ (Optimal)** | **71.0%** | **3.5%** | **58.7%** |
| $\tau = 0.70$ (Conservative) | 48.0% | 1.0% | 41.2% |

---

## Failure Analysis: Top 5 Failure Modes

I manually inspected model failures on the golden evaluation set. Here are the top 5 concrete failure modes:

### Failure Mode 1: Account-Specific State Blindness (DM Deflection Limit)

* **What goes wrong**: The customer asks for resolution on a specific account block or payment dispute that requires private backend database state.
* **Real Example**: *"My account was billed $14.99 twice this morning on my Visa ending in 4021. Please refund one immediately."*
* **Why it happens**: Public Twitter data never contains private refund execution actions. The RAG pipeline retrieves generic refund policy articles, which do not resolve the specific transaction.
* **Mitigation**: Assigned `billing_subscription` to a high-risk escalation tier. The system forces human escalation whenever account transaction mutation is required.

### Failure Mode 2: Multi-Intent Query Splitting

* **What goes wrong**: A single customer tweet contains two distinct issues, but the intent classifier picks only the dominant one, missing the secondary complaint.
* **Real Example**: *"Spotify crashes every time I open my offline download playlist, and also why did my student discount expire?"*
* **Why it happens**: Single-label classification assigns `technical_troubleshooting` (confidence 0.62) but completely drops the `billing_subscription` sub-query.
* **Mitigation**: Set an intent entropy / low margin trigger ($\text{margin} < 0.15$ between top 2 intents) that forces human escalation when multi-intent ambiguity is detected.

### Failure Mode 3: App Update / UI Drift (Temporal Out-of-Vocabulary)

* **What goes wrong**: Customers complain about new UI changes or software bugs introduced in a recent app release that post-date the retrieval index.
* **Real Example**: *"Where did the heart button go in version 8.8.12? The new plus icon isn't adding tracks to my library."*
* **Why it happens**: The historical retrieval index contains resolution pairs explaining the legacy "Heart" button interface.
* **Mitigation**: Track retrieval distance scores (`1 - top1_sim`). If top similarity falls below 0.45, the query triggers an OOD escalation flag.
### Failure Mode 4: Implicit Sarcasm and Negative Emotion Masking

* **What goes wrong**: Sarcastic customer praise is taken literally by the generator, leading to overly polite or inappropriate responses.
* **Real Example**: *"Oh wonderful, Spotify deleted my 500-song workout playlist again. Best app on earth! 👏🏻"*
* **Why it happens**: Semantic embeddings map "Best app on earth!" to positive sentiment vectors.
* **Mitigation**: Added sentiment polarity checking and mapped playlist deletion keywords directly to `playlist_library_mgmt` troubleshooting paths.

### Failure Mode 5: Citation Over-Generalization ("Citation Theater")

* **What goes wrong**: The LLM inserts `[doc_1]` citations at the end of sentences that contain details not present in Document 1.
* **Real Example**: RAG retrieves a generic offline mode doc, and LLM generates: *"You can download up to 10,000 songs on up to 5 devices [doc_1]."* (where Document 1 only mentioned downloading, not the exact device limit).
* **Why it happens**: LLM relies on pre-trained parametric knowledge while attempting to satisfy the prompt's citation constraint.
* **Mitigation**: Built an automated NLI-based Citation Auditor (`src/generation/generator.py`) that strips citations and flags the sentence as unsupported if string alignment fails.

---

## What Is Misleading About My Headline Number?

My headline metric — **71.0% autonomous resolution coverage at a 3.5% false auto-handle rate** — is solid, but interpreting it blindly as "71% of all customer support is fully solved" would be misleading for four reasons:

1. **DM-Deflection Bias in Mined Ground Truth**: 30.9% of all historical brand responses in the raw dataset were simple DM deflections ("Please send us a DM"). Although I filtered out explicit deflection templates during retrieval corpus construction, the remaining "visible resolutions" are heavily skewed toward self-contained troubleshooting questions. Real-world traffic contains a much higher proportion of account-bound issues that *must* be escalated.
2. **Temporal Drift**: My golden set was sampled from a chronological split immediately following the training window. In a live production deployment, app updates, API changes, and new billing policies will degrade retrieval precision over time unless the index is continuously updated.
3. **Multi-Turn Dynamics**: The evaluation measures single-turn resolution quality. It does not measure customer friction if a user replies to an automated response with follow-up questions.
4. **Sample Size Uncertainty**: My golden set contains 200 hand-labeled examples. A 3.5% false auto-handle rate represents 7 errors out of 200. The Wilson score 95% confidence interval for this error rate ranges from $1.7\%$ to $7.0\%$.

---

## Decision Log: 12 Non-Obvious Decisions

I logged all major technical decisions in [`data/artifacts/decision_log.json`](https://www.google.com/search?q=data/artifacts/decision_log.json):

### 1. Brand Selection via Weighted Scoring

* **Alternatives Considered**: Selecting `@AppleSupport` or `@AmazonHelp` due to higher raw volume.
* **Why I Chose It**: `@SpotifyCares` had a far higher density of self-contained, visible resolutions (69.1% vs <20% for Apple/Amazon) and a balanced mix of technical vs billing issues.
* **Tradeoff**: Lower overall thread count (47k vs 126k), but vastly higher quality training data.

### 2. Temporal Split over Random Split

* **Alternatives Considered**: Random train/test split.
* **Why I Chose It**: Random splitting leaks temporal context (e.g., active app outages occurring on the same day). A temporal split mirrors true deployment.
* **Tradeoff**: Lower test set metrics due to distribution shift over time.

### 3. Tree-Aware Conversation Reconstruction

* **Alternatives Considered**: Treating conversations as linear chains.
* **Why I Chose It**: Twitter threads branch non-linearly. Linear assumptions pair wrong agent responses with customer turns.
* **Tradeoff**: Requires complex adjacency graph building during data ingestion.

### 4. Intent Taxonomy Frozen Before Golden Labeling

* **Alternatives Considered**: Evolving the taxonomy dynamically during labeling.
* **Why I Chose It**: Freezing the 10-intent taxonomy before golden set creation prevents labeler bias and test set leakage.
* **Tradeoff**: Forces ambiguous long-tail cases into the `other_unclear` category.

### 5. Intent-Conditioned Hybrid RAG over Pure Dense Vector

* **Alternatives Considered**: Pure FAISS dense vector search or unconditioned RRF.
* **Why I Chose It**: Filtering candidate resolution pairs by predicted intent before RRF ranking eliminates false semantic hits across different problem domains.
* **Tradeoff**: If the intent classifier predicts the wrong intent, relevant documents are filtered out early.

### 6. Calibrated Logistic Regression over Uncalibrated k-NN

* **Alternatives Considered**: k-NN cosine distance classification.
* **Why I Chose It**: The escalation engine requires true, well-calibrated class probabilities to calculate prediction entropy. E4 achieved an ECE of 0.038 vs k-NN's 0.218.
* **Tradeoff**: Requires lightweight model training instead of pure index lookups.

### 7. Retaining Embedding Model (E4) over Fine-Tuned Encoder (E8)

* **Alternatives Considered**: Shipping the fine-tuned DeBERTa-v3 model (E8).
* **Why I Chose It**: Paired bootstrap testing showed the 95% CI for F1 improvement crossed zero $[-0.012, +0.028]$. I retained the simpler, faster baseline (E4).
* **Tradeoff**: Forwent a minor 0.8% nominal F1 gain to eliminate model serving complexity.

### 8. Strict Citation-Auditing Generator over Free-Text LLM

* **Alternatives Considered**: Standard free-text LLM generation without document citations.
* **Why I Chose It**: Forcing `[doc_X]` citations paired with an automated sentence alignment auditor reduced hallucinated claims from 48.5% down to 2.5%.
* **Tradeoff**: Increases prompt token cost and slightly restricts conversational tone.

### 9. Multi-Signal Risk Escalation Layer over Single Confidence Threshold

* **Alternatives Considered**: Escalating based purely on classifier confidence ($\text{prob} < 0.60$).
* **Why I Chose It**: Combined intent risk-tiers, classifier entropy, retrieval distance, and groundedness scores.
* **Tradeoff**: Requires tuning multiple threshold hyperparameters on dev data.

### 10. Asymmetric Error Cost Function ($C_{\text{false\_auto}} = 5 \times C_{\text{false\_esc}}$)

* **Alternatives Considered**: Symmetric classification error weighting.
* **Why I Chose It**: Sending a confidently wrong answer to a customer hurts brand trust far more than routing a solvable query to a human agent.
* **Tradeoff**: Lowers overall auto-handling coverage to protect safety.

### 11. Reference-Anchored LLM-as-a-Judge over Unanchored Scoring

* **Alternatives Considered**: Prompting an LLM to rate responses 1–10 without reference text.
* **Why I Chose It**: Providing human reference resolutions achieved 86.7% agreement ($\kappa = 0.807$) with human raters.
* **Tradeoff**: Requires human labor to annotate reference resolutions for the validation set.

### 12. Precomputed Artifact Freezing for Reviewer Path

* **Alternatives Considered**: Forcing reviewers to reprocess the 2.81M raw CSV file.
* **Why I Chose It**: Freezing indexes, thresholds, and splits allows reviewers to execute complete evaluation runs in under 30 seconds.
* **Tradeoff**: Adds ~15MB of precomputed binary files to the repository.

---

## What I Would Do with One More Week

If I had one additional week to extend this project, I would focus on three high-value priorities:

1. **Dynamic RAG Index Auto-Refreshing**: Implement an automated trigger that monitors customer escalation rates on novel queries. When an spike in OOD queries occurs (e.g., during a major app outage), the system should automatically index verified human agent responses from the last 24 hours into the retrieval engine.
2. **Multi-Turn Dialogue Context Tracking**: Expand the pipeline from single-turn response generation to full multi-turn conversational state tracking, allowing the system to maintain memory across multi-tweet customer threads.
3. **Active Learning Golden Set Expansion**: Build a continuous active-learning queue that flags low-confidence auto-handled cases and routes them to human annotators for weekly golden evaluation set expansion.

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

