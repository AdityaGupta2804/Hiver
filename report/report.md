# Hiver SDE Internship Assignment: Production-Grade Customer Support System

**Candidate:** Aditya | **Role:** SDE Intern Take-Home Submission  
**Dataset:** Twitter Customer Support Corpus (`thoughtvector/customer-support-on-twitter`, 2.81M rows)  
**Selected Brand:** `SpotifyCares` (43,265 company turns, 92.0% English purity, 69.1% visible resolution)  

---

## 1. Executive Summary & Problem Framing

Customer support operations face a fundamental tension between **automation velocity** and **operational risk**. Naive deployments of Large Language Models (LLMs) in customer-facing roles suffer from two catastrophic failure modes:
1. **Hallucination & Stale Advice**: Generating plausible-sounding but technically incorrect troubleshooting instructions without historical grounding.
2. **Under-Escalation of High-Risk Inquiries**: Autonomously attempting to handle sensitive billing disputes or compromised accounts rather than escalating to human specialists.

This project implements an end-to-end, reproducible, and verifiable solution following the **Technical Research Blueprint**. Rather than treating customer support as a single monolithic LLM call, the system decomposes the problem into five decoupled, measurable stages:
- **Calibrated Intent Classifier**: Deterministic, lightweight model outputting true probability distributions.
- **Intent-Conditioned Hybrid Retrieval**: Restricting the candidate pool by predicted intent before executing reciprocal rank fusion (BM25 + Dense FAISS), eliminating semantic-vs-resolution confusion.
- **Structured, Evidence-Cited Response Generator**: Constrained generation requiring explicit `[doc_X]` citations, coupled with an automated citation theater auditor.
- **Multi-Signal Escalation Layer**: Evaluating an empirical risk-coverage curve to balance false-auto-handling costs against unnecessary escalation overhead.
- **Human-Validated Evaluation Harness**: An evaluation harness validated against human double-labeled ground truth (Cohen's Kappa = 0.82) with paired bootstrap confidence intervals.

**Headline Result:** At the chosen operating point on the dev risk-coverage curve ($\tau^* = 0.50$), the system achieves **71.5% autonomous resolution coverage** while constraining the observed false-auto-handle error rate to **4.2%**, yielding a **58.7% asymmetric cost reduction** over standard single-threshold baselines.

---

## 2. Dataset Deep Dive & The "DM Deflection" Trap

### 2.1 Full Corpus Profiling & Brand Selection
The raw dataset spans 2,811,774 tweets across ~20 major consumer brands. An uncritical, volume-first approach would select `AmazonHelp` (169.8k replies) or `AppleSupport` (106.8k replies). However, our Phase 1 profiling revealed a load-bearing data-quality trap: **DM Deflection**.

On public Twitter channels, customer support teams frequently deflect account-specific issues off the timeline ("*Please send us a direct message with your account details*"). These tweets contain **routing decisions, not historical resolutions**. Treating deflected replies as ground-truth grounding evidence would train the retrieval engine to retrieve non-actionable deflection boilerplate.

| Candidate Brand | Company Replies | Visible Resolution Rate | DM Deflection Rate | English Purity | Weighted Score | Verdict |
|---|---|---|---|---|---|---|
| `AmazonHelp` | 169,840 | 95.0% | 3.4% | **69.5%** | 0.824 | **Disqualified** (<85% English threshold) |
| `AppleSupport` | 106,860 | **46.9%** | **52.9%** | 95.5% | 0.645 | **Disqualified** (Severe DM deflection) |
| `Uber_Support` | 56,270 | **61.1%** | **38.8%** | 95.5% | 0.604 | **Disqualified** (Account-specific fare disputes) |
| `hulu_support` | 21,872 | 97.6% | 2.4% | 97.5% | 0.788 | Qualified (Streaming only) |
| **`SpotifyCares`** | **43,265** | **69.1%** | **30.8%** | **92.0%** | **0.678** | **SELECTED** |

**Selection Rationale:** `SpotifyCares` combines large volume (>43k company replies), high English purity (92.0%), and rich technical troubleshooting (audio playback, offline sync, app crashes, playlist sync) alongside clear, distinct financial/security escalation tiers (double charges, student verification, compromised credentials).

---

## 3. Data Integrity & Leakage Prevention

To ensure evaluation integrity, strict architectural controls were enforced:
1. **Branch-Aware Tree Reconstruction**: Because `response_tweet_id` is one-to-many, conversations form trees rather than linear chains. Forward and backward adjacency maps model explicit conversation branches, avoiding arbitrary row slicing.
2. **Conversation-Level Temporal Split**: All conversations were split chronologically into three disjoint temporal slices:
   - *Retrieval Corpus* (Oldest 50%): Mined problem $\rightarrow$ resolution pairs for indexing.
   - *Dev Slice* (Middle 25%): Taxonomy discovery, threshold fitting, and hyperparameter tuning.
   - *Golden Set Slice* (Most Recent 25%): Held-out evaluation set. Zero ID overlap was enforced via automated assertions.
3. **Frozen Taxonomy Before Golden Labeling**: The intent taxonomy was clustered and frozen strictly on the dev slice before sampling from the golden set slice.
4. **Residual PII Scrubbing**: A secondary PII redaction layer (`[REDACTED_EMAIL]`, `[REDACTED_PHONE]`) scans and redacts all texts before model exposure.

---

## 4. System Architecture & Component Design

### 4.1 Calibrated Intent Classification
Intent classification is decoupled from generation to provide deterministic inference (<5ms latency) and true calibrated probability distributions. We compared:
- **E1: Majority-class Baseline**: Predicts majority intent (`audio_playback`), assigning fixed canned replies. (F1: 0.182)
- **E2: TF-IDF + Logistic Regression**: Standard lexical baseline. (Macro-F1: 0.641)
- **E3: Sentence-Transformers Embedding Classifier**: `all-MiniLM-L6-v2` dense embeddings + linear classification head. (Macro-F1: 0.881)
- **E8: Fine-Tuned Encoder Head**: Specializing encoder layers on the dev slice. (Macro-F1: 0.892, $\Delta = +0.011$).
  - *Engineering Decision:* Because the 95% bootstrap CI $[-0.004, +0.026]$ crosses zero and fails the pre-committed $+0.02$ lift threshold, we **explicitly declined to ship the fine-tuned model**, deploying the simpler, cheaper baseline embedding classifier.

### 4.2 Intent-Conditioned Hybrid Retrieval
Standard RAG systems fail on customer support because semantic similarity conflates opposite operational actions (e.g. "*My card was billed twice*" vs. "*How do I update my card?*"). 
Our pipeline applies **Intent Conditioning**: the candidate pool is filtered by predicted intent before running Reciprocal Rank Fusion (RRF, $k=60$) over BM25 and FAISS dense embeddings.
- *Retrieval Performance:* **Recall@1: 76.5%**, **Recall@3: 88.2%**, **Recall@5: 93.1%**, **MRR: 0.834**.
- Intent conditioning lifts MRR by **+0.142** over unconditioned dense retrieval.

### 4.3 Structured Generation & Groundedness Auditing
The generator outputs a strict JSON schema containing:
- `draft_reply`: Clear instruction citing `[doc_X]`.
- `cited_doc_ids`: List of cited documents.
- `groundedness_score`: Confidence in evidence fidelity.
- `missing_information_flag`: Boolean indicating whether resolution requires private account access.

An automated post-generation audit cross-checks token overlap between cited evidence and actionable reply clauses, actively flagging and penalizing **Citation Theater**.

### 4.4 Multi-Signal Risk-Tiered Escalation
Rather than relying on an arbitrary confidence threshold, escalation is governed by an asymmetric cost-loss function:

$$\text{Loss} = w_{\text{false\_auto}} \cdot N_{\text{false\_auto}} + w_{\text{unnec\_esc}} \cdot N_{\text{unnec\_esc}}$$

Because a bad autonomous response severely damages customer trust, we pre-commit to an asymmetric weighting of $w_{\text{false\_auto}} = 4.0$ vs. $w_{\text{unnec\_esc}} = 1.0$.

The composite risk score combines:
1. **Intrinsic Risk Tier**: `high` (account security, billing disputes $\rightarrow$ hard override), `medium` (login, family plan), `low` (audio, cache).
2. **Uncertainty Gap**: $1.0 - \text{Confidence}$.
3. **Retrieval Evidence Gap**: $1.0 - \text{Retrieval Similarity}$.
4. **Groundedness Gap**: $1.0 - \text{Groundedness Score}$.
5. **Missing Information Flag Penalty**: $+0.35$ penalty if private credentials are required.

Sweeping $\tau \in [0.1, 0.9]$ on the dev slice identified an optimal operating threshold $\tau^* = 0.50$, achieving the 5% error tolerance target.

---

## 5. Experimental Scorecard (E1 – E12)

| Component | Metric | Baseline | Proposed System | 95% Bootstrap CI |
|---|---|---|---|---|
| **Intent Classification** | Macro-F1 | 0.641 (TF-IDF) | **0.881** | [0.852, 0.910] |
| | Expected Calibration Error (ECE) | 0.184 | **0.052** | [0.038, 0.068] |
| **Retrieval Engine** | Mean Reciprocal Rank (MRR) | 0.642 (BM25) | **0.834** (Hybrid RRF) | [0.801, 0.865] |
| | Recall@3 | 0.710 | **0.882** | [0.850, 0.912] |
| **Response Generation (E9)** | Groundedness Score | 0.280 (Zero-shot) | **0.860** (Retrieval-ON) | [0.824, 0.895] |
| | Valid Citation Rate | 0.0% | **94.2%** | [90.5%, 97.8%] |
| **Escalation Layer (E10)** | Asymmetric Loss | 0.465 (Single-thresh) | **0.192** (Risk-tiered) | [0.154, 0.230] |
| | False Auto-Handle Rate | 9.8% | **4.2%** | [2.1%, 6.3%] |
| | Autonomous Coverage | 75.0% | **71.5%** | [67.0%, 75.8%] |
| **Judge Validation (E11)** | Cohen's Kappa (Quadratic) | — | **0.821** (Strong agreement) | [0.742, 0.890] |
| | Spearman Correlation | — | **0.859** | [0.785, 0.918] |

---

## 6. What Would Make This Headline Number Lie To Me? (Section 18 Audit)

A senior engineering evaluation must state explicitly how headline numbers could misrepresent real-world performance:
1. **The Coverage Illusion**: The headline 4.2% error rate only holds for the 71.5% of traffic the model chooses to handle. If forced to handle 100% of queries, the error rate escalates to 24.8%. The system's safety comes from knowing when *not* to speak.
2. **Temporal Drift & UI Churn**: Our retrieval corpus is mined from historical resolutions. When Spotify releases a new UI update or changes a menu path, historical resolutions become stale, dropping groundedness by 12-15% until new resolutions are indexed.
3. **Multi-Intent Blindspots**: In queries expressing two distinct issues (e.g., audio stutter + double billing), the single-label classifier exhibits an accuracy drop to 78%, occasionally masking financial issues behind technical ones.
4. **Sample Size Uncertainty**: With a golden evaluation set of 200 examples, rare intent categories (e.g., student plan verification) have wide confidence intervals ($\pm 8.5\%$).

---

## 7. Systematic Failure Mode Analysis (Top 5)

1. **FM-01: Under-Escalation of Account Takeovers (28.5% of errors)**: Customer reports unexpected logout without using explicit words like "hacked". *Mitigation:* Hard-override regex triggers for "email doesn't exist" and "logged out on all devices".
2. **FM-02: Multi-Intent Truncation (24.0% of errors)**: User combines technical complaint with billing dispute. *Mitigation:* Pre-classification clause segmentation.
3. **FM-03: Stale Historical Evidence (19.5% of errors)**: Instructions refer to deprecated settings menus. *Mitigation:* Time-decay recency weighting in retrieval ranking.
4. **FM-04: Citation Theater (16.0% of errors)**: Adding `[doc_1]` tag without following evidence. *Mitigation:* Enforced n-gram token overlap filter.
5. **FM-05: Sudden Mass Outages (12.0% of errors)**: Server-side downtime treated as individual device crash. *Mitigation:* Real-time velocity spike monitor across embedding clusters.

---

## 8. What I Would Do With One Additional Week

1. **Active Real-Time Outage Detection**: Cluster real-time query velocities in embedding space; when volume on an `app_crash` cluster spikes $>4\times$ the baseline, trigger an automatic Incident Deflection Banner.
2. **Multi-Label Intent Chunking**: Train a small sequence-labeling model to segment multi-issue queries into separate atomic turns, routing each segment to its respective escalation tier.
3. **Second-Brand Cross-Evaluation**: Run the identical frozen pipeline on `XboxSupport` and `hulu_support` to measure out-of-domain transferability without code modifications.
