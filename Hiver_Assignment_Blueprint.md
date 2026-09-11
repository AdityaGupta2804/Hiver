# Hiver SDE Internship Assignment — Technical Research Blueprint

**Scope of this document:** research + architecture + evaluation design + experiment plan + implementation roadmap. No code, no pseudocode. Everything here is meant to eliminate architectural uncertainty before you open Cursor.

**How to use it:** Section order follows the exact 30-header structure you specified, so you can trace every requirement back to a section. Where the assignment prompt asked for sub-analyses (leakage audit, ablation design, error taxonomy, adversarial review, final scorecard), they're nested inside the most relevant top-level section rather than added as new top-level numbers, so the document stays navigable.

**One important discipline maintained throughout:** the brand is *not* permanently selected here. Section 4 gives you a scoring framework and a data-grounded shortlist with hypotheses about which brands should score well, but the actual selection happens after you run the profiling script on the real CSV (Roadmap Phase 1). Everything downstream is written to be brand-agnostic until that point.

---

# Executive Summary

**What the assignment is really testing.** Hiver isn't grading you on whether you can call an LLM API. They're grading whether you can (a) take an unlabeled, messy, real dataset and impose a defensible structure on it, (b) build a system that knows the difference between "I can answer this" and "I should not touch this," and (c) prove both of those things to a skeptical reviewer who will actively hunt for leakage, cherry-picking, and vanity metrics. The phrase "the proof is worth more than the system" is not throat-clearing — it is the rubric.

**What I recommend building, in one sentence.** A retrieval-grounded response system (hybrid BM25 + embedding retrieval over mined `problem → resolution` pairs, filtered by a small, cheaply-trained intent classifier) that generates structured, evidence-cited replies with an LLM, wrapped in a risk-based escalation layer whose thresholds are chosen from a risk-coverage curve rather than guessed — evaluated by a golden set built specifically to contain the cases where the system *should* fail, scored with a multi-dimensional LLM judge that is itself validated against human labels before its scores are trusted.

**What I recommend *not* building:** a fine-tuned generation model (the data doesn't support it — see Section 9), a cross-encoder reranker unless ablation proves it earns its latency and complexity cost (Section 10), a single collapsed "quality score" (it hides which component is actually failing), and any architecture-first design that wasn't justified by a measurement.

**The single biggest risk in this dataset** is not noise or slang — it's that a large fraction of "resolved" conversations are not resolutions at all. Brand accounts frequently deflect to DMs ("please send us a private message") the moment an issue becomes account-specific, which means the *actual* fix is invisible to this dataset. Any brand-selection or retrieval-corpus decision that ignores this will silently build a grounding system on top of ungrounded boilerplate. This shapes almost every downstream decision in this document.

**The single biggest risk in the evaluation** is that with a 150–250 example golden set, a 2–3 point difference between two systems is very likely noise, not signal. Section 18 and the Decision Log treat this explicitly rather than let a single flattering headline number stand unchallenged.

---

# 1. What This Assignment Is Actually Testing

Reading the assignment's own language closely: it names three deliverable *systems* (classify, ground, escalate) but spends proportionally more of its rules and constraints on *proof* — a golden set with a documented methodology, an evaluation harness with automated metrics *and* an LLM judge *and* evidence the judge agrees with humans, a mandatory "what is misleading about my headline number" section, and a decision log. A fresher submission optimizes step 1 (build something that mostly works) and treats step 2 (prove it works) as an afterthought — a confusion matrix and a "94% accuracy!" banner. This assignment is explicitly designed to catch that pattern and reward the opposite: an unremarkable model with rigorous, honest measurement beats an impressive-looking model with sloppy measurement, because sloppy measurement is indistinguishable from a lucky roll or a hidden leak, and Hiver has told you they'll be looking for exactly that.

Three concrete implications for the design:

1. **Evaluation is not a downstream module — it is designed first**, and the golden set's sampling methodology has to be defensible on its own, independent of any specific model result. This blueprint follows that order: Sections 13–18 (evaluation) are logically prior to Section 19 (architecture), even though architecture is described first for readability. In the actual roadmap (Section 27), evaluation-set construction happens *before* the final model is tuned.
2. **Every headline number needs an attached "what would make this number lie to me" note.** This isn't a formality section to skip — it's the thing that most differentiates a senior engineer's submission from a fresher's.
3. **The interesting engineering is in the escalation and grounding layers, not the LLM call.** Anyone can prompt an LLM to draft a support reply. The differentiator is knowing, with evidence, when *not* to trust that draft.

---

# 2. Dataset Deep Dive

## 2.1 Provenance and what it actually is

The dataset ("Customer Support on Twitter," Kaggle `thoughtvector/customer-support-on-twitter`) was released in early 2018 as, in the original authors' words, a corpus meant to "aid innovation in natural language understanding and conversational models, and for the study of modern customer support practices." Multiple independent academic reproductions (a customer-support MRC/re-ranking paper and an empathetic-chatbot paper, both citing it directly) consistently report the same shape: roughly 2.8–3 million tweets, spanning about 20 major consumer brands, covering the same well-known set (Apple, Amazon, Uber, Delta, Spotify, American Airlines, and similar). This cross-source agreement is worth noting precisely because the assignment tells you not to trust a Kaggle listing at face value — here, independent parties who used the raw file for unrelated research corroborate the shape, which is a reasonable (though not certain) signal of integrity for the core columns.

## 2.2 Exact schema

Every row is one tweet. Confirmed columns (cross-checked against a public reproduction that inspects the raw CSV directly):

| Column | Type | Meaning | Data-quality implication |
|---|---|---|---|
| `tweet_id` | int | Anonymized unique tweet ID | Used as the join key for thread reconstruction; not sequential-safe for time ordering |
| `author_id` | string | Anonymized user ID. Brand accounts keep their real handle-like string (e.g. `sprintcare`, `AppleSupport`); customer accounts are anonymized numeric strings | This is the *only* reliable way to separate "which side of the conversation" beyond the `inbound` flag — non-numeric `author_id` = a company |
| `inbound` | bool | `True` if the tweet is inbound *to* a company (i.e., sent by a customer) | Confirmed against real rows: company replies have `inbound=False` and a real brand handle as `author_id` |
| `created_at` | string timestamp | When the tweet was sent, formatted like a raw Twitter API timestamp | All sampled timestamps carry a `+0000` UTC offset — good, means no timezone-normalization step is required, but must still be parsed with the Twitter-style format string, not ISO |
| `text` | string | Tweet content, with `@mentions` pointing at anonymized IDs; sensitive strings (emails, phone numbers) are already replaced with mask tokens like `**email**` by the dataset publisher | This means basic PII scrubbing was *already done upstream* — you inherit it, you don't need to build it, but you must verify it (mask failures happen) before using any text in a prompt or a report |
| `response_tweet_id` | string, comma-separated IDs | IDs of tweets that reply to this one | **Can be more than one ID.** A tweet can have multiple replies — threads branch, they are not linear chains. This is the single most important structural fact for Phase reconstruction. |
| `in_response_to_tweet_id` | float/nullable | ID of the tweet this one replies to | Null for a thread's root message |

There is **no label column of any kind** — no sentiment, no intent, no "resolved" flag, no category. Every derived label (intent, resolution status, escalation risk) is something you construct; none of it is provided, which matches the assignment's explicit instruction that you define intents yourself.

## 2.3 Scale and brand distribution (as reported by independent reproductions)

One public reanalysis that loaded the actual file and ran `.value_counts()` on company-authored tweets reports (top 20, tweet counts on the company side of the conversation): AppleSupport ≈126.5k, AmazonHelp ≈119.1k, Uber_Support ≈72.5k, AmericanAir ≈49.3k, SpotifyCares ≈47.7k, comcastcares ≈46.2k, Delta ≈45.3k, Tesco ≈43.8k, TMobileHelp ≈36.6k, SouthwestAir ≈36.3k, British_Airways ≈34.1k, Ask_Spectrum ≈32.2k, UPSHelp ≈27.4k, sprintcare ≈25.8k, hulu_support ≈24.7k, VirginTrains ≈23.8k, XboxSupport ≈22.6k, AskTarget ≈21.0k, GWRHelp ≈20.0k, ATVIAssist (Activision) ≈19.6k. Treat these as directional, not gospel — they come from one analyst's cleaning pipeline, not the dataset publisher, and your own profiling script (Roadmap Phase 1) is the actual source of truth for brand selection. But they're useful for pre-narrowing the search space before you spend compute on full profiling.

## 2.4 Conversation / thread reconstruction

Because `response_tweet_id` is one-to-many, the "conversation" is really a **tree**, not a list. Reconstructing it correctly requires:

- Treating each tweet with `in_response_to_tweet_id = null` and `inbound = True` as a potential thread root.
- Walking `response_tweet_id` forward, but explicitly handling the branch case: if a customer tweet gets two different company replies (common when two different support agents pick it up, or a bot auto-reply plus a human reply), naively taking "the next row in file order" silently picks an arbitrary branch and can attribute the wrong reply to the wrong turn.
- Handling **orphaned references**: an `in_response_to_tweet_id` that points to a `tweet_id` not present anywhere in the export. This happens when the referenced tweet was deleted before the dataset was scraped, or belongs to a conversation branch outside the exported window. These threads must be treated as **truncated**, not as clean single-turn conversations — silently treating "first turn with no visible parent" as "this is definitely where the conversation started" will misclassify plenty of mid-conversation turns as fresh complaints.
- Deduplicating **exact and near-duplicate text**. Boilerplate brand replies ("We're sorry to hear that. Please DM us your account details so we can help.") appear at extremely high frequency and are near-identical apart from a mention token. Two independent reanalyses that deduplicated on exact tweet text both ended up removing on the order of a third of rows as duplicates — this is not a minor cleaning step, it materially changes both the retrieval corpus and any "resolution availability" statistic you compute.

## 2.5 The trap that matters most: DM deflection

Read the actual reply text of any high-volume brand and a large share of company replies are a variant of "please send us a DM / private message with your account info." This is a structural property of using Twitter for support: anything that requires touching a real account (refund, password reset, order lookup) legally and practically has to move off the public timeline. The consequence for this assignment:

- **The dataset frequently does not contain the actual fix.** It contains the *routing decision* ("this needs private handling"), not the resolution. Treating a DM-deflection reply as a "historical resolution" to retrieve and ground future answers on is actively wrong — it would train the system to always say "please DM us," which is a real but nearly useless behavior to imitate at scale.
- This must be an explicit filter in the retrieval-corpus construction step (Section 10) and an explicit factor in brand scoring (Section 4): **brands whose issues are inherently self-contained and don't require account-touching (troubleshooting-style: "restart your router," "here's the known outage," "try this in-app setting") will have a much higher rate of genuinely retrievable resolutions than brands whose issues are inherently account-specific (billing disputes, lost card, flight rebooking).**

## 2.6 Other confirmed and expected noise sources

- **Non-English content.** The dataset is described as "mostly English," and public cleaning pipelines that ran language detection found it necessary to filter — implying a non-trivial non-English tail (French/Spanish-language brand handles among the ~20 brands, and code-switching within otherwise-English tweets).
- **Bot-like/templated messages** beyond DM deflection — automated outage notices, "we are aware of the issue, see status page" broadcasts sent identically to many customers within a short time window. These inflate apparent "resolution" counts if not deduplicated at the template level (not just exact string level — templates often interpolate a ticket ID or mention).
- **Very short / low-information messages** ("@AppleSupport lol", "still broken", emoji-only) that carry essentially no intent signal on their own and need thread context to be interpretable at all.
- **Slang, sarcasm, all-caps venting, and heavy emoji use**, expected and normal for a public-complaint medium; sentiment work on this exact dataset (independent Twitter CS sentiment/topic-modeling analysis) reports the expected pattern — customer-initiated tweets skew negative, and sentiment measurably improves after a company response, with meaningful variance by brand (e.g., southwest and airline-brand comparisons in that analysis showed different starting sentiment and different amounts of sentiment recovery).
- **Multi-issue conversations**, where a single thread pivots ("also, why was I charged twice") mid-conversation — this breaks any assumption that one thread = one intent, and must be explicitly designed for in the taxonomy (Section 6) rather than discovered as a surprise during labeling.
- **Ordering problems from timestamp ties and reply-timing races** — company support teams sometimes issue two replies within one second of each other from an internal queue, and comma-separated `response_tweet_id` values are not guaranteed to be in a meaningful order.

## 2.7 What this means for "how reliable is the dataset"

Reliable for: raw column semantics (`inbound`, ID linkage, timestamp format), scale, breadth of brands, and English-dominance. **Not reliable for**: assuming any given thread represents a full, resolved interaction; assuming text similarity implies resolution similarity; assuming the volume-leading brand is the best choice; assuming random sampling gives you a representative "resolved case." Every one of these has a concrete mitigation described in the relevant later section — this section exists so those mitigations aren't ad-hoc patches discovered mid-project, but decisions made up front.

---

# 3. Data Quality and Risks

Section 2 already surfaced the mechanisms. This section turns them into a risk register you can actually check off during implementation — each row is something the profiling script (Roadmap Phase 1) must measure, not assume.

| # | Risk | How it manifests | Detection method | Mitigation |
|---|---|---|---|---|
| 1 | DM-deflection masquerading as resolution | Company reply text matches a "please DM us" template | Regex/embedding-cluster match against a small set of known deflection templates; flag any reply below a length threshold that contains "DM", "direct message", "private message" | Exclude from retrieval corpus; track separately as a "deflected, no visible resolution" bucket used for brand scoring, not for grounding |
| 2 | Exact/near-duplicate tweets inflating volume and retrieval hit rate | Same text (mass outage complaints, canned replies) repeated hundreds–thousands of times | Exact-text dedup + MinHash/embedding near-dup clustering | Deduplicate before computing any "brand has N conversations" statistic; cap retrieval corpus to one representative per duplicate cluster to avoid one template dominating nearest-neighbor search |
| 3 | Broken thread reconstruction from branching replies | `response_tweet_id` has multiple comma-separated values | Explicitly model conversation as a tree; never assume "next row" order | Build an explicit tree-walk with branch IDs; pick the primary branch by earliest-timestamp company reply, log the rest, don't silently drop them |
| 4 | Orphaned / truncated threads | `in_response_to_tweet_id` points to a `tweet_id` absent from the export | Join check: does the parent ID exist in the file? | Mark as "truncated context" — exclude from golden set unless the truncation itself is the labeled phenomenon (rare, don't build for it) |
| 5 | Non-English or code-switched text | Non-ASCII heavy text, language-detector low-confidence English | Fast language ID (e.g., a compact langid model) at ≥0.85 confidence threshold, matching practice used in independent reproductions of this exact dataset | Drop or route to a separate "non-English, out of scope" bucket; document the drop rate, don't silently discard without measuring how much you lost |
| 6 | Bot/broadcast messages (outage notices) | Identical text to many customers in a short time window, one-to-many timing pattern | Time-windowed duplicate-text clustering | Exclude from per-issue retrieval corpus (they're announcements, not case-specific resolutions); optionally keep as a distinct "known outage" intent |
| 7 | Multi-issue / topic-drift threads | Thread pivots to a second unrelated complaint mid-conversation | Heuristic: large semantic-similarity drop between consecutive customer turns in the same thread | Label at the turn level, not the thread level, for intent; note as an explicit "multi-intent" case type when building the golden set |
| 8 | Very short / no-signal messages | <4 tokens, emoji-only, pure venting | Token-length threshold | Keep a slice of these in the golden set deliberately (they're a real production case), but exclude them from taxonomy *discovery* (they'd only add noise to clustering) |
| 9 | Mass near-duplicate complaints during outages create leakage | Ten near-identical "app is down" tweets, some ending up in train/retrieval, some in golden set, purely by chance | Conversation-level (not tweet-level) and near-duplicate-aware split, not pure random split | See temporal/conversation-level split strategy in Section 17 |
| 10 | Sensitive-data mask failures | Publisher already masks emails/phones with tokens like `**email**`, but masking is regex-based and can miss edge cases (partial numbers, non-US phone formats, order numbers that look like account IDs) | Run your own PII scan over the final retrieval corpus and golden set before anything goes into a prompt or a report | Treat the publisher's masking as a first pass, not a guarantee; re-scan and re-mask before any text leaves your pipeline (including into LLM API calls) |

**Overall reliability verdict:** the dataset is reliable as a *raw event log* of public support interactions, and unreliable as a *resolution corpus* unless you actively filter for genuine, visible resolutions. This is the load-bearing conclusion of Phases 1–3 combined, and it is the reason brand selection (Section 4) weighs "visible resolution rate" as heavily as volume.

---

# 4. Brand Selection Strategy

## 4.1 Why volume-first selection is wrong here

The obvious move — pick AppleSupport or AmazonHelp because they have the most tweets — is close to a trap given Section 2.5. Both are consumer-electronics/marketplace giants whose non-trivial issues ("my order," "my device," "my account") almost always require touching a private account, so a large fraction of their visible replies are exactly the DM-deflection pattern that produces no usable ground truth. High volume with low *visible-resolution density* gives you a large but shallow corpus — lots of conversations, few retrievable answers.

## 4.2 Scoring framework

Score every candidate brand (start from the ~20-brand shortlist in Section 2.3, confirm real numbers from your own profiling pass) on these weighted factors. Weights below are a starting point — set them once, before looking at final scores, and don't retune them to make a favorite brand win (that would itself be a form of the "tuning baselines until they lose" problem the assignment explicitly warns against, applied to yourself).

| Factor | Weight | What it measures | How to compute cheaply |
|---|---|---|---|
| Conversation volume (post-dedup) | 10% | Enough data to support disjoint retrieval-corpus / dev / golden splits without starving any of them | Count unique root threads after near-dup collapse |
| Visible-resolution rate | 25% | % of company replies that are *not* DM-deflection and *not* a pure apology with no content | Template-match filter from Section 3, row 1 |
| Issue diversity | 15% | Whether a meaningful, non-trivial intent taxonomy is even possible | Quick embedding + k-means sweep on customer turns; count distinct, well-separated clusters at a fixed silhouette threshold |
| Self-containedness of typical resolution | 15% | Whether resolving the issue plausibly requires only public information (troubleshooting steps, known outages, policy facts) vs. private account state | Manual read of a random 30-reply sample; score the fraction that give an actionable, account-independent instruction |
| Escalation-worthy issue presence | 10% | Whether the brand's issue mix includes genuinely high-risk categories (billing, security, safety) so the escalation layer has something real to prove | Keyword/embedding scan for billing/security/safety terms in customer turns |
| Noise level | 10% | Non-English rate, bot/broadcast rate, average message length | Language-ID pass + duplicate-cluster size distribution |
| English purity | 10% | Directly gates whether the taxonomy and generation prompts need multilingual handling | Language-ID pass (reuse above) |
| Repetition of issues (good) | 5% | High repetition of the *same* issue types (not the same *text*) means retrieval has enough historical precedent per intent to be useful | Cluster size distribution — many small singleton clusters is bad, a moderate number of medium clusters is good |

Compute a 0–1 normalized score per factor, take the weighted sum, and rank. **Explicitly disqualify** any brand where visible-resolution rate is below a floor (recommend 20%) regardless of its total score — a brand can't compensate for "we don't actually have grounding material" with volume or diversity.

## 4.3 What makes a brand a bad choice (explicit anti-patterns)

- **High volume, high DM-deflection rate** (the Apple/Amazon trap above) — looks rich, is shallow.
- **Extremely narrow issue space** (e.g., a brand where 90% of traffic is one recurring outage) — makes the intent taxonomy trivial and the retrieval problem trivial, which undersells the "hard parts" the assignment wants to see engineered.
- **Heavy multilingual mix** — adds a real NLP problem (language detection, potentially multilingual embeddings) that is orthogonal to what's being graded and burns time without proving anything relevant.
- **Too few threads after dedup** to support three or four disjoint slices (retrieval corpus, taxonomy-dev, golden set, judge-validation subset) without overlap — remember every one of those must be genuinely disjoint (Section 17).

## 4.4 Working hypothesis to test first (not a final decision)

Based on the public secondary-analysis numbers in Section 2.3 and the self-containedness heuristic in 4.2, the brands worth profiling *first* are ones whose issue domain is technical/troubleshooting rather than account/billing-heavy: **gaming support handles (`XboxSupport`, and if present in your pull, `AskPlayStation`)** and **`SpotifyCares`** are good candidates to check first — technical troubleshooting ("error code," "can't download," "app crashing," "controller won't sync") tends to produce genuinely reusable, account-independent resolution text, mid-size volume (tens of thousands post-dedup) is enough without being unwieldy for a 15-minute reproducible pipeline, and there's a natural, real escalation tier (account bans, payment/refund disputes, account security) sitting alongside the low-risk troubleshooting tier, which gives the escalation layer something non-trivial to prove. Airlines (`Delta`, `AmericanAir`, `British_Airways`) are a reasonable second choice — good issue diversity (delays, booking, baggage, service) but a higher share of PNR-specific issues that route to DM. Telecom (`TMobileHelp`, `sprintcare`, `Ask_Spectrum`) sit in between. **This is a hypothesis to confirm against your own profiling numbers, not a commitment** — run Roadmap Phase 1's scoring script on the actual data before freezing the choice, and if a brand you didn't expect scores better on visible-resolution rate, take that result over this prior.

---

# 5. Problem Decomposition

The assignment names three capabilities; underneath them are five genuinely separable ML problems with different right answers for "what kind of model should do this." Treating the whole thing as "one LLM call that does everything" is the generic-submission failure mode this document is explicitly trying to avoid.

| Sub-problem | Is it one model, many, or a pipeline? | Best-fit approach (with reasoning) |
|---|---|---|
| **A. Intent classification** | Standalone, upstream of everything else | A small, calibratable classifier (embedding + linear head, or a lightweight fine-tuned encoder — decided empirically in Section 9), **not** an LLM call at inference time. Reasoning: intent classification needs to be deterministic, cheap, fast, and — critically — needs a real probability distribution over classes for the escalation layer's confidence signal. An LLM asked to output a label with a "confidence" is not well-calibrated by default and is expensive/slow to call for every message. |
| **B. Historical-resolution retrieval / grounding** | A dedicated retrieval pipeline, intent-conditioned, sitting between classification and generation | Hybrid lexical (BM25) + dense embedding retrieval over a curated `(problem, resolution)` pair corpus, filtered by predicted intent before ranking. Not a pure LLM task — retrieval quality is measurable independently (Recall@K, MRR) and should be, because generation quality is a function of retrieval quality and you need to isolate which one is failing. |
| **C. Response generation** | LLM, but constrained and structured, not open-ended | An LLM is the right tool *here specifically* because natural-language drafting conditioned on retrieved evidence is exactly what LLMs are good at, and the alternative (template filling) can't handle the long tail of phrasing. But it must be constrained to cite which retrieved evidence it used (Section 11), so groundedness is checkable, not just plausible-sounding. |
| **D. Auto-handle vs. escalate** | A dedicated decision layer, downstream of A–C, combining signals from all three | Not a classifier trained to predict "auto/escalate" directly (there's no reliable ground truth label for that at scale, and it would hide *why* — see Section 12) but a **rule-and-threshold decision function over calibrated signals**: intent confidence, retrieval strength, groundedness score, risk tier, and novelty/OOD signal. Thresholds are fit from data (risk-coverage curve), not model weights. |
| **E. Evaluation** | Its own system, not a byproduct | A harness with independent metrics per component (A–D each get their own metrics) plus an end-to-end LLM judge that is itself validated against a human-labeled subset. Treated as a first-class deliverable, built before the final architecture is locked (Section 13). |

**For each sub-problem, the "could X work better" checklist actually applied:**

- **Traditional ML (TF-IDF/logistic regression) for intent?** Yes, as a strong baseline (Section 7) and possibly as the shipped classifier if a fine-tuned encoder doesn't beat it by a real margin (Section 9) — don't assume more complex wins.
- **Embeddings for intent?** Yes — embedding + k-NN or embedding + linear head is the most likely production choice; cheap, no training loop needed for the k-NN variant, decent quality on a domain this narrow.
- **A smaller classifier for intent?** Yes, likely the winner — see Section 9's fine-tuning investigation.
- **An LLM for intent?** Only as a zero/few-shot *baseline* to sanity-check the trained classifier, and as an OOD/"other" fallback for genuinely novel phrasing the trained classifier wasn't exposed to — not as the primary path.
- **Hybrid for retrieval?** Yes — pure dense retrieval alone conflates "similar words" with "similar resolution" (the "my card isn't working" problem in Section 10); BM25 catches exact product/error-code terms dense embeddings can blur.
- **Rules anywhere?** Yes, in exactly two places: (1) the DM-deflection / boilerplate filter (Section 3) is a rule, not a model, because it doesn't need to be one; (2) the hard risk-tier override in escalation ("if the message mentions account security or a payment dispute, escalate regardless of confidence") is a rule, because some categories should never be auto-handled no matter how confident the model is.
- **Confidence calibration anywhere?** Yes — this is the single most important cross-cutting technique in the whole system, feeding directly into escalation (Section 12) and into how the intent metric is reported (Section 15).

---

# 6. Intent Taxonomy Strategy

## 6.1 Methodology

1. **Discover, don't guess.** Run embedding-based clustering (embed customer-turn text with a general-purpose sentence embedding model, cluster with a density-aware method like HDBSCAN rather than a fixed-k method like plain k-means, since you don't know the true number of intents ahead of time) over the taxonomy-discovery slice only — never over the golden set (leakage, Section 17).
2. **LLM-assisted labeling of clusters, human-reviewed.** For each discovered cluster, sample ~10–15 representative messages and ask an LLM to propose a short, human-readable label and a one-line definition. Then a human (you) reviews every proposed label — merges near-duplicate clusters, discards clusters that are actually noise (e.g., a cluster that's just "thank you" replies), and splits clusters that are secretly two different issues.
3. **Validate the taxonomy**, don't just accept the first pass: check that every intent has a clear, one-sentence definition that a different labeler could apply consistently; check for near-duplicate intents (two labels describing the same underlying issue with different wording) and merge them; check coverage by re-running assignment on a fresh sample and measuring what fraction falls outside every defined bucket.
4. **Always include an explicit `other/unclear` bucket.** Real traffic always has a long tail that doesn't fit a clean taxonomy — pretending otherwise produces a classifier that's either overconfident on tail cases or that the taxonomy quietly excludes from evaluation (a leakage-adjacent problem: don't let your golden set only ever contain in-taxonomy examples).
5. **Freeze the taxonomy before the golden set is finalized.** This is the most important sequencing rule in this whole section — see the explicit leakage note below.

## 6.2 Granularity target

Do not optimize for a large number of intents. A support intent taxonomy for one brand's Twitter traffic, done well, should land in roughly the **8–15 intent range** plus the `other` bucket — enough to preserve genuinely different resolution paths (a "refund request" needs a different retrieval pool and a different escalation risk tier than "how do I redeem a promo code"), not so many that classes become sparse and indistinguishable (splitting "refund for late delivery" and "refund for wrong item" into separate intents is very likely over-granular; they resolve the same way). The right test for "is this its own intent" is **"would the retrieval corpus and the escalation risk tier for this be different from a nearby intent?"** — if the answer is no, merge.

## 6.3 Ambiguous and multi-intent messages

Label these explicitly during human review rather than forcing a single tag: allow a `primary_intent` plus an optional `secondary_intent` field. For classification purposes, evaluate against `primary_intent` as the main metric, but keep the secondary tag around because it's exactly the kind of case the golden set should stress-test (Section 14) and exactly the kind of case that should push the escalation layer toward "escalate" (ambiguity is itself a risk signal, Section 12).

## 6.4 Class imbalance

Twitter support traffic is naturally power-law distributed (a handful of intents dominate). Two consequences: (1) macro-F1, not accuracy, is the right headline classification metric (Section 15) — accuracy would let a model that just predicts the majority class look strong; (2) the golden set must be intent-stratified, not purely random, specifically so rare-but-important intents (e.g., account security) get enough golden examples to produce a non-trivial per-class estimate, even though they're rare in raw traffic.

## 6.5 The leakage rule that governs this whole section

**The golden evaluation set must never be looked at, sampled from, or influence any decision, during taxonomy discovery.** Concretely: discover and freeze the taxonomy using only the taxonomy-discovery slice (Section 17's split); only *after* the taxonomy is frozen do you sample the golden set (from a later, disjoint time slice) and label it against the now-fixed intent list. If a golden-set example doesn't fit any frozen intent, it goes into `other` — you do not go back and add a new intent because the golden set revealed one. If you discover during golden-set labeling that the taxonomy genuinely has a hole, that's a legitimate finding to report in "what I'd do with one more week" (Section 30), not a license to quietly patch the taxonomy using test-set knowledge.

---

# 7. Baseline Strategies

## 7.1 The two required baselines

**Baseline 1 — genuinely trivial.** Majority-class intent prediction (always predict the single most frequent intent in the training distribution) combined with a fixed canned reply per intent (no retrieval, no personalization) and a fixed escalation rule ("never auto-handle" or "auto-handle everything" — report both trivial extremes, since they bound the risk-coverage curve). **What this proves:** the absolute floor. If the final system doesn't clear this by a wide, statistically defensible margin, nothing else in the report matters.

**Baseline 2 — simple but real.** TF-IDF + linear classifier (logistic regression or linear SVM) for intent, paired with a keyword/BM25-only retrieval (no embeddings, no reranking) for grounding, and a single fixed confidence threshold on the classifier's predicted probability for escalation. **What this proves:** how much of the final system's improvement comes from "using an LLM/embeddings at all" versus classical, cheap, fully-explainable methods. This is the baseline most likely to actually be competitive on intent classification specifically — TF-IDF+linear models are a genuinely strong, well-documented baseline for topic-constrained text classification, and losing to it should be a real possibility you test for, not something you assume away.

## 7.2 Additional baseline candidates worth running (not both required, but cheap and informative)

- **Zero-shot LLM classification** (no examples, just the intent list and definitions in the prompt) — proves how much the trained classifier's advantage comes from actually seeing brand-specific training data versus general LLM world knowledge.
- **Few-shot LLM classification** (a handful of labeled examples per intent in the prompt) — the natural middle point between zero-shot LLM and the trained classifier; useful for the fine-tuning-worth-it argument in Section 9.
- **Embedding nearest-neighbor classifier** (no training loop — just embed and take the majority label of the k nearest labeled examples) — a near-zero-effort strong baseline that a fine-tuned model needs to beat to justify the extra complexity.
- **For response generation specifically**, the meaningful baseline is **"LLM with no retrieval"** (same prompt, same model, zero historical grounding) — this isolates exactly how much of the final response quality is coming from the grounding pipeline versus the LLM's own general knowledge, which is the single most important ablation for defending the "grounded in historical resolutions" claim in an interview.

## 7.3 What each baseline proves, stated explicitly (so the report doesn't have to re-derive this)

| Baseline | Proves |
|---|---|
| Majority-class + canned reply | The absolute floor; anything must beat this by a real margin |
| TF-IDF + linear / BM25-only | How much value comes from embeddings/LLMs vs. classical methods |
| Zero-shot LLM classification | How much of the classifier's skill is brand-specific learning vs. general LLM knowledge |
| Embedding k-NN | The no-training-loop ceiling a fine-tuned model must clear to justify its complexity |
| LLM with no retrieval | How much of response quality is retrieval vs. the LLM's own general knowledge — the core "grounding" claim's baseline |

## 7.4 Baseline fairness (so these numbers survive scrutiny)

Every baseline runs on **exactly the same golden set**, with **the same available information** (no baseline gets to see intent labels the final system doesn't also get, and vice versa), the same preprocessing (whatever text-cleaning step is applied to the final system's input is applied identically to baseline input), and the same output contract (if the final system must produce a structured `{intent, draft_reply, decision, reason}` object, so does every baseline, even a trivial one — a baseline that isn't asked to satisfy the same output schema and is only compared informally isn't a fair comparison). Baseline prompts (for the LLM baselines) are fixed and frozen **before** the final system's results are computed — never re-tune a baseline's prompt after seeing that it lost, and disclose in the decision log that this rule was followed.

---

# 8. Model Strategy

## 8.1 Comparison across the seven candidate strategies, per sub-problem

| Strategy | Intent quality | Response quality | Grounding | Hallucination risk | Latency | Cost | Repro. | Explainability | Data needed | Fresher-laptop feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. Commercial LLM API only | Good, but poorly calibrated confidence | Good fluency, weak faithfulness without retrieval | None by default | High without constraints | Medium (network call) | Per-token, adds up if used for every intent call too | Low (API nondeterminism) | Low (no visible decision boundary) | None | Yes |
| 2. Open-source instruct model (self-hosted) | Similar to (1), lower quality at small sizes | Similar, usually behind top commercial models at accessible sizes | None by default | High without constraints | Depends on hardware | Free at inference, GPU cost otherwise | Higher (fixed weights) | Low | None | Only with a rented GPU |
| 3. Embedding model + retrieval | N/A directly, but powers B | N/A directly | This *is* the grounding mechanism | Low (retrieval itself doesn't hallucinate, though bad matches mislead downstream generation) | Low (a similarity search) | Cheap, mostly one-time embedding cost | High | High (you can show the retrieved neighbor) | Needs a curated corpus, not labels | Yes |
| 4. Traditional classifier (TF-IDF+linear) | Solid on the intent task specifically | N/A | N/A | N/A | Very low | Free | Very high | Very high | A few hundred labeled examples per class ideally | Yes, trivially |
| 5. Small fine-tuned classifier (encoder + LoRA or full fine-tune) | Likely best if enough labeled data exists | N/A | N/A | N/A | Very low | Cheap one-time training | High if seeds/config recorded | Medium-high | A few hundred to low-thousands of labeled examples | Yes, on CPU-friendly small encoders or a modest free-tier GPU |
| 6. Fine-tuned language model for generation | N/A | Risky — see Section 9 | Risky — see Section 9 | Depends heavily on training-data quality, which Section 2.5 shows is compromised | Can be low if small model | Real training cost | Medium | Low-medium | Thousands of clean (problem, resolution) pairs — **this dataset largely does not have them cleanly**, see Section 9 | Marginal, and not advisable here |
| 7. Hybrid (trained classifier + retrieval + LLM generation only) | Best of (4/5) | Best of (1) constrained by (3) | Strong, because (3) is explicit and checkable | Lower — generation is constrained to cite retrieved evidence | Sum of a cheap classifier call, a cheap retrieval call, and one LLM call | Only one LLM call per message (for generation), not per stage | High for stages 1&2, LLM-API-typical for stage 3 | High for stages 1&2, medium for stage 3 (citations make it partially explainable) | Moderate | Yes |

## 8.2 Recommendation: Strategy 7 (hybrid), decomposed as recommended in Section 5

This is not the "safe middle option" chosen to avoid commitment — it's the only strategy in the table that scores well on every column that the assignment explicitly says it's grading (explainability, reproducibility, reasonable cost, "engineered rather than assembled from tutorials"). Pure Strategy 1 ("call an LLM API for everything") is exactly the generic submission the assignment is implicitly warning against — it would score fine on response fluency and terribly on the assignment's actual rubric items: explainability of the escalation decision, reproducibility of a nondeterministic single-call pipeline, and cost at scale (every intent classification, every escalation decision, and every response draft going through the same expensive nondeterministic call).

**Where an LLM API is genuinely the right tool, and only there:** final response drafting, conditioned on retrieved evidence, with structured output (Section 11). That's it. Intent classification and retrieval do not need an LLM call at inference time once the classifier is trained and the corpus is indexed.

## 8.3 What would change this recommendation

If the selected brand's issue space turns out (during Roadmap Phase 1 profiling) to have very high intent diversity with very few examples per intent, a trained classifier can't be fit reliably — at that point, falling back to zero/few-shot LLM classification (Strategy 1's classification role) becomes the right call, and that fallback path is exactly why Section 7's baseline suite includes zero-shot and few-shot LLM classification: so you have real numbers to make that call from, rather than guessing.

---

# 9. Fine-Tuning Investigation

This section exists specifically so the submission isn't another "LLM API + RAG + prompt" project with no training component — but the goal is to find where fine-tuning is *actually* justified by the data, not to add it for appearances (rule 5 of your own constraints: do not over-engineer; rule 12: be skeptical of impressive-sounding approaches that don't measurably help).

## 9.1 The five architectures, compared honestly

| Option | Description | Verdict |
|---|---|---|
| A. Pure LLM API | Zero-shot/few-shot for everything | Rejected as primary strategy — see Section 8.2. Kept only as a baseline. |
| B. RAG + LLM | Retrieval-grounded generation, classical/embedding intent classifier | **This is the default position going in** — the burden of proof is on any more complex option to beat it by a real, CI-backed margin |
| C. Fine-tuned classifier + RAG + LLM | Same as B, but intent classification is a fine-tuned small encoder instead of embedding-kNN/TF-IDF | **Worth testing, not worth assuming.** See 9.2. |
| D. Fine-tuned classifier + retrieval + smaller generation model (no big commercial LLM for drafting) | Same as C, but a small fine-tuned LM drafts responses instead of a commercial LLM | **Not recommended.** See 9.3 — this is exactly where the data quality problem in Section 2.5 bites hardest. |
| E. Fully open-source stack | Self-hosted embeddings + self-hosted LLM everywhere | **Not recommended for a take-home under a laptop/cloud-budget constraint** — it doesn't improve any of the graded dimensions (Section 8.1's columns) and adds real infrastructure risk (hosting, GPU access, latency) for no measurable benefit. Worth one sentence in the report as "considered and rejected," not worth building. |

## 9.2 Should you fine-tune the intent classifier? Investigate, don't assume.

**The case for yes:** a small encoder (e.g., a DistilBERT/DeBERTa-v3-small class model) fine-tuned (full fine-tune or LoRA — LoRA preferred purely for speed/reproducibility on limited compute, since the marginal quality gap between LoRA and full fine-tuning is small at these model sizes and LoRA trains and reruns much faster, which matters for hitting the 15-minute reproducibility budget) directly on brand-specific historical language should, in principle, beat a generic embedding model's k-NN classification, because it can specialize to brand-specific phrasing, product names, and error-code vocabulary that a general-purpose embedding model was never trained to weight heavily.

**The case for no, or "it won't matter":** the amount of *clean, human-reviewed* labeled data available for training is small — realistically low hundreds to low thousands of examples once you exclude the golden set and the human-review-required cluster-labeling step is itself the bottleneck, not compute. Fine-tuning gains over a strong embedding baseline shrink fast as labeled data shrinks, and a well-tuned embedding+linear-head or embedding-kNN classifier is a genuinely strong baseline at this data scale, not a weak one.

**The actual answer is an experiment, not a guess** (this is Experiment 8 in Section 23): train both, evaluate both on the same frozen dev set with the same metric (macro-F1, plus calibration/ECE since the escalation layer needs calibrated probabilities either way), and only ship the fine-tuned classifier if it wins by a margin that survives a bootstrap confidence interval on the dev set. **If it doesn't win by a real margin, ship the simpler embedding-based classifier and say so explicitly in the report** — "we tested fine-tuning; it did not produce a statistically meaningful improvement over the embedding baseline at this data scale, so we shipped the simpler model" is a *stronger* engineering statement than a fine-tuned model with no comparison, because it's exactly the kind of evidence-based restraint the assignment is asking to see (constraint 13: "if research reveals my idea is inferior to a simpler approach, tell me explicitly").

If fine-tuning wins, the concrete plan: **model** a small pretrained encoder (parameter count in the tens of millions, not billions — full fine-tuning is feasible on a single modest GPU or even patient CPU training at this size); **training data** the human-reviewed intent-labeled cluster samples from Section 6, explicitly excluding golden-set and judge-validation-set examples; **objective** standard cross-entropy classification with label smoothing (helps calibration, which matters downstream); **split** a held-out dev slice from the same labeled pool, stratified by intent; **LoRA vs. full fine-tune** LoRA first (cheaper, faster iteration, easier to reproduce in the 15-minute budget), escalate to full fine-tuning only if LoRA underperforms and there's compute budget left; **hyperparameters to sweep** learning rate and number of epochs primarily (2–4 values each is enough at this scale — this is not a hyperparameter-search-heavy problem); **evaluation** macro-F1 and expected calibration error (ECE) against the embedding baseline, on the same dev slice, with a bootstrap CI on the difference; **overfitting risk** real, given small data — mitigate with early stopping on the dev slice and label smoothing, and treat a large train/dev gap as a stop signal, not something to push through with more epochs; **catastrophic forgetting** a real but secondary risk here since the model is being specialized for a narrow classification head, not repurposed for open-ended generation — low priority to investigate further given the scale.

## 9.3 Should you fine-tune a response-generation model? No — and here is the specific evidence, not a general aversion to fine-tuning.

This is the more important conclusion in this section, and it follows directly from Section 2.5, not from a generic "fine-tuning is risky" heuristic. Fine-tuning a generator requires clean `(customer problem, brand resolution)` pairs at real scale. This dataset's biggest structural flaw is that a large share of what looks like a "resolution" is actually a DM-deflection or a content-free apology — training a generation model on that corpus, even after filtering, risks two specific failure modes: (1) the model learns to imitate boilerplate ("I'm sorry to hear that, please reach out via DM") as a default completion pattern, which is a real behavior the brand exhibits but a nearly useless one to automate; (2) even after aggressively filtering to the visible-resolution subset, the *remaining* clean pairs are a small, biased sample (skewed toward whichever issue types happen not to require private account access), so a generator fine-tuned on them would be well-calibrated only for that biased subset and silently worse everywhere else — and you would have no clean way to detect that blind spot without the exact retrieval-and-groundedness-checking machinery that RAG already gives you for free. In other words: **the RAG approach's per-answer evidence citation is a mechanism for catching exactly the failure mode that fine-tuned generation would hide inside the model's weights.** Given that, plus the added training cost, added nondeterminism, and added interview-defense burden ("how do you know fine-tuning helped, given the label quality issues you yourself identified?"), fine-tuning a generator is explicitly **not recommended**, and the report should state this conclusion with the reasoning above rather than omitting the option as if it wasn't considered.

---

# 10. Historical Grounding / Retrieval Strategy

## 10.1 Retrieval unit: what actually gets stored and searched

Investigated four candidate units (Section 7 of the prompt): raw `customer message → brand reply` pairs, `full conversation → final resolution`, `intent → canonical examples`, and `issue → successful historical resolution`. **Recommendation: `issue → successful historical resolution` pairs, intent-tagged, mined from full conversations but stored as a compact pair, not the whole thread.** Reasoning:

- Raw `message → immediate reply` pairs are too noisy — the "immediate reply" is very often an acknowledgment or a DM-deflection (Section 2.5), not the resolution.
- Full `conversation → final resolution` preserves context but makes retrieval and later citation-checking harder (you'd be retrieving and citing a whole thread when only two turns of it matter), and increases prompt length/cost for no real benefit once the resolution has been extracted.
- The winning design extracts, per resolved thread, a compact structured record: `{customer_problem_summary, brand_resolution_text, intent_tag, resolution_type (self-contained vs. required-account-action), source_thread_id}`. This keeps the corpus small, keeps citations traceable back to a real source thread (needed for the groundedness check in Section 11), and lets metadata filtering (by `intent_tag` and `resolution_type`) happen cheaply before any similarity search runs.

## 10.2 Retrieval pipeline design

**Hybrid BM25 + dense embedding retrieval, intent-filtered, with reranking only if ablation earns it.**

1. **Pre-filter by predicted intent** (from Section 8's classifier) — restrict the candidate pool to same-intent (and optionally, adjacent/related-intent) resolution pairs before any ranking happens. This is the direct fix for the "my card isn't working could have many different underlying causes" problem named in the prompt: semantic similarity alone doesn't distinguish "card not activating" from "card declined at checkout" from "card physically damaged," but they're different intents with different resolutions, and intent-conditioning the retrieval pool fixes exactly that failure mode before it can happen.
2. **Rank within the filtered pool with a hybrid score**: BM25 (catches exact terms — product names, error codes, specific policy words that dense embeddings can blur past) combined with dense embedding cosine similarity (catches paraphrase and semantic equivalence BM25 misses). A simple weighted combination or a reciprocal-rank-fusion combination is enough; this is not a place to over-engineer a learned fusion model at this project's scale.
3. **Retrieval happens after intent classification, not before or in parallel** — because the filtering in step 1 depends on the intent prediction. (The prompt explicitly asks whether retrieval should happen before, after, or both relative to classification; the answer here is "after," specifically because the intent-conditioning is the mechanism that solves the semantic-similarity-vs-resolution-similarity problem — doing it in parallel would forgo that filter.)
4. **Reranking (cross-encoder) is a candidate, not a commitment.** Add it as Experiment 5 in the ablation plan (Section 23) and only keep it in the final architecture if it produces a measurable Recall@3/nDCG improvement over hybrid retrieval alone that's worth its added latency. If it doesn't clear that bar, cut it and say so in the decision log — this is a direct instance of the assignment's own instruction (26.5): "if reranking adds only complexity but negligible improvement, explicitly recommend removing it."

## 10.3 Distinguishing semantic similarity from resolution similarity — the design mechanism, concretely

Beyond intent-conditioning, two additional guards: (a) **resolution-type filtering** — don't let a self-contained troubleshooting answer be retrieved as evidence for a message whose flagged signals (mentions of "charged," "refund," "account") suggest it needs an account-specific resolution, and vice versa; (b) **evidence agreement check at generation time** — if the top-K retrieved resolutions for a query disagree with each other in a way the generator can detect (two different fixes suggested for what looks like the same problem), that disagreement itself becomes an escalation signal (Section 12), rather than being silently averaged away by the LLM into a confident-sounding but potentially wrong answer.

## 10.4 Retrieval evaluation, treated as its own ML problem (not folded into end-to-end response scoring)

For every golden-set example, the harness must record, independently of what the generator does with it: **what was retrieved**, and (from the golden-set labeling process) **what should have been retrieved / whether the top result was actually useful for resolving the issue** — this ground truth is created during golden-set labeling (Section 14), not invented after the fact. Metrics: Recall@1/@3/@5/@10, MRR, nDCG, and a **resolution-usefulness rate** distinct from semantic relevance — a human labeler (you) marks each top retrieved item not just "topically related" but "would this actually have informed a correct resolution," which is the direct operationalization of the semantic-vs-resolution distinction from 10.3.

## 10.5 Retrieval ablation (see also Section 23 for the full experiment table)

Compare, on the same query set with the same ground truth: (A) BM25 alone, (B) dense embedding alone, (C) hybrid A+B, (D) hybrid + reranking, (E) intent-aware filtering added to hybrid, (F) intent + resolution-type metadata filtering added on top of E. Report Recall@3 and nDCG for each. Expect the biggest single jump at step E (intent-conditioning) given the analysis in 10.1–10.3 — if the data doesn't show that, that's an important, reportable finding in itself, not something to paper over.

---

# 11. Response Generation Strategy

## 11.1 Structured, evidence-cited generation — not free-text drafting

The generator's prompt receives: the customer message, the predicted intent (with the classifier's confidence), and the top-K retrieved `(problem, resolution)` records with their source-thread IDs. It is instructed to produce a **structured output**, not a free-text reply, specifically: `{draft_reply, cited_evidence_ids: [...], claims: [...], confidence_self_report, missing_information_flag}`. The `cited_evidence_ids` field is the load-bearing piece — it lets the evaluation harness mechanically check, for every generated response, whether the substantive claims in `draft_reply` actually trace back to a retrieved record, rather than relying on a judge's subjective read of "does this sound grounded."

## 11.2 What "grounded" means operationally, and how it's checked without just trusting the LLM's self-report

A response is "grounded" if every actionable claim in it (a specific instruction, a specific policy statement, a specific number) maps to content present in one of the cited evidence records. This is checked two ways, not one: (1) a **cheap automated check** — does the draft actually contain content overlapping with the cited record, or did the model cite an ID but ignore it and answer from general knowledge (a real failure mode with LLMs asked to self-report citations); (2) the **LLM judge's groundedness dimension** (Section 16), which is a semantic check the automated overlap check can't do alone (paraphrase can be faithful without lexical overlap). Neither check is sufficient alone — the automated check catches "citation theater" (citing evidence but not using it), the judge check catches "faithful paraphrase" that the automated check would falsely flag.

## 11.3 Handling missing information and avoiding invented policy

If retrieval returns nothing above a similarity floor for the predicted intent, or retrieved records disagree (Section 10.3), the generator is instructed to set `missing_information_flag: true` and produce a response that explicitly acknowledges uncertainty rather than fabricating a plausible-sounding policy — and, critically, `missing_information_flag: true` is one of the direct inputs to the escalation decision (Section 12), not just a cosmetic field. This is the mechanism that answers the assignment's explicit question, "can the system recognize that it doesn't know?" — it's not a separate uncertainty-detection model, it's a structural consequence of forcing retrieval-conditioned, evidence-cited generation in the first place.

## 11.4 Privacy

Because retrieved historical examples come from real (if anonymized) past conversations, the generator's prompt explicitly instructs it never to reproduce order numbers, tracking numbers, or any masked-PII tokens verbatim from a retrieved record into a new customer's draft reply — a final regex-based PII re-scan (Section 3, row 10) runs on the generated draft before it's considered complete, as a hard backstop independent of the LLM following instructions correctly.

## 11.5 Why not templates, why not pure LLM, why this hybrid

Pure templates can't cover the phrasing diversity of real customer messages and would look exactly like the "assembled from tutorials" pattern the assignment wants to avoid. Pure unconstrained LLM generation (no retrieval, no structure) is fluent but unfalsifiable — you cannot mechanically check whether it's grounded, only whether it sounds plausible, which is precisely the gap an LLM judge alone (without groundedness-specific structure to check against) would fail to catch reliably. The structured, evidence-cited hybrid is the only option of the three that makes "is this answer actually grounded" a checkable property rather than a vibe.

---

# 12. Escalation and Trust Strategy

This is the component the assignment signals matters most ("likely one of the most important parts of the assignment"), and it's where a solo candidate can most clearly demonstrate engineering maturity versus a generic submission's single arbitrary confidence threshold.

## 12.1 Escalation as a decision-theoretic problem, not a classification problem

There is no reliable ground-truth label for "should this have been auto-handled" at scale — you cannot cheaply obtain it for the full traffic volume, only for the golden set. So escalation is not trained as a supervised classifier; it's a **decision function over calibrated signals**, whose thresholds are chosen from the golden/dev set using a risk-coverage analysis, and whose behavior is fully inspectable (you can always answer "why did the function decide this" by reading off the signal values, unlike a black-box classifier trained end-to-end to output auto/escalate).

## 12.2 The signals that feed the decision, and why each one is there

| Signal | What it captures | Why it matters here specifically |
|---|---|---|
| Intent confidence (calibrated) | How sure the classifier is | Base signal, but insufficient alone — see 12.3 |
| Retrieval strength (top-1 similarity, agreement across top-K) | Whether relevant precedent actually exists | Directly operationalizes "historical resolution availability" from the prompt |
| Groundedness score (from Section 11.2's checks) | Whether the *draft reply itself* is actually supported by evidence, not just whether evidence was found | Catches cases where retrieval succeeded but generation still drifted off-evidence |
| `missing_information_flag` (Section 11.3) | Explicit generator self-report of insufficient/conflicting evidence | A direct, cheap, high-signal flag |
| Intent risk tier (a manually assigned property of each intent, not learned) | Whether this *category* of issue should ever be auto-handled regardless of confidence | Encodes domain judgment a purely statistical model can't derive on its own — e.g., account-security and payment-dispute intents are tagged high-risk by definition |
| Novelty/OOD signal (distance from the query to its nearest labeled/retrieved neighbor, relative to a fitted distribution) | Whether this message resembles anything the system has seen | Directly operationalizes "can the model know when it doesn't know" |
| Conflicting-evidence flag (Section 10.3) | Whether top retrieved resolutions disagree | Catches the specific failure mode of confidently averaging over contradictory precedent |

## 12.3 Why not a single confidence threshold

A single threshold on intent-classifier confidence answers "how sure is the classifier," which is a necessary but nowhere-near-sufficient condition for safety — a message can have very high intent confidence ("this is definitely a refund request") while having zero good retrieved precedent, or while belonging to a risk tier that should never be automated regardless of confidence. Collapsing all of that into one number throws away exactly the information a reviewer will ask about ("why should this message be auto-handled?" — Section 26's adversarial list). The decision function instead is structured as: **hard risk-tier override first** (if high-risk tier → always escalate, full stop, no threshold negotiation) → **then** a combined score over the remaining signals, thresholded at an operating point chosen from the risk-coverage curve (12.4).

## 12.4 Threshold selection: risk-coverage curves, not guessed constants

On the dev set (never the golden set — thresholds are fit before golden-set evaluation, then frozen), sweep the combined confidence/groundedness score and, at each candidate threshold, compute **coverage** (% of messages the system would auto-handle at that threshold) and the **observed error rate among auto-handled messages** at that threshold (using the dev set's human-reviewed correctness labels). Plot this as a risk-coverage curve. **Pick the operating threshold that satisfies a stated error tolerance** (e.g., "we choose the threshold that keeps the auto-handled error rate ≤ 3%, and report whatever coverage that buys us" — not the other way around, i.e. not "we choose the threshold that gives us the highest coverage and hope the error rate is acceptable"). This directly implements the assignment's suggested headline framing: **"At X% auto-handling coverage, what is the observed error rate?"** instead of a bare classifier accuracy number that says nothing about deployment safety.

## 12.5 Are the two error types equally costly? No — say so explicitly and quantify it.

A **false auto-handle** (confidently wrong answer sent to a real customer) is worse than a **false escalation** (a case the system could have handled safely gets sent to a human anyway) — the first actively damages trust and can cause real harm (wrong policy stated, wrong refund promised); the second only costs a human agent's time, which is the existing status quo cost the system is trying to reduce, not eliminate. Concretely, weight false-auto-handle roughly 3–5x worse than false-escalation in any single scalar cost function you might report (state this weighting explicitly as an assumption, don't hide it), and prefer to report the two error rates *separately* rather than only a blended cost — a blended number is exactly the kind of "collapsing everything into one score" the final scorecard (Section 15) is designed to avoid.

## 12.6 Escalation reason as structured data, not prose

Every escalation decision emits a structured reason object, e.g. `{decision: "escalate", triggers: ["high_risk_intent_tier"], intent_confidence: 0.41, retrieval_top1_sim: 0.38, groundedness_score: 0.52, missing_information_flag: true}` — never a vague prose sentence alone. This makes "why did I escalate this?" mechanically answerable from logs, which is both a good production practice and directly what the assignment asks for ("design the escalation reason as structured data rather than vague prose").

---

# 13. Evaluation-First Design

## 13.1 Why this is designed before the final model, in practice

Concretely, "evaluation first" means: the taxonomy is frozen (Section 6.5), the data splits are frozen (Section 17), and the golden set's sampling methodology is written down and followed *before* you look at how any candidate model performs on it. If you build the model first and then construct the golden set by "grabbing some examples that look reasonable," you've made cherry-picking possible even if you didn't intend it — the discipline here is procedural, not a matter of good intentions.

## 13.2 Split strategy and why temporal, not random

Investigated: random split, conversation-level split, user-level split, temporal split, and hybrid. **Recommendation: a temporal split at the conversation level, in this order along the timeline**: retrieval corpus (oldest slice) → taxonomy-discovery/dev slice (middle) → golden set (most recent slice) → judge-validation subset (a labeled sub-slice of the golden set). Two independent reasons, not one:

1. **Realism.** The deployed system will always face issues that occur *after* whatever data it was built from. A random split lets the model implicitly "see the future" relative to any individual test example (its neighbors in time, which may describe the same live outage or the same just-announced policy change, can land on either side of a random split). A temporal split is the only one that actually asks the question the assignment cares about: "what happens on tomorrow's unseen customer issues?"
2. **Leakage specific to this dataset.** Section 2's mass-near-duplicate-during-outages pattern means random splitting risks putting ten near-identical complaints from the same five-minute window on both sides of a split — inflating retrieval and generation scores in a way that has nothing to do with genuine generalization. A temporal split, combined with the near-duplicate collapsing already required in Section 3, closes this specific hole; a random split does not.

Conversation-level (not tweet-level) splitting is layered on top of the temporal split for the same reason — no thread's turns should be split across two different roles (e.g., corpus and eval) even within the same rough time window.

## 13.3 What "good" means for this brand — made explicit, not left implicit

Before building anything, write down, in the report, a short brand-specific definition of "good" grounded in what the brand's traffic actually looks like (from Section 4's profiling): e.g., for a troubleshooting-heavy brand, "good" weights groundedness and correct escalation of account-risk cases more heavily than stylistic tone; for an airline, "good" weights correctly distinguishing self-contained answers (baggage policy) from must-escalate answers (rebooking, refund) more heavily. This becomes the stated rationale behind the metric weighting in Section 15 — a scorecard without this framing looks arbitrary even if the individual metrics are sound.

---

# 14. Golden Evaluation Set

## 14.1 Sampling strategy: stratified, not purely random, with deliberate hard-case oversampling

150–250 examples cannot be "just random" without accidentally being mostly easy, common-intent cases (because raw traffic is power-law distributed, Section 6.4) — a purely random sample would make the golden set look like a headline-flattering, mostly-easy benchmark by construction, which is exactly the "evaluation set not representing production traffic" trap named in Section 18. Recommended composition for a ~200-example set:

| Slice | Approx. share | Purpose |
|---|---|---|
| Intent-stratified "normal" cases | ~45% | Baseline competence across every defined intent, including rare ones (minimum floor per intent, not proportional-only, so rare intents aren't statistically invisible) |
| Deliberately hard cases (short/ambiguous/typo-heavy/slang, identified by heuristics from Section 3) | ~15% | Stress the system on real noise, not just clean traffic |
| Multi-intent cases | ~10% | Tests whether the system handles topic-drift/multi-issue threads sanely rather than silently picking one |
| Low-retrieval-evidence cases (bottom-similarity by running retrieval offline before finalizing the set) | ~10% | Tests whether the system correctly recognizes "I don't have good precedent" (Section 12) |
| Conflicting-evidence cases (top retrieved resolutions disagree, identified offline) | ~5% | Tests the conflicting-evidence escalation trigger specifically |
| Cases that should escalate (high-risk intent tier) | ~10% | Tests escalation precision/recall directly, not just as a byproduct |
| Most-recent-window "fresh" cases, untouched by any upstream step | ~5% | A dedicated temporal-realism check, distinct from the general temporal split |

Document this table, with actual counts, in the report's methodology note — this *is* the "short note explaining sampling and labelling methodology" the assignment requires, and it's exactly the kind of thing that pre-empts the "how did you choose these examples?" adversarial question (Section 26).

## 14.2 A separate hard/stress subset, reported separately, never blended into the main number

In addition to the composition above (which mixes hard cases *into* one golden set), also report a headline split of **overall performance** vs. **hard-subset performance** (the hard/ambiguous/low-evidence/conflicting-evidence rows above, reported as their own sub-metric) — this makes weaknesses visible rather than diluted into one average, directly following the assignment's instruction not to let a hard-case subset "inflate or replace the main metric."

## 14.3 Labeling methodology

For every golden example, a human (you) labels: `primary_intent` (and `secondary_intent` if applicable), `should_escalate` (ground truth judgment on whether this genuinely warrants a human, independent of what any model would decide), `reference_resolution_summary` (what a correct response should convey, in your own words, used as the judge's reference point — Section 16), and `case_type_tags` (which of the table 14.1 categories this example was sampled for, so you can always slice results by case type later). Label using the raw thread as the source of truth for what actually happened, not using any model's suggestion — never let a model draft a label for you to "just confirm," which quietly reintroduces the leakage/bias the golden set exists to avoid.

## 14.4 Human evaluation budget — spend it where automated metrics are weakest

With limited time, prioritize human labeling effort on: (1) the full golden set's ground-truth labels (non-negotiable, this is the golden set itself); (2) a smaller **judge-validation subset** (recommend ~40–60 of the golden examples, oversampled toward the hard/ambiguous slices where a judge is most likely to disagree with a human) double-labeled by you at two different times (or, if feasible, by a second person) specifically to compute human-vs-human and human-vs-judge agreement (Section 16); pointwise rating (not pairwise) is the right choice here given the small scale and the need for absolute pass/fail judgments per dimension (auto-handle safety needs an absolute bar, not just "which of two responses is better"). Disagreements on the double-labeled subset are resolved by re-reading the source thread and picking the label a third, disinterested read supports — document any case where you changed your own first label, since that's itself useful signal about which case types are genuinely ambiguous even for a human.

---

# 15. Metrics

## 15.1 Per-component metrics (never collapsed into one number by default)

**Intent classification:** macro-F1 as the headline (not accuracy — Section 6.4), plus per-class precision/recall (surfaces which specific intents are weak, which raw accuracy hides), a confusion matrix, and expected calibration error (ECE) since the escalation layer consumes this model's confidence directly.

**Retrieval:** Recall@1/@3/@5/@10, MRR, and the resolution-usefulness rate defined in Section 10.4 — report both semantic Recall@K *and* usefulness rate side by side specifically to make the semantic-vs-resolution distinction visible in the numbers, not just in prose.

**Generation:** never one "quality" score. Separate dimensions: correctness, groundedness, relevance, completeness, tone, and — treated as pass/fail gates, not scored on a sliding scale — unsupported-claim rate and hallucination rate. A response that reads well but invents a policy is a **fail**, full stop, regardless of how well it scores on fluency/tone; encode this as an explicit rule in the judge rubric (Section 16), not left to a judge's holistic 1–10 impression.

**Escalation:** report at the chosen operating point from the risk-coverage curve (Section 12.4): coverage (% auto-handled), observed error rate among auto-handled cases (the single most safety-relevant number in the whole system), escalation precision and recall against the golden set's `should_escalate` ground truth, and the full risk-coverage curve itself as a figure (not just the one chosen point) so a reviewer can see the tradeoff space you picked from, not just the point you landed on.

**Judge:** human agreement (Section 16), and judge self-consistency (same input scored twice — see 16.3).

**System:** latency (per stage and end-to-end), cost (per message, broken down by which stage spent it), and reproducibility (does re-running the frozen pipeline on the frozen golden set reproduce the headline number within the sampling noise established in Section 18).

## 15.2 Final scorecard — the single consolidated view, explicitly not a single score

| Category | Metric | Reported as |
|---|---|---|
| Data quality | Deduplication/leakage rate, usable-conversation rate (visible-resolution rate from Section 4) | % with description of what was excluded and why |
| Intent | Macro-F1, per-class P/R, ECE, `other`-bucket detection rate | Table + confusion matrix |
| Retrieval | Recall@1/3/5/10, MRR, resolution-usefulness rate | Table, plus the ablation delta from Section 10.5 |
| Generation | Correctness, groundedness, unsupported-claim rate (pass/fail gate), human-rated helpfulness | Table, dimension-by-dimension, no blended score |
| Escalation | Coverage at chosen operating point, observed error rate at that point, escalation precision/recall, full risk-coverage curve | Table + one figure |
| Judge | Human agreement (Cohen's kappa / Spearman as appropriate — 16.2), judge self-consistency | Table |
| System | p50/p95 latency per stage, cost per message, reproducibility check result | Table |

**Explicitly named:** PRIMARY SUCCESS METRIC = observed error rate among auto-handled cases at the chosen coverage point (this is the number that answers "is this safe to trust," which is the assignment's real question). SECONDARY METRICS = macro-F1, Recall@3, groundedness rate. SAFETY METRICS = false-auto-handle rate specifically (separated from the blended error rate, per Section 12.5), unsupported-claim rate. DIAGNOSTIC METRICS = everything else in the table above, useful for debugging but not headline-worthy on their own. This separation is itself a decision-log entry (Section 25) — it's exactly what stops the report from "collapsing everything into one score unless there is a strong reason," per the assignment's own instruction.

---

# 16. LLM-as-Judge + Human Agreement

## 16.1 Rubric design — structured, not "rate 1–10"

The judge receives, per example: the customer message, the predicted intent, the retrieved evidence with source IDs, the generated response with its cited evidence IDs, and — critically — the human-written `reference_resolution_summary` from Section 14.3, making this a **reference-based** judge, not a bare pointwise judge with no anchor. It scores, independently, on a small fixed set of dimensions (correctness, groundedness, relevance, completeness, tone, and a binary unsupported-claim flag) rather than one holistic score, and is explicitly instructed to flag `unsupported_claim: true` whenever the response asserts something not traceable to the cited evidence *even if the response otherwise reads as excellent* — this is how a fluent-but-hallucinated answer is prevented from scoring well on the strength of writing quality alone (the assignment's specific concern: "the judge should be prevented from rewarding hallucinated but convincing answers"). Pointwise, reference-based scoring is preferred over pairwise here because the real deployment question is absolute ("is this answer safe/correct"), not relative ("is this answer better than that one") — pairwise is the right tool for comparing two candidate systems against each other (and is used for that, in the ablation experiments in Section 23), but the wrong tool for the core "is this trustworthy" question.

## 16.2 Judge validation against humans

Using the judge-validation subset from Section 14.4 (~40–60 double-labeled examples), compute agreement between the human labels and the judge's scores, separately per dimension. Use **Cohen's kappa** for the binary/categorical dimensions (unsupported-claim flag, pass/fail correctness) and **Spearman correlation** for any ordinal-scaled dimension (a 1–5 tone or completeness scale) — Cohen's kappa is the right tool for categorical agreement because it corrects for chance agreement, which matters when one category (e.g., "no unsupported claims") is much more common than the other; Spearman is the right tool for ordinal scales because it doesn't assume the scale is linearly meaningful, only that rank order matters, which is the honest assumption for a 1–5 human/LLM rating scale. Report the actual numbers, not just "the judge agrees well" — if kappa lands in a mediocre range (say, below ~0.4–0.5) on any dimension, that dimension's judge score should be reported with an explicit caveat in every subsequent table it appears in, not silently trusted.

## 16.3 Judge biases investigated, and the specific mitigation for each

| Bias | Mitigation |
|---|---|
| Position bias (order of options/evidence affects the score) | Not applicable to pointwise scoring in the main pipeline; explicitly controlled for (randomized order) in the pairwise ablation comparisons in Section 23 |
| Verbosity bias (longer responses score higher regardless of quality) | Explicit rubric instruction to penalize unnecessary length and reward concision; spot-check by correlating response length with judge score on the validation subset — a strong positive correlation with no corresponding jump in human-rated quality is a red flag to report, not hide |
| Self-preference (if the judge model is the same family as the generator) | Where feasible, use a different model (or at least a materially different prompt/version) for judging than for generating, and note explicitly in the report if budget constraints forced the same model for both, as a named limitation |
| Reference-answer bias (judge over-rewards responses that lexically resemble the reference) | Rubric explicitly instructs the judge to score paraphrases that convey the same resolution as fully correct, and the automated groundedness check (Section 11.2) — which checks meaning, not phrasing — acts as a cross-check against this specific failure |
| Prompt sensitivity / judge variance | Score a fixed subset twice with the same prompt (self-consistency check) — large swings between the two runs on the same input is itself a finding to report, not something to average away silently |

## 16.4 If the judge disagrees with humans a lot

Don't hide it. Investigate which dimension is disagreeing and why (usually: judges over-trust fluent phrasing, and under-detect subtle factual drift that a human catches because they know the brand's real policy) — and either tighten that dimension's rubric with a more explicit rule, or, if tightening doesn't close the gap, report that dimension's score with a stated reliability caveat rather than presenting it at face value in the headline scorecard.

---

# 17. Data Leakage Prevention

## 17.1 Formal split boundaries — what each dataset can and cannot see

| Split | Created from | Frozen when | Can access | Must never access |
|---|---|---|---|---|
| Retrieval corpus | Oldest temporal slice, post-dedup, visible-resolution-filtered | After brand selection + cleaning, before taxonomy discovery | Its own contents; the frozen intent taxonomy (applied after the fact, for metadata tagging) | Any golden-set or judge-validation example, at any point, including during re-indexing |
| Taxonomy-discovery / dev slice | Middle temporal slice | After taxonomy is finalized and reviewed (Section 6.5) | Its own contents, for clustering, threshold-fitting (Section 12.4), and baseline/model dev-time tuning | Golden set contents; must not be re-used *after* the golden set is created to "adjust" anything |
| Golden evaluation set | Most recent temporal slice, sampled per Section 14.1's stratified plan | Immediately after labeling is complete (Section 14.3) — no further edits, ever | Nothing upstream depends on it; it is read-only from this point forward | It must never be added to the retrieval corpus, never used to pick embedding models, never used to adjust the taxonomy, never used to re-tune escalation thresholds |
| Judge-validation subset | A labeled sub-slice of the golden set, double-annotated | At the same time as the golden set | Used only to compute judge-human agreement | Not used to tune the judge's rubric *after* seeing disagreement on this specific subset in a way that overfits the rubric to it — tune the rubric conceptually, re-validate on a fresh slice if you do, don't iterate directly against the same 40–60 examples repeatedly |

## 17.2 Leakage paths checked, one by one

Exact/near-duplicate tweets across splits (closed by near-duplicate-aware, temporal, conversation-level splitting — Section 3 row 9, Section 13.2); same conversation appearing in multiple splits (closed by conversation-level, not tweet-level, splitting); historical response retrieval surfacing a golden-set example's own resolution (structurally impossible if the retrieval corpus is built only from the pre-golden-set temporal slice, but worth an explicit automated check: assert zero overlap in source thread IDs between the retrieval corpus and the golden set before finalizing); golden-set examples entering the retrieval index (same check); golden-set examples influencing prompt construction (few-shot prompt examples, if used anywhere, are drawn only from the dev slice, never the golden set — an explicit rule, checked by asserting no ID overlap); golden-set examples influencing intent taxonomy (Section 6.5's ordering rule); LLM-generated labels derived from evaluation examples (the golden set is always human-labeled from the raw thread, never model-suggested-then-confirmed, per 14.3); temporal leakage / future information (closed by the temporal split itself); embedding-index contamination (the embedding index for retrieval is built once, from the frozen retrieval corpus, before golden-set evaluation begins, and not rebuilt afterward using any information learned from evaluation results); manual example-selection bias (closed by the documented, pre-committed stratified sampling plan in Section 14.1, followed mechanically rather than by eyeballing "good" examples).

## 17.3 Information-flow boundary, stated once, for reference

At evaluation time, the system has access to: the frozen retrieval corpus, the frozen trained/fitted classifier and thresholds, and the incoming golden-set message. It has access to nothing else — not the golden set's own labels, not any other golden-set example, not any information from a time point later than the retrieval corpus's cutoff. Every component in Sections 8–12 was designed to respect this boundary by construction (temporal retrieval corpus, dev-only threshold fitting, human-only golden labeling), which is why this section is a verification pass over decisions already made elsewhere, not a late patch.

---

# 18. "What Is Misleading About My Headline Number?"

This section is written as it will appear in the final report — a direct, unhedged accounting of ways the headline number could look excellent while the system is actually weak, paired with the specific safeguard already built into the design above.

| Way the number could lie | Concretely, how | Safeguard already in place |
|---|---|---|
| Class imbalance | Reporting accuracy on a power-law intent distribution rewards always guessing the majority intent | Macro-F1 is the headline classification metric, not accuracy (Section 15.1) |
| Data leakage / train-test contamination | Retrieval corpus or dev slice overlapping the golden set | Temporal, conversation-level split with explicit zero-ID-overlap assertions (Section 17) |
| Duplicate conversations inflating apparent performance | Mass near-identical outage complaints landing on both sides of a naive split | Near-duplicate collapsing before any split is made (Section 3, row 2 & 9) |
| Retrieval contamination | A golden-set example's own resolution ending up retrievable for itself | Retrieval corpus is built only from the pre-golden-set temporal slice (Section 17.1) |
| Easy examples dominating the metric | A purely random golden set is mostly common, easy cases | Stratified sampling with deliberate hard-case oversampling (Section 14.1) |
| Evaluation set not representing production traffic | Golden set skewed toward whatever was convenient to label | Documented, pre-committed sampling plan followed mechanically, reported with counts (Section 14.1) |
| LLM judge bias | Judge rewards fluent, hallucinated answers, or over-trusts response length | Human-agreement validation (Section 16.2), explicit unsupported-claim gate independent of fluency (Section 16.1), verbosity-bias spot-check (Section 16.3) |
| Historical responses being inconsistent | Two retrieved "resolutions" for the same issue type actually contradict each other, and the number doesn't reflect that | Conflicting-evidence detection feeds escalation directly (Section 10.3, 12.2), and a dedicated golden-set slice targets this (Section 14.1) |
| Majority-class dominance | Same root cause as class imbalance, applied to the escalation decision (e.g., "escalate everything" trivially minimizes false-auto-handle rate while providing zero automation value) | Coverage is reported alongside error rate, always as a pair, never error rate alone (Section 12.4, 15.2) |
| Cherry-picked examples | Report showcases only flattering failure/success examples | Failure-mode selection is systematic, not hand-picked (Section 24's procedure), and golden-set composition is fixed *before* any model result is seen |
| Measuring response similarity instead of correctness | A metric like ROUGE/BLEU against a reference would reward lexical overlap, not whether the advice given was actually right | Explicitly rejected as a metric; correctness and groundedness are scored against meaning (judge + automated citation check), not surface text overlap (Section 11.2, 16.1) |
| Measuring offline performance without measuring abstention | Reporting classifier accuracy alone says nothing about whether the system knows when to not answer | Escalation is a first-class, separately reported metric with its own scorecard row, not an afterthought (Section 15.2) |
| High accuracy but unsafe auto-handling | A classifier can be 90% accurate while the 10% it gets wrong happen to be the highest-stakes cases (billing, security) | Risk-tier override sits *above* confidence-based thresholding, not blended into one score (Section 12.2–12.3) |
| Artificially clean preprocessing | Aggressive cleaning removes exactly the noisy, realistic traffic the system needs to be robust to | Deliberate hard-case and noisy-message slices are kept *in* the golden set, not cleaned away (Section 14.1) |
| Using test examples in prompt construction | Few-shot examples accidentally drawn from the golden set | Explicit rule + ID-overlap assertion restricting few-shot examples to the dev slice only (Section 17.2) |
| Using future information | A retrieved "historical" resolution that postdates the query in real time | Structurally prevented by the temporal split (Section 13.2, 17.1) |

## Statistical uncertainty at n ≈ 150–250 — stated plainly, not glossed over

With a golden set this size, a proportion metric (say, an observed 90% safe-auto-handle rate) carries a **bootstrap/Wilson 95% confidence interval on the order of ±4–6 percentage points** at typical proportions in this range — narrower near 0% or 100%, wider near 50%. Concretely: **do not report "System A (91%) beats System B (88%)" as a real difference without a paired significance check** — with overlapping ~±5-point intervals, a 3-point gap on 200 examples is very plausibly noise. The correct practice, used throughout the experiment plan (Section 23): **paired comparison on the identical example set** (both systems scored on exactly the same golden examples, differences computed per-example, then a paired bootstrap or a McNemar-style test on the paired differences) rather than comparing two independent-looking summary percentages — paired comparison has much more statistical power at this sample size than comparing two unpaired confidence intervals, and it's the only way small-sample differences on this golden set can be reported honestly. Every headline number in the final report is accompanied by its dataset size, its sampling methodology (pointer to Section 14.1), the baseline it's compared against, and — wherever a comparison is claimed — the outcome of a paired significance check, not just the two raw numbers side by side.

---

# 19. Recommended End-to-End Architecture

## 19.1 Data-flow diagram

```
Raw Twitter CSV (all brands)
        |
Schema validation + PII re-scan
        |
Language filter (English-confidence >= 0.85)
        |
Near-duplicate / template collapse (exact + embedding-cluster)
        |
Thread reconstruction (tree-walk, branch-aware, orphan-flagged)
        |
Brand profiling + scoring (Section 4) --> BRAND SELECTED
        |
DM-deflection / boilerplate filter (visible-resolution extraction)
        |
Temporal split: [retrieval corpus | taxonomy-dev slice | golden set | judge-validation subset]
        |
Intent discovery (embedding clustering on dev slice, LLM-assisted labeling, human review) --> TAXONOMY FROZEN
        |
Golden set sampled (stratified, Section 14.1) and human-labeled --> GOLDEN SET FROZEN
        |                                        |
   [Retrieval corpus]                    [Dev slice: classifier + threshold fitting]
        |                                        |
Resolution-pair extraction                Intent classifier trained/fitted
 (problem, resolution, intent_tag,               |
  resolution_type, source_id)                    |
        |                                        |
   Hybrid index (BM25 + embeddings)               |
        |                                        |
        +--------------- at inference -----------+
                          |
                 Incoming customer message
                          |
                 Intent classifier (calibrated)
                          |
          Intent-filtered hybrid retrieval (+ optional reranker, if earned)
                          |
       Response generator (LLM, structured, evidence-cited)
                          |
        Groundedness / PII / citation-integrity checks
                          |
        Risk + confidence engine (Section 12)
                          |
              AUTO-HANDLE  <---or--->  ESCALATE (+ structured reason)
                          |
              Evaluation harness scores every stage independently
```

## 19.2 Component-by-component specification

**1. Data cleaning & thread reconstruction.** Input: raw CSV rows. Output: a tree-structured conversation object per thread, with dedup/language/orphan flags attached. Responsibility: turn a flat tweet log into analyzable conversations without silently mis-threading branches. Algorithm: rule-based tree walk + embedding near-dup clustering + fastText-style language ID. Why it exists: without it, every downstream statistic (brand scoring, resolution rate) is built on a wrong unit of analysis. Alternatives considered: treating rows as already-linear conversations (rejected — Section 2.4 shows this actively mis-threads branching replies). Failure modes: mis-attributed branch, false orphan flag on a thread that's actually complete but references a tweet outside the export window. Evaluation: manual spot-check of N reconstructed threads against the raw CSV. Cost/latency: one-time batch job, not in the online path.

**2. Brand profiling & scoring.** Input: cleaned, threaded corpus. Output: a ranked brand scorecard (Section 4.2) and the selected brand. Responsibility: replace "pick the biggest brand" with a defensible, pre-committed rubric. Why: prevents the DM-deflection trap from silently steering the whole project onto a shallow corpus. Alternatives: pick by volume alone (rejected, Section 4.1). Cost: one-time, cheap (mostly counting and a small clustering pass per candidate brand).

**3. Intent taxonomy discovery.** Input: taxonomy-dev slice, brand-filtered. Output: a frozen, human-reviewed intent list with definitions. Responsibility: impose the structure the assignment explicitly asks you to define yourself. Algorithm: HDBSCAN over sentence embeddings, LLM-assisted cluster labeling, human merge/split/validate pass. Alternatives: pure LDA topic modeling (weaker cluster separation on short, noisy text than embedding-based clustering in practice); fixed-k k-means (rejected — true intent count unknown ahead of time). Failure modes: over-merged clusters hiding a real distinction; under-merged clusters splitting one issue into near-duplicate labels. Evaluation: re-run assignment on a fresh sample, check coverage and consistency. Cost: one-time, dominated by human review time, not compute.

**4. Intent classifier.** Input: customer message text. Output: calibrated probability distribution over the frozen intent list (+ `other`). Responsibility: fast, deterministic, explainable intent signal feeding both retrieval filtering and escalation. Algorithm: embedding + linear head or lightweight fine-tuned encoder — **decided empirically per Experiment 8 (Section 23)**, not assumed. Alternatives: LLM zero/few-shot (kept as baseline and OOD fallback, not primary — Section 8.3). Failure modes: confident-but-wrong on genuinely ambiguous/multi-intent messages; miscalibration on rare intents with few training examples. Evaluation: macro-F1, per-class P/R, ECE, on the frozen dev slice, then confirmed once on the golden set. Cost/latency: milliseconds, negligible.

**5. Retrieval corpus & hybrid index.** Input: retrieval-corpus slice, post visible-resolution filtering. Output: an indexed set of `(problem, resolution, intent_tag, resolution_type, source_id)` records, searchable by BM25 and dense embedding similarity. Responsibility: the actual "historical grounding" material. Why this unit, why hybrid: Section 10.1–10.2. Failure modes: a genuinely novel issue with no good precedent in the corpus (handled by the novelty signal in Section 12.2, not swept under the rug). Evaluation: Recall@K/MRR/resolution-usefulness rate (Section 10.4). Cost: one-time embedding + indexing pass; per-query cost at inference is a similarity search, low-latency.

**6. Response generator.** Input: message, intent, top-K filtered/ranked retrieved records. Output: structured `{draft_reply, cited_evidence_ids, claims, missing_information_flag}`. Responsibility: produce a faithful, checkable draft, not just a fluent one. Algorithm: commercial LLM API call with a constrained, structured-output prompt. Alternatives: fine-tuned generator (rejected — Section 9.3); template-only (rejected — Section 11.5). Failure modes: citation theater (cites but doesn't use evidence — caught by 11.2's automated check); fabricated specifics under thin evidence (caught by the judge's unsupported-claim gate). Evaluation: correctness/groundedness/unsupported-claim rate (Section 15.1). Cost/latency: the single most expensive and highest-latency stage — one LLM call per message, budget and cache accordingly (Section 21).

**7. Groundedness / PII / citation-integrity checks.** Input: the generator's structured output plus the cited records. Output: a groundedness score and a hard PII-scan pass/fail. Responsibility: a cheap, deterministic backstop that doesn't rely on trusting the LLM's self-report. Failure modes: false negative on a faithful paraphrase with no lexical overlap (mitigated by pairing this with the judge's semantic groundedness check, Section 11.2). Cost: negligible, rule-based.

**8. Risk + confidence engine (escalation).** Input: all upstream signals (Section 12.2's table). Output: `AUTO-HANDLE` or `ESCALATE` plus a structured reason object. Responsibility: the decision-theoretic core of the system's safety story. Algorithm: hard risk-tier override, then a thresholded combined score fit from a risk-coverage curve on the dev set (Section 12.4). Alternatives: single confidence threshold (rejected, Section 12.3); an end-to-end-trained auto/escalate classifier (rejected — no reliable ground-truth label at scale, and it would hide the "why," Section 12.1). Failure modes: an unanticipated risk category not covered by the manually-assigned risk tiers (a real gap — flagged as a limitation, not hidden). Evaluation: coverage vs. observed error rate at the chosen operating point, escalation precision/recall (Section 15.1). Cost/latency: negligible, a few arithmetic comparisons over already-computed signals.

**9. Evaluation harness.** Input: golden set + judge-validation subset + every upstream component's intermediate outputs (not just the final decision). Output: the full scorecard (Section 15.2), the risk-coverage curve, the ablation table (Section 23), and the failure-mode report (Section 24). Responsibility: the deliverable the assignment weighs most heavily. Cost: dominates neither latency nor per-message cost (it's offline), but dominates *engineering time* — allocate accordingly in the roadmap (Section 27).

---

# 20. Technology Stack

Every choice below is justified by "solves an actual, named problem above," not by novelty. For a take-home under a 15-minute reproducibility budget, simplicity and reproducibility are weighted above raw performance ceiling wherever they trade off.

| Layer | Recommended choice | Why (tied to a problem above) |
|---|---|---|
| Language | Python | Ecosystem fit for every other layer below; no reason to deviate |
| Data processing | Pandas for the golden set and dev-scale work; consider DuckDB/Polars only for the one-time full-corpus profiling/cleaning pass over millions of rows, where Pandas alone would be slow and memory-heavy | Matches the assignment's own "preprocess the larger dataset once, use a curated subset for the reproducible pipeline" instruction (Section 17 of the original prompt) — the one-time heavy pass and the fast reproducible pass are different jobs with different tools |
| Classical ML | scikit-learn | TF-IDF+linear baseline (Section 7), embedding+linear-head classifier — no need for a heavier framework at this scale |
| Fine-tuning (if Experiment 8 justifies it) | Hugging Face `transformers` + a LoRA library (e.g., PEFT) | Standard, well-documented, reproducible; LoRA specifically for training-time and reproducibility budget (Section 9.2) |
| Embeddings | A general-purpose sentence-embedding model, commercial-API or open, chosen by a quick quality/cost/latency comparison, not assumed | Powers retrieval (Section 10) and clustering (Section 6); the specific model matters less than having one fixed, versioned choice that's recorded for reproducibility |
| Vector index | FAISS (local, no server to run, no extra infra) over Chroma/Qdrant/pgvector for this project's scale | At tens of thousands of corpus records, a local FAISS index is sufficient and keeps the 15-minute reproduction path free of an extra database service to stand up — a server-backed vector DB is solving a scale problem this project doesn't have |
| Keyword retrieval | BM25 (a lightweight, standard implementation) | Needed for the hybrid retrieval design (Section 10.2); no justification for anything heavier |
| LLM | Commercial API for generation, chosen once and version-pinned; not automatically the newest/largest available — pick based on the cost/latency/quality tradeoff actually measured in Experiment 6 | Avoids the "automatically recommend the biggest model" trap the assignment explicitly warns against |
| Fine-tuning for generation | Not used (Section 9.3) | N/A |
| Experiment tracking | A flat, versioned config file (e.g., YAML) per experiment plus a simple results log (a CSV/JSON of metric outputs per run) — not a full experiment-tracking platform | At this project's scale, a heavyweight tracker (e.g., a hosted MLOps platform) adds setup overhead with no proportionate benefit; a config-in, results-out convention is enough to satisfy the reproducibility requirements in Section 22 |
| Evaluation | Custom harness (this is the deliverable, not a library you import) built on top of scikit-learn's metric functions for the classical metrics, plus a small structured-prompt module for the LLM judge | The evaluation harness's design *is* the graded artifact; outsourcing it to an off-the-shelf eval library would undercut exactly what's being assessed |
| Testing | Unit tests for the deterministic components (thread reconstruction, dedup, the escalation decision function, the PII scan) where correctness is checkable without a model; integration tests that run the full pipeline on a tiny fixture subset and assert it completes and produces schema-valid output; no unit tests attempting to assert exact LLM output (nondeterministic, not testable that way) | Matches what's actually testable; testing an LLM's exact text output is a waste of effort, testing the deterministic scaffolding around it is not |
| Packaging | A single `requirements.txt`/lockfile-pinned environment, a `config/` directory for frozen splits and thresholds, a `Makefile` or a couple of documented shell entry points for "reproduce headline results" | Matches the 15-minute reproduction requirement directly (Section 22) |

No microservices, no orchestration framework, no agent framework, no message queue — none of them solve a problem this project actually has, and each one would only add surface area a reviewer could reasonably ask "why does this exist?" about, with no good answer.

---

# 21. Compute and Cost

**Local requirements:** CPU is sufficient for cleaning, thread reconstruction, TF-IDF/embedding-kNN baselines, BM25 indexing, and even LoRA fine-tuning of a small encoder given enough patience (a few minutes to tens of minutes, not hours, at the scale in Section 9.2). A modest GPU (including a free-tier cloud notebook GPU) meaningfully speeds up fine-tuning and embedding generation over the full candidate-brand corpus, but is not a hard requirement for the reproducible headline path.

**One-time vs. repeated costs, kept separate:** the *full-corpus* profiling/cleaning/brand-scoring pass (Section 4) is a one-time job over the full multi-brand dataset — expect this to be the single most compute- and time-heavy step in the whole project, and it is explicitly **not** part of the 15-minute reproducibility budget (Section 22 addresses this directly: cache its output). Everything downstream of brand selection operates on the much smaller, already-filtered single-brand corpus, which is where the 15-minute budget applies.

**Embedding generation cost:** a one-time pass over the (post-dedup, post-filter) single-brand retrieval corpus — tens of thousands of short texts at most — is cheap regardless of whether you use a commercial embedding API (low, roughly fractions-of-a-cent-per-thousand-tokens-scale) or a local open embedding model (compute-only, no per-call cost). Cache the resulting vectors to disk; never re-embed the same frozen corpus on every run.

**LLM API cost, where it actually matters:** the generation stage is the one recurring per-message cost in the online path. Budget this deliberately: the reproducible demo path should run the full pipeline over the golden set only (150–250 calls), not the full multi-hundred-thousand-tweet corpus — running generation over the full corpus is neither necessary (you only need generation quality measured on the golden set) nor affordable/fast for a 15-minute budget.

**Fine-tuning cost (if Experiment 8 justifies keeping it):** a small-encoder LoRA run over low-thousands of examples is inexpensive in both time and, if using a rented GPU, dollar cost — order of a normal work session, not a multi-day job.

**Storage:** dominated by the raw CSV (multi-gigabyte-scale for the full multi-brand dataset) versus a tiny footprint for the filtered single-brand corpus, the embedding index, and the golden set — store the raw CSV once, cache every derived artifact (cleaned corpus, embeddings, index, taxonomy, thresholds) so nothing expensive is ever silently recomputed.

**The 15-minute reproduction constraint, designed for directly:** cache the expensive one-time steps (full-corpus cleaning, brand scoring, taxonomy discovery, corpus embedding, index building, classifier fitting/training) as versioned artifacts checked into or fetchable by the repo. The reproducible path a reviewer runs is: load cached artifacts → run inference + evaluation over the golden set (150–250 examples, one classifier call + one retrieval call + one LLM call each) → produce the scorecard. This is the only path that plausibly fits 15 minutes, and it's why Section 22's reproducibility design treats "what's cached vs. what's recomputed on every run" as the central design question, not an afterthought.

---

# 22. Reproducibility

**What must be frozen and version-recorded, explicitly:** the intent taxonomy (Section 6.5), the data splits and their exact membership (as a list of thread/tweet IDs, not just "the first 60%" — timestamps alone aren't a sufficient audit trail), the embedding model name/version, the LLM model name/version/temperature/any other sampling parameter used for generation and for judging, the escalation thresholds (with the risk-coverage curve that produced them saved as an artifact, not just the final number), and random seeds for every stage that has one (clustering initialization, classifier training, any sampling in the LLM calls where determinism settings are exposed).

**Handling LLM nondeterminism honestly:** commercial LLM APIs are not perfectly deterministic even at temperature 0 across API versions/dates. The mitigation is not to pretend this doesn't exist — it's to (1) pin the exact model version string used, (2) set temperature to 0 (or the lowest available) for both generation and judging to minimize (not eliminate) run-to-run variance, (3) report the judge's self-consistency check (Section 16.3) as the empirical measure of how much variance actually remains, and (4) frame the headline number with its confidence interval (Section 18) precisely because some of that interval's width is coming from this irreducible source, not just from sample size. State this explicitly in the README rather than implying the pipeline is bit-for-bit deterministic when it structurally cannot be.

**What the README should instruct the evaluator to do, concretely, to reproduce headline results in under 15 minutes:** (1) install the pinned environment; (2) download or point to the cached artifacts (cleaned single-brand corpus, embeddings, index, fitted classifier, frozen thresholds, frozen taxonomy, frozen golden set); (3) run one entry-point command that loads those artifacts and executes inference + the full evaluation harness over the golden set only; (4) print/save the scorecard (Section 15.2) and the risk-coverage curve figure. Everything upstream of step 2 (the expensive full-corpus profiling and one-time training) is documented as a separate, clearly-labeled "how the cached artifacts were produced" section of the README — runnable, but explicitly *not* part of the 15-minute path, so a reviewer never accidentally kicks off a multi-hour job expecting a quick check.

---

# 23. Experiment Plan

Ordered by information gained per unit of time — cheap, high-signal experiments first, expensive/marginal ones last, and later experiments are allowed to be skipped if an earlier one already answers the question well enough.

| ID | Hypothesis | What changes | Held constant | Dataset | Metric | Decision rule |
|---|---|---|---|---|---|---|
| E1 | The trivial baseline is genuinely weak | Majority-class + canned reply vs. nothing | — | Golden set | Macro-F1, safe-auto-handle rate | Establishes the floor; no pass/fail, just a recorded number everything else must clear |
| E2 | TF-IDF+linear meaningfully beats majority-class for intent | Classifier only | Everything else fixed at trivial-baseline settings | Dev slice, confirmed on golden | Macro-F1 | Must clear E1 by a paired-significant margin to proceed past "trivial is fine" |
| E3 | Embedding-kNN beats TF-IDF+linear | Classifier only | Same as E2 | Dev slice | Macro-F1, ECE | Whichever wins becomes the new baseline for E8 |
| E4 | Zero/few-shot LLM classification is competitive with E2/E3 | Classifier only | Same | Dev slice | Macro-F1, plus $ and latency cost | Informs Section 8.3's fallback decision; not expected to win outright given brand-specific vocabulary |
| E5 | Hybrid BM25+embedding retrieval beats either alone | Retrieval method only | Fixed classifier (winner of E2-E4), fixed generator off | Retrieval-eval query set (Section 10.4) | Recall@3, MRR, resolution-usefulness rate | Hybrid must clear both single-method baselines by a real margin or the system reverts to whichever single method matched hybrid |
| E6 | Intent-conditioned filtering measurably improves retrieval over hybrid alone | Retrieval filtering only | Fixed hybrid ranking from E5 | Same | Recall@3, resolution-usefulness rate | This is the experiment expected to show the largest single jump (Section 10.5) — if it doesn't, that's a first-class reportable finding, not a bug to hide |
| E7 | Reranking improves retrieval enough to justify its latency | Add cross-encoder rerank on top of E6's winner | Everything else fixed | Same | Recall@3/nDCG delta vs. added latency | Keep only if the Recall/nDCG gain clears a threshold you set *before* running this experiment (avoid post-hoc justification) |
| E8 | Fine-tuning the intent classifier beats the best non-fine-tuned option (E2-E4 winner) | Classifier only | Fixed retrieval/generation | Dev slice, confirmed on golden | Macro-F1, ECE, paired bootstrap CI on the delta | Ship the fine-tuned model only if it wins by a margin the CI supports (Section 9.2); otherwise ship the simpler winner and report why |
| E9 | Retrieval-grounded generation beats no-retrieval generation | Generator's retrieval context on/off | Fixed everything else (best classifier, best retrieval) | Golden set | Groundedness, correctness, unsupported-claim rate | This is the core "grounding" claim's evidence — must show a real, paired-significant gap or the grounding claim in the report is unsupported |
| E10 | The escalation decision function beats a single fixed confidence threshold | Full risk-based decision function vs. single-threshold baseline | Fixed upstream signals | Golden set, at matched coverage levels | Observed error rate at matched coverage | Compare at the *same* coverage point on both, not just overall accuracy — an apples-to-apples risk-coverage comparison |
| E11 | The judge agrees adequately with humans | Judge scoring vs. human scoring on the validation subset | — | Judge-validation subset | Cohen's kappa / Spearman per dimension (Section 16.2) | If agreement is weak on a dimension, that dimension is flagged with a reliability caveat everywhere it's reported, not silently trusted |
| E12 (ablation) | Each architectural component earns its complexity | Full system minus one component at a time: {reranker, intent-conditioning, hybrid retrieval (vs. dense-only), fine-tuned classifier (vs. E8's simpler alternative), groundedness checker, escalation layer (vs. fixed threshold)} | Everything else fixed | Golden set | Component-specific metric (retrieval metrics for retrieval-related removals, safe-auto-handle rate for escalation-related removals) | Any component whose removal doesn't measurably hurt the relevant metric is a candidate for cutting from the final architecture — report this table even for components you keep, since "we tested removing it and it mattered" is itself evidence |

**Ablation table format, explicitly, since this is one of the assignment's specific requests:** report a single table with rows = {Full system, − reranker, − intent conditioning, − hybrid retrieval, − fine-tuned classifier, − groundedness checker, − escalation layer} and columns = {the 2–3 metrics each removal is expected to affect}, so the question "which parts of this architecture actually matter" has one direct, scannable answer rather than being scattered across the report.

---

# 24. Failure Analysis Methodology

## 24.1 A systematic procedure, not five invented examples

1. Run the full evaluation harness over the golden set.
2. Collect every example where any component's output was flagged as incorrect (wrong intent, poor retrieval per the resolution-usefulness label, a judge-flagged unsupported claim, or an escalation decision that disagreed with the golden set's `should_escalate` label).
3. Tag each collected error with an **error-taxonomy root cause** (below) — not "LLM made a mistake," which the assignment explicitly calls out as not an engineering diagnosis.
4. Group by root cause and by intent/case-type tag from Section 14.1.
5. Quantify: count and % of total errors per root-cause bucket.
6. Select the five buckets with the largest count (or, if a low-count bucket is disproportionately high-risk — e.g., a single false-auto-handle on an account-security case — elevate it regardless of raw count, and say so explicitly, since raw frequency is not the same as impact).
7. Pull 2–3 real example transcripts per selected bucket, quoting the actual input/output pair (not paraphrased into something cleaner than it really was).
8. For each, write a specific hypothesis about the mechanism ("the classifier confused these two intents because their training examples share vocabulary X" — not a vague "the model struggled").
9. For each hypothesis, propose a concrete fix.
10. For each proposed fix, name the risk it introduces (e.g., "merging these two intents fixes the confusion but loses the ability to route billing disputes differently from delivery disputes") — never propose a fix as free.

## 24.2 Error taxonomy — the root-cause buckets

| Bucket | Definition | Example symptom |
|---|---|---|
| Data quality | The underlying training/retrieval data itself was wrong, missing, or misleading (e.g., a DM-deflection reply that slipped through the filter) | Retrieved "resolution" is actually a content-free apology |
| Intent classification | The predicted intent was wrong, given a reasonable taxonomy | Billing question misclassified as a general complaint |
| Retrieval | The intent was right but the retrieved evidence, though topically related, wasn't actually useful for resolution | Retrieved an answer for a different sub-case of the same intent |
| Reranking | Correct candidates existed in the retrieved set but were ranked below a worse one | Top-1 result is worse than the top-3 result by human judgment |
| Generation | Evidence was good, but the generator didn't use it faithfully | Draft ignores the cited evidence and answers from general knowledge |
| Grounding/citation | The generator cited evidence it didn't actually rely on, or made a claim the evidence doesn't support | `cited_evidence_ids` present but claim doesn't match cited text |
| Escalation | Upstream signals were reasonable but the decision function's threshold or logic was wrong | A low-confidence, low-evidence case was still auto-handled |
| Evaluation | The judge or the automated check itself was wrong, not the system | Judge penalizes a correct paraphrase for not lexically matching the reference |
| Ambiguous human label | The golden-set label itself is contestable on a second read | Re-reading the thread, a different intent tag is equally defensible |

Reporting the *distribution* across these buckets (not just five isolated anecdotes) is itself a finding — e.g., "60% of generation-stage errors trace back to a retrieval-stage root cause" is a materially more useful statement than five unconnected examples, and it directly informs which component gets attention in "what I'd do with one more week" (Section 30).

---

# 25. Decision Log

Twelve non-obvious decisions, in the format the assignment asks for. These are the decisions most likely to come up in an interview, and Section 26 cross-references several of them directly.

**1. Decision:** Brand selected by a weighted scoring rubric that penalizes DM-deflection-heavy brands, not by raw tweet volume.
**Alternatives:** pick the highest-volume brand (Apple/Amazon).
**Reason:** volume without visible-resolution density produces a shallow grounding corpus (Section 2.5, 4.1).
**Evidence:** manual read of sampled replies from high-volume brands showing a high DM-deflection rate.
**Tradeoff:** the chosen brand likely has less raw volume than the top candidate, meaning a smaller retrieval corpus.

**2. Decision:** Intent taxonomy frozen before the golden set is sampled or labeled.
**Alternatives:** iterate on the taxonomy while building the golden set.
**Reason:** prevents test-set information from silently shaping the labels the model is evaluated against (Section 6.5).
**Evidence:** the assignment's own explicit leakage warning about the taxonomy.
**Tradeoff:** a genuine taxonomy gap discovered during golden-set labeling can't be patched immediately; it becomes a documented limitation instead.

**3. Decision:** Temporal, conversation-level split instead of random split.
**Alternatives:** random split (simpler to implement).
**Reason:** random splitting under-represents the real deployment condition (unseen future issues) and is vulnerable to mass-near-duplicate leakage specific to this dataset (Section 13.2).
**Evidence:** the near-duplicate outage-complaint pattern observed during data-quality review.
**Tradeoff:** a temporal split can make the golden set's intent distribution slightly different from the full corpus's distribution (recent traffic may skew toward whatever issues were current near the end of the collection window) — mitigated by, but not fully solved by, stratified sampling.

**4. Decision:** Intent classification done by a small trained model (embedding-based or fine-tuned encoder, decided empirically), not an LLM call.
**Alternatives:** zero/few-shot LLM classification for everything.
**Reason:** need deterministic, cheap, calibratable confidence for the escalation layer; LLM confidence self-reports are not well-calibrated by default (Section 8.2).
**Evidence:** Experiment E4 measures the LLM option directly rather than assuming it's worse.
**Tradeoff:** requires a labeled training/dev set and a training step the pure-LLM approach wouldn't need.

**5. Decision:** Fine-tuning is applied (if at all) only to the intent classifier, never to the response generator.
**Alternatives:** fine-tune an end-to-end generator on brand-specific reply data.
**Reason:** the retrieval corpus's own resolution data is compromised by the DM-deflection problem (Section 2.5); fine-tuning a generator on it risks learning to imitate boilerplate non-answers, and hides that risk inside model weights instead of exposing it as a checkable retrieval/citation signal (Section 9.3).
**Evidence:** manual audit of "resolution" text showing a large deflection share.
**Tradeoff:** forgoes any latency/cost benefit a smaller fine-tuned generator might have offered over a commercial LLM call.

**6. Decision:** Retrieval is intent-conditioned (filter by predicted intent before ranking), not pure similarity search.
**Alternatives:** rank the whole corpus by embedding similarity alone.
**Reason:** semantic similarity conflates surface-similar-but-operationally-different issues (the "my card isn't working" problem, Section 10.1–10.3).
**Evidence:** Experiment E6 measures the improvement directly.
**Tradeoff:** if the classifier's intent prediction is wrong, retrieval inherits that error — a compounding-error risk explicitly tracked in the error taxonomy (Section 24.2).

**7. Decision:** Reranking is included only conditionally, based on ablation results (Experiment E7), not by default.
**Alternatives:** include a cross-encoder reranker unconditionally, since "more retrieval sophistication" is a common default in RAG write-ups.
**Reason:** added latency/complexity must earn its keep with a measured improvement (assignment's own instruction, 26.5).
**Evidence:** E7's Recall@3/nDCG delta versus a pre-committed threshold.
**Tradeoff:** if kept, adds latency and one more component to explain and maintain; if cut, forgoes whatever marginal ranking improvement it might have offered.

**8. Decision:** Response generation outputs a structured, evidence-cited object, not free text.
**Alternatives:** let the LLM produce a plain-text reply.
**Reason:** makes groundedness mechanically checkable rather than dependent entirely on a judge's subjective read (Section 11.1–11.2).
**Evidence:** the specific "citation theater" failure mode (citing but not using evidence) is only detectable this way.
**Tradeoff:** more complex prompt engineering and output parsing than free-text generation.

**9. Decision:** Escalation is a hand-specified risk-tiered decision function with data-fit thresholds, not a trained auto/escalate classifier.
**Alternatives:** train a supervised classifier to predict auto-vs-escalate directly.
**Reason:** no reliable ground-truth label exists for this at scale, and a trained end-to-end classifier would hide *why* it decided what it decided, which directly conflicts with the assignment's ask for a structured, explainable escalation reason (Section 12.1, 12.6).
**Evidence:** the assignment's explicit instruction not to rely on an arbitrary threshold, generalized here to "don't rely on an opaque model either."
**Tradeoff:** more manual design work (defining risk tiers, wiring signals together) than training a single classifier would require.

**10. Decision:** False-auto-handle and false-escalation are reported as two separate rates, with an explicit stated asymmetric cost weighting, not blended into one score.
**Alternatives:** report a single blended escalation-accuracy number.
**Reason:** the two error types have genuinely different real-world costs (Section 12.5), and blending them hides which one the system is actually trading off.
**Evidence:** the assignment's explicit question, "is a bad auto-response worse than an unnecessary escalation?"
**Tradeoff:** a more complex scorecard with more numbers to explain, versus one clean headline figure.

**11. Decision:** The LLM judge is validated against a human-double-labeled subset before its scores are trusted anywhere else in the report.
**Alternatives:** use the judge's scores directly as ground truth once the rubric looks reasonable.
**Reason:** an unvalidated judge is exactly the "LLM judge bias" failure mode the assignment names explicitly as something a skeptical reviewer will look for (Section 16.2, 18).
**Evidence:** Cohen's kappa / Spearman computed on the validation subset (Experiment E11).
**Tradeoff:** consumes part of the limited human-labeling budget that could otherwise have gone toward a larger golden set.

**12. Decision:** The headline metric is "observed error rate among auto-handled cases at a chosen coverage point," not "classifier accuracy" or "average judge score."
**Alternatives:** report the highest, most flattering single number available (e.g., average judge score across all dimensions).
**Reason:** directly answers the deployment-safety question the assignment frames as central ("can I trust this system enough to let it act"), rather than a number that could be high while the system is still unsafe to deploy (Section 15.2, 18).
**Evidence:** the risk-coverage curve itself, plus the explicit "what's misleading about my headline number" analysis.
**Tradeoff:** a less immediately impressive-sounding number than a single "94% accuracy" headline would be — a deliberate choice to prioritize honesty over polish.

---

# 26. Interview Defense

## 26.1 Major-decision Q&A (likely question / strong answer / weak answer to avoid)

**"Why this brand?"** Strong: "I scored candidate brands on a weighted rubric — volume, visible-resolution rate, issue diversity, self-containedness, escalation-worthy risk presence, noise level — and disqualified brands whose visible-resolution rate fell below a floor, because I found that the highest-volume brands deflect most account-specific issues to DMs, which means the dataset doesn't actually contain their resolutions." Weak: "It had a lot of tweets."

**"Why this intent taxonomy?"** Strong: "I discovered it from embedding clusters on a dev slice, validated coverage and consistency on a held-out sample, and explicitly froze it before touching the golden set to avoid shaping labels around what I already knew the model would see." Weak: "I made up categories that seemed reasonable."

**"Why these baselines?"** Strong: name the specific thing each baseline isolates (Section 7.3's table) — "the trivial baseline is the floor everything must clear; the TF-IDF/BM25 baseline isolates how much value comes from embeddings/LLMs at all; the no-retrieval LLM baseline isolates how much of response quality is retrieval versus the model's own knowledge." Weak: "Because the assignment required two baselines."

**"Why did you use an LLM here [generation], and why didn't you use a smaller model?"** Strong: natural-language drafting conditioned on retrieved evidence is exactly the LLM's strength, and the cost/latency of one LLM call per message (versus per every stage) was deliberately minimized by keeping classification and retrieval outside the LLM path. Weak: "LLMs are state of the art."

**"Why didn't you fine-tune [the generator]? / Why did you fine-tune [the classifier, if you did]?"** Strong: point directly to Section 9's evidence — the retrieval corpus's own resolution data is compromised by DM-deflection, so fine-tuning a generator on it risks learning to imitate non-answers; the classifier, by contrast, only needed a labeled signal that's cheaper and cleaner to obtain, and was only kept if Experiment E8 showed a real, CI-backed improvement over the simpler baseline. Weak: "Fine-tuning sounded more advanced."

**"How do you know fine-tuning helped?"** Strong: name the specific paired comparison and confidence interval from E8, and state plainly what you'd have done if it hadn't cleared the bar (ship the simpler model). Weak: "It got a higher number."

**"How do you know the retrieved response was actually relevant?"** Strong: distinguish semantic relevance (Recall@K/MRR) from resolution-usefulness (a separately labeled dimension, Section 10.4) and report both — a high Recall@K with a low usefulness rate is itself a finding you'd surface, not hide.

**"What happens when there is no similar historical conversation?"** Strong: the novelty/OOD signal and `missing_information_flag` (Sections 11.3, 12.2) push the decision toward escalation rather than letting the generator improvise; walk through a real golden-set example tagged as a low-evidence case.

**"What happens when two historical conversations give contradictory answers?"** Strong: the conflicting-evidence detector (Section 10.3) is a direct escalation trigger, and a dedicated golden-set slice (Section 14.1) exists specifically to test this.

**"How do you know the LLM judge is reliable?"** Strong: name the specific kappa/Spearman numbers from the human-validation subset and state, without hedging, whether any dimension fell short and what you did about it (tightened the rubric, or flagged that dimension with a caveat) — never claim the judge is simply "good" without a number.

**"What happens when the classifier is confident but wrong?"** Strong: this is exactly why confidence alone doesn't drive escalation — the risk-tier override and the groundedness/conflicting-evidence checks are independent signals that can catch a confidently-wrong classification downstream, and the error taxonomy (Section 24.2) explicitly tracks how often this actually occurs.

**"Why should this message be auto-handled? What is the cost of a false auto-response?"** Strong: walk through the risk-coverage curve and the stated asymmetric cost weighting (Section 12.5) — "we chose to accept X% coverage in exchange for keeping the auto-handled error rate under Y%, and we weight a false auto-handle as roughly 3-5x worse than an unnecessary escalation because the former actively damages trust."

**"Why this operating threshold and not a different one?"** Strong: it's the point on the risk-coverage curve that satisfies a pre-stated error tolerance, fit on the dev set before the golden set was touched — show the curve, not just the point.

**"What would happen on tomorrow's unseen customer issues? How much of your performance depends on historical repetition?"** Strong: this is exactly what the temporal split (Section 13.2) and the "fresh" golden-set slice (Section 14.1) are designed to measure directly — report that slice's performance specifically, not just the aggregate.

**"What happens when the customer uses completely different wording?"** Strong: the hybrid retrieval design (dense embeddings specifically to catch paraphrase BM25 would miss) plus the novelty signal for the case where wording is different *and* the underlying issue is genuinely novel, not just rephrased.

**"Can the system recognize that it doesn't know?"** Strong: yes, by construction — `missing_information_flag`, novelty scoring, and conflicting-evidence detection all feed escalation; this isn't a separate "confidence" bolt-on, it's a structural property of forcing retrieval-conditioned, evidence-cited generation (Section 11.3).

## 26.2 Where an interviewer could reasonably challenge the architecture (said out loud, not left for them to find)

The manually-assigned intent risk tiers (Section 12.2) encode judgment that wasn't learned from data and could miss a genuinely risky category that wasn't anticipated — a real gap, disclosed rather than hidden. The escalation decision function's combined score, if it uses a hand-set weighting across signals rather than a learned combination, trades some potential accuracy for interpretability — a defensible tradeoff, but one to name proactively. The golden set, at 150–250 examples, cannot support fine-grained per-intent statistics for the rarest intents with tight confidence — acknowledged directly in Section 18's uncertainty discussion, not glossed over. And the brand-selection scoring rubric's weights (Section 4.2) were chosen once, up front, based on judgment about what should matter — a different, equally reasonable weighting could produce a different brand ranking, and that's disclosed rather than presented as if the rubric were itself an objective measurement.

---

# 27. Complete Implementation Roadmap

No code in this document, per the brief — this is the sequencing and acceptance criteria for what you'll build in Cursor.

**Phase 0 — Environment + dataset acquisition.** Objective: a working, pinned environment and the raw CSV in hand. Tasks: set up the pinned Python environment; download the full multi-brand CSV; verify row count and column names against Section 2.2's confirmed schema. Inputs: none. Outputs: raw CSV + environment. Files/modules eventually needed: `env/`, `data/raw/`. Acceptance criteria: schema matches Section 2.2 exactly; row count is in the expected multi-million range.

**Phase 1 — Dataset profiling (full corpus).** Objective: real numbers to replace every hypothesis in Sections 2–4 with measured facts. Tasks: language ID pass; exact + near-duplicate detection and collapse; thread reconstruction (branch-aware tree walk); per-brand statistics (volume post-dedup, DM-deflection rate, rough issue-diversity via a quick clustering pass, noise level). Outputs: a brand scorecard (Section 4.2's table, filled in with real numbers). Acceptance criteria: every factor in the scoring rubric has a real, computed value per candidate brand, not a guess.

**Phase 2 — Brand selection.** Objective: commit to one brand, with the scorecard as the written justification. Tasks: apply the pre-committed weights; apply the visible-resolution-rate floor as a hard disqualifier; select. Acceptance criteria: the selection is fully explainable from Phase 1's numbers alone, with no post-hoc reweighting.

**Phase 3 — Single-brand cleaning + conversation reconstruction.** Objective: a clean, threaded, single-brand corpus. Tasks: filter to the selected brand; re-run language/dedup/thread-reconstruction at brand scale with tighter manual spot-checking (small enough now to sanity-check by eye); extract visible-resolution pairs (filter out DM-deflection and content-free replies). Acceptance criteria: a manual spot-check of ~30 random reconstructed threads shows correct branch handling and correct visible-resolution filtering.

**Phase 4 — Temporal split.** Objective: the frozen retrieval-corpus / dev-slice / (pre-golden) boundary. Tasks: sort by time; cut into the ordered slices from Section 13.2; record exact thread-ID membership per slice as a versioned artifact. Acceptance criteria: zero ID overlap between slices, verified by an explicit assertion, not by inspection.

**Phase 5 — Intent discovery + taxonomy freeze.** Objective: the frozen intent list. Tasks: embedding clustering on the dev slice; LLM-assisted cluster labeling; human merge/split/validate pass; freeze. Acceptance criteria: taxonomy is written down with one-sentence definitions per intent, re-running assignment on a fresh dev sample shows consistent coverage, and this happens strictly before Phase 6.

**Phase 6 — Golden set construction.** Objective: the frozen, human-labeled 150–250 example evaluation set. Tasks: sample per Section 14.1's stratified plan from the golden-set temporal slice only; label `primary_intent`/`secondary_intent`, `should_escalate`, `reference_resolution_summary`, `case_type_tags`; sample and double-label the judge-validation subset. Acceptance criteria: the sampling composition table (Section 14.1) matches the actual counts achieved; the set is then treated as read-only for the rest of the project.

**Phase 7 — Baseline models.** Objective: E1–E4 run and recorded. Tasks: implement majority-class+canned-reply, TF-IDF+linear, embedding-kNN, zero/few-shot LLM classification; evaluate each on the dev slice. Acceptance criteria: a results table exists before any "final" model is built, per the evaluation-first discipline in Section 13.

**Phase 8 — Retrieval system.** Objective: E5–E7 run and the retrieval architecture locked. Tasks: build the resolution-pair corpus and hybrid index; run the retrieval ablation; decide on reranking based on E7's result. Acceptance criteria: Recall@K/MRR/resolution-usefulness numbers exist for every ablation row in Section 10.5's table.

**Phase 9 — Response generation.** Objective: the structured, evidence-cited generator working end-to-end on top of the locked retrieval system. Tasks: build the structured-output prompt; build the automated citation/groundedness check; run E9 (retrieval on/off). Acceptance criteria: every generated output validates against the structured schema, and E9's groundedness delta is measured.

**Phase 10 — Escalation.** Objective: the risk-based decision function, thresholds fit from the dev slice. Tasks: assign risk tiers per intent; wire up the signal set (Section 12.2); fit thresholds via the risk-coverage curve on the dev slice; run E10 against the single-threshold baseline. Acceptance criteria: a risk-coverage curve figure exists, and the chosen operating point is justified by a stated error-tolerance target, not picked after seeing golden-set results.

**Phase 11 — Fine-tuning experiment (E8).** Objective: a real answer to "should the classifier be fine-tuned." Tasks: train the small encoder (LoRA first); compare against the best of Phase 7's non-fine-tuned options with a paired bootstrap CI. Acceptance criteria: a ship/don't-ship decision is made and written down with the supporting numbers, regardless of which way it goes.

**Phase 12 — Evaluation harness + golden-set run.** Objective: the full scorecard (Section 15.2), computed once, on the now-fully-locked system, over the golden set. Tasks: run every component's metric; run the LLM judge; run E11 (judge validation); assemble the ablation table (E12). Acceptance criteria: every number in Section 15.2's scorecard has a value, and every claim has an attached confidence interval where Section 18 calls for one.

**Phase 13 — Failure analysis.** Objective: the top-five failure modes, found systematically. Tasks: run the 10-step procedure in Section 24.1 over the golden-set results from Phase 12. Acceptance criteria: every reported failure mode has a root-cause tag from the taxonomy (24.2), a quantified frequency, and a real example transcript — not an invented one.

**Phase 14 — Final experiments / polish.** Objective: anything E1–E12 revealed as worth a second pass (e.g., a rubric tightened after E11, a threshold re-examined after seeing the risk-coverage curve on the dev slice — never re-tuned against the golden set itself). Acceptance criteria: any change made here is logged in the decision log with a reason.

**Phase 15 — README / reproducibility packaging.** Objective: the 15-minute reproduction path. Tasks: cache every expensive artifact; write the two-part README (Section 22's "reproduce headline results" fast path + "how the cached artifacts were produced" slow path); pin versions. Acceptance criteria: a clean-environment dry run of the fast path actually completes in under 15 minutes.

**Phase 16 — Report.** Objective: the ≤6-page report. Tasks: assemble from material already produced in Phases 1–14 (problem framing from Section 1/13.3, "what good means" from 13.3, the two baselines from Phase 7, the top-5 failure modes from Phase 13, the misleading-headline-number section directly from Section 18, and the one-more-week section from Section 30 below). Acceptance criteria: every section the assignment requires is present, and nothing in the report states a number without its dataset size, methodology pointer, and baseline comparison, per Section 18's discipline.

Each phase's acceptance criteria must pass before starting the next — this is a genuinely sequential roadmap (evaluation infrastructure and frozen splits have to exist before models are meaningfully compared), not a checklist that can be reordered for convenience.

---

# 28. Final Recommended Architecture

**RECOMMENDED SYSTEM:** a hybrid pipeline — a small, empirically-chosen intent classifier (trained, not an LLM call) feeding an intent-conditioned hybrid (BM25 + embedding) retrieval system over a mined `problem → resolution` corpus that has been explicitly filtered to exclude DM-deflection and content-free replies, feeding a single commercial-LLM generation call constrained to structured, evidence-cited output, wrapped in a risk-tiered escalation decision function whose thresholds come from a risk-coverage curve fit on a dev set — evaluated by a golden set built with deliberate hard-case oversampling and scored by an LLM judge that is itself validated against human labels before being trusted.

**Why this is the best tradeoff, stated once, plainly:** it is the only option in Section 8.1's comparison table that scores well on every column the assignment is actually grading — explainability, reproducibility, reasonable cost, and a genuine (not decorative) answer to "why is this better than an LLM API with basic RAG." The differentiator from a generic submission is concentrated in exactly three places, and an interviewer's attention should be steered there: (1) intent-conditioned retrieval that explicitly solves the semantic-vs-resolution-similarity problem, (2) an escalation layer that is a decision-theoretic risk-coverage system rather than a single guessed threshold, and (3) an evaluation harness that validates its own judge and reports honest uncertainty rather than one flattering percentage.

**Rolled-up pointers to every sub-answer, per the assignment's own requested checklist:** complete architecture — Section 19; data/model flow — Section 19.1; dataset preparation — Roadmap Phases 0–4; intent discovery — Section 6, Roadmap Phase 5; intent classification — Section 8, 9.2; retrieval architecture — Section 10; response generation — Section 11; escalation architecture — Section 12; evaluation architecture — Sections 13–18; golden set — Section 14; judge methodology — Section 16; human validation — Section 14.4, 16.2; baselines — Section 7; leakage prevention — Section 17; reproducibility — Section 22; technology stack — Section 20; compute — Section 21; cost — Section 21; bottlenecks — the full-corpus profiling pass (Phase 1) and the LLM generation stage's latency/cost are the two most likely bottlenecks, both explicitly budgeted around rather than discovered late; failure modes — Section 24; security/privacy — Section 3 row 10, Section 11.4; repository architecture — Section 20's packaging row, Section 22; experiment plan — Section 23; report structure — Roadmap Phase 16; decision log — Section 25; interview defense — Section 26.

---

# 29. What NOT to Build

Stated explicitly, because the report is required to name what was deliberately left out, and because restraint here is itself evidence of judgment: **no fine-tuned response generator** (Section 9.3 — the data doesn't support it, and RAG's citation mechanism already catches the failure mode fine-tuning would hide). **No cross-encoder reranker by default** — included only if E7 earns it, cut otherwise (Section 10.2, 23). **No microservice architecture, message queue, or agent framework** — nothing in this project's scale or requirements needs them, and each one would be unexplainable weight in an interview (Section 20). **No self-hosted open-source LLM stack** — doesn't move any graded metric and adds real infrastructure risk for a solo, time-boxed take-home (Section 9.1, option E). **No single collapsed "quality score"** — every section from 15 onward deliberately keeps dimensions separate because collapsing them is exactly how misleading headline numbers get produced (Section 18). **No end-to-end-trained auto/escalate classifier** — no reliable ground-truth label exists at the scale that would require, and it would remove the explainability the escalation layer exists to provide (Section 12.1). **No flashy UI** — the assignment's own priority list explicitly excludes this; effort goes into the evaluation harness instead. **No attempt to cover all ~20 brands** — one well-chosen brand, deeply and honestly evaluated, is worth more than a shallow multi-brand demo (Section 4).

---

# 30. What I Would Do With One Additional Week

In priority order, given everything above: **(1) Expand the golden set's rare-intent and conflicting-evidence slices specifically** — the confidence intervals in Section 18 are widest exactly where the golden set is thinnest, and a second week is best spent narrowing those intervals rather than adding a new architectural component. **(2) Run a second, independent human labeler over a larger overlap subset** to get a real inter-annotator-agreement baseline for the golden set itself, not just for the judge — right now, human-vs-human agreement is only measured on the small judge-validation subset; a bigger version of that check would tell you how much of any human-vs-judge disagreement is actually judge error versus inherent label ambiguity. **(3) Revisit the manually-assigned intent risk tiers with a small red-team pass** — deliberately try to construct messages designed to slip past the escalation logic (ambiguous phrasing that hides a security-relevant issue, e.g.), and use whatever gets through to harden the risk-tier assignment and the conflicting-evidence detector. **(4) Test a second brand through the same pipeline** end-to-end (not to ship, just to evaluate) to see how much of the current system's performance is brand-specific tuning versus genuinely general architecture — this would directly answer a natural follow-up interview question ("would this work for a different brand?") with evidence instead of assertion. **(5) Revisit the reranker decision (E7) with a larger retrieval-eval query set**, since the current one is sized for a fast reproducible pipeline and a null result there could simply reflect low power, not a genuine absence of benefit.
