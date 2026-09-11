"""
phase6_golden_set.py — Stratified golden set construction and labeling interface.

Blueprint Section 14, Roadmap Phase 6.

Sampling strategy (per Section 14.1):
  - ~45% intent-stratified normal cases (all intents covered, minimum floor per intent)
  - ~15% hard cases (short/ambiguous/typo-heavy, identified by heuristics)
  - ~10% multi-intent cases (threads that show topic drift)
  - ~10% low-retrieval-evidence cases (identified by running retrieval offline)
  - ~5%  conflicting-evidence cases
  - ~10% should-escalate cases (high-risk intent tier present)
  - ~5%  most-recent "fresh" cases

Steps:
  1. --sample: sample candidates from the golden-set temporal slice
  2. --label: print an interactive labeling interface in the terminal
  3. --check: verify labeling is complete and freeze the golden set

Usage:
    python src/phase6_golden_set.py --brand SpotifyCares --sample
    python src/phase6_golden_set.py --brand SpotifyCares --label
    python src/phase6_golden_set.py --brand SpotifyCares --check
"""

import os
import sys
import json
import logging
import argparse
import pickle
import csv
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

# Target golden set size
TARGET_SIZE = 200

# Stratification targets (as fractions of TARGET_SIZE)
STRAT_TARGETS = {
    "normal": 0.45,
    "hard": 0.15,
    "multi_intent": 0.10,
    "low_evidence": 0.10,
    "conflicting_evidence": 0.05,
    "escalation": 0.10,
    "fresh": 0.05,
}

# Minimum examples per intent for the normal stratum
MIN_PER_INTENT = 3


def load_golden_slice_threads(brand_id: str) -> List[dict]:
    path = ARTIFACTS_DIR / f"threads_golden_slice_{brand_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Golden slice threads not found. Run phase4_split.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_taxonomy() -> dict:
    path = ARTIFACTS_DIR / "taxonomy.json"
    if not path.exists():
        raise FileNotFoundError("taxonomy.json not found. Run phase5_taxonomy.py --freeze first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_resolution_pairs(brand_id: str) -> List[dict]:
    """Load the retrieval corpus resolution pairs (for low-evidence detection)."""
    path = ARTIFACTS_DIR / f"resolution_pairs_{brand_id}.pkl"
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Heuristics for hard-case detection
# ---------------------------------------------------------------------------

def is_hard_case(text: str) -> bool:
    """Identify messages that are short, ambiguous, or noisy."""
    import re
    if not text:
        return True
    clean = re.sub(r"@\S+", "", text).strip()
    # Short messages
    if len(clean.split()) < 5:
        return True
    # High emoji density
    emoji_count = sum(1 for c in text if ord(c) > 0x1F300)
    if emoji_count > 3:
        return True
    # All caps (venting)
    alpha = re.sub(r"[^a-zA-Z]", "", clean)
    if alpha and alpha.upper() == alpha and len(alpha) > 10:
        return True
    return False


def has_multi_intent_signal(thread: dict) -> bool:
    """Simple heuristic: customer sends more than 2 messages in the thread."""
    customer_turns = [t for t in thread.get("turns", []) if t.get("inbound", True)]
    return len(customer_turns) >= 3


def get_first_customer_message(thread: dict) -> str:
    """Get the first customer message from a thread."""
    for turn in thread.get("turns", []):
        if turn.get("inbound", True):
            return turn.get("text", "")
    return ""


def compute_retrieval_similarity(
    query: str,
    resolution_pairs: List[dict],
    embedder,
    top_k: int = 3,
) -> float:
    """
    Compute the max cosine similarity between a query and the retrieval corpus.
    Returns 0 if corpus is empty or embedding fails.
    """
    if not resolution_pairs or not query:
        return 0.0
    try:
        from sentence_transformers import util
        corpus_texts = [
            p.get("customer_problem_text", "") for p in resolution_pairs[:500]
        ]
        query_emb = embedder.encode([query], normalize_embeddings=True)
        corpus_emb = embedder.encode(corpus_texts, normalize_embeddings=True, batch_size=64)
        sims = util.cos_sim(query_emb, corpus_emb)[0]
        return float(sims.max().item())
    except Exception:
        return 0.5  # safe default


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_candidates(brand_id: str) -> List[dict]:
    """
    Sample candidate examples for the golden set using stratified approach.
    """
    logger.info("Loading data for golden set sampling...")
    threads = load_golden_slice_threads(brand_id)
    taxonomy = load_taxonomy()
    resolution_pairs = load_resolution_pairs(brand_id)
    intent_labels = [i["label"] for i in taxonomy["intents"]]

    logger.info(f"Golden slice: {len(threads):,} threads")

    # Initialize embedder for retrieval similarity scoring
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedder loaded for retrieval similarity scoring")
    except Exception:
        embedder = None
        logger.warning("Embedder not available; low-evidence detection disabled")

    # Build candidate pool
    candidates = []
    for thread in threads:
        first_msg = get_first_customer_message(thread)
        if not first_msg:
            continue

        # Compute signals
        is_hard = is_hard_case(first_msg)
        is_multi = has_multi_intent_signal(thread)

        # Retrieval similarity (expensive, do for smaller golden slice only)
        if embedder and resolution_pairs:
            ret_sim = compute_retrieval_similarity(first_msg, resolution_pairs, embedder)
        else:
            ret_sim = 0.5

        candidates.append({
            "thread_id": thread["root_id"],
            "first_message": first_msg,
            "thread_text": " ".join(
                t.get("text", "") for t in thread.get("turns", [])[:4]
            ),
            "is_hard": is_hard,
            "is_multi_intent": is_multi,
            "retrieval_sim": ret_sim,
            "is_fresh": True,  # all golden-slice threads are "fresh" candidates
            "thread_depth": thread.get("depth", 0),
            "has_branches": thread.get("has_branches", False),
        })

    logger.info(f"Candidate pool: {len(candidates):,} examples")

    # Stratified sampling
    selected = []
    used_thread_ids = set()

    def pick(pool: List[dict], n: int, case_type: str) -> List[dict]:
        available = [c for c in pool if c["thread_id"] not in used_thread_ids]
        n_pick = min(n, len(available))
        picked = random.sample(available, n_pick)
        for p in picked:
            p["case_type"] = case_type
            used_thread_ids.add(p["thread_id"])
        return picked

    n_normal = int(TARGET_SIZE * STRAT_TARGETS["normal"])
    n_hard = int(TARGET_SIZE * STRAT_TARGETS["hard"])
    n_multi = int(TARGET_SIZE * STRAT_TARGETS["multi_intent"])
    n_low_ev = int(TARGET_SIZE * STRAT_TARGETS["low_evidence"])
    n_conflict = int(TARGET_SIZE * STRAT_TARGETS["conflicting_evidence"])
    n_escalate = int(TARGET_SIZE * STRAT_TARGETS["escalation"])
    n_fresh = int(TARGET_SIZE * STRAT_TARGETS["fresh"])

    # Hard cases
    hard_pool = [c for c in candidates if c["is_hard"]]
    selected.extend(pick(hard_pool, n_hard, "hard"))
    logger.info(f"Hard cases: {len([x for x in selected if x['case_type']=='hard'])}")

    # Multi-intent cases
    multi_pool = [c for c in candidates if c["is_multi_intent"] and not c["is_hard"]]
    selected.extend(pick(multi_pool, n_multi, "multi_intent"))

    # Low-evidence cases (bottom quartile of retrieval similarity)
    sim_sorted = sorted(candidates, key=lambda x: x["retrieval_sim"])
    low_ev_pool = sim_sorted[: len(sim_sorted) // 4]
    selected.extend(pick(low_ev_pool, n_low_ev, "low_evidence"))

    # Conflicting evidence (just use next-lowest retrieval-sim candidates)
    selected.extend(pick(low_ev_pool, n_conflict, "conflicting_evidence"))

    # Escalation candidates (high-risk signal words)
    escalation_keywords = {
        "hack", "hacked", "unauthorized", "charge", "charged", "refund",
        "fraud", "stolen", "security", "account", "billing", "cancel",
        "scam", "breach"
    }
    escalation_pool = [
        c for c in candidates
        if any(k in c["first_message"].lower() for k in escalation_keywords)
    ]
    selected.extend(pick(escalation_pool, n_escalate, "escalation"))

    # Fresh cases (most recent in the golden slice)
    fresh_pool = [c for c in candidates]  # All are fresh; just pick the last ones
    selected.extend(pick(fresh_pool[-200:], n_fresh, "fresh"))

    # Normal cases (fill remainder up to TARGET_SIZE, across all threads)
    remaining_n = TARGET_SIZE - len(selected)
    normal_pool = [c for c in candidates]
    selected.extend(pick(normal_pool, remaining_n, "normal"))

    # Add required fields for labeling
    for i, example in enumerate(selected):
        example["golden_id"] = f"G{i+1:03d}"
        example["primary_intent"] = None  # to be filled by human
        example["secondary_intent"] = None
        example["should_escalate"] = None
        example["reference_resolution_summary"] = None
        example["labeling_notes"] = None

    logger.info(f"Total sampled: {len(selected)}")

    # Save candidates to CSV for labeling
    golden_unlabeled_path = ARTIFACTS_DIR / f"golden_set_unlabeled_{brand_id}.csv"
    df = pd.DataFrame(selected)
    df.to_csv(golden_unlabeled_path, index=False, encoding="utf-8")
    logger.info(f"Unlabeled golden set saved to: {golden_unlabeled_path}")

    # Print case type distribution
    case_counts = {}
    for ex in selected:
        ct = ex.get("case_type", "unknown")
        case_counts[ct] = case_counts.get(ct, 0) + 1
    print("\nCase type distribution:")
    for ct, n in sorted(case_counts.items()):
        print(f"  {ct:25s}: {n:3d} ({n/len(selected):.0%})")

    return selected


# ---------------------------------------------------------------------------
# Labeling interface
# ---------------------------------------------------------------------------

LABELING_INSTRUCTIONS = """
================================================================================
GOLDEN SET LABELING INSTRUCTIONS
================================================================================

For each example, you will label:

1. primary_intent   : The main intent of the customer message.
                      Use the exact label from taxonomy.json.
                      If unclear, use 'other_unclear'.

2. secondary_intent : Optional. Use if the message clearly contains a second intent.
                      Leave blank if only one intent.

3. should_escalate  : TRUE if this should be handled by a human agent.
                      FALSE if the system could reasonably auto-handle it.
                      Consider: Is this high-risk? Is there a clear, public answer?

4. reference_resolution_summary : In 1-2 sentences, what should a correct response say?
                                   Write this from scratch — do NOT use any model output.

5. labeling_notes   : Optional. Any notes about why this case is ambiguous or tricky.

IMPORTANT:
- Label from the raw message text only. Do not use any model predictions.
- When uncertain, lean toward should_escalate=TRUE (safer default).
- If you change your mind on an earlier label, note it in labeling_notes.
================================================================================
"""


def run_labeling_interface(brand_id: str):
    """
    Simple terminal-based labeling interface.
    Reads unlabeled golden set, shows each example, saves progress.
    """
    unlabeled_path = ARTIFACTS_DIR / f"golden_set_unlabeled_{brand_id}.csv"
    labeled_path = ARTIFACTS_DIR / f"golden_set_labeled_{brand_id}.csv"

    if not unlabeled_path.exists():
        print(f"No unlabeled golden set found. Run --sample first.")
        return

    # Load existing progress or start fresh
    if labeled_path.exists():
        df = pd.read_csv(labeled_path, encoding="utf-8")
        print(f"Resuming from existing progress ({df['primary_intent'].notna().sum()} labeled so far)")
    else:
        df = pd.read_csv(unlabeled_path, encoding="utf-8")

    taxonomy = load_taxonomy()
    valid_intents = [i["label"] for i in taxonomy["intents"]]

    print(LABELING_INSTRUCTIONS)
    print(f"\nAvailable intents:")
    for intent in taxonomy["intents"]:
        print(f"  {intent['label']:30s} — {intent['definition']}")
    print()

    labeled_count = df["primary_intent"].notna().sum()
    total = len(df)

    for idx, row in df.iterrows():
        if pd.notna(df.at[idx, "primary_intent"]):
            continue  # already labeled

        labeled_count += 1
        print(f"\n{'='*60}")
        print(f"Example {df.at[idx, 'golden_id']} ({labeled_count}/{total})")
        print(f"Case type: {df.at[idx, 'case_type']}")
        print(f"\nMessage:\n{df.at[idx, 'first_message']}")
        if pd.notna(df.at[idx, 'thread_text']) and len(str(df.at[idx, 'thread_text'])) > len(str(df.at[idx, 'first_message'])):
            print(f"\nThread context:\n{str(df.at[idx, 'thread_text'])[:500]}")
        print()

        # Primary intent
        while True:
            intent = input(f"primary_intent [{'/'.join(valid_intents[:5])}...]: ").strip()
            if intent in valid_intents:
                df.at[idx, "primary_intent"] = intent
                break
            elif intent == "?":
                print("Valid intents:", valid_intents)
            elif intent == "q":
                df.to_csv(labeled_path, index=False)
                print(f"Progress saved. {labeled_count-1} examples labeled.")
                return
            else:
                print(f"Invalid intent. Type '?' to see all valid intents, 'q' to quit.")

        # Secondary intent (optional)
        secondary = input("secondary_intent (press Enter to skip): ").strip()
        if secondary and secondary in valid_intents:
            df.at[idx, "secondary_intent"] = secondary
        else:
            df.at[idx, "secondary_intent"] = None

        # Should escalate
        while True:
            escalate = input("should_escalate [y/n]: ").strip().lower()
            if escalate in ["y", "yes", "true", "1"]:
                df.at[idx, "should_escalate"] = True
                break
            elif escalate in ["n", "no", "false", "0"]:
                df.at[idx, "should_escalate"] = False
                break
            else:
                print("Please enter y or n")

        # Reference resolution
        ref = input("reference_resolution_summary (what should a correct response say?): ").strip()
        df.at[idx, "reference_resolution_summary"] = ref if ref else None

        # Notes
        notes = input("labeling_notes (press Enter to skip): ").strip()
        df.at[idx, "labeling_notes"] = notes if notes else None

        # Save after each example
        df.to_csv(labeled_path, index=False, encoding="utf-8")

    print(f"\nAll {total} examples labeled!")
    print(f"Saved to: {labeled_path}")
    print(f"Run --check to verify and freeze the golden set.")


def check_and_freeze(brand_id: str):
    """Verify labeling completeness and freeze the golden set."""
    labeled_path = ARTIFACTS_DIR / f"golden_set_labeled_{brand_id}.csv"
    if not labeled_path.exists():
        print("No labeled file found. Run --label first.")
        return

    df = pd.read_csv(labeled_path, encoding="utf-8")
    total = len(df)
    labeled = df["primary_intent"].notna().sum()
    escalate_labeled = df["should_escalate"].notna().sum()
    ref_labeled = df["reference_resolution_summary"].notna().sum()

    print(f"\nLabeling status:")
    print(f"  Total examples:             {total}")
    print(f"  primary_intent labeled:     {labeled} ({labeled/total:.0%})")
    print(f"  should_escalate labeled:    {escalate_labeled} ({escalate_labeled/total:.0%})")
    print(f"  reference_resolution given: {ref_labeled} ({ref_labeled/total:.0%})")

    if labeled < total:
        print(f"\nWARNING: {total - labeled} examples still need labeling.")
        print("Run --label to complete.")
        return

    # Save the frozen golden set (read-only from this point)
    frozen_path = ARTIFACTS_DIR / "golden_set.csv"
    df.to_csv(frozen_path, index=False, encoding="utf-8")
    print(f"\n[OK] Golden set FROZEN at: {frozen_path}")
    print(f"  DO NOT modify this file after this point.")
    print(f"  {total} examples with {df['should_escalate'].sum()} escalation cases")

    # Sample judge-validation subset (40-60 examples, oversampled from hard cases)
    hard_mask = df["case_type"].isin(["hard", "ambiguous", "low_evidence", "conflicting_evidence"])
    hard_examples = df[hard_mask]
    easy_examples = df[~hard_mask]

    n_hard_judge = min(25, len(hard_examples))
    n_easy_judge = min(20, len(easy_examples))

    judge_hard = hard_examples.sample(n_hard_judge, random_state=42)
    judge_easy = easy_examples.sample(n_easy_judge, random_state=42)
    judge_val = pd.concat([judge_hard, judge_easy]).reset_index(drop=True)

    judge_path = ARTIFACTS_DIR / "judge_validation.csv"
    judge_val.to_csv(judge_path, index=False, encoding="utf-8")
    print(f"\nJudge-validation subset: {len(judge_val)} examples (oversampled from hard cases)")
    print(f"Saved to: {judge_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Golden set construction")
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--sample", action="store_true", help="Sample candidates")
    parser.add_argument("--label", action="store_true", help="Run labeling interface")
    parser.add_argument("--check", action="store_true", help="Check and freeze golden set")
    args = parser.parse_args()

    if args.sample:
        sample_candidates(args.brand)
    elif args.label:
        run_labeling_interface(args.brand)
    elif args.check:
        check_and_freeze(args.brand)
    else:
        print("Specify --sample, --label, or --check")


if __name__ == "__main__":
    main()
