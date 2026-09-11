"""
phase3_cleaning.py — Single-brand cleaning and thread reconstruction.

Blueprint Roadmap Phase 3.

After brand selection, this script:
1. Filters the full CSV to the selected brand's conversations only.
2. Runs language filtering on customer tweets.
3. Collapses exact + near-duplicates.
4. Reconstructs all conversations as branch-aware trees.
5. Extracts visible-resolution pairs (filtering out DM deflections).
6. Saves the cleaned corpus to data/artifacts/.

Usage:
    python src/phase3_cleaning.py --brand SpotifyCares
    python src/phase3_cleaning.py --brand SpotifyCares --skip-near-dedup
"""

import os
import sys
import json
import logging
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.thread_utils import reconstruct_all_threads
from src.utils.dedup_utils import (
    normalize_for_dedup,
    exact_dedup_texts,
    near_dedup_minhash,
    classify_company_reply,
    is_dm_deflection,
)
from src.utils.lang_utils import filter_english_texts
from src.utils.pii_utils import scan_corpus_for_pii, redact_pii

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def find_dataset_csv() -> Path:
    env_path = os.environ.get("TWCS_CSV_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    candidates = [
        PROJECT_ROOT / "datasets" / "twcs" / "twcs.csv",
        PROJECT_ROOT / "datasets" / "twcs.csv",
        PROJECT_ROOT / "data" / "raw" / "twcs.csv",
        PROJECT_ROOT / "kagglehub" / "datasets" / "thoughtvector" /
        "customer-support-on-twitter" / "versions" / "10" / "twcs" / "twcs.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

CSV_PATH = find_dataset_csv()

ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def is_company_author(author_id: str) -> bool:
    try:
        float(author_id)
        return False
    except (ValueError, TypeError):
        return True


def load_brand_conversations(df: pd.DataFrame, brand_id: str) -> pd.DataFrame:
    """
    Extract all tweets that belong to conversations involving the selected brand.
    A conversation involves the brand if any turn was authored by brand_id OR
    is a direct customer reply to a brand tweet.
    """
    # Find all tweet IDs that the brand replied to (and the replies themselves)
    company_rows = df[df["author_id"].astype(str) == brand_id]
    brand_replied_to = set(
        company_rows["in_response_to_tweet_id"]
        .dropna()
        .astype(int)
        .tolist()
    )
    brand_tweet_ids = set(company_rows["tweet_id"].astype(int).tolist())

    # Get customer tweets that started conversations with this brand
    # (those that the brand directly replied to, plus any roots)
    relevant_tweet_ids = brand_tweet_ids | brand_replied_to

    # Expand: also include tweets that are in the same thread as any brand tweet.
    # Walk backwards through in_response_to_tweet_id to find thread roots.
    # This is a simplified version — full thread reconstruction handles the rest.
    valid_parents = df[df["in_response_to_tweet_id"].notna()][["tweet_id", "in_response_to_tweet_id"]]
    id_to_parent = dict(zip(valid_parents["tweet_id"].astype(int), valid_parents["in_response_to_tweet_id"].astype(int)))

    # For each brand-adjacent tweet, walk up to the root
    to_include = set(relevant_tweet_ids)
    for tweet_id in list(relevant_tweet_ids):
        current = tweet_id
        depth = 0
        while current in id_to_parent and depth < 50:
            parent = id_to_parent[current]
            to_include.add(parent)
            current = parent
            depth += 1

    brand_df = df[df["tweet_id"].astype(int).isin(to_include)].copy()
    logger.info(
        f"Brand '{brand_id}': extracted {len(brand_df):,} tweets "
        f"from {len(df):,} total"
    )
    return brand_df


def extract_resolution_pairs(threads: List[dict], brand_id: str) -> List[dict]:
    """
    From reconstructed threads, extract visible-resolution pairs:
    (customer_problem_summary, brand_resolution_text, source_thread_id)

    A valid resolution pair requires:
    - At least one customer turn and one non-DM-deflection brand turn
    - The brand turn must be classified as 'resolution' (not 'dm_deflection' or 'content_free')

    Returns a list of resolution pair dicts.
    """
    pairs = []

    for thread in threads:
        turns = thread.get("turns", [])
        if not turns:
            continue

        # Collect customer and company turns in order
        customer_turns = [t for t in turns if t.get("inbound", True)]
        company_turns = [t for t in turns if not t.get("inbound", True)]

        if not customer_turns or not company_turns:
            continue

        # Check if any company turn is a real resolution
        resolution_turns = [
            t for t in company_turns
            if classify_company_reply(t.get("text", "")) == "resolution"
        ]

        if not resolution_turns:
            continue

        # Build the problem text from first 1-2 customer turns
        problem_text = " ".join(
            t.get("text", "") for t in customer_turns[:2]
        ).strip()

        # Use the first resolution turn as the resolution text
        resolution_turn = resolution_turns[0]
        resolution_text = resolution_turn.get("text", "").strip()

        # Re-scan for PII before storing
        problem_text = redact_pii(problem_text)
        resolution_text = redact_pii(resolution_text)

        pairs.append({
            "source_thread_id": thread["root_id"],
            "customer_problem_text": problem_text,
            "brand_resolution_text": resolution_text,
            "resolution_tweet_id": resolution_turn["tweet_id"],
            "thread_depth": thread.get("depth", 0),
            "has_branches": thread.get("has_branches", False),
            # intent_tag will be filled in Phase 5 (taxonomy discovery)
            "intent_tag": None,
            # resolution_type: 'self_contained' vs 'account_specific' — TBD in Phase 5
            "resolution_type": None,
        })

    logger.info(f"Extracted {len(pairs):,} resolution pairs from {len(threads):,} threads")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Single-brand cleaning")
    parser.add_argument(
        "--brand",
        type=str,
        required=True,
        help="Brand author_id (e.g. 'SpotifyCares')",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
    )
    parser.add_argument(
        "--skip-near-dedup",
        action="store_true",
        help="Skip near-dedup (use for quick testing)",
    )
    args = parser.parse_args()

    brand_id = args.brand
    logger.info(f"Phase 3: Cleaning for brand '{brand_id}'")

    # Load full CSV
    logger.info("Loading full CSV...")
    df = pd.read_csv(args.csv, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows")

    # Extract brand conversations
    brand_df = load_brand_conversations(df, brand_id)

    # Language filter: customer tweets only
    customer_mask = brand_df["inbound"].astype(str).str.lower().isin(["true", "1"])
    customer_df = brand_df[customer_mask].copy()
    customer_texts = customer_df["text"].fillna("").astype(str).tolist()

    en_indices, lang_stats = filter_english_texts(customer_texts)
    logger.info(f"Language filter stats: {lang_stats}")

    # Keep all company rows + English customer rows
    english_customer_ids = set(
        customer_df.iloc[en_indices]["tweet_id"].astype(int).tolist()
    )
    company_ids = set(
        brand_df[~customer_mask]["tweet_id"].astype(int).tolist()
    )
    keep_ids = english_customer_ids | company_ids
    filtered_df = brand_df[brand_df["tweet_id"].astype(int).isin(keep_ids)].copy()
    logger.info(f"After language filter: {len(filtered_df):,} rows")

    # Exact dedup on customer tweet texts
    cust_filtered = filtered_df[filtered_df["inbound"].astype(str).str.lower().isin(["true","1"])].copy()
    cust_texts = cust_filtered["text"].fillna("").astype(str).tolist()
    keep_exact, exact_clusters = exact_dedup_texts(cust_texts)
    n_exact_dups = len(cust_texts) - len(keep_exact)
    logger.info(f"Exact dedup: removed {n_exact_dups:,} duplicate customer tweets")

    # Near-dedup (optional)
    if not args.skip_near_dedup and len(keep_exact) > 1000:
        normalized = [normalize_for_dedup(cust_texts[i]) for i in keep_exact]
        canonical_idx, cluster_map = near_dedup_minhash(normalized, threshold=0.85)
        keep_after_near = [keep_exact[i] for i in canonical_idx]
        n_near_dups = len(keep_exact) - len(keep_after_near)
        logger.info(f"Near-dedup: removed {n_near_dups:,} additional near-duplicate customer tweets")
        final_keep_indices = keep_after_near
    else:
        final_keep_indices = keep_exact
        logger.info("Near-dedup skipped")

    # Reconstruct threads
    # We use the full filtered_df (company + English customer tweets) for reconstruction
    logger.info("Reconstructing threads...")
    threads = reconstruct_all_threads(filtered_df, brand_id)

    # Filter to threads with at least one company reply
    threads_with_reply = [t for t in threads if t.get("has_company_reply", False)]
    logger.info(
        f"Threads: {len(threads):,} total, "
        f"{len(threads_with_reply):,} with company reply"
    )

    # Extract resolution pairs
    resolution_pairs = extract_resolution_pairs(threads_with_reply, brand_id)

    # PII scan on resolution pairs
    all_texts = (
        [p["customer_problem_text"] for p in resolution_pairs] +
        [p["brand_resolution_text"] for p in resolution_pairs]
    )
    pii_stats = scan_corpus_for_pii(all_texts)
    logger.info(f"PII scan: {pii_stats['texts_with_pii']} texts with residual PII (re-redacted)")

    # Save artifacts
    threads_path = ARTIFACTS_DIR / f"clean_threads_{brand_id}.pkl"
    with open(threads_path, "wb") as f:
        pickle.dump(threads, f)
    logger.info(f"Threads saved to: {threads_path}")

    pairs_path = ARTIFACTS_DIR / f"resolution_pairs_{brand_id}.pkl"
    with open(pairs_path, "wb") as f:
        pickle.dump(resolution_pairs, f)
    logger.info(f"Resolution pairs saved to: {pairs_path}")

    # Save summary
    summary = {
        "brand_id": brand_id,
        "total_raw_rows": len(df),
        "brand_rows": len(brand_df),
        "after_language_filter": len(filtered_df),
        "threads_total": len(threads),
        "threads_with_company_reply": len(threads_with_reply),
        "resolution_pairs": len(resolution_pairs),
        "language_stats": lang_stats,
        "pii_scan": {k: v for k, v in pii_stats.items() if k != "flagged_indices"},
    }
    summary_path = ARTIFACTS_DIR / f"phase3_summary_{brand_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to: {summary_path}")

    print(f"\nPhase 3 complete for '{brand_id}':")
    print(f"  Threads reconstructed: {len(threads):,}")
    print(f"  With company reply:     {len(threads_with_reply):,}")
    print(f"  Resolution pairs:       {len(resolution_pairs):,}")
    print(f"\nArtifacts in: {ARTIFACTS_DIR}")

    return threads, resolution_pairs


if __name__ == "__main__":
    main()
