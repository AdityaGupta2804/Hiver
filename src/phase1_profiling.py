"""
phase1_profiling.py — Full-corpus brand profiling and scoring.

Blueprint Section 4 + Roadmap Phase 1.

Runs language ID, exact+near-dedup detection, thread reconstruction,
per-brand stats, DM-deflection rate, and computes the weighted brand scorecard.

Outputs (saved to data/artifacts/):
    brand_scores.json       — scored brand table
    profiling_summary.json  — raw stats per brand

Usage:
    python src/phase1_profiling.py [--csv path/to/twcs.csv] [--sample N]

The --sample flag limits to the first N rows for quick testing.
Without it, runs the full corpus (may take several minutes for millions of rows).
"""

import os
import sys
import json
import logging
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dedup_utils import (
    normalize_for_dedup,
    exact_dedup_texts,
    is_dm_deflection,
    classify_company_reply,
)
from src.utils.lang_utils import detect_language

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

# ---------------------------------------------------------------------------
# Brand scoring weights (pre-committed per blueprint Section 4.2)
# These must NOT be changed after looking at brand scores.
# ---------------------------------------------------------------------------
SCORING_WEIGHTS = {
    "conversation_volume_score": 0.10,
    "visible_resolution_rate": 0.25,
    "issue_diversity_score": 0.15,
    "self_containedness_score": 0.15,
    "escalation_worthy_presence": 0.10,
    "noise_level_score": 0.10,
    "english_purity_score": 0.10,
    "repetition_score": 0.05,
}

# Minimum visible resolution rate floor: disqualify brands below this
RESOLUTION_RATE_FLOOR = 0.20

# ---------------------------------------------------------------------------
# Known company author_ids (non-numeric = company account)
# (We use the heuristic: brand accounts have alphabetic/non-numeric author_id)
# ---------------------------------------------------------------------------

def is_company_author(author_id: str) -> bool:
    """Company accounts have alphabetic author_ids (like 'sprintcare', 'AmazonHelp')."""
    try:
        float(author_id)
        return False  # purely numeric = customer
    except (ValueError, TypeError):
        return True  # contains letters = company


# ---------------------------------------------------------------------------
# Step 1: Load and validate schema
# ---------------------------------------------------------------------------

def load_and_validate(csv_path: Path, sample_n: Optional[int] = None) -> pd.DataFrame:
    """Load the CSV and validate against the expected schema."""
    logger.info(f"Loading CSV from: {csv_path}")

    expected_columns = {
        "tweet_id", "author_id", "inbound", "created_at",
        "text", "response_tweet_id", "in_response_to_tweet_id"
    }

    if sample_n:
        df = pd.read_csv(csv_path, nrows=sample_n, low_memory=False)
        logger.info(f"Loaded {len(df):,} rows (sample mode)")
    else:
        # Read in chunks to handle large file gracefully
        chunks = []
        for chunk in pd.read_csv(csv_path, chunksize=100_000, low_memory=False):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df):,} rows (full corpus)")

    # Schema validation
    missing_cols = expected_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    logger.info(f"Schema valid. Columns: {list(df.columns)}")
    logger.info(f"Shape: {df.shape}")

    return df


# ---------------------------------------------------------------------------
# Step 2: Per-brand statistics computation
# ---------------------------------------------------------------------------

def compute_brand_stats(df: pd.DataFrame) -> Dict[str, dict]:
    """
    Compute raw statistics for each brand. These feed directly into the
    scoring rubric.
    """
    logger.info("Computing per-brand statistics...")

    # Identify company rows
    df = df.copy()
    df["is_company"] = df["author_id"].astype(str).apply(is_company_author)

    # Parse inbound flag robustly
    df["inbound_bool"] = df["inbound"].astype(str).str.lower().map(
        {"true": True, "1": True, "false": False, "0": False}
    ).fillna(df["inbound"].astype(bool))

    # Parse timestamps
    df["created_at_parsed"] = pd.to_datetime(
        df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
        utc=True,
    )

    # Find company accounts (used as "brand" identifier)
    company_rows = df[df["is_company"]].copy()
    brands = company_rows["author_id"].value_counts()
    logger.info(f"Found {len(brands)} distinct company author_ids")
    logger.info(f"Top 20 brands:\n{brands.head(20).to_string()}")

    brand_stats = {}

    for brand, _ in brands.items():
        # All rows mentioning or authored by this brand
        brand_company_rows = company_rows[company_rows["author_id"] == brand]

        # Find customer rows that this brand replied to
        brand_reply_tweet_ids = set(
            brand_company_rows["in_response_to_tweet_id"]
            .dropna()
            .astype(int)
            .tolist()
        )

        # Customer tweets directed at this brand (those the brand replied to)
        customer_rows_for_brand = df[
            df["tweet_id"].astype(str).apply(
                lambda x: int(float(x)) if x.replace(".", "").isdigit() else -1
            ).isin(brand_reply_tweet_ids)
        ]

        total_company_replies = len(brand_company_rows)
        total_customer_tweets = len(customer_rows_for_brand)

        if total_company_replies == 0:
            continue

        # --- DM deflection rate ---
        reply_texts = brand_company_rows["text"].fillna("").astype(str).tolist()
        reply_classifications = [classify_company_reply(t) for t in reply_texts]
        dm_count = reply_classifications.count("dm_deflection")
        content_free_count = reply_classifications.count("content_free")
        resolution_count = reply_classifications.count("resolution")
        visible_resolution_rate = resolution_count / total_company_replies

        # --- Exact dedup: unique reply texts (measures template repetition) ---
        normalized_replies = [normalize_for_dedup(t) for t in reply_texts]
        unique_normalized = set(normalized_replies)
        exact_dup_rate = 1.0 - (len(unique_normalized) / max(len(reply_texts), 1))

        # --- Language quality: sample customer tweets ---
        sample_customer_texts = (
            customer_rows_for_brand["text"]
            .dropna()
            .astype(str)
            .sample(min(200, len(customer_rows_for_brand)), random_state=42)
            .tolist()
            if len(customer_rows_for_brand) > 0
            else []
        )
        if sample_customer_texts:
            lang_results = [detect_language(t) for t in sample_customer_texts]
            english_count = sum(
                1 for lang, prob in lang_results
                if lang == "en" and prob >= 0.85
            )
            english_purity = english_count / len(sample_customer_texts)
        else:
            english_purity = 0.0

        # --- Average reply length (quality signal for self-containedness) ---
        avg_reply_length = (
            brand_company_rows["text"]
            .dropna()
            .astype(str)
            .str.len()
            .mean()
        )

        brand_stats[brand] = {
            "total_company_replies": total_company_replies,
            "total_customer_tweets": total_customer_tweets,
            "dm_deflection_count": dm_count,
            "dm_deflection_rate": dm_count / total_company_replies,
            "content_free_count": content_free_count,
            "content_free_rate": content_free_count / total_company_replies,
            "resolution_count": resolution_count,
            "visible_resolution_rate": visible_resolution_rate,
            "exact_dup_rate": exact_dup_rate,
            "english_purity": english_purity,
            "avg_reply_length_chars": avg_reply_length,
        }

        logger.info(
            f"  {brand:25s} | replies={total_company_replies:6d} | "
            f"resolution_rate={visible_resolution_rate:.1%} | "
            f"dm_rate={dm_count/total_company_replies:.1%} | "
            f"en={english_purity:.1%}"
        )

    return brand_stats


# ---------------------------------------------------------------------------
# Step 3: Score each brand using the pre-committed weighted rubric
# ---------------------------------------------------------------------------

def score_brands(brand_stats: Dict[str, dict]) -> pd.DataFrame:
    """
    Convert raw stats to 0-1 factor scores and apply the pre-committed weights.
    Returns a DataFrame sorted by total score descending.
    """
    rows = []

    # Extract values for normalization
    volumes = [s["total_company_replies"] for s in brand_stats.values()]
    res_rates = [s["visible_resolution_rate"] for s in brand_stats.values()]
    en_purity = [s["english_purity"] for s in brand_stats.values()]
    avg_lens = [s["avg_reply_length_chars"] for s in brand_stats.values()]
    dup_rates = [s["exact_dup_rate"] for s in brand_stats.values()]

    max_vol = max(volumes) if volumes else 1
    max_len = max(avg_lens) if avg_lens else 1

    for brand, stats in brand_stats.items():
        # --- Factor 1: Conversation volume (normalized 0-1) ---
        volume_score = stats["total_company_replies"] / max_vol

        # --- Factor 2: Visible resolution rate (0-1, directly) ---
        resolution_rate = stats["visible_resolution_rate"]

        # --- Factor 3: Issue diversity ---
        # Proxy: low exact-dup rate means more diverse reply content
        # (The real clustering-based diversity runs later per-brand)
        # For now: 1 - dup_rate, normalized
        diversity_score = 1.0 - stats["exact_dup_rate"]

        # --- Factor 4: Self-containedness ---
        # Proxy: average reply length (longer = more likely actionable)
        # Normalize to 0-1 across brands
        self_contain_score = min(stats["avg_reply_length_chars"] / max_len, 1.0)

        # --- Factor 5: Escalation-worthy presence ---
        # We can't easily compute this without embedding the full corpus here.
        # Use placeholder: 0.5 for all. Will be refined in Phase 3 per-brand.
        escalation_score = 0.5  # placeholder

        # --- Factor 6: Noise level (1 = low noise = good) ---
        # Proxy: 1 - dm_rate (lower DM rate = less noise in the corpus)
        noise_score = 1.0 - stats["dm_deflection_rate"]

        # --- Factor 7: English purity ---
        english_score = stats["english_purity"]

        # --- Factor 8: Repetition of issues (good repetition) ---
        # Counterintuitively, moderate dup rate = more repeating issue types
        # A very low dup rate = everything is unique (bad for retrieval)
        # A very high dup rate = one canned reply (also bad)
        # Optimal: moderate repetition. Score highest around dup_rate 0.3-0.6.
        dr = stats["exact_dup_rate"]
        repetition_score = 1.0 - abs(dr - 0.4) / 0.6  # peaks at 0.4

        # Apply hard disqualification floor
        disqualified = resolution_rate < RESOLUTION_RATE_FLOOR

        # Weighted total score
        total = (
            SCORING_WEIGHTS["conversation_volume_score"] * volume_score
            + SCORING_WEIGHTS["visible_resolution_rate"] * resolution_rate
            + SCORING_WEIGHTS["issue_diversity_score"] * diversity_score
            + SCORING_WEIGHTS["self_containedness_score"] * self_contain_score
            + SCORING_WEIGHTS["escalation_worthy_presence"] * escalation_score
            + SCORING_WEIGHTS["noise_level_score"] * noise_score
            + SCORING_WEIGHTS["english_purity_score"] * english_score
            + SCORING_WEIGHTS["repetition_score"] * repetition_score
        )

        if disqualified:
            total = 0.0  # zero score for disqualified brands

        rows.append({
            "brand": brand,
            "total_company_replies": stats["total_company_replies"],
            "visible_resolution_rate": round(resolution_rate, 3),
            "dm_deflection_rate": round(stats["dm_deflection_rate"], 3),
            "english_purity": round(stats["english_purity"], 3),
            "avg_reply_length_chars": round(stats["avg_reply_length_chars"], 1),
            "exact_dup_rate": round(stats["exact_dup_rate"], 3),
            "volume_score": round(volume_score, 3),
            "resolution_score": round(resolution_rate, 3),
            "diversity_score": round(diversity_score, 3),
            "self_contain_score": round(self_contain_score, 3),
            "noise_score": round(noise_score, 3),
            "english_score": round(english_score, 3),
            "repetition_score": round(repetition_score, 3),
            "weighted_total": round(total, 3),
            "disqualified": disqualified,
        })

    df_scores = pd.DataFrame(rows).sort_values("weighted_total", ascending=False)
    return df_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Brand profiling and scoring")
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Path to the twcs.csv file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, only use the first N rows (for quick testing)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 1: Dataset Profiling and Brand Scoring")
    logger.info("=" * 60)

    # Load and validate
    df = load_and_validate(args.csv, sample_n=args.sample)

    # Basic stats
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Inbound (customer) tweets: {df['inbound'].astype(str).str.lower().isin(['true','1']).sum():,}")
    logger.info(f"Outbound (company) tweets: {df['inbound'].astype(str).str.lower().isin(['false','0']).sum():,}")

    # Per-brand stats
    brand_stats = compute_brand_stats(df)

    # Score brands
    df_scores = score_brands(brand_stats)

    # Print the scorecard
    print("\n" + "=" * 80)
    print("BRAND SCORECARD (sorted by weighted total, disqualified brands at bottom)")
    print("=" * 80)
    display_cols = [
        "brand", "total_company_replies", "visible_resolution_rate",
        "dm_deflection_rate", "english_purity", "weighted_total", "disqualified"
    ]
    print(df_scores[display_cols].to_string(index=False))
    print()

    # Top 5 non-disqualified candidates
    qualified = df_scores[~df_scores["disqualified"]]
    print("TOP 5 QUALIFIED CANDIDATES:")
    print(qualified[display_cols].head(5).to_string(index=False))
    print()

    # Save outputs
    scores_path = ARTIFACTS_DIR / "brand_scores.json"
    df_scores.to_json(scores_path, orient="records", indent=2)
    logger.info(f"Brand scores saved to: {scores_path}")

    stats_path = ARTIFACTS_DIR / "brand_raw_stats.json"
    with open(stats_path, "w") as f:
        json.dump(brand_stats, f, indent=2, default=str)
    logger.info(f"Raw stats saved to: {stats_path}")

    # Also save as CSV for easy reading
    df_scores.to_csv(ARTIFACTS_DIR / "brand_scores.csv", index=False)

    logger.info("\nPhase 1 complete. Review brand_scores.csv/json and then run Phase 2 to select the brand.")

    return df_scores, brand_stats


if __name__ == "__main__":
    main()
