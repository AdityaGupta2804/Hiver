"""
phase4_split.py — Temporal, conversation-level data splitting.

Blueprint Section 13.2, Roadmap Phase 4.

Splits the brand's thread corpus into disjoint temporal slices:
  1. Retrieval corpus (oldest) — for building the resolution index
  2. Taxonomy/dev slice (middle) — for taxonomy discovery and threshold fitting
  3. Golden-set slice (most recent) — for golden set sampling and evaluation

Key rules:
- Split at the CONVERSATION level (thread root timestamp), not tweet level.
- Zero ID overlap between slices — enforced by an assertion.
- Exact thread ID membership saved as a versioned artifact (not just a % cutoff).

Default split ratios (configurable):
    retrieval: 50%, dev: 25%, golden: 25%

Usage:
    python src/phase4_split.py --brand SpotifyCares
"""

import os
import sys
import json
import logging
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"


def load_threads(brand_id: str) -> List[dict]:
    path = ARTIFACTS_DIR / f"clean_threads_{brand_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No threads found at {path}. Run phase3_cleaning.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def sort_threads_by_time(threads: List[dict]) -> List[dict]:
    """Sort threads by their earliest timestamp (the conversation's start time)."""
    def get_timestamp(thread):
        ts = thread.get("earliest_timestamp")
        if ts is None:
            return pd.Timestamp.min.tz_localize("UTC")
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts

    return sorted(threads, key=get_timestamp)


def temporal_split(
    threads: List[dict],
    retrieval_frac: float = 0.50,
    dev_frac: float = 0.25,
    golden_frac: float = 0.25,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split sorted threads into retrieval / dev / golden slices.
    Fractions must sum to ~1.0.
    """
    assert abs(retrieval_frac + dev_frac + golden_frac - 1.0) < 1e-6, \
        "Split fractions must sum to 1.0"

    n = len(threads)
    n_retrieval = int(n * retrieval_frac)
    n_dev = int(n * dev_frac)
    n_golden = n - n_retrieval - n_dev

    retrieval = threads[:n_retrieval]
    dev = threads[n_retrieval : n_retrieval + n_dev]
    golden = threads[n_retrieval + n_dev :]

    logger.info(
        f"Split: retrieval={len(retrieval):,}, dev={len(dev):,}, golden={len(golden):,} "
        f"(total={n:,})"
    )
    return retrieval, dev, golden


def check_zero_overlap(
    retrieval: List[dict],
    dev: List[dict],
    golden: List[dict],
):
    """
    Assert that no thread root_id appears in more than one split.
    This is the leakage-prevention assertion from Section 17.
    """
    ids_r = {t["root_id"] for t in retrieval}
    ids_d = {t["root_id"] for t in dev}
    ids_g = {t["root_id"] for t in golden}

    overlap_rd = ids_r & ids_d
    overlap_rg = ids_r & ids_g
    overlap_dg = ids_d & ids_g

    if overlap_rd or overlap_rg or overlap_dg:
        raise AssertionError(
            f"LEAKAGE DETECTED: overlapping thread IDs between splits! "
            f"retrieval∩dev={len(overlap_rd)}, "
            f"retrieval∩golden={len(overlap_rg)}, "
            f"dev∩golden={len(overlap_dg)}"
        )
    logger.info("✓ Zero-overlap assertion passed: no thread appears in more than one split")


def get_time_range(threads: List[dict]) -> Tuple:
    """Return (min_timestamp, max_timestamp) for a list of threads."""
    timestamps = [
        t.get("earliest_timestamp") for t in threads
        if t.get("earliest_timestamp") is not None
    ]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Temporal conversation split")
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--retrieval-frac", type=float, default=0.50)
    parser.add_argument("--dev-frac", type=float, default=0.25)
    parser.add_argument("--golden-frac", type=float, default=0.25)
    args = parser.parse_args()

    brand_id = args.brand
    logger.info(f"Phase 4: Temporal split for brand '{brand_id}'")

    # Load threads from Phase 3
    threads = load_threads(brand_id)
    logger.info(f"Loaded {len(threads):,} threads")

    # Sort by time
    threads_sorted = sort_threads_by_time(threads)
    logger.info("Threads sorted by earliest timestamp")

    # Split
    retrieval, dev, golden = temporal_split(
        threads_sorted,
        retrieval_frac=args.retrieval_frac,
        dev_frac=args.dev_frac,
        golden_frac=args.golden_frac,
    )

    # Zero-overlap assertion
    check_zero_overlap(retrieval, dev, golden)

    # Time range reporting
    r_start, r_end = get_time_range(retrieval)
    d_start, d_end = get_time_range(dev)
    g_start, g_end = get_time_range(golden)

    logger.info(f"Retrieval corpus:  {r_start} → {r_end}")
    logger.info(f"Dev slice:         {d_start} → {d_end}")
    logger.info(f"Golden-set slice:  {g_start} → {g_end}")

    # Save the split ID membership (the authoritative audit trail)
    split_ids = {
        "brand_id": brand_id,
        "retrieval_thread_ids": [t["root_id"] for t in retrieval],
        "dev_thread_ids": [t["root_id"] for t in dev],
        "golden_thread_ids": [t["root_id"] for t in golden],
        "retrieval_time_range": [str(r_start), str(r_end)],
        "dev_time_range": [str(d_start), str(d_end)],
        "golden_time_range": [str(g_start), str(g_end)],
        "split_fractions": {
            "retrieval": args.retrieval_frac,
            "dev": args.dev_frac,
            "golden": args.golden_frac,
        },
        "counts": {
            "retrieval": len(retrieval),
            "dev": len(dev),
            "golden": len(golden),
            "total": len(threads_sorted),
        },
    }

    split_path = ARTIFACTS_DIR / f"split_ids_{brand_id}.json"
    with open(split_path, "w") as f:
        json.dump(split_ids, f, indent=2, default=str)
    logger.info(f"Split IDs saved to: {split_path}")

    # Also save the actual thread lists as pickles for downstream use
    for name, split_threads in [("retrieval", retrieval), ("dev", dev), ("golden_slice", golden)]:
        path = ARTIFACTS_DIR / f"threads_{name}_{brand_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(split_threads, f)
        logger.info(f"  {name}: {len(split_threads):,} threads -> {path}")

    print(f"\nPhase 4 complete for '{brand_id}':")
    print(f"  Retrieval corpus:  {len(retrieval):,} threads ({r_start} -> {r_end})")
    print(f"  Dev slice:         {len(dev):,} threads ({d_start} -> {d_end})")
    print(f"  Golden-set slice:  {len(golden):,} threads ({g_start} -> {g_end})")
    print(f"\nSplit IDs frozen at: {split_path}")
    return split_ids


if __name__ == "__main__":
    main()
