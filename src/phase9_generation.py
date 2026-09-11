"""
phase9_generation.py — Response Generation & Experiment E9 (Retrieval ON vs. OFF).

Blueprint Section 11, Roadmap Phase 9.

Implements:
1. Retrieval-grounded response generation using structured prompt & schema.
2. Automated citation verification and groundedness checking.
3. Experiment E9: Compare responses with retrieval evidence vs. without retrieval evidence.
4. Logs generation scorecard and groundedness delta.

Usage:
    python src/phase9_generation.py --brand SpotifyCares
"""

import os
import sys
import json
import logging
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.generation_utils import generate_structured_response, verify_groundedness_and_citations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def load_retrieval_corpus(brand_id: str) -> List[Dict]:
    path = ARTIFACTS_DIR / f"retrieval_corpus_{brand_id}.pkl"
    if not path.exists():
        path = ARTIFACTS_DIR / f"resolution_pairs_{brand_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Retrieval corpus not found for brand {brand_id}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_golden_set() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "golden_set.csv"
    if not path.exists():
        raise FileNotFoundError("golden_set.csv not found. Complete Phase 6 first.")
    return pd.read_csv(path)


def run_experiment_e9(brand_id: str, sample_size: int = 50) -> Dict[str, Any]:
    logger.info(f"Running Experiment E9 (Retrieval Ablation) on {brand_id}...")
    corpus = load_retrieval_corpus(brand_id)
    golden_df = load_golden_set()

    eval_sample = golden_df.head(sample_size).to_dict(orient="records")

    retrieval_on_results = []
    retrieval_off_results = []

    for item in eval_sample:
        customer_text = item.get("customer_text", item.get("first_message", item.get("text", "")))
        intent = item.get("primary_intent", "other")

        # Top 3 corpus matches for the intent
        matching_pairs = [c for c in corpus if c.get("intent", "").lower() == intent.lower()]
        if not matching_pairs:
            matching_pairs = corpus[:3]
        top_k_evidence = matching_pairs[:3]

        # Condition A: Retrieval ON
        resp_on = generate_structured_response(customer_text, intent, top_k_evidence)
        retrieval_on_results.append({
            "tweet_id": item.get("tweet_id"),
            "intent": intent,
            "response": resp_on,
            "groundedness": resp_on.get("groundedness_score", 0.0),
            "citations_valid": resp_on.get("groundedness_audit", {}).get("citation_consistent", False),
        })

        # Condition B: Retrieval OFF (empty evidence)
        resp_off = generate_structured_response(customer_text, intent, [])
        retrieval_off_results.append({
            "tweet_id": item.get("tweet_id"),
            "intent": intent,
            "response": resp_off,
            "groundedness": resp_off.get("groundedness_score", 0.0),
            "citations_valid": False,
        })

    avg_groundedness_on = sum(r["groundedness"] for r in retrieval_on_results) / len(retrieval_on_results)
    avg_groundedness_off = sum(r["groundedness"] for r in retrieval_off_results) / len(retrieval_off_results)
    citation_rate_on = sum(1 for r in retrieval_on_results if r["citations_valid"]) / len(retrieval_on_results)

    e9_summary = {
        "experiment": "E9_retrieval_groundedness_ablation",
        "brand_id": brand_id,
        "sample_size": len(eval_sample),
        "retrieval_on": {
            "avg_groundedness_score": round(avg_groundedness_on, 3),
            "valid_citation_rate": round(citation_rate_on, 3),
        },
        "retrieval_off": {
            "avg_groundedness_score": round(avg_groundedness_off, 3),
            "valid_citation_rate": 0.0,
        },
        "groundedness_lift": round(avg_groundedness_on - avg_groundedness_off, 3),
    }

    out_path = EXPERIMENTS_DIR / "e9_generation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(e9_summary, f, indent=2)

    logger.info(f"E9 results saved to: {out_path}")
    print("\n========================================================")
    print("EXPERIMENT E9: RETRIEVAL ABLATION RESULTS")
    print(f"  Retrieval ON Groundedness:  {avg_groundedness_on:.3f}")
    print(f"  Retrieval OFF Groundedness: {avg_groundedness_off:.3f}")
    print(f"  Groundedness Lift:          +{e9_summary['groundedness_lift']:.3f}")
    print(f"  Valid Citation Rate:        {citation_rate_on:.1%}")
    print("========================================================\n")

    return e9_summary


def main():
    parser = argparse.ArgumentParser(description="Phase 9: Response Generation & E9")
    parser.add_argument("--brand", type=str, default="SpotifyCares")
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()

    run_experiment_e9(args.brand, args.sample_size)


if __name__ == "__main__":
    main()
