"""
retrieval_utils.py — Utilities for hybrid retrieval (BM25 + Dense FAISS + Reciprocal Rank Fusion).

Blueprint Section 10.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize and normalize text for BM25 indexing/querying."""
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    tokens = re.findall(r"\b[a-z0-9_]{2,}\b", text)
    return tokens


def reciprocal_rank_fusion(
    bm25_ranked_ids: List[int],
    dense_ranked_ids: List[int],
    k: int = 60,
    top_n: int = 10,
) -> List[Tuple[int, float]]:
    """
    Combines BM25 and Dense ranking lists using Reciprocal Rank Fusion (RRF).
    Score(d) = sum(1 / (k + rank_i(d)))
    """
    scores: Dict[int, float] = {}

    for rank, doc_id in enumerate(bm25_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc_id in enumerate(dense_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


def compute_retrieval_metrics(
    retrieved_ids: List[int],
    ground_truth_ids: List[int],
    k_list: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    Computes Recall@K, Precision@K, and MRR.
    """
    metrics = {}
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return {f"recall@{k}": 0.0 for k in k_list}

    for k in k_list:
        top_k = retrieved_ids[:k]
        hits = len(set(top_k) & gt_set)
        metrics[f"recall@{k}"] = hits / min(len(gt_set), k)
        metrics[f"precision@{k}"] = hits / k if k > 0 else 0.0

    # Mean Reciprocal Rank (MRR)
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids):
        if doc_id in gt_set:
            mrr = 1.0 / (rank + 1)
            break
    metrics["mrr"] = mrr

    return metrics
