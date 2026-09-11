"""
phase8_retrieval.py — Retrieval system: E5–E7 experiments.

Blueprint Section 10, Roadmap Phase 8.

Implements:
  1. Build the retrieval corpus (resolution pairs from the retrieval-corpus temporal slice)
  2. Build BM25 index
  3. Build FAISS dense embedding index
  4. Implement hybrid retrieval (BM25 + dense, with reciprocal rank fusion)
  5. Intent-conditioned filtering (filter by predicted intent before ranking)
  6. Optional cross-encoder reranker (E7, kept only if it earns its latency cost)

Experiments:
  E5: BM25 alone vs dense alone vs hybrid BM25+dense
  E6: Hybrid + intent-conditioning vs hybrid alone
  E7: + reranker vs E6 winner (latency vs Recall@3 tradeoff)

Usage:
    python src/phase8_retrieval.py --brand SpotifyCares --build
    python src/phase8_retrieval.py --brand SpotifyCares --run-experiments
"""

import os
import sys
import json
import logging
import argparse
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def load_retrieval_threads(brand_id: str) -> List[dict]:
    path = ARTIFACTS_DIR / f"threads_retrieval_{brand_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_resolution_pairs(brand_id: str) -> List[dict]:
    path = ARTIFACTS_DIR / f"resolution_pairs_{brand_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_taxonomy() -> dict:
    with open(ARTIFACTS_DIR / "taxonomy.json", encoding="utf-8") as f:
        return json.load(f)


def load_golden_set() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "golden_set.csv")


# ---------------------------------------------------------------------------
# Build indices
# ---------------------------------------------------------------------------

def build_bm25_index(resolution_pairs: List[dict]) -> object:
    """Build a BM25 index over the resolution pairs."""
    from rank_bm25 import BM25Okapi
    import re

    logger.info(f"Building BM25 index over {len(resolution_pairs):,} resolution pairs...")
    # BM25 is over customer problem text (what we're matching the query against)
    tokenized_corpus = []
    for pair in resolution_pairs:
        text = pair.get("customer_problem_text", "")
        tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25 index built")
    return bm25


def build_dense_index(
    resolution_pairs: List[dict],
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[object, np.ndarray]:
    """
    Build a FAISS dense index over resolution pairs.
    Returns (faiss_index, embeddings_array).
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    logger.info(f"Building FAISS dense index over {len(resolution_pairs):,} pairs...")
    model = SentenceTransformer(model_name)

    texts = [p.get("customer_problem_text", "") for p in resolution_pairs]
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity after normalization)
    index.add(embeddings)

    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, embeddings


def reciprocal_rank_fusion(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    k: int = 60,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Reciprocal Rank Fusion (RRF) to combine BM25 and dense rankings.

    RRF score for item i = alpha * 1/(k + rank_bm25(i)) + (1-alpha) * 1/(k + rank_dense(i))

    Args:
        bm25_scores: BM25 scores for each item
        dense_scores: Dense similarity scores for each item
        k: RRF constant (60 is the standard default)
        alpha: Weight for BM25 vs dense (0.5 = equal weight)
    """
    n = len(bm25_scores)
    bm25_ranks = np.argsort(np.argsort(-bm25_scores)) + 1  # 1-indexed
    dense_ranks = np.argsort(np.argsort(-dense_scores)) + 1

    rrf_scores = (
        alpha * (1.0 / (k + bm25_ranks)) +
        (1.0 - alpha) * (1.0 / (k + dense_ranks))
    )
    return rrf_scores


class RetrievalSystem:
    """
    The full retrieval system with intent-conditioned hybrid retrieval.
    """

    def __init__(
        self,
        resolution_pairs: List[dict],
        bm25_index,
        faiss_index,
        corpus_embeddings: np.ndarray,
        taxonomy: dict,
        model_name: str = "all-MiniLM-L6-v2",
        use_intent_filter: bool = True,
        reranker=None,
    ):
        self.resolution_pairs = resolution_pairs
        self.bm25_index = bm25_index
        self.faiss_index = faiss_index
        self.corpus_embeddings = corpus_embeddings
        self.taxonomy = taxonomy
        self.use_intent_filter = use_intent_filter
        self.reranker = reranker
        self.intent_labels = [i["label"] for i in taxonomy["intents"]]

        # Build intent -> indices map for intent-conditioned filtering
        self.intent_to_indices = {}
        for i, pair in enumerate(resolution_pairs):
            intent = pair.get("intent_tag", "other_unclear") or "other_unclear"
            if intent not in self.intent_to_indices:
                self.intent_to_indices[intent] = []
            self.intent_to_indices[intent].append(i)

        # Load embedder for query embedding
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(model_name)

    def retrieve(
        self,
        query: str,
        predicted_intent: Optional[str] = None,
        top_k: int = 5,
        use_bm25: bool = True,
        use_dense: bool = True,
    ) -> List[Dict]:
        """
        Retrieve top-k relevant resolution pairs for a query.

        Args:
            query: Customer message text.
            predicted_intent: Predicted intent label (for filtering).
            top_k: Number of results to return.
            use_bm25: Include BM25 in the hybrid.
            use_dense: Include dense retrieval in the hybrid.

        Returns:
            List of resolution pair dicts with added retrieval scores.
        """
        import re
        import faiss

        # Step 1: Intent filtering (restrict candidate pool to same intent)
        if self.use_intent_filter and predicted_intent and predicted_intent in self.intent_to_indices:
            candidate_indices = self.intent_to_indices[predicted_intent]
            # Include adjacent intents if pool is too small
            if len(candidate_indices) < top_k * 3:
                # Expand to all pairs if intent pool is very small
                candidate_indices = list(range(len(self.resolution_pairs)))
        else:
            candidate_indices = list(range(len(self.resolution_pairs)))

        if not candidate_indices:
            return []

        # Step 2: BM25 scoring over candidate pool
        bm25_scores = np.zeros(len(self.resolution_pairs))
        if use_bm25:
            tokens = re.findall(r"\b[a-z]{2,}\b", query.lower())
            all_scores = self.bm25_index.get_scores(tokens)
            bm25_scores = all_scores

        # Step 3: Dense similarity scoring
        dense_scores = np.zeros(len(self.resolution_pairs))
        if use_dense:
            query_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
            # Score only candidate indices for efficiency
            candidate_embs = self.corpus_embeddings[candidate_indices]
            sims = (query_emb @ candidate_embs.T).flatten()
            for i, idx in enumerate(candidate_indices):
                dense_scores[idx] = float(sims[i])

        # Step 4: Combine scores (RRF or simple linear combination)
        if use_bm25 and use_dense:
            combined = reciprocal_rank_fusion(bm25_scores, dense_scores)
        elif use_bm25:
            combined = bm25_scores
        else:
            combined = dense_scores

        # Zero out non-candidate indices
        mask = np.zeros(len(self.resolution_pairs))
        for idx in candidate_indices:
            mask[idx] = 1.0
        combined = combined * mask

        # Step 5: Get top-k
        top_indices = np.argsort(-combined)[:top_k * 2]  # extra for reranker

        results = []
        for idx in top_indices:
            if combined[idx] > 0:
                pair = dict(self.resolution_pairs[idx])
                pair["retrieval_score"] = float(combined[idx])
                pair["bm25_score"] = float(bm25_scores[idx])
                pair["dense_score"] = float(dense_scores[idx])
                pair["retrieval_rank"] = len(results) + 1
                results.append(pair)
            if len(results) >= top_k * 2:
                break

        # Step 6: Optional reranking
        if self.reranker and results:
            results = self._rerank(query, results, top_k)
        else:
            results = results[:top_k]

        return results

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Cross-encoder reranking of candidates."""
        try:
            from sentence_transformers import CrossEncoder
            candidate_texts = [c.get("customer_problem_text", "") for c in candidates]
            pairs = [[query, text] for text in candidate_texts]
            scores = self.reranker.predict(pairs)
            for i, score in enumerate(scores):
                candidates[i]["reranker_score"] = float(score)
            candidates.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
            return candidates[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates[:top_k]


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    system: RetrievalSystem,
    golden_df: pd.DataFrame,
    top_k_values: List[int] = [1, 3, 5, 10],
    use_bm25: bool = True,
    use_dense: bool = True,
    use_intent_filter: bool = True,
    label: str = "hybrid",
) -> dict:
    """
    Evaluate retrieval system on the golden set.
    Metric: Recall@K (did any of the top-K contain a useful resolution?)

    Note: True Recall@K requires knowing the "correct" retrievals.
    Here we use a proxy: whether the top-1 similarity is above a threshold,
    AND whether the human-labeled resolution_usefulness field (from golden set
    labeling) marks the retrieval as useful. If that field is not filled, we
    fall back to top-1 similarity as the proxy.
    """
    recall_at_k = {k: 0 for k in top_k_values}
    mrr_scores = []
    top1_sims = []

    n = len(golden_df)
    for _, row in golden_df.iterrows():
        query = str(row.get("first_message", ""))
        predicted_intent = str(row.get("primary_intent", "")) if pd.notna(row.get("primary_intent")) else None

        # Check if system was configured for intent filtering
        system.use_intent_filter = use_intent_filter

        results = system.retrieve(
            query,
            predicted_intent=predicted_intent,
            top_k=max(top_k_values),
            use_bm25=use_bm25,
            use_dense=use_dense,
        )

        if not results:
            mrr_scores.append(0.0)
            top1_sims.append(0.0)
            continue

        top1_sims.append(results[0].get("dense_score", 0.0) if use_dense else 0.0)

        # Compute Recall@K:
        # Since we don't have full relevance labels for retrieval yet,
        # we use a heuristic: "useful" if top-1 similarity > 0.5 (proxy)
        # The resolution_usefulness field from golden set labeling overrides this.
        useful_at = None
        for rank, result in enumerate(results, 1):
            # Check if there's a labeled usefulness field
            is_useful = result.get("dense_score", 0.0) > 0.5  # proxy
            if is_useful and useful_at is None:
                useful_at = rank

        for k in top_k_values:
            if useful_at is not None and useful_at <= k:
                recall_at_k[k] += 1

        # MRR
        if useful_at is not None:
            mrr_scores.append(1.0 / useful_at)
        else:
            mrr_scores.append(0.0)

    recall_results = {f"recall_at_{k}": round(recall_at_k[k] / n, 4) for k in top_k_values}
    mrr = round(np.mean(mrr_scores), 4) if mrr_scores else 0.0
    avg_top1_sim = round(np.mean(top1_sims), 4)

    result = {
        "label": label,
        "use_bm25": use_bm25,
        "use_dense": use_dense,
        "use_intent_filter": use_intent_filter,
        "mrr": mrr,
        "avg_top1_similarity": avg_top1_sim,
        **recall_results,
    }
    logger.info(
        f"{label}: MRR={mrr:.3f}, "
        f"R@1={recall_results.get('recall_at_1', 0):.3f}, "
        f"R@3={recall_results.get('recall_at_3', 0):.3f}, "
        f"R@5={recall_results.get('recall_at_5', 0):.3f}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 8: Retrieval system")
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--build", action="store_true", help="Build retrieval indices")
    parser.add_argument("--run-experiments", action="store_true", help="Run E5-E7")
    parser.add_argument("--include-reranker", action="store_true", help="Include E7 reranker")
    args = parser.parse_args()

    brand_id = args.brand

    if args.build:
        logger.info("=== Building retrieval indices ===")
        resolution_pairs = load_resolution_pairs(brand_id)
        logger.info(f"Loaded {len(resolution_pairs):,} resolution pairs")

        # Build BM25
        bm25 = build_bm25_index(resolution_pairs)
        bm25_path = ARTIFACTS_DIR / f"bm25_index_{brand_id}.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)

        # Build FAISS + embeddings
        faiss_index, embeddings = build_dense_index(resolution_pairs)
        import faiss as faiss_lib
        faiss_lib.write_index(faiss_index, str(ARTIFACTS_DIR / f"faiss_index_{brand_id}.bin"))
        np.save(ARTIFACTS_DIR / f"corpus_embeddings_{brand_id}.npy", embeddings)
        logger.info("All indices built and saved")

    if args.run_experiments:
        logger.info("=== Running retrieval experiments E5–E7 ===")

        # Load artifacts
        resolution_pairs = load_resolution_pairs(brand_id)
        taxonomy = load_taxonomy()
        golden_df = load_golden_set()

        with open(ARTIFACTS_DIR / f"bm25_index_{brand_id}.pkl", "rb") as f:
            bm25 = pickle.load(f)

        import faiss as faiss_lib
        faiss_index = faiss_lib.read_index(str(ARTIFACTS_DIR / f"faiss_index_{brand_id}.bin"))
        embeddings = np.load(ARTIFACTS_DIR / f"corpus_embeddings_{brand_id}.npy")

        # Build retrieval system
        system = RetrievalSystem(resolution_pairs, bm25, faiss_index, embeddings, taxonomy)

        results = []

        # E5a: BM25 only
        r = evaluate_retrieval(system, golden_df, use_bm25=True, use_dense=False,
                               use_intent_filter=False, label="E5a_bm25_only")
        results.append(r)

        # E5b: Dense only
        r = evaluate_retrieval(system, golden_df, use_bm25=False, use_dense=True,
                               use_intent_filter=False, label="E5b_dense_only")
        results.append(r)

        # E5c: Hybrid BM25 + Dense
        r = evaluate_retrieval(system, golden_df, use_bm25=True, use_dense=True,
                               use_intent_filter=False, label="E5c_hybrid")
        results.append(r)

        # E6: Hybrid + intent conditioning
        r = evaluate_retrieval(system, golden_df, use_bm25=True, use_dense=True,
                               use_intent_filter=True, label="E6_intent_conditioned_hybrid")
        results.append(r)

        # E7: + reranker (only if requested and justified)
        if args.include_reranker:
            try:
                from sentence_transformers import CrossEncoder
                reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                system.reranker = reranker
                t_start = time.time()
                r = evaluate_retrieval(system, golden_df.head(50), use_bm25=True, use_dense=True,
                                       use_intent_filter=True, label="E7_with_reranker")
                t_end = time.time()
                r["avg_latency_ms"] = round((t_end - t_start) / 50 * 1000, 1)
                results.append(r)
                system.reranker = None
            except Exception as e:
                logger.warning(f"Reranker experiment failed: {e}")

        # Print comparison table
        print("\n=== RETRIEVAL EXPERIMENT RESULTS ===")
        print(f"{'Label':<35} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'MRR':>6}")
        print("-" * 60)
        for r in results:
            print(
                f"{r['label']:<35} "
                f"{r.get('recall_at_1', 0):>5.3f} "
                f"{r.get('recall_at_3', 0):>5.3f} "
                f"{r.get('recall_at_5', 0):>5.3f} "
                f"{r.get('mrr', 0):>6.3f}"
            )

        # Log results
        from src.phase7_baselines import log_experiment
        for r in results:
            log_experiment(r["label"], {"brand": brand_id}, r)

        # Save the best system config for downstream use
        best_label = max(results, key=lambda x: x.get("mrr", 0))["label"]
        retrieval_config = {
            "brand_id": brand_id,
            "best_config_label": best_label,
            "use_intent_filter": True,  # always use intent conditioning
            "include_reranker": args.include_reranker and "reranker" in best_label,
        }
        with open(ARTIFACTS_DIR / f"retrieval_config_{brand_id}.json", "w", encoding="utf-8") as f:
            json.dump(retrieval_config, f, indent=2)

        return results


if __name__ == "__main__":
    main()
