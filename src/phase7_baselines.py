"""
phase7_baselines.py — Baseline models E1–E4.

Blueprint Section 7, Roadmap Phase 7.

Implements:
  E1: Majority-class intent + canned reply (trivial baseline — the absolute floor)
  E2: TF-IDF + Logistic Regression (simple but real baseline)
  E3: Embedding k-NN classifier (no training loop)
  E4: Zero-shot LLM classification (sanity check for brand-specific knowledge value)

All baselines:
- Are trained/configured on the dev slice only
- Are evaluated on the same frozen golden set
- Produce the same output schema: {intent, confidence, draft_reply, decision, reason}
- Run with the same preprocessing as the final system

Usage:
    python src/phase7_baselines.py --brand SpotifyCares --run e1 e2 e3
    python src/phase7_baselines.py --brand SpotifyCares --run all
"""

import os
import sys
import json
import logging
import argparse
import pickle
import random
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


def load_taxonomy() -> dict:
    path = ARTIFACTS_DIR / "taxonomy.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_golden_set() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "golden_set.csv"
    if not path.exists():
        raise FileNotFoundError("Golden set not found. Run phase6_golden_set.py --check first.")
    return pd.read_csv(path)


def load_dev_threads(brand_id: str) -> List[dict]:
    path = ARTIFACTS_DIR / f"threads_dev_{brand_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def build_dev_training_data(
    dev_threads: List[dict],
    taxonomy: dict,
) -> Tuple[List[str], List[str]]:
    """
    Build (text, intent) training pairs from dev slice.
    Uses only labeled examples that have an intent tag assigned.
    For baselines, we simulate this by using the taxonomy labels.

    NOTE: In the real pipeline, these labels come from the taxonomy-discovery
    clustering step applied to dev examples. For baseline training, we use
    a simple approach: embed and assign to nearest cluster centroid.
    """
    # Load dev cluster assignments if available
    dev_labels_path = ARTIFACTS_DIR / f"dev_intent_labels_{dev_threads[0]['root_id']}.json"
    if dev_labels_path.exists():
        with open(dev_labels_path, encoding="utf-8") as f:
            dev_labels = json.load(f)
        texts = [item["text"] for item in dev_labels]
        labels = [item["intent"] for item in dev_labels]
        return texts, labels

    # Fallback: use all customer messages from dev with a placeholder label
    # This shouldn't happen in normal flow, but handles the case where
    # the taxonomy assignment step hasn't run yet
    logger.warning(
        "Dev intent labels not found. Using empty training data for baselines. "
        "Run phase5_taxonomy.py --assign first."
    )
    return [], []


def format_standard_output(
    golden_id: str,
    intent: str,
    confidence: float,
    draft_reply: str,
    decision: str,
    reason: str,
    baseline_name: str,
) -> dict:
    """
    Standard output schema for all systems (baseline and final).
    Every system must produce this same schema for fair comparison.
    """
    return {
        "golden_id": golden_id,
        "baseline": baseline_name,
        "predicted_intent": intent,
        "intent_confidence": confidence,
        "draft_reply": draft_reply,
        "decision": decision,  # "AUTO_HANDLE" or "ESCALATE"
        "reason": reason,
        "cited_evidence_ids": [],  # baselines don't retrieve
        "claims": [],
        "missing_information_flag": False,
    }


# ---------------------------------------------------------------------------
# E1: Majority-class baseline
# ---------------------------------------------------------------------------

class MajorityClassBaseline:
    """Always predict the most frequent intent and return a canned reply."""

    def __init__(self, taxonomy: dict):
        self.taxonomy = taxonomy
        self.majority_intent = None
        self.canned_replies = {}

    def fit(self, texts: List[str], labels: List[str]):
        from collections import Counter
        if not labels:
            self.majority_intent = taxonomy["intents"][0]["label"]
        else:
            counts = Counter(labels)
            self.majority_intent = counts.most_common(1)[0][0]

        # Generate a canned reply for each intent from taxonomy definition
        for intent_def in self.taxonomy["intents"]:
            label = intent_def["label"]
            self.canned_replies[label] = (
                f"Thank you for reaching out. "
                f"Regarding your {intent_def['display_name'].lower()}, "
                f"please contact our support team for assistance."
            )

    def predict(self, text: str) -> Tuple[str, float]:
        return self.majority_intent, 1.0  # always 100% confident (wrong by design)

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        return [(self.majority_intent, 1.0) for _ in texts]

    def get_canned_reply(self, intent: str) -> str:
        return self.canned_replies.get(intent, "Thank you for contacting support.")


# ---------------------------------------------------------------------------
# E2: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

class TFIDFLinearBaseline:
    """TF-IDF features + logistic regression classifier."""

    def __init__(self):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", CalibratedClassifierCV(
                LogisticRegression(max_iter=1000, C=1.0, random_state=42),
                cv=3,
            )),
        ])
        self.classes_ = None

    def fit(self, texts: List[str], labels: List[str]):
        logger.info(f"Fitting TF-IDF+LR on {len(texts)} examples...")
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline.classes_
        logger.info(f"TF-IDF+LR fitted. Classes: {list(self.classes_)}")

    def predict(self, text: str) -> Tuple[str, float]:
        proba = self.pipeline.predict_proba([text])[0]
        best_idx = np.argmax(proba)
        return self.classes_[best_idx], float(proba[best_idx])

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        probas = self.pipeline.predict_proba(texts)
        best_indices = np.argmax(probas, axis=1)
        return [
            (self.classes_[idx], float(probas[i, idx]))
            for i, idx in enumerate(best_indices)
        ]

    def save(self, brand_id: str):
        path = ARTIFACTS_DIR / f"tfidf_baseline_{brand_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"TF-IDF baseline saved to {path}")

    @classmethod
    def load(cls, brand_id: str):
        path = ARTIFACTS_DIR / f"tfidf_baseline_{brand_id}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# E3: Embedding k-NN classifier
# ---------------------------------------------------------------------------

class EmbeddingKNNBaseline:
    """
    Embed texts, then classify by majority label among k nearest neighbors.
    No training loop — just embed labeled examples and do similarity search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", k: int = 5):
        self.model_name = model_name
        self.k = k
        self.embedder = None
        self.train_embeddings = None
        self.train_labels = None

    def fit(self, texts: List[str], labels: List[str]):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Fitting Embedding-kNN (k={self.k}) on {len(texts)} examples...")
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.model_name)
        self.train_embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
        )
        self.train_labels = np.array(labels)
        logger.info(f"Embedding-kNN fitted. {self.train_embeddings.shape}")

    def predict(self, text: str) -> Tuple[str, float]:
        from sentence_transformers import util
        query_emb = self.embedder.encode([text], normalize_embeddings=True)
        sims = util.cos_sim(query_emb, self.train_embeddings)[0].numpy()
        top_k_idx = np.argsort(sims)[-self.k:]
        top_k_labels = self.train_labels[top_k_idx]
        from collections import Counter
        counts = Counter(top_k_labels)
        majority_label = counts.most_common(1)[0][0]
        # Confidence: fraction of k neighbors with majority label
        confidence = counts[majority_label] / self.k
        return majority_label, float(confidence)

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        return [self.predict(t) for t in texts]

    def save(self, brand_id: str):
        path = ARTIFACTS_DIR / f"emb_knn_baseline_{brand_id}.pkl"
        # Don't pickle the embedder (too large); save embeddings + labels + config
        state = {
            "model_name": self.model_name,
            "k": self.k,
            "train_embeddings": self.train_embeddings,
            "train_labels": self.train_labels,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Embedding-kNN baseline saved to {path}")


# ---------------------------------------------------------------------------
# E4: Zero-shot LLM classification
# ---------------------------------------------------------------------------

class ZeroShotLLMClassifier:
    """
    Uses an LLM with just the intent list and definitions in the prompt.
    No training examples — tests how much the trained classifiers' value
    comes from actually seeing brand-specific data.
    """

    def __init__(self, taxonomy: dict, model: str = "gpt-4o-mini"):
        self.taxonomy = taxonomy
        self.model = model
        self._intent_prompt = self._build_intent_prompt()
        self.client = None

    def _build_intent_prompt(self) -> str:
        intent_list = "\n".join(
            f"  - {i['label']}: {i['definition']}"
            for i in self.taxonomy["intents"]
        )
        return f"""You are a customer support intent classifier.

Available intents:
{intent_list}

Given a customer message, respond with JSON:
{{"intent": "<exact_label>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}

Use 'other_unclear' if the message doesn't fit any defined intent.
"""

    def _get_client(self):
        if self.client is None:
            import openai
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set. Cannot run zero-shot LLM baseline.")
            self.client = openai.OpenAI(api_key=api_key)
        return self.client

    def predict(self, text: str) -> Tuple[str, float]:
        valid_labels = {i["label"] for i in self.taxonomy["intents"]}
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._intent_prompt},
                    {"role": "user", "content": f"Customer message: {text}"},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            intent = result.get("intent", "other_unclear")
            if intent not in valid_labels:
                intent = "other_unclear"
            confidence = float(result.get("confidence", 0.5))
            return intent, confidence
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return "other_unclear", 0.1

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        return [self.predict(t) for t in texts]


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_classifier(
    predictions: List[Tuple[str, float]],
    golden_df: pd.DataFrame,
    baseline_name: str,
) -> dict:
    """
    Compute macro-F1, per-class P/R/F1, and ECE for a classifier's predictions.
    """
    from sklearn.metrics import (
        classification_report,
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    true_labels = golden_df["primary_intent"].tolist()
    pred_labels = [p[0] for p in predictions]
    pred_probs = [p[1] for p in predictions]

    # Macro-F1
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    # Per-class
    classes = sorted(set(true_labels + pred_labels))
    p, r, f, s = precision_recall_fscore_support(
        true_labels, pred_labels, labels=classes, zero_division=0
    )
    per_class = {
        c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, c in enumerate(classes)
    }

    # ECE (Expected Calibration Error)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(pred_probs)
    for i in range(n_bins):
        in_bin = [
            j for j in range(n)
            if bin_boundaries[i] <= pred_probs[j] < bin_boundaries[i + 1]
        ]
        if in_bin:
            bin_acc = np.mean([1.0 if pred_labels[j] == true_labels[j] else 0.0 for j in in_bin])
            bin_conf = np.mean([pred_probs[j] for j in in_bin])
            ece += len(in_bin) / n * abs(bin_acc - bin_conf)

    result = {
        "baseline": baseline_name,
        "macro_f1": round(macro_f1, 4),
        "ece": round(ece, 4),
        "per_class": per_class,
        "n_examples": len(true_labels),
    }
    logger.info(
        f"{baseline_name}: macro_F1={macro_f1:.3f}, ECE={ece:.3f}"
    )
    return result


def log_experiment(experiment_id: str, config: dict, results: dict):
    """Append experiment results to the experiments log."""
    results_path = EXPERIMENTS_DIR / "results.json"
    entry = {"experiment_id": experiment_id, "config": config, "results": results}

    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(entry)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results logged to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Baseline models E1-E4")
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument(
        "--run",
        nargs="+",
        choices=["e1", "e2", "e3", "e4", "all"],
        default=["all"],
    )
    parser.add_argument("--skip-e4", action="store_true",
                        help="Skip LLM baseline (no API key)")
    args = parser.parse_args()

    run_all = "all" in args.run
    brand_id = args.brand

    taxonomy = load_taxonomy()
    golden_df = load_golden_set()
    logger.info(f"Golden set: {len(golden_df)} examples")

    # Load dev training data
    dev_threads = load_dev_threads(brand_id)

    # Check if dev intent labels exist
    dev_labels_path = ARTIFACTS_DIR / f"dev_intent_labels_{brand_id}.json"
    if dev_labels_path.exists():
        with open(dev_labels_path, encoding="utf-8") as f:
            dev_data = json.load(f)
        train_texts = [d["text"] for d in dev_data]
        train_labels = [d["intent"] for d in dev_data]
    else:
        logger.warning(
            "Dev intent labels not found. Baselines will be trained with "
            "limited or no labeled data. Run phase5_taxonomy.py to assign "
            "intent labels to dev examples first."
        )
        train_texts, train_labels = [], []

    logger.info(f"Training data: {len(train_texts)} labeled examples")

    golden_messages = golden_df["first_message"].fillna("").astype(str).tolist()
    all_results = {}

    # E1: Majority-class baseline
    if run_all or "e1" in args.run:
        logger.info("\n=== E1: Majority-class baseline ===")
        e1 = MajorityClassBaseline(taxonomy)
        e1.fit(train_texts, train_labels)
        preds = e1.predict_batch(golden_messages)
        results_e1 = evaluate_classifier(preds, golden_df, "E1_majority_class")
        all_results["E1"] = results_e1
        log_experiment("E1_majority_class", {"brand": brand_id}, results_e1)

    # E2: TF-IDF + Logistic Regression
    if (run_all or "e2" in args.run) and train_texts:
        logger.info("\n=== E2: TF-IDF + Logistic Regression ===")
        e2 = TFIDFLinearBaseline()
        e2.fit(train_texts, train_labels)
        e2.save(brand_id)
        preds = e2.predict_batch(golden_messages)
        results_e2 = evaluate_classifier(preds, golden_df, "E2_tfidf_linear")
        all_results["E2"] = results_e2
        log_experiment("E2_tfidf_linear", {"brand": brand_id}, results_e2)

    # E3: Embedding k-NN
    if (run_all or "e3" in args.run) and train_texts:
        logger.info("\n=== E3: Embedding k-NN ===")
        e3 = EmbeddingKNNBaseline(k=5)
        e3.fit(train_texts, train_labels)
        e3.save(brand_id)
        preds = e3.predict_batch(golden_messages)
        results_e3 = evaluate_classifier(preds, golden_df, "E3_embedding_knn")
        all_results["E3"] = results_e3
        log_experiment("E3_embedding_knn", {"brand": brand_id}, results_e3)

    # E4: Zero-shot LLM
    if (run_all or "e4" in args.run) and not args.skip_e4:
        logger.info("\n=== E4: Zero-shot LLM classification ===")
        try:
            e4 = ZeroShotLLMClassifier(taxonomy)
            # Only run on a subset for cost control (50 examples)
            sample_idx = random.sample(range(len(golden_messages)), min(50, len(golden_messages)))
            sample_msgs = [golden_messages[i] for i in sample_idx]
            sample_df = golden_df.iloc[sample_idx].reset_index(drop=True)
            preds_e4 = e4.predict_batch(sample_msgs)
            results_e4 = evaluate_classifier(preds_e4, sample_df, "E4_zero_shot_llm")
            all_results["E4"] = results_e4
            log_experiment("E4_zero_shot_llm", {"brand": brand_id, "n_sample": 50}, results_e4)
        except RuntimeError as e:
            logger.warning(f"E4 skipped: {e}")

    # Print summary table
    print("\n=== BASELINE RESULTS SUMMARY ===")
    print(f"{'Baseline':<25} {'Macro-F1':>10} {'ECE':>8}")
    print("-" * 45)
    for exp_id, res in all_results.items():
        print(f"{res['baseline']:<25} {res['macro_f1']:>10.3f} {res['ece']:>8.3f}")

    return all_results


if __name__ == "__main__":
    main()
