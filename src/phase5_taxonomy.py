"""
phase5_taxonomy.py — Intent taxonomy discovery from the dev slice.

Blueprint Section 6, Roadmap Phase 5.

Steps:
1. Embed all customer tweets in the dev slice using sentence-transformers.
2. Cluster with HDBSCAN (density-aware, no fixed k needed).
3. For each cluster, sample representative messages and propose a label
   using an LLM (if API key available) or heuristic keyword summary.
4. Save the proposed taxonomy and present it for human review.
5. After human approval, freeze the taxonomy (write taxonomy.json).

The taxonomy is frozen BEFORE the golden set is touched.

Usage:
    # Step 1: generate cluster proposals
    python src/phase5_taxonomy.py --brand SpotifyCares --propose

    # Step 2: after you edit the proposals file, freeze the taxonomy
    python src/phase5_taxonomy.py --brand SpotifyCares --freeze
"""

import os
import sys
import json
import logging
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Optional
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
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dev_threads(brand_id: str) -> List[dict]:
    path = ARTIFACTS_DIR / f"threads_dev_{brand_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Dev threads not found. Run phase4_split.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_customer_messages(threads: List[dict]) -> List[Dict]:
    """
    Extract all customer-inbound messages from dev threads.
    Returns list of dicts with thread_id, tweet_id, text.
    We only cluster CUSTOMER messages (not brand replies) since we're
    discovering customer intent, not brand response patterns.
    """
    messages = []
    for thread in threads:
        for turn in thread.get("turns", []):
            if turn.get("inbound", True):  # customer message
                text = turn.get("text", "").strip()
                if len(text) > 20:  # skip very short messages
                    messages.append({
                        "thread_id": thread["root_id"],
                        "tweet_id": turn["tweet_id"],
                        "text": text,
                    })
    return messages


def embed_messages(messages: List[Dict], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed customer messages using sentence-transformers.
    Caches embeddings to disk.
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Embedding {len(messages):,} messages with {model_name}...")
    model = SentenceTransformer(model_name)
    texts = [m["text"] for m in messages]

    # Batch encode
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int = 5,
) -> np.ndarray:
    """
    Cluster embeddings with HDBSCAN. Returns cluster labels array.
    Label -1 = noise (not assigned to any cluster).
    """
    import hdbscan

    logger.info(
        f"Clustering {len(embeddings):,} embeddings with HDBSCAN "
        f"(min_cluster_size={min_cluster_size})..."
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(
        f"HDBSCAN found {n_clusters} clusters, {n_noise:,} noise points "
        f"({n_noise/len(labels):.1%} of messages)"
    )
    return labels


def sample_cluster_messages(
    messages: List[Dict],
    labels: np.ndarray,
    n_samples: int = 12,
) -> Dict[int, List[str]]:
    """
    For each cluster, return up to n_samples representative messages.
    """
    clusters = {}
    for i, (msg, label) in enumerate(zip(messages, labels)):
        lbl = int(label)
        if lbl == -1:
            continue
        if lbl not in clusters:
            clusters[lbl] = []
        clusters[lbl].append(msg["text"])

    # Sample
    sampled = {}
    for label, texts in clusters.items():
        idx = np.random.choice(len(texts), size=min(n_samples, len(texts)), replace=False)
        sampled[int(label)] = [texts[int(i)] for i in idx]

    return sampled


def propose_labels_with_llm(
    cluster_samples: Dict[int, List[str]],
    brand_id: str,
) -> Dict[int, Dict]:
    """
    Use an LLM to propose a label and one-line definition for each cluster.
    Falls back to keyword extraction if no API key is available.
    """
    try:
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)
        use_llm = True
        logger.info("Using LLM for cluster label proposals")
    except Exception as e:
        logger.warning(f"LLM not available ({e}), using keyword-based labels")
        use_llm = False

    proposals = {}

    for cluster_id in sorted(cluster_samples.keys()):
        samples = cluster_samples[cluster_id]
        sample_text = "\n".join(f"- {s}" for s in samples[:10])

        if use_llm:
            prompt = f"""You are analyzing customer support tweets sent to {brand_id}.
Below are {len(samples)} example customer messages that form a cluster.
Based on these messages, identify the customer intent.

Messages:
{sample_text}

Respond in JSON with exactly these fields:
{{
  "label": "short_snake_case_label (e.g. 'playback_issue')",
  "display_name": "Human readable name (e.g. 'Playback Issues')",
  "definition": "One sentence: what complaint or request this intent covers",
  "example": "A representative example from the messages above"
}}"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                proposal = json.loads(response.choices[0].message.content)
                proposal["cluster_id"] = int(cluster_id)
                proposal["n_messages"] = int(len(cluster_samples[cluster_id]))
                proposal["sample_messages"] = samples[:5]
                proposals[int(cluster_id)] = proposal
                logger.info(f"  Cluster {cluster_id}: {proposal.get('label', '?')}")
            except Exception as e:
                logger.warning(f"  LLM failed for cluster {cluster_id}: {e}")
                proposals[int(cluster_id)] = _keyword_label(cluster_id, samples)
        else:
            proposals[int(cluster_id)] = _keyword_label(cluster_id, samples)

    return proposals


def _keyword_label(cluster_id: int, samples: List[str]) -> Dict:
    """Simple keyword-based label as fallback."""
    from collections import Counter
    import re
    words = []
    for s in samples:
        words.extend(re.findall(r"\b[a-z]{4,}\b", s.lower()))
    # Remove very common stopwords
    stops = {"this", "that", "with", "have", "your", "from", "they", "will",
              "what", "when", "been", "were", "just", "also", "more", "some",
              "would", "could", "about", "help", "please", "thank", "still"}
    keywords = [w for w in words if w not in stops]
    top_words = [w for w, _ in Counter(keywords).most_common(5)]
    label = "_".join(top_words[:3]) if top_words else f"cluster_{cluster_id}"
    return {
        "cluster_id": int(cluster_id),
        "label": label,
        "display_name": " ".join(w.title() for w in top_words[:3]) if top_words else f"Cluster {cluster_id}",
        "definition": f"Auto-labeled from keywords: {', '.join(top_words)}",
        "example": samples[0] if samples else "",
        "n_messages": int(len(samples)),
        "sample_messages": samples[:5],
    }


def run_propose(brand_id: str, model_name: str = "all-MiniLM-L6-v2"):
    """Run clustering and generate cluster label proposals for human review."""
    logger.info(f"=== Phase 5: Intent Discovery for '{brand_id}' ===")

    # Load dev threads
    dev_threads = load_dev_threads(brand_id)
    logger.info(f"Loaded {len(dev_threads):,} dev threads")

    # Extract customer messages
    messages = extract_customer_messages(dev_threads)
    logger.info(f"Extracted {len(messages):,} customer messages for clustering")

    # Embed
    embed_cache = ARTIFACTS_DIR / f"dev_embeddings_{brand_id}.npy"
    msg_cache = ARTIFACTS_DIR / f"dev_messages_{brand_id}.pkl"

    if embed_cache.exists() and msg_cache.exists():
        logger.info("Loading cached embeddings...")
        embeddings = np.load(embed_cache)
        with open(msg_cache, "rb") as f:
            messages = pickle.load(f)
    else:
        embeddings = embed_messages(messages, model_name)
        np.save(embed_cache, embeddings)
        with open(msg_cache, "wb") as f:
            pickle.dump(messages, f)
        logger.info(f"Embeddings cached to {embed_cache}")

    # Cluster
    labels = cluster_embeddings(embeddings)

    # Sample cluster messages
    cluster_samples = sample_cluster_messages(messages, labels)
    logger.info(f"Sampling from {len(cluster_samples)} clusters...")

    # Propose labels
    proposals = propose_labels_with_llm(cluster_samples, brand_id)

    # Save noise messages separately
    noise_messages = [m for m, l in zip(messages, labels) if l == -1]

    # Build proposals file for human review
    proposals_data = {
        "brand_id": brand_id,
        "status": "PENDING_HUMAN_REVIEW",
        "instructions": (
            "Review each cluster below. You may:\n"
            "  1. Keep a cluster as-is\n"
            "  2. MERGE two clusters by giving them the same label\n"
            "  3. SPLIT a cluster by editing its definition to be narrower\n"
            "  4. DELETE a cluster by setting 'keep': false\n"
            "  5. RENAME/REDEFINE any cluster\n"
            "  6. Add an 'other' bucket if needed\n"
            "Always include an 'other_unclear' intent even if not in the clusters.\n"
            "Aim for 8-15 intents total.\n"
            "After editing, run: python src/phase5_taxonomy.py --brand <brand> --freeze"
        ),
        "n_clusters_proposed": len(proposals),
        "n_noise_messages": len(noise_messages),
        "clusters": list(proposals.values()),
    }

    proposals_path = ARTIFACTS_DIR / f"taxonomy_proposals_{brand_id}.json"
    with open(proposals_path, "w", encoding="utf-8") as f:
        json.dump(proposals_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nProposals written to: {proposals_path}")
    print(f"\n{'='*60}")
    print(f"PHASE 5: CLUSTER PROPOSALS READY FOR REVIEW")
    print(f"{'='*60}")
    print(f"File: {proposals_path}")
    print(f"Clusters found: {len(proposals)}")
    print(f"Noise messages: {len(noise_messages):,}")
    print(f"\nProposed intents:")
    for p in proposals.values():
        print(f"  [{p['cluster_id']:2d}] {p.get('label','?')}: {p.get('definition','')[:80]}")
    print(f"\nEdit the proposals file and then run:")
    print(f"  python src/phase5_taxonomy.py --brand {brand_id} --freeze")


def run_freeze(brand_id: str):
    """Read the human-reviewed proposals and write the frozen taxonomy.json."""
    proposals_path = ARTIFACTS_DIR / f"taxonomy_proposals_{brand_id}.json"
    if not proposals_path.exists():
        raise FileNotFoundError(f"Proposals file not found: {proposals_path}")

    with open(proposals_path, "r", encoding="utf-8") as f:
        proposals_data = json.load(f)

    clusters = proposals_data.get("clusters", [])
    kept = [c for c in clusters if c.get("keep", True) is not False]

    # Ensure 'other_unclear' is always present
    labels_present = {c.get("label") for c in kept}
    if "other_unclear" not in labels_present:
        kept.append({
            "label": "other_unclear",
            "display_name": "Other / Unclear",
            "definition": (
                "Message does not fit any defined intent, or intent is unclear "
                "from the available text."
            ),
            "example": "",
            "cluster_id": -1,
        })

    # Build the frozen taxonomy
    taxonomy = {
        "brand_id": brand_id,
        "frozen": True,
        "intents": [],
    }
    for i, cluster in enumerate(kept):
        taxonomy["intents"].append({
            "id": i,
            "label": cluster["label"],
            "display_name": cluster.get("display_name", cluster["label"].replace("_", " ").title()),
            "definition": cluster.get("definition", ""),
            "example": cluster.get("example", ""),
        })

    taxonomy_path = ARTIFACTS_DIR / "taxonomy.json"
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    logger.info(f"\nTaxonomy FROZEN at: {taxonomy_path}")
    print(f"\n{'='*60}")
    print(f"TAXONOMY FROZEN — {len(taxonomy['intents'])} intents")
    print(f"{'='*60}")
    for intent in taxonomy["intents"]:
        print(f"  [{intent['id']:2d}] {intent['label']}: {intent['definition'][:80]}")
    print(f"\nDo NOT edit taxonomy.json after this point without explicitly re-running --freeze.")
    print(f"The golden set must be sampled AFTER this freeze. Proceeding to Phase 6 is safe.")

    return taxonomy


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Intent taxonomy discovery")
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--propose", action="store_true", help="Run clustering + propose labels")
    parser.add_argument("--freeze", action="store_true", help="Freeze the reviewed taxonomy")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    if args.propose:
        run_propose(args.brand, model_name=args.model)
    elif args.freeze:
        run_freeze(args.brand)
    else:
        print("Specify --propose or --freeze")
        print("Example: python src/phase5_taxonomy.py --brand SpotifyCares --propose")


if __name__ == "__main__":
    main()
