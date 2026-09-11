"""
judge_utils.py — LLM-as-a-judge rubric, scoring, and human validation metrics (Cohen's kappa, Spearman).

Blueprint Section 16 & Experiment E11.
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

JUDGE_RUBRIC_PROMPT = """You are an expert customer support quality auditor evaluating an AI assistant's reply.
Score each dimension from 1 to 5 based on the strict rubric below.

Customer Problem:
"{customer_problem}"

Retrieved Reference Resolution:
"{reference_resolution}"

Generated Reply:
"{generated_reply}"

DIMENSIONS:
1. Accuracy (1-5): Does the advice address the actual root cause correctly?
   1: Completely wrong or harmful advice.
   3: Partially relevant but vague or missing key step.
   5: Completely accurate, flawless troubleshooting steps.

2. Groundedness (1-5): Is the reply faithful to the retrieved evidence without making up unsupported claims?
   1: Full hallucination or contradicts evidence.
   3: Mostly grounded with minor unverified extrapolation.
   5: Strictly grounded in and supported by the retrieved evidence.

3. Completeness & Actionability (1-5): Can the customer take action immediately without follow-up questions?
   1: Unusable, non-actionable boilerplate.
   3: Gives direction but leaves customer guessing specific steps.
   5: Clear, step-by-step actionable guidance.

4. Tone & Professionalism (1-5): Empathic, professional, respectful.
   1: Rude, robotic, or dismissive.
   3: Neutral, standard corporate tone.
   5: Highly empathetic, courteous, and reassuring.

Output JSON:
{
  "accuracy": int,
  "groundedness": int,
  "completeness": int,
  "tone": int,
  "explanation": "Brief 1-2 sentence justification"
}
"""


def heuristic_judge_scorer(
    customer_problem: str,
    reference_resolution: str,
    generated_reply: str,
    is_escalated: bool = False,
) -> Dict[str, Any]:
    """
    Deterministic rule-based judge scorer for offline validation and fallback.
    Matches human evaluation rubric closely based on text overlap, keyword presence, and tone markers.
    """
    reply_lower = generated_reply.lower()
    ref_lower = reference_resolution.lower()
    prob_lower = customer_problem.lower()

    # Accuracy: overlap with reference resolution keywords
    ref_words = set(re.findall(r"\b[a-z]{4,}\b", ref_lower))
    reply_words = set(re.findall(r"\b[a-z]{4,}\b", reply_lower))
    overlap = len(ref_words & reply_words) if ref_words else 0
    overlap_ratio = overlap / max(1, len(ref_words))

    if overlap_ratio > 0.4:
        accuracy = 5
    elif overlap_ratio > 0.2:
        accuracy = 4
    elif overlap_ratio > 0.05:
        accuracy = 3
    else:
        accuracy = 2

    # Groundedness: citation tokens and absence of generic hallucination
    has_citations = bool(re.search(r"\[doc_\d+\]", generated_reply))
    if has_citations and overlap_ratio > 0.15:
        groundedness = 5
    elif has_citations or overlap_ratio > 0.1:
        groundedness = 4
    elif overlap_ratio > 0.05:
        groundedness = 3
    else:
        groundedness = 2

    # Completeness: actionable verbs and reasonable length
    action_verbs = ["try", "check", "restart", "log", "reinstall", "update", "go to", "settings", "click", "select"]
    has_action = any(v in reply_lower for v in action_verbs)
    if len(generated_reply) > 80 and has_action:
        completeness = 5
    elif len(generated_reply) > 40:
        completeness = 4
    elif is_escalated:
        completeness = 4  # escalating appropriately is complete
    else:
        completeness = 2

    # Tone: polite greetings and closing
    polite_markers = ["hi", "hello", "sorry", "help", "please", "thanks", "let us know", "happy to"]
    polite_count = sum(1 for m in polite_markers if m in reply_lower)
    if polite_count >= 3:
        tone = 5
    elif polite_count >= 1:
        tone = 4
    else:
        tone = 3

    return {
        "accuracy": accuracy,
        "groundedness": groundedness,
        "completeness": completeness,
        "tone": tone,
        "overall": round((accuracy + groundedness + completeness + tone) / 4.0, 2),
        "explanation": "Evaluated on keyword alignment, citation fidelity, and actionability.",
    }


def compute_judge_validation_metrics(
    human_scores: List[int],
    judge_scores: List[int],
) -> Dict[str, float]:
    """
    Computes inter-annotator agreement metrics:
    - Quadratic weighted Cohen's Kappa
    - Spearman rank correlation
    - Exact match rate
    - Within-1-point agreement rate
    """
    if len(human_scores) < 2:
        return {}

    kappa = cohen_kappa_score(human_scores, judge_scores, weights="quadratic")
    spearman_corr, p_val = spearmanr(human_scores, judge_scores)

    exact_matches = sum(1 for h, j in zip(human_scores, judge_scores) if h == j)
    within_one = sum(1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= 1)
    n = len(human_scores)

    return {
        "cohen_kappa_quadratic": round(float(kappa), 4) if not np.isnan(kappa) else 0.0,
        "spearman_correlation": round(float(spearman_corr), 4) if not np.isnan(spearman_corr) else 0.0,
        "exact_match_rate": round(exact_matches / n, 4),
        "within_1_point_rate": round(within_one / n, 4),
        "sample_size": n,
    }
