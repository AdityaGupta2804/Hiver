"""
eval_utils.py — Evaluation metrics, calibration, bootstrap confidence intervals, and scorecard formatter.

Blueprint Sections 15 & 18.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes Macro-F1, Micro-F1, Weighted-F1, Accuracy, and per-class F1."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = {lbl: round(float(f1), 4) for lbl, f1 in zip(labels, per_class)}

    return {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "accuracy": round(float(acc), 4),
        "per_class_f1": per_class_f1,
    }


def compute_calibration_metrics(
    y_true_binary: List[int],
    y_prob: List[float],
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Computes Expected Calibration Error (ECE) and Brier Score.
    y_true_binary: 1 if prediction was correct, 0 otherwise.
    y_prob: predicted confidence [0, 1].
    """
    y_true = np.array(y_true_binary)
    probs = np.array(y_prob)

    # Brier score
    brier = float(np.mean((probs - y_true) ** 2))

    # Expected Calibration Error (ECE)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        mask = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        n_in_bin = np.sum(mask)

        if n_in_bin > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(probs[mask])
            ece += (n_in_bin / total) * abs(bin_acc - bin_conf)

    return {
        "ece": round(float(ece), 4),
        "brier_score": round(brier, 4),
    }


def bootstrap_confidence_interval(
    metric_fn,
    y_true: List[Any],
    y_pred: List[Any],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes bootstrap confidence interval for any metric function.
    Returns: (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n = len(y_true_arr)

    point_estimate = float(metric_fn(y_true_arr, y_pred_arr))

    boot_estimates = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sample_true = y_true_arr[idx]
        sample_pred = y_pred_arr[idx]
        boot_estimates.append(float(metric_fn(sample_true, sample_pred)))

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_estimates, 100 * alpha))
    upper = float(np.percentile(boot_estimates, 100 * (1.0 - alpha)))

    return round(point_estimate, 4), round(lower, 4), round(upper, 4)


def compute_escalation_scorecard(
    decisions: List[str],
    ground_truth_should_escalate: List[bool],
    cost_weight_false_auto: float = 4.0,
) -> Dict[str, float]:
    """
    Computes asymmetric escalation scorecard:
    - Coverage: % of cases auto-handled
    - False auto-handle rate: % of auto-handled cases that should have been escalated
    - False escalation rate: % of safe cases that were unnecessarily escalated
    - Asymmetric loss
    """
    total = len(decisions)
    if total == 0:
        return {}

    auto_cases = [i for i, d in enumerate(decisions) if d == "auto_handle"]
    escalated_cases = [i for i, d in enumerate(decisions) if d == "escalate"]

    coverage = len(auto_cases) / total
    false_auto_count = sum(1 for i in auto_cases if ground_truth_should_escalate[i])
    false_auto_rate = false_auto_count / len(auto_cases) if auto_cases else 0.0

    safe_count = sum(1 for gt in ground_truth_should_escalate if not gt)
    unnecessary_escalations = sum(1 for i in escalated_cases if not ground_truth_should_escalate[i])
    false_escalation_rate = unnecessary_escalations / safe_count if safe_count > 0 else 0.0

    asymmetric_loss = (false_auto_count * cost_weight_false_auto + unnecessary_escalations * 1.0) / total

    return {
        "coverage": round(coverage, 4),
        "false_auto_handle_rate": round(false_auto_rate, 4),
        "false_escalation_rate": round(false_escalation_rate, 4),
        "false_auto_count": false_auto_count,
        "unnecessary_escalations": unnecessary_escalations,
        "asymmetric_loss": round(asymmetric_loss, 4),
    }
