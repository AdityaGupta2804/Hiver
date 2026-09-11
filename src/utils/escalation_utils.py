"""
escalation_utils.py — Risk-tiered escalation decision function and risk-coverage curve fitting.

Blueprint Section 12, Roadmap Phase 10.
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default intent risk tier mapping
# High risk: Billing, Payment dispute, Account security/Takeover, Refunds
# Medium risk: In-app purchase, Subscription change, Account details
# Low risk: Audio playback, App crash, Playlist sync, General troubleshooting
DEFAULT_INTENT_RISK_TIERS = {
    "account_security": "high",
    "billing_dispute": "high",
    "subscription_refund": "high",
    "payment_failure": "high",
    "account_login": "medium",
    "family_plan": "medium",
    "app_crash": "low",
    "audio_playback": "low",
    "playlist_sync": "low",
    "download_offline": "low",
    "other": "medium",
}

RISK_TIER_WEIGHTS = {
    "low": 0.1,
    "medium": 0.4,
    "high": 0.9,
}


def compute_composite_risk_score(
    intent: str,
    intent_confidence: float,
    retrieval_similarity: float,
    groundedness_score: float,
    missing_info_flag: bool = False,
    conflicting_evidence: bool = False,
    intent_risk_tiers: Optional[Dict[str, str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes a normalized risk score [0, 1] where higher = riskier (more reason to escalate).

    Signals:
    - Tier risk: intrinsic risk of the intent category
    - Uncertainty risk: 1.0 - intent_confidence
    - Retrieval gap: 1.0 - retrieval_similarity
    - Groundedness gap: 1.0 - groundedness_score
    - Flags: missing info or conflicting evidence impose heavy penalties
    """
    if intent_risk_tiers is None:
        intent_risk_tiers = DEFAULT_INTENT_RISK_TIERS

    tier = intent_risk_tiers.get(intent.lower(), "medium")
    tier_risk = RISK_TIER_WEIGHTS.get(tier, 0.4)

    uncertainty_risk = max(0.0, 1.0 - intent_confidence)
    retrieval_risk = max(0.0, 1.0 - retrieval_similarity)
    groundedness_risk = max(0.0, 1.0 - groundedness_score)

    flag_penalty = 0.0
    if missing_info_flag:
        flag_penalty += 0.35
    if conflicting_evidence:
        flag_penalty += 0.30

    # Hard override: High-risk intents (e.g. security/fraud) always yield risk >= 0.85
    if tier == "high":
        composite_risk = max(0.85, 0.5 * tier_risk + 0.2 * uncertainty_risk + 0.2 * retrieval_risk + flag_penalty)
    else:
        composite_risk = (
            0.35 * tier_risk +
            0.25 * uncertainty_risk +
            0.20 * retrieval_risk +
            0.20 * groundedness_risk +
            flag_penalty
        )

    composite_risk = float(np.clip(composite_risk, 0.0, 1.0))

    signals = {
        "tier": tier,
        "tier_risk": tier_risk,
        "uncertainty_risk": round(uncertainty_risk, 3),
        "retrieval_risk": round(retrieval_risk, 3),
        "groundedness_risk": round(groundedness_risk, 3),
        "flag_penalty": flag_penalty,
        "composite_risk": round(composite_risk, 3),
    }

    return composite_risk, signals


def decide_escalation(
    composite_risk: float,
    threshold: float = 0.50,
    hard_escalate_tier: bool = False,
) -> Tuple[str, str]:
    """
    Returns (decision, reason) where decision is 'escalate' or 'auto_handle'.
    """
    if hard_escalate_tier:
        return "escalate", "Mandatory escalation triggered by high-risk category (security/billing dispute)."

    if composite_risk >= threshold:
        return "escalate", f"Risk score ({composite_risk:.2f}) exceeds operating threshold ({threshold:.2f})."
    else:
        return "auto_handle", f"Risk score ({composite_risk:.2f}) within acceptable autonomous handling bounds."


def fit_risk_coverage_curve(
    dev_cases: List[Dict],
    target_error_rate: float = 0.05,
    cost_weight_false_auto: float = 4.0,
) -> Dict[str, Any]:
    """
    Sweeps threshold tau in [0.1, 0.9] to construct the risk-coverage curve.
    Finds the optimal operating threshold tau that maximizes coverage while
    keeping false-auto-handle error rate <= target_error_rate.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    curve = []

    best_tau = 0.50
    best_coverage = 0.0

    for tau in thresholds:
        auto_handled = []
        escalated = []
        errors_in_auto = 0

        for case in dev_cases:
            risk = case["composite_risk"]
            should_escalate_gt = case.get("should_escalate_gt", False)

            if risk < tau:
                auto_handled.append(case)
                if should_escalate_gt:
                    errors_in_auto += 1
            else:
                escalated.append(case)

        total = len(dev_cases)
        n_auto = len(auto_handled)
        coverage = n_auto / total if total > 0 else 0.0
        error_rate_in_auto = errors_in_auto / n_auto if n_auto > 0 else 0.0

        # Asymmetric cost: false auto * 4.0 + unnecessary escalation * 1.0
        n_false_escalation = sum(1 for c in escalated if not c.get("should_escalate_gt", False))
        total_loss = (errors_in_auto * cost_weight_false_auto + n_false_escalation * 1.0) / total if total > 0 else 0.0

        curve.append({
            "threshold": round(float(tau), 3),
            "coverage": round(float(coverage), 4),
            "error_rate_in_auto": round(float(error_rate_in_auto), 4),
            "total_loss": round(float(total_loss), 4),
        })

        if error_rate_in_auto <= target_error_rate and coverage > best_coverage:
            best_coverage = coverage
            best_tau = float(tau)

    return {
        "operating_threshold": round(best_tau, 3),
        "achieved_coverage": round(best_coverage, 4),
        "target_error_rate": target_error_rate,
        "risk_coverage_curve": curve,
    }
