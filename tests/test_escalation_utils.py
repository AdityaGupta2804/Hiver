"""
Unit tests for escalation decision function and composite risk scoring.
"""

import pytest
from src.utils.escalation_utils import (
    compute_composite_risk_score,
    decide_escalation,
    fit_risk_coverage_curve,
)


def test_high_risk_tier_triggers_escalation():
    risk, signals = compute_composite_risk_score(
        intent="account_security",
        intent_confidence=0.98,  # even with 98% confidence
        retrieval_similarity=0.90,
        groundedness_score=0.95,
    )
    assert risk >= 0.85
    decision, reason = decide_escalation(risk, threshold=0.50, hard_escalate_tier=True)
    assert decision == "escalate"


def test_low_risk_high_confidence_auto_handles():
    risk, signals = compute_composite_risk_score(
        intent="audio_playback",
        intent_confidence=0.95,
        retrieval_similarity=0.90,
        groundedness_score=0.92,
    )
    assert risk < 0.35
    decision, reason = decide_escalation(risk, threshold=0.50, hard_escalate_tier=False)
    assert decision == "auto_handle"


def test_missing_info_forces_escalation():
    risk, signals = compute_composite_risk_score(
        intent="audio_playback",
        intent_confidence=0.90,
        retrieval_similarity=0.40,
        groundedness_score=0.30,
        missing_info_flag=True,
    )
    assert risk >= 0.50
    decision, reason = decide_escalation(risk, threshold=0.50)
    assert decision == "escalate"
