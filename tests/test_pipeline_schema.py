"""
Unit tests for end-to-end pipeline schema and output consistency.
"""

import pytest
from src.pipeline import CustomerSupportPipeline


def test_pipeline_output_schema():
    pipeline = CustomerSupportPipeline(brand_id="SpotifyCares")
    query = "My music keeps stuttering and stopping on Android"

    result = pipeline.process_query(query)

    expected_keys = [
        "input_query",
        "redacted_query",
        "pii_detected",
        "predicted_intent",
        "intent_confidence",
        "decision",
        "decision_reason",
        "composite_risk_score",
        "risk_signals",
        "draft_reply",
        "cited_doc_ids",
        "groundedness_score",
    ]

    for key in expected_keys:
        assert key in result, f"Missing key: {key}"

    assert result["decision"] in ["auto_handle", "escalate"]
    assert 0.0 <= result["composite_risk_score"] <= 1.0
    assert 0.0 <= result["groundedness_score"] <= 1.0
    assert isinstance(result["cited_doc_ids"], list)


def test_pipeline_escalates_security():
    pipeline = CustomerSupportPipeline(brand_id="SpotifyCares")
    security_query = "Someone changed my password and email without my authorization!"

    result = pipeline.process_query(security_query)
    assert result["predicted_intent"] == "account_security"
    assert result["decision"] == "escalate"
    assert result["composite_risk_score"] >= 0.70
