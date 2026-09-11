"""
Unit tests for PII scanning and redaction.
"""

import pytest
from src.utils.pii_utils import redact_pii, scan_corpus_for_pii


def test_redact_pii_email():
    text = "My email is user.name@example.com please help"
    redacted = redact_pii(text)
    assert "[REDACTED_EMAIL]" in redacted
    assert "user.name@example.com" not in redacted


def test_redact_pii_phone():
    text = "Call me at 555-123-4567 regarding my account"
    redacted = redact_pii(text)
    assert "[REDACTED_PHONE]" in redacted
    assert "555-123-4567" not in redacted


def test_scan_corpus_for_pii():
    corpus = [
        "Normal text without personal info",
        "Reach me at test@spotify.com immediately",
    ]
    stats = scan_corpus_for_pii(corpus)
    assert stats["texts_with_pii"] == 1
    assert stats["total_texts"] == 2
