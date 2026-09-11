"""
Unit tests for exact dedup and near-dup detection.
"""

import pytest
from src.utils.dedup_utils import (
    normalize_for_dedup,
    exact_dedup_texts,
    classify_company_reply,
    is_dm_deflection,
)


def test_normalize_for_dedup():
    text = "@SpotifyCares Hey! Can you help me? https://t.co/xyz123"
    norm = normalize_for_dedup(text)
    assert "@spotifycares" not in norm
    assert "help" in norm


def test_exact_dedup_texts():
    texts = [
        "Spotify is down for me",
        "Spotify is down for me",
        "Different issue here",
        "Spotify is down for me",
    ]
    keep, clusters = exact_dedup_texts(texts)
    assert len(keep) == 2
    assert 0 in keep
    assert 2 in keep


def test_dm_deflection():
    dm_text = "Sorry to hear that! Please send us a DM with your account details so we can investigate."
    res_text = "Hi there! Try clearing your cache in Settings -> Storage -> Clear Cache."

    assert is_dm_deflection(dm_text) is True
    assert is_dm_deflection(res_text) is False
    assert classify_company_reply(dm_text) == "dm_deflection"
    assert classify_company_reply(res_text) == "resolution"
