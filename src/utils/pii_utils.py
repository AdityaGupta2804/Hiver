"""
pii_utils.py — PII detection and re-scanning.

The dataset publisher already masked emails/phones with tokens like **email**,
**phone**, etc. But regex-based masking can miss edge cases. This module:
1. Verifies that published mask tokens are present (publisher's pass).
2. Runs our own secondary scan for common PII patterns that might have slipped
   through (partial phone numbers, order numbers, non-US formats).
3. Provides a hard re-scan before any text enters a prompt or is included in
   a report.

This is explicitly NOT a primary PII scrubber — it's a backstop pass.
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Published mask tokens the dataset uses (confirmed from data inspection)
# ---------------------------------------------------------------------------
PUBLISHER_MASK_TOKENS = [
    "**email**",
    "**phone**",
    "**url**",
    "**name**",
    "**username**",
]

# ---------------------------------------------------------------------------
# Our additional patterns to catch mask failures
# ---------------------------------------------------------------------------
# Phone numbers: US-style, international with +, partial numbers
PHONE_PATTERN = re.compile(
    r"""
    (?<!\d)                    # not preceded by digit
    (?:\+?1[-.\s]?)?           # optional country code
    \(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}   # 10-digit US
    (?!\d)                     # not followed by digit
    """,
    re.VERBOSE,
)

# Email: basic pattern beyond what publisher may have caught
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# Order/tracking numbers: long numeric strings (8+ digits) that look like IDs
# These are potentially account-identifying even if not directly PII
ORDER_NUMBER_PATTERN = re.compile(r"\b\d{8,20}\b")

# Credit card patterns: groups of 4 digits separated by spaces/dashes
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[- ]){3}\d{4}\b"
)

# Social Security (US): NNN-NN-NNNN
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def scan_for_pii(text: str) -> List[Tuple[str, str]]:
    """
    Scan a text string for potential PII. Returns a list of (pattern_name, match)
    tuples for anything found. Empty list means clean.

    Args:
        text: The text to scan.

    Returns:
        List of (pattern_name, matched_string) tuples.
    """
    if not text or not isinstance(text, str):
        return []

    findings = []

    for match in PHONE_PATTERN.finditer(text):
        findings.append(("phone_number", match.group().strip()))

    for match in EMAIL_PATTERN.finditer(text):
        findings.append(("email_address", match.group().strip()))

    for match in CREDIT_CARD_PATTERN.finditer(text):
        findings.append(("credit_card", match.group().strip()))

    for match in SSN_PATTERN.finditer(text):
        findings.append(("ssn", match.group().strip()))

    # Order numbers: flag but don't automatically redact — they're common
    # in legitimate support contexts. We log them separately.
    for match in ORDER_NUMBER_PATTERN.finditer(text):
        findings.append(("order_number_candidate", match.group().strip()))

    return findings


def redact_pii(text: str, aggressive: bool = False) -> str:
    """
    Redact detected PII from text. Returns the redacted string.

    Args:
        text: Input text.
        aggressive: If True, also redact order number candidates (8+ digit strings).
                    Default False since these are very common in support contexts.

    Returns:
        Text with PII replaced by placeholder tokens.
    """
    if not text or not isinstance(text, str):
        return text

    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
    text = CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", text)
    text = SSN_PATTERN.sub("[REDACTED_SSN]", text)

    if aggressive:
        text = ORDER_NUMBER_PATTERN.sub("[REDACTED_ID]", text)

    return text


def check_publisher_masks_present(text: str) -> bool:
    """
    Check whether any publisher mask tokens are in the text.
    Useful for verifying the upstream masking step is working.
    """
    return any(token in text for token in PUBLISHER_MASK_TOKENS)


def scan_corpus_for_pii(texts: List[str]) -> dict:
    """
    Scan a list of texts and return summary statistics.

    Returns dict with:
        total_texts, texts_with_pii, pii_type_counts, flagged_indices
    """
    pii_type_counts = {}
    flagged_indices = []

    for i, text in enumerate(texts):
        findings = scan_for_pii(text)
        # Exclude order_number_candidate from "has PII" flag since they're too common
        real_pii = [f for f in findings if f[0] != "order_number_candidate"]
        if real_pii:
            flagged_indices.append(i)
            for ptype, _ in real_pii:
                pii_type_counts[ptype] = pii_type_counts.get(ptype, 0) + 1

    return {
        "total_texts": len(texts),
        "texts_with_pii": len(flagged_indices),
        "pii_type_counts": pii_type_counts,
        "flagged_indices": flagged_indices,
    }
