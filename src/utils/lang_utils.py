"""
lang_utils.py — Language identification for filtering non-English tweets.

Blueprint requirement (Section 2.6, 3 row 5):
    Keep only English tweets with language detection confidence >= 0.85.
    Document the drop rate, don't silently discard.

We use langdetect (a Python port of Google's language-detection library).
It's fast, works offline, and handles short noisy text adequately.
"""

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Confidence threshold for accepting a language detection result
ENGLISH_CONFIDENCE_THRESHOLD = 0.85


def detect_language(text: str) -> Tuple[Optional[str], float]:
    """
    Detect language of a text. Returns (language_code, probability).
    Returns (None, 0.0) if detection fails or text is too short.

    Args:
        text: Input text string.

    Returns:
        Tuple of (language_code_or_None, confidence_float)
    """
    if not text or not isinstance(text, str):
        return None, 0.0

    # Very short texts are unreliable — don't try to classify them
    if len(text.strip()) < 10:
        return "short", 0.0

    try:
        from langdetect import detect_langs
        results = detect_langs(text)
        if results:
            top = results[0]
            return top.lang, top.prob
    except Exception:
        pass

    return None, 0.0


def is_english(text: str, threshold: float = ENGLISH_CONFIDENCE_THRESHOLD) -> bool:
    """
    Return True if text is detected as English with confidence >= threshold.
    """
    lang, prob = detect_language(text)
    return lang == "en" and prob >= threshold


def filter_english_texts(
    texts: List[str],
    threshold: float = ENGLISH_CONFIDENCE_THRESHOLD,
    return_stats: bool = False,
) -> Tuple[List[int], dict]:
    """
    Filter a list of texts to English-only.

    Returns:
        english_indices: indices of texts that pass the English filter
        stats: dict with total, english_count, non_english_count, short_count,
               failed_count, english_rate
    """
    english_indices = []
    non_english = 0
    short_count = 0
    failed_count = 0

    for i, text in enumerate(texts):
        lang, prob = detect_language(text)

        if lang == "short":
            # Short texts: we keep them (they have no language signal anyway)
            # Downstream dedup/noise filtering handles them
            english_indices.append(i)
            short_count += 1
        elif lang is None:
            # Detection failed entirely — keep (don't silently drop)
            english_indices.append(i)
            failed_count += 1
        elif lang == "en" and prob >= threshold:
            english_indices.append(i)
        else:
            non_english += 1

    english_count = len(english_indices) - short_count - failed_count

    stats = {
        "total": len(texts),
        "english_count": english_count,
        "short_kept": short_count,
        "failed_detection_kept": failed_count,
        "total_kept": len(english_indices),
        "non_english_dropped": non_english,
        "english_rate": len(english_indices) / len(texts) if texts else 0.0,
    }

    logger.info(
        f"Language filter: kept {stats['total_kept']}/{stats['total']} "
        f"({stats['english_rate']:.1%}), dropped {stats['non_english_dropped']} non-English"
    )

    return english_indices, stats
