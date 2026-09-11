"""
dedup_utils.py — Exact and near-duplicate tweet detection.

Two kinds of duplicates matter:
1. EXACT duplicates: identical text strings (canned replies sent to many users).
2. NEAR duplicates: same template with minor interpolations (username mentions,
   ticket numbers differ but the rest is identical).

For near-dup detection we use MinHash LSH (datasketch library). This runs
efficiently at millions of rows without pairwise comparison.

Blueprint reference: Section 2.4 "deduplicating exact and near-duplicate text"
and Section 3 risk table row 2.
"""

import re
import logging
import hashlib
from typing import List, Dict, Set, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text normalization for dedup comparison
# ---------------------------------------------------------------------------

# Regex to strip @mentions (anonymized anyway)
MENTION_PATTERN = re.compile(r"@\S+")
# Regex to collapse whitespace
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    """
    Normalize tweet text for deduplication comparison:
    - Lowercase
    - Remove @mentions (they differ between users but carry the same template)
    - Collapse whitespace
    - Strip leading/trailing whitespace

    We DON'T remove punctuation because canned replies often differ only by
    a URL or a brand-specific word — keeping those helps distinguish them.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = MENTION_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def exact_dedup_texts(texts: List[str]) -> Tuple[List[int], Dict[str, List[int]]]:
    """
    Find exact duplicates (after normalization).

    Returns:
        to_keep: indices of the first occurrence of each unique text
        clusters: dict mapping canonical text -> list of all indices with that text
    """
    seen: Dict[str, int] = {}  # canonical text -> first index
    clusters: Dict[str, List[int]] = {}
    to_keep: List[int] = []

    for i, text in enumerate(texts):
        normalized = normalize_for_dedup(text)
        if normalized not in seen:
            seen[normalized] = i
            to_keep.append(i)
            clusters[normalized] = [i]
        else:
            clusters[normalized].append(i)

    return to_keep, clusters


# ---------------------------------------------------------------------------
# MinHash LSH near-duplicate detection
# ---------------------------------------------------------------------------

def _make_shingles(text: str, k: int = 3) -> Set[str]:
    """Create character-level k-shingles from text."""
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def near_dedup_minhash(
    texts: List[str],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> Tuple[List[int], Dict[int, int]]:
    """
    Near-duplicate detection using MinHash LSH.

    Returns:
        canonical_indices: one representative index per cluster
        cluster_map: index -> canonical index for every duplicate
                     (identity mapping for canonical indices)

    Args:
        texts: List of (already normalized) text strings.
        threshold: Jaccard similarity threshold (0.8 = very conservative,
                   catches near-identical templates but not paraphrases).
        num_perm: Number of permutations for MinHash (higher = more accurate,
                  slower). 128 is a good balance at this scale.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.warning("datasketch not installed; falling back to exact dedup only")
        canonical_indices = list(range(len(texts)))
        cluster_map = {i: i for i in range(len(texts))}
        return canonical_indices, cluster_map

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []

    for i, text in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        shingles = _make_shingles(text)
        for shingle in shingles:
            m.update(shingle.encode("utf8"))
        minhashes.append(m)

        try:
            lsh.insert(str(i), m)
        except ValueError:
            # Already inserted (shouldn't happen, but guard for safety)
            pass

    # Find clusters
    cluster_map: Dict[int, int] = {}  # index -> canonical index
    visited: Set[int] = set()
    canonical_indices: List[int] = []

    for i in range(len(texts)):
        if i in visited:
            continue
        # i is canonical
        cluster_map[i] = i
        canonical_indices.append(i)
        visited.add(i)

        # Find near-duplicates
        neighbors = lsh.query(minhashes[i])
        for n_str in neighbors:
            n = int(n_str)
            if n not in visited and n != i:
                cluster_map[n] = i
                visited.add(n)

    logger.info(
        f"Near-dedup: {len(texts)} texts -> {len(canonical_indices)} unique "
        f"(threshold={threshold})"
    )
    return canonical_indices, cluster_map


# ---------------------------------------------------------------------------
# DM-deflection detection (blueprint Section 2.5, 3 row 1)
# ---------------------------------------------------------------------------

DM_DEFLECTION_PATTERNS = [
    r"\bdm\b",
    r"\bdirect\s+message\b",
    r"\bprivate\s+message\b",
    r"\bsend\s+us\s+a\s+message\b",
    r"\bplease\s+message\b",
    r"\bplease\s+reach\s+out\b",
    r"\bcontact\s+us\s+privately\b",
    r"\bprivately\b.*\bassist\b",
]

DM_DEFLECTION_REGEX = re.compile(
    "|".join(DM_DEFLECTION_PATTERNS), re.IGNORECASE
)

# Short-reply threshold: below this, likely an acknowledgment with no content
SHORT_REPLY_THRESHOLD = 30  # characters after stripping mentions


def is_dm_deflection(text: str) -> bool:
    """
    Return True if this company reply is a DM deflection (asks user to DM).
    Used to filter the retrieval corpus.
    """
    if not text:
        return False
    return bool(DM_DEFLECTION_REGEX.search(text))


def is_content_free_reply(text: str) -> bool:
    """
    Return True if a company reply carries essentially no resolution content.
    Includes:
    - Very short replies (acknowledgment only)
    - DM deflections
    - Pure apologies with no actionable content
    """
    if not text:
        return True
    clean = MENTION_PATTERN.sub("", text).strip()
    if len(clean) < SHORT_REPLY_THRESHOLD:
        return True
    if is_dm_deflection(text):
        return True
    # Pure apologies (no actionable words)
    apology_pattern = re.compile(
        r"^[\s@\w]*(sorry|apologize|apologies)[\s.,!]*$", re.IGNORECASE
    )
    if apology_pattern.match(clean):
        return True
    return False


def classify_company_reply(text: str) -> str:
    """
    Classify a company reply into one of three categories:
    - 'dm_deflection': asks user to send a DM
    - 'content_free': short acknowledgment / apology with no resolution
    - 'resolution': potentially contains a useful resolution

    Used for brand scoring (visible resolution rate) and retrieval corpus
    construction.
    """
    if is_dm_deflection(text):
        return "dm_deflection"
    if is_content_free_reply(text):
        return "content_free"
    return "resolution"
