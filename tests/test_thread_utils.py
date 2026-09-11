"""
Unit tests for thread reconstruction and branch handling.
"""

import pytest
import pandas as pd
from src.utils.thread_utils import build_adjacency, find_thread_roots, reconstruct_thread


def test_build_adjacency():
    df = pd.DataFrame({
        "tweet_id": [1, 2, 3],
        "in_response_to_tweet_id": [None, 1, 1],
        "response_tweet_id": ["2,3", None, None],
        "inbound": [True, False, False],
        "author_id": ["115657", "SpotifyCares", "SpotifyCares"],
    })

    parent_to_children, child_to_parent, all_ids = build_adjacency(df)

    assert 1 in all_ids
    assert 2 in all_ids
    assert 3 in all_ids
    assert child_to_parent[2] == 1
    assert child_to_parent[3] == 1
    assert set(parent_to_children[1]) == {2, 3}


def test_find_thread_roots():
    df = pd.DataFrame({
        "tweet_id": [10, 11, 12],
        "in_response_to_tweet_id": [None, 10, 999],  # 999 is orphaned
        "inbound": [True, False, True],
        "author_id": ["cust1", "SpotifyCares", "cust2"],
    })
    all_ids = {10, 11, 12}

    roots = find_thread_roots(df, all_ids, "SpotifyCares")
    assert 10 in roots
    assert 12 in roots  # orphaned start
    assert 11 not in roots  # company reply
