"""
thread_utils.py — Branch-aware Twitter conversation tree reconstruction.

Every "conversation" is a tree rooted at a tweet with no parent.
We walk the tree forward using response_tweet_id (child IDs of each tweet).
A tweet can have multiple children — those are branches.
We flag orphaned references (parent not in the dataset).
"""

import pandas as pd
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build parent→children and child→parent maps from raw dataframe
# ---------------------------------------------------------------------------

def build_adjacency(df: pd.DataFrame) -> Tuple[Dict, Dict, set]:
    """
    Build forward and backward adjacency maps from the raw tweet dataframe.

    Returns:
        parent_to_children: tweet_id -> list of child tweet_ids
        child_to_parent: tweet_id -> parent tweet_id (or None)
        all_ids: set of all tweet_ids in this slice
    """
    all_ids = set(df["tweet_id"].dropna().astype(int).tolist())

    # Forward map: tweet -> its children
    parent_to_children: Dict[int, List[int]] = defaultdict(list)
    # Backward map: tweet -> its parent
    child_to_parent: Dict[int, Optional[int]] = {}

    for _, row in df.iterrows():
        tweet_id = int(row["tweet_id"])
        parent_id = row["in_response_to_tweet_id"]

        if pd.notna(parent_id):
            parent_id = int(parent_id)
            child_to_parent[tweet_id] = parent_id
            parent_to_children[parent_id].append(tweet_id)
        else:
            child_to_parent[tweet_id] = None

    # Also parse response_tweet_id (the comma-separated forward-link column)
    # This is mostly redundant with the above but catches any inconsistencies.
    for _, row in df.iterrows():
        tweet_id = int(row["tweet_id"])
        resp_ids_raw = row.get("response_tweet_id", None)

        if pd.notna(resp_ids_raw):
            resp_ids_str = str(resp_ids_raw).strip()
            if resp_ids_str:
                for rid_str in resp_ids_str.split(","):
                    rid_str = rid_str.strip()
                    if rid_str:
                        try:
                            rid = int(float(rid_str))
                            if rid not in parent_to_children[tweet_id]:
                                parent_to_children[tweet_id].append(rid)
                        except ValueError:
                            pass

    return dict(parent_to_children), child_to_parent, all_ids


def find_thread_roots(df: pd.DataFrame, all_ids: set, brand_id: str) -> List[int]:
    """
    Find thread roots: inbound=True tweets with no parent in the dataset,
    or whose parent is not in all_ids (orphaned start).
    We only consider customer-initiated roots (inbound=True).
    """
    inbound_mask = df["inbound"].astype(str).str.lower().isin(["true", "1"])
    inbound_df = df[inbound_mask]

    roots = []
    for _, row in inbound_df.iterrows():
        tweet_id = int(row["tweet_id"])
        parent_id = row["in_response_to_tweet_id"]

        if pd.isna(parent_id):
            # No parent: clean root
            roots.append(tweet_id)
        else:
            parent_id = int(parent_id)
            if parent_id not in all_ids:
                # Parent referenced but missing from dataset: orphaned root
                roots.append(tweet_id)

    return roots


# ---------------------------------------------------------------------------
# Tree walk: reconstruct each conversation as a list of turns
# ---------------------------------------------------------------------------

def reconstruct_thread(
    root_id: int,
    parent_to_children: Dict[int, List[int]],
    tweet_lookup: Dict[int, dict],
    all_ids: set,
    max_depth: int = 30,
) -> dict:
    """
    Walk the conversation tree from root_id and return a structured object.

    The tree can branch (one customer tweet gets two brand replies).
    Strategy: take the earliest-timestamp branch at each branch point.
    Log all branches so we don't silently lose them.

    Returns a dict:
        root_id, turns (ordered list of tweet dicts), is_orphaned,
        has_branches, branch_count, depth
    """
    turns = []
    branch_count = 0
    is_orphaned = (root_id not in all_ids)

    # BFS queue: (tweet_id, depth)
    queue = [(root_id, 0)]
    visited = set()

    while queue:
        current_id, depth = queue.pop(0)

        if current_id in visited or depth > max_depth:
            continue
        visited.add(current_id)

        if current_id not in tweet_lookup:
            # Referenced but not in dataset
            continue

        tweet = tweet_lookup[current_id].copy()
        tweet["depth"] = depth
        turns.append(tweet)

        children = parent_to_children.get(current_id, [])

        if len(children) > 1:
            branch_count += 1
            # Sort children by timestamp to pick primary branch deterministically
            valid_children = [
                c for c in children if c in tweet_lookup
            ]
            try:
                valid_children.sort(
                    key=lambda c: tweet_lookup[c].get("created_at_parsed", 0)
                )
            except Exception:
                pass  # Fall back to existing order if parsing fails
            # Add primary branch first, then secondaries
            for child in valid_children:
                queue.append((child, depth + 1))
        else:
            for child in children:
                queue.append((child, depth + 1))

    return {
        "root_id": root_id,
        "turns": turns,
        "is_orphaned": is_orphaned,
        "has_branches": branch_count > 0,
        "branch_count": branch_count,
        "depth": max(t["depth"] for t in turns) if turns else 0,
    }


# ---------------------------------------------------------------------------
# High-level: reconstruct all threads for a brand slice
# ---------------------------------------------------------------------------

def reconstruct_all_threads(df: pd.DataFrame, brand_id: str) -> List[dict]:
    """
    Main entry point. Takes a brand-filtered dataframe and returns
    a list of reconstructed thread objects.

    Each thread contains:
        root_id, turns (list of tweet dicts), is_orphaned, has_branches,
        branch_count, depth, has_company_reply, turn_count,
        earliest_timestamp, latest_timestamp
    """
    logger.info(f"Reconstructing threads for brand: {brand_id}, rows: {len(df)}")

    # Parse timestamps for sorting
    df = df.copy()
    df["created_at_parsed"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )

    # Build lookup: tweet_id -> row dict
    tweet_lookup = {}
    for _, row in df.iterrows():
        tweet_id = int(row["tweet_id"])
        tweet_lookup[tweet_id] = {
            "tweet_id": tweet_id,
            "author_id": str(row["author_id"]),
            "inbound": bool(row["inbound"]) if pd.notna(row["inbound"]) else True,
            "created_at": str(row["created_at"]),
            "created_at_parsed": row["created_at_parsed"],
            "text": str(row["text"]) if pd.notna(row["text"]) else "",
            "in_response_to_tweet_id": (
                int(row["in_response_to_tweet_id"])
                if pd.notna(row["in_response_to_tweet_id"])
                else None
            ),
        }

    parent_to_children, child_to_parent, all_ids = build_adjacency(df)
    roots = find_thread_roots(df, all_ids, brand_id)

    logger.info(f"Found {len(roots)} thread roots")

    threads = []
    for root_id in roots:
        thread = reconstruct_thread(
            root_id, parent_to_children, tweet_lookup, all_ids
        )

        # Enrich thread metadata
        turns = thread["turns"]
        if turns:
            timestamps = [
                t["created_at_parsed"]
                for t in turns
                if t["created_at_parsed"] is not pd.NaT
                and t["created_at_parsed"] is not None
            ]
            thread["earliest_timestamp"] = min(timestamps) if timestamps else None
            thread["latest_timestamp"] = max(timestamps) if timestamps else None

            thread["has_company_reply"] = any(
                not t["inbound"] for t in turns
            )
            thread["turn_count"] = len(turns)
            thread["company_turn_count"] = sum(1 for t in turns if not t["inbound"])
            thread["customer_turn_count"] = sum(1 for t in turns if t["inbound"])
        else:
            thread["earliest_timestamp"] = None
            thread["latest_timestamp"] = None
            thread["has_company_reply"] = False
            thread["turn_count"] = 0
            thread["company_turn_count"] = 0
            thread["customer_turn_count"] = 0

        threads.append(thread)

    logger.info(
        f"Reconstructed {len(threads)} threads, "
        f"{sum(t['has_branches'] for t in threads)} with branches, "
        f"{sum(t['is_orphaned'] for t in threads)} orphaned"
    )
    return threads
