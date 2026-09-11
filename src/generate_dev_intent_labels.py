"""
generate_dev_intent_labels.py — Assign intent labels to dev messages using keyword rules.

Reads  : data/artifacts/dev_messages_SpotifyCares.pkl
Writes : data/artifacts/dev_intent_labels_SpotifyCares.json

This enables E2 (TF-IDF+LR) and E3 (Embedding k-NN) baselines to train on
labeled dev data rather than falling back to empty training sets.
"""

import re
import sys
import json
import logging
import random
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

# Re-use the same keyword rules as in auto_label_golden_set.py
INTENT_RULES = [
    ("account_security", [
        r"\bhack(ed)?\b", r"\bstolen\b", r"\bunauthorize[d]?\b",
        r"\bcompromise[d]?\b", r"\bsecurity\b",
        r"\bemail (was |has been )?changed\b", r"\bchanged (my )?email\b",
        r"\blogged out (of )?all\b",
        r"\baccount (was |got )?hacked\b", r"\bfraud\b", r"\bscam\b",
        r"\bunauthorized (activity|access|charge|change)\b",
    ]),
    ("subscription_refund", [
        r"\brefund\b", r"\bget (my )?money back\b",
    ]),
    ("billing_dispute", [
        r"\bcharg(e[d]?|ing)\b", r"\bbill(ed|ing)?\b", r"\bpayment\b",
        r"\breceipt\b", r"\bdouble (charge|bill)\b", r"\binvoice\b",
        r"\b(paid|pay).{0,20}(still free|no premium)\b",
        r"\bcharged.{0,30}(free|no premium|cancel)\b",
        r"\bcanceled?.{0,20}still (getting )?charge\b",
    ]),
    ("account_login", [
        r"\b(can.t|cannot|unable to) log (in|into)\b",
        r"\bwon.t log (in|into)\b",
        r"\bfacebook.{0,30}(log|sign|connect|link)\b",
        r"\b(deleted|removed).{0,20}facebook\b",
        r"\bcan.t (access|get into|open) my account\b",
        r"\bwont log in\b", r"\bcan't login\b",
    ]),
    ("subscription_upgrade", [
        r"\bstudent\b", r"\bhulu\b", r"\bfamily plan\b", r"\bpremium duo\b",
        r"\b(eligible|avail|qualify|upgrade).{0,30}premium\b",
        r"\b0\.99\b", r"\b99p\b",
        r"\b3 months\b.{0,30}premium\b",
    ]),
    ("app_crash", [
        r"\bcrash(ing|ed)?\b", r"\bfreez(es|ing|ed)\b",
        r"\b(iphone x|iphonex)\b", r"\boptimize[d]?.{0,20}iphone\b",
        r"\bupdate.{0,20}(support|compatible|iphone x)\b",
        r"\bapp.{0,20}(not working|broken|crashes|crash)\b",
    ]),
    ("audio_playback", [
        r"\bpaus(ing|ed|es)\b", r"\bbuffer(ing)?\b", r"\bstutter(ing)?\b",
        r"\bsound (cut|off|out|stops)\b",
        r"\bmusic.{0,20}(stop|pause|cut|interrupt|drop)\b",
        r"\bplayback\b", r"\baudio\b.{0,30}(issue|problem|broken)\b",
    ]),
    ("playlist_content_missing", [
        r"\breputation\b",
        r"\b(where is|find|missing).{0,30}(album|song|artist|playlist)\b",
        r"\bnot (on|in) spotify\b",
        r"\b(when will|coming to).{0,30}spotify\b",
        r"\badd.{0,20}(song|album|artist)\b",
        r"\bplaylist.{0,20}(missing|gone|disappeared|deleted|lost)\b",
        r"\bmy playlist(s)? (is|are) (gone|missing|disappeared)\b",
    ]),
    ("download_offline", [
        r"\boffline\b", r"\bdownload(ed|ing|s)?\b",
        r"\bdownloaded.{0,20}(song|music|playlist).{0,20}(gone|missing|disappeared)\b",
        r"\bplay.{0,15}offline\b",
    ]),
]


def classify_intent(text: str) -> str:
    t = text.lower()
    for intent, patterns in INTENT_RULES:
        for pat in patterns:
            if re.search(pat, t):
                return intent
    return "other_unclear"


def main(brand_id: str = "SpotifyCares"):
    msgs_path = ARTIFACTS_DIR / f"dev_messages_{brand_id}.pkl"
    if not msgs_path.exists():
        raise FileNotFoundError(f"dev_messages not found: {msgs_path}")

    with open(msgs_path, "rb") as f:
        messages: List[Dict] = pickle.load(f)

    logger.info(f"Loaded {len(messages):,} dev messages")

    labeled = []
    intent_counts: Dict[str, int] = {}
    for msg in messages:
        text = msg.get("text", "")
        intent = classify_intent(text)
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        labeled.append({
            "thread_id": msg.get("thread_id"),
            "tweet_id": msg.get("tweet_id"),
            "text": text,
            "intent": intent,
        })

    logger.info("Intent distribution over dev messages:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {intent:<35} {count:>6} ({count/len(labeled):.1%})")

    out_path = ARTIFACTS_DIR / f"dev_intent_labels_{brand_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False)

    logger.info(f"\nDev intent labels saved to: {out_path}")
    logger.info(f"Total labeled: {len(labeled):,} dev messages")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dev intent labels")
    parser.add_argument("--brand", type=str, default="SpotifyCares")
    args = parser.parse_args()
    main(args.brand)
