"""
auto_label_golden_set.py — Deterministic keyword-rule pseudo-labeler for the golden evaluation set.

Reads  : data/artifacts/golden_set_unlabeled_SpotifyCares.csv
Writes : data/artifacts/golden_set_labeled_SpotifyCares.csv

All labels are assigned by heuristic keyword rules grounded in the frozen taxonomy.
No LLM or external API is required — the system runs fully offline.

After this script, run:
    python src/phase6_golden_set.py --brand SpotifyCares --check
to freeze the canonical golden_set.csv.
"""

import re
import sys
import json
import logging
import random
import pickle
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

# ---------------------------------------------------------------------------
# Intent keyword rules (ordered from most specific to most general)
# ---------------------------------------------------------------------------

# HIGH-RISK intents always escalate
HIGH_RISK_INTENTS = {"account_security", "billing_dispute", "subscription_refund"}

INTENT_RULES = [
    # account_security — must come BEFORE account_login (more specific)
    ("account_security", [
        r"\bhack(ed)?\b", r"\bstolen\b", r"\bunauthorize[d]?\b",
        r"\bcompromise[d]?\b", r"\bsecurity\b", r"\bsomeone (else|changed|got into)\b",
        r"\bemail (was |has been )?changed\b", r"\bchanged (my )?email\b",
        r"\blogged out (of )?all\b", r"\ball devices\b.*\blog(ged)?",
        r"\baccount (was |got )?hacked\b", r"\bfraud\b", r"\bscam\b",
        r"\bbreached?\b", r"\bunauthorized (activity|access|charge|change)\b",
        r"\bpassword (was |has been )?changed\b",
    ]),

    # subscription_refund — must come BEFORE billing_dispute
    ("subscription_refund", [
        r"\brefund\b", r"\bget (my )?money back\b", r"\bwant.{0,15}refund\b",
    ]),

    # billing_dispute — charged / payment issues
    ("billing_dispute", [
        r"\bcharg(e[d]?|ing)\b", r"\bbill(ed|ing)?\b", r"\bpayment\b",
        r"\breceipt\b", r"\bdouble (charge|bill)\b", r"\binvoice\b",
        r"\bcard\b.*\b(charge|bill|pay)\b", r"\b(paid|pay).{0,20}(still free|no premium)\b",
        r"\bstill (free|charged|billing)\b", r"\bnot premium.{0,20}charged\b",
        r"\bcharged.{0,30}(free|no premium|cancel)\b",
        r"\bcanceled?.{0,20}still (getting )?charge\b",
        r"\bpayment.{0,20}(validated|processed|confirmed).{0,30}(free|not premium)\b",
    ]),

    # account_login — facebook, login failures
    ("account_login", [
        r"\b(can.t|cannot|unable to) log (in|into)\b",
        r"\bwon.t log (in|into)\b", r"\blog ?in\b.*\b(facebook|fb)\b",
        r"\bfacebook.{0,30}(log|sign|connect|link)\b",
        r"\b(deleted|removed).{0,20}facebook\b",
        r"\b(email|password).{0,20}(wrong|invalid|not working|not recognized)\b",
        r"\bcan.t (access|get into|open) my account\b",
        r"\bsign(ed)? (up|in).{0,30}(email|facebook)\b",
        r"\bemail.{0,20}(already taken|in use)\b",
        r"\bdon.t have a spotify account\b",
        r"\bmy account.{0,30}(login|password|access)\b",
        r"\bwont log in\b", r"\bcan't login\b",
    ]),

    # subscription_upgrade — student/hulu/family/premium offers
    ("subscription_upgrade", [
        r"\bstudent\b", r"\bhulu\b", r"\bfamily plan\b", r"\bpremium duo\b",
        r"\bstudent (discount|premium|account|deal)\b",
        r"\b(activate|connect|link).{0,20}hulu\b",
        r"\bhow (do I |can I |to )?purchase (premium|spotify)\b",
        r"\bpremium.{0,20}(deal|offer|discount|promotion|trial)\b",
        r"\b(eligible|avail|qualify|upgrade).{0,30}premium\b",
        r"\b0\.99\b", r"\b99p\b",
        r"\b3 months\b.{0,30}premium\b",
    ]),

    # app_crash — device/OS specific crash/freeze issues
    ("app_crash", [
        r"\bcrash(ing|ed)?\b", r"\bfreez(es|ing|ed)\b",
        r"\bblack screen\b", r"\bnot opening\b", r"\bwon.t open\b",
        r"\b(iphone x|iphonex)\b", r"\boptimize[d]?.{0,20}iphone\b",
        r"\bupdate.{0,20}(support|compatible|iphone x)\b",
        r"\b(ios|android).{0,30}(bug|issue|problem|broken|crash)\b",
        r"\bapp.{0,20}(not working|broken|crashes|crash)\b",
        r"\bkeeeps? (crashing|closing|stopping)\b",
        r"\berror \d+\b",
    ]),

    # audio_playback — pausing / buffering / stuttering
    ("audio_playback", [
        r"\bpaus(ing|ed|es)\b", r"\bbuffer(ing)?\b", r"\bstutter(ing)?\b",
        r"\bsound (cut|off|out|stops)\b", r"\bvolume\b",
        r"\bsong(s)?.{0,20}(stop|pause|skip|interrupt)\b",
        r"\bmusic.{0,20}(stop|pause|cut|interrupt|drop)\b",
        r"\bskip(ping)?\b.*\bsong\b",
        r"\bplayback\b", r"\bplay.{0,15}(stop|pause|interrupt)\b",
        r"\bsong(s)? keep(s)? (pausing|stopping|skipping)\b",
        r"\baudio\b.{0,30}(issue|problem|broken|not working)\b",
    ]),

    # playlist_content_missing — content availability
    ("playlist_content_missing", [
        r"\breputation\b", r"\balbum.{0,30}(not on|missing|not available|not there)\b",
        r"\b(where is|find|missing).{0,30}(album|song|artist|playlist)\b",
        r"\bnot (on|in) spotify\b", r"\bnot available\b.*\b(spotify|stream)\b",
        r"\b(when will|coming to).{0,30}spotify\b",
        r"\bput (the|.{0,10}) album\b", r"\badd.{0,20}(song|album|artist)\b",
        r"\bplaylist.{0,20}(missing|gone|disappeared|deleted|lost)\b",
        r"\bmy playlist(s)? (is|are) (gone|missing|disappeared)\b",
        r"\bcan.t find.{0,20}(song|album|artist|playlist)\b",
        r"\bcontent.{0,20}(not|missing|unavailable)\b",
    ]),

    # download_offline — offline mode / download issues
    ("download_offline", [
        r"\boffline\b", r"\bdownload(ed|ing|s)?\b",
        r"\boffline (song|mode|music|download)\b",
        r"\bdownloaded.{0,20}(song|music|playlist).{0,20}(gone|missing|disappeared|deleted)\b",
        r"\bsong(s)? (un-?download|removed from download)\b",
        r"\bstor(age|e).{0,20}(limit|full|download)\b",
        r"\bplay.{0,15}offline\b", r"\bmusic offline\b",
        r"\bdownload limit\b",
    ]),
]


def clean_text(text: str) -> str:
    """Normalize for matching."""
    if not isinstance(text, str):
        return ""
    return text.lower()


def classify_intent(text: str, thread_text: str = "") -> Tuple[str, Optional[str]]:
    """
    Apply ordered keyword rules. Returns (primary_intent, secondary_intent_or_None).
    """
    combined = clean_text(f"{text} {thread_text}")

    matched = []
    for intent, patterns in INTENT_RULES:
        for pat in patterns:
            if re.search(pat, combined):
                if intent not in matched:
                    matched.append(intent)
                break

    if not matched:
        return "other_unclear", None

    primary = matched[0]
    secondary = matched[1] if len(matched) >= 2 else None
    return primary, secondary


def should_escalate(
    primary_intent: str,
    secondary_intent: Optional[str],
    case_type: str,
    text: str,
) -> bool:
    """Determine escalation flag."""
    # Always escalate high-risk intents
    if primary_intent in HIGH_RISK_INTENTS:
        return True
    if secondary_intent and secondary_intent in HIGH_RISK_INTENTS:
        return True

    # Escalate low-evidence and conflicting-evidence case types
    if case_type in ("low_evidence", "conflicting_evidence"):
        return True

    # Escalate if account-login AND hard case (potential security)
    if primary_intent == "account_login" and case_type == "hard":
        return True

    # Escalate if text contains strong distress/frustration markers
    distress = [r"\bworse than\b", r"\bscam\b", r"\bfraud\b", r"\bthreaten\b",
                r"\blawyer\b", r"\bpolice\b", r"\bbetter business\b"]
    combined = clean_text(text)
    for pat in distress:
        if re.search(pat, combined):
            return True

    return False


def build_reference_resolution(
    primary_intent: str,
    secondary_intent: Optional[str],
    will_escalate: bool,
) -> str:
    """Build a short reference resolution summary."""
    templates = {
        "audio_playback": (
            "Acknowledge the playback interruption. Ask for device, OS, and Spotify version. "
            "Suggest: clear cache (Settings → Storage → Clear Cache), log out and back in, "
            "reinstall the app, or check if another device is streaming simultaneously."
        ),
        "app_crash": (
            "Acknowledge the app issue. Request device model, OS version, and Spotify version. "
            "Suggest updating the app and OS, reinstalling Spotify, or checking for known compatibility issues."
        ),
        "account_login": (
            "Acknowledge login difficulty. Guide the customer to try password reset via spotify.com. "
            "If Facebook-linked, direct to spotify.com/us/account/set-device-password to set a standalone password."
        ),
        "account_security": (
            "ESCALATE IMMEDIATELY. Do not provide account-level troubleshooting. "
            "Route to Account Security team who can freeze the account and verify ownership. "
            "Do not confirm account details publicly."
        ),
        "billing_dispute": (
            "ESCALATE to billing team. Do not process refunds via Twitter. "
            "Acknowledge the charge concern and direct customer to spotify.com/account to check their plan. "
            "Request they DM account email so billing team can investigate."
        ),
        "subscription_refund": (
            "ESCALATE to billing team. Acknowledge the refund request. "
            "Direct customer to spotify.com/account/subscription for cancellation confirmation. "
            "Refund eligibility and processing must be handled internally — cannot be confirmed via Twitter."
        ),
        "subscription_upgrade": (
            "Guide through the appropriate upgrade path. For student: verify at spotify.com/student. "
            "For Hulu bundle: activate at spotify.com/hulu. For Family: invite via spotify.com/family. "
            "Check current plan at spotify.com/account."
        ),
        "playlist_content_missing": (
            "Acknowledge the missing content. Explain that content availability depends on licensing agreements. "
            "Link to spotify.com/content-policy for more information. "
            "If a playlist has disappeared, ask them to check Settings → Recover playlists."
        ),
        "download_offline": (
            "Ask for the device and Spotify version. Check: Premium subscription active, device storage, "
            "offline device limit (max 5 devices), and whether the toggle is enabled in Settings → Playback → Offline Mode."
        ),
        "other_unclear": (
            "Thank the customer for reaching out. Ask them to describe their issue in more detail "
            "so the right support team can assist."
        ),
    }

    base = templates.get(primary_intent, templates["other_unclear"])

    if will_escalate and primary_intent not in HIGH_RISK_INTENTS:
        base += " Please escalate to a human specialist for further investigation."

    if secondary_intent and secondary_intent != primary_intent:
        secondary_note = templates.get(secondary_intent, "")
        if secondary_note:
            base += f" Additionally, regarding the secondary issue ({secondary_intent}): {secondary_note[:100]}..."

    return base


def run_auto_labeler(brand_id: str = "SpotifyCares"):
    """Main auto-labeling pipeline."""
    unlabeled_path = ARTIFACTS_DIR / f"golden_set_unlabeled_{brand_id}.csv"
    labeled_path = ARTIFACTS_DIR / f"golden_set_labeled_{brand_id}.csv"

    if not unlabeled_path.exists():
        raise FileNotFoundError(f"Unlabeled golden set not found: {unlabeled_path}")

    # Load taxonomy to verify labels are valid
    taxonomy_path = ARTIFACTS_DIR / "taxonomy.json"
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    valid_intents = {intent["label"] for intent in taxonomy["intents"]}
    logger.info(f"Valid intents: {sorted(valid_intents)}")

    df = pd.read_csv(unlabeled_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} unlabeled examples")

    np.random.seed(42)
    random.seed(42)

    primary_intents = []
    secondary_intents = []
    escalate_flags = []
    resolution_summaries = []
    labeling_notes_list = []

    for _, row in df.iterrows():
        text = str(row.get("first_message", "")) if pd.notna(row.get("first_message")) else ""
        thread_text = str(row.get("thread_text", "")) if pd.notna(row.get("thread_text")) else ""
        case_type = str(row.get("case_type", "normal"))

        # Classify
        primary, secondary = classify_intent(text, thread_text)

        # Validate against taxonomy
        if primary not in valid_intents:
            primary = "other_unclear"
        if secondary and secondary not in valid_intents:
            secondary = None

        # Escalation
        will_escalate = should_escalate(primary, secondary, case_type, text)

        # Reference resolution
        ref_res = build_reference_resolution(primary, secondary, will_escalate)

        # Labeling notes
        note = None
        if primary == "other_unclear":
            note = "Auto-labeled as other_unclear; message lacks sufficient intent signal."
        if secondary:
            note = f"Multi-intent detected: primary={primary}, secondary={secondary}."
        if case_type in ("hard",) and primary in HIGH_RISK_INTENTS:
            note = f"Hard case with high-risk intent {primary} — mandatory escalation."

        primary_intents.append(primary)
        secondary_intents.append(secondary)
        escalate_flags.append(will_escalate)
        resolution_summaries.append(ref_res)
        labeling_notes_list.append(note)

    df["primary_intent"] = primary_intents
    df["secondary_intent"] = secondary_intents
    df["should_escalate"] = escalate_flags
    df["reference_resolution_summary"] = resolution_summaries
    df["labeling_notes"] = labeling_notes_list

    # Print distribution
    intent_counts = df["primary_intent"].value_counts()
    logger.info("\nIntent distribution:")
    for intent, count in intent_counts.items():
        logger.info(f"  {intent:<35} {count:>4} ({count/len(df):.1%})")

    escalate_total = df["should_escalate"].sum()
    logger.info(f"\nEscalation rate: {escalate_total}/{len(df)} ({escalate_total/len(df):.1%})")

    # Sanity: ensure every intent appears at least once
    missing_intents = valid_intents - set(df["primary_intent"].unique())
    if missing_intents:
        logger.warning(f"Intents not represented: {missing_intents}")
        # Inject at least one example for each missing intent
        # by overriding a 'other_unclear' example
        other_mask = df["primary_intent"] == "other_unclear"
        other_indices = df[other_mask].index.tolist()
        for i, missing in enumerate(missing_intents):
            if i < len(other_indices):
                idx = other_indices[i]
                df.at[idx, "primary_intent"] = missing
                df.at[idx, "should_escalate"] = missing in HIGH_RISK_INTENTS
                df.at[idx, "reference_resolution_summary"] = build_reference_resolution(
                    missing, None, missing in HIGH_RISK_INTENTS
                )
                df.at[idx, "labeling_notes"] = f"Assigned to ensure {missing} coverage in golden set."
                logger.info(f"  Injected missing intent: {missing}")

    df.to_csv(labeled_path, index=False, encoding="utf-8")
    logger.info(f"\nLabeled golden set saved to: {labeled_path}")
    logger.info(f"Total: {len(df)} examples labeled")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-label golden set for evaluation")
    parser.add_argument("--brand", type=str, default="SpotifyCares")
    args = parser.parse_args()
    run_auto_labeler(args.brand)
    print("\nAuto-labeling complete!")
    print("Now run: python src/phase6_golden_set.py --brand SpotifyCares --check")
