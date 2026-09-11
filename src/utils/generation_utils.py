"""
generation_utils.py — Structured response generation with evidence citations and groundedness check.

Blueprint Section 11, Roadmap Phase 9.
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Any
from urllib import request, error

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful, accurate, and concise customer support assistant for a brand.
Your responses MUST be strictly grounded in the provided historical resolutions.
RULES:
1. Every actionable instruction or factual statement must be cited with the evidence tag [doc_X] (e.g., [doc_1]).
2. Do not invent troubleshooting steps or policies not found in the evidence.
3. If the retrieved evidence does NOT contain enough information to resolve the issue, or if the issue requires private account inspection (e.g. password reset, card charge dispute), set "missing_information_flag" to true and recommend escalation.
4. Always output valid JSON adhering exactly to the requested schema.
"""

GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "draft_reply": {"type": "string"},
        "cited_doc_ids": {"type": "array", "items": {"type": "integer"}},
        "groundedness_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "missing_information_flag": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["draft_reply", "cited_doc_ids", "groundedness_score", "missing_information_flag", "reasoning"],
}


def build_generation_prompt(
    customer_text: str,
    predicted_intent: str,
    retrieved_pairs: List[Dict],
) -> str:
    """Builds prompt containing customer query and numbered retrieval evidence."""
    evidence_text = ""
    for idx, pair in enumerate(retrieved_pairs, 1):
        problem = pair.get("customer_problem_text", "").strip()
        res = pair.get("brand_resolution_text", "").strip()
        evidence_text += f"[doc_{idx}] Problem: {problem}\nResolution: {res}\n\n"

    if not evidence_text.strip():
        evidence_text = "No historical resolutions found for this query.\n"

    user_content = f"""Customer Query:
"{customer_text}"

Identified Intent: {predicted_intent}

Retrieved Historical Resolutions:
{evidence_text}

Instructions:
Draft a response in the requested JSON format:
{{
  "draft_reply": "Your clear, empathetic, and concise reply citing [doc_X]",
  "cited_doc_ids": [list of 1-based doc numbers cited in the reply],
  "groundedness_score": float between 0.0 and 1.0,
  "missing_information_flag": true or false,
  "reasoning": "Brief explanation of how the evidence supports the reply"
}}
"""
    return user_content


def verify_groundedness_and_citations(
    draft_reply: str,
    cited_doc_ids: List[int],
    retrieved_pairs: List[Dict],
) -> Dict[str, Any]:
    """
    Validates citation consistency:
    - Checks that cited doc IDs exist in the retrieved pool
    - Checks whether citations actually appear in text ([doc_X])
    - Checks overlap between cited evidence tokens and reply tokens
    """
    found_tags = [int(m) for m in re.findall(r"\[doc_(\d+)\]", draft_reply)]
    valid_doc_count = len(retrieved_pairs)
    valid_cited = [d for d in cited_doc_ids if 1 <= d <= valid_doc_count]

    # Citation consistency
    has_valid_citations = len(valid_cited) > 0 and set(found_tags) == set(valid_cited)

    # Word overlap with cited documents
    overlap_score = 0.0
    if valid_cited:
        reply_words = set(re.findall(r"\b[a-z]{3,}\b", draft_reply.lower()))
        evidence_words = set()
        for doc_id in valid_cited:
            pair = retrieved_pairs[doc_id - 1]
            text = (pair.get("brand_resolution_text", "") + " " + pair.get("customer_problem_text", "")).lower()
            evidence_words.update(re.findall(r"\b[a-z]{3,}\b", text))

        if reply_words and evidence_words:
            overlap = len(reply_words & evidence_words) / len(reply_words)
            overlap_score = min(1.0, overlap * 1.5)

    citation_theater = len(found_tags) > 0 and overlap_score < 0.15

    return {
        "valid_citations": valid_cited,
        "citation_tags_in_text": found_tags,
        "citation_consistent": has_valid_citations,
        "evidence_overlap_score": round(overlap_score, 3),
        "citation_theater_detected": citation_theater,
    }


def call_llm_api(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Optional[str]:
    """Calls OpenAI or OpenRouter API if an API key is available."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    if not api_key:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if "openrouter.ai" in base_url:
        headers["HTTP-Referer"] = "https://github.com/hiver-assignment"
        headers["X-Title"] = "Hiver Customer Support Assistant"

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    try:
        req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"LLM API call failed: {e}. Falling back to deterministic generator.")
        return None


def fallback_deterministic_generator(
    customer_text: str,
    predicted_intent: str,
    retrieved_pairs: List[Dict],
) -> Dict[str, Any]:
    """
    Deterministic offline response generator adhering to the exact schema.
    Used for offline reproduction or when no API key is provided.
    """
    if not retrieved_pairs:
        return {
            "draft_reply": "I'm sorry to hear you're experiencing this issue. Let me route you to a support specialist who can investigate your account directly.",
            "cited_doc_ids": [],
            "groundedness_score": 0.0,
            "missing_information_flag": True,
            "reasoning": "No relevant historical resolutions were found in the retrieval index.",
        }

    best_pair = retrieved_pairs[0]
    best_resolution = best_pair.get("brand_resolution_text", "").strip()

    # If the resolution is too brief or asks for DM, flag missing information
    if "dm" in best_resolution.lower() or len(best_resolution) < 20:
        return {
            "draft_reply": f"Hi there, this issue may require verifying your account details. [doc_1] We will escalate this to a customer specialist.",
            "cited_doc_ids": [1],
            "groundedness_score": 0.60,
            "missing_information_flag": True,
            "reasoning": "Historical resolution defers to private channel; requires human specialist.",
        }

    reply = f"Hi! Based on verified troubleshooting steps: {best_resolution} [doc_1]. Please let us know if that helps!"
    return {
        "draft_reply": reply,
        "cited_doc_ids": [1],
        "groundedness_score": 0.90,
        "missing_information_flag": False,
        "reasoning": "Grounded directly in highest-ranking historical resolution pair.",
    }


def generate_structured_response(
    customer_text: str,
    predicted_intent: str,
    retrieved_pairs: List[Dict],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Generates a structured response object with validation & groundedness check.
    """
    prompt = build_generation_prompt(customer_text, predicted_intent, retrieved_pairs)
    api_response = call_llm_api(prompt, model=model)

    result = None
    if api_response:
        try:
            parsed = json.loads(api_response)
            if all(k in parsed for k in ["draft_reply", "cited_doc_ids", "groundedness_score", "missing_information_flag"]):
                result = parsed
        except Exception as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")

    if not result:
        result = fallback_deterministic_generator(customer_text, predicted_intent, retrieved_pairs)

    # Run groundedness verification
    audit = verify_groundedness_and_citations(
        result["draft_reply"],
        result["cited_doc_ids"],
        retrieved_pairs,
    )
    result["groundedness_audit"] = audit

    # If citation theater or inconsistent, adjust groundedness
    if audit["citation_theater_detected"]:
        result["groundedness_score"] = min(result["groundedness_score"], 0.3)

    return result
