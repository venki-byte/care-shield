"""Adversarial Threat Auditor & Safety Intelligence.

Executes scoped adversarial search queries using SerpApi Google Search to discover
CDSCO regulatory warnings, product recall notices, counterfeit seizures, and spurious batch alerts.
Applies false-positive mitigation filters before asserting risk factors.
Strictly distinguishes between live authenticated searches and demo replay.
"""

import re
import logging
from typing import List, Optional

import config
from schemas import SafetyIntelResult

logger = logging.getLogger(__name__)

# High-conviction legal & regulatory alert triggers
HIGH_CONVICTION_PHRASES = [
    "seized by authorities",
    "fake batch",
    "cdsco warning",
    "consumer warning",
    "not of standard quality",
    "spurious drug",
    "spurious batch",
    "counterfeit raid",
    "recalled by manufacturer",
    "alert issued by cdsco",
    "counterfeit medical device",
    "police seized",
    "substandard quality",
    "adulterated",
]


def audit_safety_intel(
    product_title: str,
    detected_brand: str,
    serpapi_key: Optional[str] = None,
    is_demo_mode: bool = False
) -> SafetyIntelResult:
    """Execute adversarial threat search with false-positive mitigation."""
    active_key = serpapi_key or config.SERPAPI_API_KEY

    # 1. Demo Mode strictly tagged
    if is_demo_mode:
        lower_title = product_title.lower()
        if "suport" in lower_title or "spurious" in lower_title or "knee" in lower_title:
            return SafetyIntelResult(
                has_active_recall=True,
                risk_factor=0.85,
                evidence_snippets=[
                    "[DEMO REPLAY FIXTURE] CDSCO State Drug Control Advisory: Spurious batches of orthopedic supports bearing fake license numbers seized in wholesale market raids.",
                    "[DEMO REPLAY FIXTURE] Consumer Safety Warning: Unauthorized imitation products circulating without verified biomechanical load testing."
                ],
                is_live_query=False,
                api_warning="Demo Replay Mode: Using pre-recorded threat intelligence fixture."
            )
        else:
            return SafetyIntelResult(
                has_active_recall=False,
                risk_factor=0.0,
                evidence_snippets=[
                    "[DEMO REPLAY FIXTURE] No active CDSCO alerts or recall notices for verified brand specification."
                ],
                is_live_query=False,
                api_warning="Demo Replay Mode: Using pre-recorded threat intelligence fixture."
            )

    # 2. Live Mode WITHOUT API Key: Honest warning, no fake data!
    if not active_key:
        return SafetyIntelResult(
            has_active_recall=False,
            risk_factor=0.0,
            evidence_snippets=[
                "Adversarial search unexecuted: SERPAPI_API_KEY is not configured in .env or sidebar. Recall and spurious batch intelligence requires API credentials."
            ],
            is_live_query=False,
            api_warning="Live threat search skipped: SERPAPI_API_KEY missing."
        )

    # 3. Live Mode WITH API Key: Real Google Search
    evidence: List[str] = []
    matched_conviction_count = 0
    warning = None

    try:
        from serpapi import GoogleSearch
        query = f'"{detected_brand}" "{product_title}" (counterfeit OR fake OR "recall notice" OR CDSCO OR spurious)'
        params = {
            "engine": "google",
            "q": query,
            "gl": "in",
            "hl": "en",
            "api_key": active_key,
            "num": 10
        }
        print("[SAFETY INTEL] Querying SerpApi Google Search for safety advisories...")

        def _fetch_safety():
            return GoogleSearch(params).get_dict()

        results = config.execute_with_retry(
            _fetch_safety,
            timeout_seconds=15.0,
            max_retries=1,
            fallback={},
            label="Safety Intel Search API"
        )

        if not results:
            warning = "Live threat search timed out or failed after retry."
        elif "error" in results:
            warning = f"SerpApi Google Search returned error: {results['error']}"
        else:
            organic = results.get("organic_results", [])
            for result in organic:
                snippet = result.get("snippet", "")
                title = result.get("title", "")
                full_text = f"{title}. {snippet}".lower()

                for phrase in HIGH_CONVICTION_PHRASES:
                    if phrase in full_text:
                        matched_conviction_count += 1
                        clean_snip = snippet.strip()
                        if clean_snip and clean_snip not in evidence:
                            evidence.append(f"[LIVE SEARCH: {phrase.upper()}]: {clean_snip}")
                        break
    except Exception as e:
        logger.error(f"Safety intel live search failed: {e}")
        warning = f"Live search failed: {e}"

    if matched_conviction_count == 0:
        return SafetyIntelResult(
            has_active_recall=False,
            risk_factor=0.0,
            evidence_snippets=[
                "Live Google Search verified: No active CDSCO warnings or counterfeit seizure reports identified."
            ],
            is_live_query=True,
            api_warning=warning
        )
    else:
        risk = min(1.0, 0.4 + (matched_conviction_count * 0.2))
        return SafetyIntelResult(
            has_active_recall=True,
            risk_factor=round(risk, 2),
            evidence_snippets=evidence[:4],
            is_live_query=True,
            api_warning=warning
        )
