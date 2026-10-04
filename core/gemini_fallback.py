"""Gemini fallback for SerpApi.

When SerpApi is unavailable (quota exhausted, error, timeout or no key), Care-Shield switches
to Gemini with **Google Search grounding**. Grounded answers are built from real web pages and
come with source URLs, so the zero-hallucination rule still holds: we only keep items the model
returns together with a price / source it found, and we label them as fallback in the UI.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import config
from schemas import BenchmarkMerchant

logger = logging.getLogger(__name__)

_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]


def _grounded(prompt: str, api_key: str) -> Tuple[str, List[Dict[str, str]]]:
    """Run a Google-Search-grounded Gemini call. Returns (text, [{title, uri}])."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
    last_err: Optional[Exception] = None
    for model in _MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            sources: List[Dict[str, str]] = []
            try:
                for ch in resp.candidates[0].grounding_metadata.grounding_chunks or []:
                    web = getattr(ch, "web", None)
                    if web and getattr(web, "uri", None):
                        sources.append({"title": getattr(web, "title", "") or "", "uri": web.uri})
            except Exception:
                pass
            return (resp.text or ""), sources
        except Exception as e:  # try the next model
            last_err = e
            logger.warning(f"Gemini grounded model {model} failed: {e}")
    raise RuntimeError(f"Gemini grounded search failed: {last_err}")


def _json_from(text: str) -> Any:
    cleaned = text.strip()
    m = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _run(prompt: str, api_key: str, label: str):
    return config.execute_with_retry(
        lambda: _grounded(prompt, api_key),
        timeout_seconds=40.0, max_retries=0, fallback=None, label=label,
    )


def gemini_price_benchmark(query: str, api_key: Optional[str]) -> List[BenchmarkMerchant]:
    """Current Indian online prices for a product, via grounded Gemini search."""
    if not api_key or not query.strip():
        return []
    prompt = (
        f"Search the web for the current selling price in India (INR) of: {query}.\n"
        "Use Indian online sellers (Amazon.in, Flipkart, Tata 1mg, Apollo Pharmacy, Netmeds, PharmEasy, etc).\n"
        "Return ONLY a JSON array of up to 6 objects: "
        '[{"seller": "...", "price_inr": 1234, "title": "listing title"}]. '
        "Include ONLY prices you actually found on a web page for this exact product. "
        "If you find none, return []. No commentary."
    )
    out = _run(prompt, api_key, "Gemini grounded prices")
    if not out:
        return []
    text, sources = out
    data = _json_from(text)
    merchants: List[BenchmarkMerchant] = []
    if not isinstance(data, list):
        return []
    for row in data:
        try:
            price = float(str(row.get("price_inr")).replace(",", ""))
            seller = str(row.get("seller") or "Online seller").strip()
        except Exception:
            continue
        if price <= 0:
            continue
        key = re.sub(r"[^a-z0-9]", "", seller.lower())[:6]
        link = next((s["uri"] for s in sources if key and key in re.sub(r"[^a-z0-9]", "", (s["title"] + s["uri"]).lower())), None)
        merchants.append(BenchmarkMerchant(name=seller, price=price, link=link, extracted_title=str(row.get("title") or "")))
    return merchants


def gemini_safety_intel(brand: str, title: str, api_key: Optional[str]) -> Tuple[List[str], bool]:
    """Recall / counterfeit warnings via grounded Gemini search. Returns (snippets, has_alert)."""
    if not api_key:
        return [], False
    prompt = (
        f"Search the web for any recall, spurious/counterfeit warning, CDSCO or state drug-control alert, "
        f"or seizure news about the product '{title}' by brand '{brand}' in India.\n"
        'Return ONLY JSON: {"has_alert": true/false, "findings": ["one-sentence finding with source name", ...]}. '
        "Only report alerts that are explicitly about this brand/product. If none, has_alert=false and findings=[]."
    )
    out = _run(prompt, api_key, "Gemini grounded safety")
    if not out:
        return [], False
    data = _json_from(out[0])
    if not isinstance(data, dict):
        return [], False
    findings = [str(f) for f in (data.get("findings") or []) if str(f).strip()][:4]
    return findings, bool(data.get("has_alert")) and bool(findings)


def gemini_news(subject: str, api_key: Optional[str]) -> List[Dict[str, str]]:
    """Recent recall / spurious headlines via grounded Gemini search."""
    if not api_key or not subject.strip():
        return []
    prompt = (
        f"Find up to 3 recent news articles about recalls, spurious/counterfeit products, or overcharging "
        f"related to '{subject}' in India. Return ONLY a JSON array: "
        '[{"title": "...", "source": "publisher", "url": "https://..."}]. If none, return [].'
    )
    out = _run(prompt, api_key, "Gemini grounded news")
    if not out:
        return []
    data = _json_from(out[0])
    if not isinstance(data, list):
        return []
    return [
        {"title": str(n.get("title", "")), "link": str(n.get("url", "")), "source": str(n.get("source", "News")), "date": ""}
        for n in data[:3] if n.get("title") and str(n.get("url", "")).startswith("http")
    ]


def gemini_price_estimate(query: str, api_key: Optional[str]) -> Optional[str]:
    """Last-resort, clearly-labelled AI estimate (NOT live data, never used for scoring)."""
    if not api_key or not query.strip():
        return None
    from google import genai

    def _call():
        client = genai.Client(api_key=api_key)
        for model in _MODELS:
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=(f"In one short sentence, what is the typical retail price range in India (INR) of: {query}? "
                              "If you are not reasonably sure, reply exactly: UNKNOWN"),
                )
                return (r.text or "").strip()
            except Exception as e:
                logger.warning(f"Gemini estimate model {model} failed: {e}")
        return None

    text = config.execute_with_retry(_call, timeout_seconds=20.0, max_retries=0, fallback=None, label="Gemini price estimate")
    return None if not text or "UNKNOWN" in text.upper() else text
