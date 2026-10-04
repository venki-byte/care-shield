"""Impact extras powered by SerpApi (live only — never fabricated).

- find_jan_aushadhi_kendras(): SerpApi Google Maps -> nearest government generic-medicine
  stores (Pradhan Mantri Bhartiya Janaushadhi Kendra), typically far cheaper than branded retail.
- fetch_brand_news(): SerpApi Google News -> latest recall / spurious-product headlines.
- compute_savings(): pure arithmetic over the existing price-parity result.
"""

import logging
from typing import Any, Dict, List, Optional

import config
from core.map_router import build_directions_url

logger = logging.getLogger(__name__)


def _search(params: Dict[str, Any], label: str) -> Dict[str, Any]:
    def _call():
        from serpapi import GoogleSearch
        return GoogleSearch(params).get_dict()
    return config.execute_with_retry(_call, timeout_seconds=15.0, max_retries=1, fallback={}, label=label) or {}


def find_jan_aushadhi_kendras(lat: float, lng: float, city: str, api_key: Optional[str]) -> List[Dict[str, Any]]:
    """Nearest Jan Aushadhi Kendras via SerpApi Google Maps. Empty list when unavailable."""
    if not api_key:
        return []
    res = _search({
        "engine": "google_maps",
        "q": "Jan Aushadhi Kendra",
        "ll": f"@{lat},{lng},13z",
        "type": "search",
        "hl": "en",
        "api_key": api_key,
    }, "Jan Aushadhi Maps")
    out = []
    for item in (res.get("local_results") or [])[:6]:
        gps = item.get("gps_coordinates") or {}
        out.append({
            "name": item.get("title", "Jan Aushadhi Kendra"),
            "address": item.get("address", f"{city}, India"),
            "rating": item.get("rating"),
            "reviews": item.get("reviews"),
            "open_state": item.get("open_state"),
            "directions_url": build_directions_url(
                item.get("title", ""), item.get("place_id"), gps.get("latitude"), gps.get("longitude"), city),
        })
    return out[:3]


def fetch_brand_news(brand: str, title: str, api_key: Optional[str]) -> List[Dict[str, str]]:
    """Latest recall / spurious headlines via SerpApi Google News. Empty list when unavailable."""
    if not api_key:
        return []
    subject = (brand or "").strip()
    if not subject or subject.lower().startswith(("not provided", "unknown")):
        subject = " ".join((title or "").split()[:3])
    if not subject:
        return []
    res = _search({
        "engine": "google_news",
        "q": f'"{subject}" (recall OR spurious OR counterfeit OR CDSCO OR overcharging)',
        "gl": "in",
        "hl": "en",
        "api_key": api_key,
    }, "Google News")
    out = []
    for n in (res.get("news_results") or [])[:3]:
        src = n.get("source")
        out.append({
            "title": n.get("title", ""),
            "link": n.get("link", ""),
            "source": (src.get("name") if isinstance(src, dict) else src) or "News",
            "date": n.get("date", ""),
        })
    return [n for n in out if n["title"] and n["link"]]


def compute_savings(store_price: Optional[float], printed_mrp: Optional[float],
                    online_median: Optional[float], cheapest_online: Optional[float]) -> Dict[str, Optional[float]]:
    """Rupee impact. Only computes figures that are backed by real inputs."""
    overcharge = None
    if store_price and printed_mrp and store_price > printed_mrp:
        overcharge = round(store_price - printed_mrp, 2)
    save_vs_cheapest = None
    # Guard against junk listings (e.g. accessories at Rs 1): cheapest must be a plausible
    # price for this product, i.e. at least 40% of the online median.
    plausible = bool(online_median) and cheapest_online is not None and cheapest_online >= 0.4 * online_median
    if store_price and cheapest_online and plausible and store_price > cheapest_online:
        save_vs_cheapest = round(store_price - cheapest_online, 2)
    save_pct = round(100 * save_vs_cheapest / store_price, 1) if save_vs_cheapest else None
    return {"illegal_overcharge": overcharge, "save_vs_cheapest": save_vs_cheapest, "save_pct": save_pct}
