"""Financial Metrology & Google Shopping Parity Checker.

Uses SerpApi Google Shopping (gl=in, hl=en) to query verified Indian pharmacy sources,
filters out accessory/bulk bundle outliers using IQR, performs legal metrology check against
printed MRP, and computes markup deviation against the cleaned online median.

Maintains strict separation between live API calls and demo benchmark playback.
"""

import re
import logging
from typing import Optional, List, Tuple
import numpy as np

import config
from schemas import ShoppingParityResult, BenchmarkMerchant

logger = logging.getLogger(__name__)


def clean_price_string(price_str: str) -> Optional[float]:
    """Parse Indian rupee formatted string into float (e.g. '₹1,499.00' -> 1499.0)."""
    if not price_str:
        return None
    cleaned = re.sub(r"[^\d.]", "", price_str.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def filter_iqr_outliers(prices: List[float]) -> Tuple[List[float], float, float]:
    """Filter out price outliers using Interquartile Range (IQR).

    Eliminates cheap spare parts/accessories (< Q25 - 1.5*IQR)
    and commercial bulk hospital carton bundles (> Q75 + 1.5*IQR).
    """
    if len(prices) < 4:
        min_val = min(prices) if prices else 0.0
        max_val = max(prices) if prices else 0.0
        return prices, min_val, max_val

    arr = np.array(sorted(prices))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr = q75 - q25

    lower_bound = max(0.0, q25 - 1.5 * iqr)
    upper_bound = q75 + 1.5 * iqr

    filtered = [p for p in prices if lower_bound <= p <= upper_bound]
    if not filtered:
        filtered = prices

    return filtered, lower_bound, upper_bound


def get_demo_benchmark_merchants(product_title: str) -> List[BenchmarkMerchant]:
    """Provide reference pharmacy benchmarks strictly for offline demo playback."""
    lower_title = product_title.lower()
    if "oximeter" in lower_title:
        return [
            BenchmarkMerchant(name="Tata 1mg [Demo Ref]", price=1549.0, link="https://www.1mg.com", extracted_title="Dr Trust Signature Series Pulse Oximeter"),
            BenchmarkMerchant(name="Apollo Pharmacy [Demo Ref]", price=1599.0, link="https://www.apollopharmacy.in", extracted_title="Dr Trust USA Finger Tip Pulse Oximeter 506"),
            BenchmarkMerchant(name="Netmeds [Demo Ref]", price=1620.0, link="https://www.netmeds.com", extracted_title="Dr Trust Oximeter 506 Professional"),
            BenchmarkMerchant(name="Amazon India [Demo Ref]", price=1499.0, link="https://www.amazon.in", extracted_title="Dr Trust USA Pulse Oximeter Signature Series"),
            BenchmarkMerchant(name="Pharmeasy [Demo Ref]", price=1580.0, link="https://www.pharmeasy.in", extracted_title="Dr Trust Pulse Oximeter"),
        ]
    else:
        return [
            BenchmarkMerchant(name="Tata 1mg [Demo Ref]", price=620.0, link="https://www.1mg.com", extracted_title="Tynor Functional Knee Support Hinged"),
            BenchmarkMerchant(name="Apollo Pharmacy [Demo Ref]", price=650.0, link="https://www.apollopharmacy.in", extracted_title="Tynor Knee Support Hinged Neoprene"),
            BenchmarkMerchant(name="Netmeds [Demo Ref]", price=635.0, link="https://www.netmeds.com", extracted_title="Tynor Hinged Knee Support Neo"),
            BenchmarkMerchant(name="Amazon India [Demo Ref]", price=599.0, link="https://www.amazon.in", extracted_title="Tynor Knee Support Hinged Neo Premium"),
        ]


def check_price_parity(
    product_title: str,
    detected_brand: str,
    scanned_price: Optional[float] = None,
    printed_mrp_inr: Optional[float] = None,
    serpapi_key: Optional[str] = None,
    is_demo_mode: bool = False
) -> ShoppingParityResult:
    """Execute financial metrology check and Google Shopping parity audit.

    In live mode, queries SerpApi if a key is provided. If no key is provided,
    strictly evaluates legal metrology against printed MRP and flags missing API credentials.
    Adheres strictly to Hard Rules 1, 2, and 3.
    """
    active_key = serpapi_key or config.SERPAPI_KEY
    benchmarks: List[BenchmarkMerchant] = []
    is_live = False
    warning = None

    # Legal Metrology Check: Is scanned asking price strictly above printed MRP?
    # If printed_mrp_inr is None or <= 0, or scanned_price is None, skip check without hallucinating!
    is_above_mrp = False
    mrp_audit_skipped = False
    if scanned_price is not None and printed_mrp_inr is not None and printed_mrp_inr > 0:
        if scanned_price > printed_mrp_inr:
            is_above_mrp = True
    else:
        mrp_audit_skipped = True

    if active_key and not is_demo_mode:
        try:
            from serpapi import GoogleSearch
            query = f"{detected_brand} {product_title}".strip()
            params = {
                "engine": "google_shopping",
                "q": query,
                "gl": "in",
                "hl": "en",
                "api_key": active_key,
                "num": 20
            }
            print(f"[PARITY CHECKER] Querying SerpApi Google Shopping for '{query}'...")

            def _fetch_shopping():
                return GoogleSearch(params).get_dict()

            results = config.execute_with_retry(
                _fetch_shopping,
                timeout_seconds=15.0,
                max_retries=1,
                fallback={},
                label="Google Shopping API"
            )

            if not results:
                warning = "Live Google Shopping call timed out or failed after retry."
            elif "error" in results:
                warning = f"SerpApi Google Shopping returned error: {results['error']}"
            else:
                shopping_items = results.get("shopping_results", [])
                for item in shopping_items:
                    raw_price = item.get("extracted_price") or item.get("price")
                    if isinstance(raw_price, (int, float)):
                        p_val = float(raw_price)
                    elif isinstance(raw_price, str):
                        p_val = clean_price_string(raw_price)
                    else:
                        p_val = None

                    if p_val and p_val > 0:
                        merchant_name = item.get("source", "Verified Online Merchant")
                        benchmarks.append(
                            BenchmarkMerchant(
                                name=merchant_name,
                                price=p_val,
                                link=item.get("link") or item.get("product_link"),
                                extracted_title=item.get("title")
                            )
                        )
                if benchmarks:
                    is_live = True
                else:
                    warning = "Live Google Shopping query returned 0 products for this title in India."
        except Exception as e:
            logger.error(f"SerpApi Google Shopping query failed: {e}")
            warning = f"Live Google Shopping call failed: {e}"

    elif is_demo_mode:
        benchmarks = get_demo_benchmark_merchants(product_title)
        warning = "Demo Replay Mode: Using benchmark fixtures."
    else:
        # Live mode without API key: Do not fake data!
        warning = "Live Google Shopping skipped: SERPAPI_API_KEY is not configured in .env or sidebar. Online median benchmark is unavailable without API credentials."

    if benchmarks:
        all_prices = [b.price for b in benchmarks]
        filtered_prices, lower_bound, upper_bound = filter_iqr_outliers(all_prices)
        online_median = float(np.median(filtered_prices)) if filtered_prices else (printed_mrp_inr or scanned_price)
        if scanned_price is not None and online_median and online_median > 0:
            markup = ((scanned_price - online_median) / online_median) * 100.0
        else:
            markup = 0.0
    else:
        filtered_prices = []
        lower_bound = 0.0
        upper_bound = 0.0
        online_median = None
        markup = 0.0

    return ShoppingParityResult(
        online_prices=filtered_prices,
        online_median=round(online_median, 2) if online_median else None,
        iqr_filtered_min=round(lower_bound, 2),
        iqr_filtered_max=round(upper_bound, 2),
        scanned_purchase_price=round(scanned_price, 2) if scanned_price is not None else None,
        is_above_printed_mrp=is_above_mrp,
        mrp_audit_skipped=mrp_audit_skipped,
        markup_percent=round(markup, 2),
        benchmark_merchants=benchmarks[:8],
        is_live_query=is_live,
        api_warning=warning
    )
