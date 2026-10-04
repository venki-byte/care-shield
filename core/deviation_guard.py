"""Regulatory & Text Auditor (Deviation Guard).

1. CDSCO MDR Regex Engine: Evaluates packaging license tokens against official Indian
   Medical Device Rules patterns (MFG/MD/YYYY/XXXXXX and IMP/MD/YYYY/XXXXXX).
2. Canonical Specification Check: Performs token fuzzy matching (Levenshtein similarity)
   against verified commercial brand packaging to flag knockoff typos and subtle spelling alterations.
   Returns exact character/token confidence percentages with strict penalty deduction.
"""

import re
import difflib
import logging
from typing import Optional, Tuple, List

import config

logger = logging.getLogger(__name__)

# Official CDSCO Medical Device Rules (MDR 2017) Regex Formats
CDSCO_MFG_PATTERN = re.compile(r"^MFG/MD/\d{4}/\d{5,6}$", re.IGNORECASE)
CDSCO_IMP_PATTERN = re.compile(r"^IMP/MD/\d{4}/\d{5,6}$", re.IGNORECASE)
CDSCO_LEGACY_PATTERN = re.compile(r"^MFG/(?:[A-Z]{2})/\d{2,4}/\d{4,6}$", re.IGNORECASE)

# Verified Canonical Brand Library for Deterministic Audits
CANONICAL_BRAND_DIRECTORY = {
    "dr trust": {
        "canonical_name": "Dr Trust",
        "canonical_products": [
            "Dr Trust Signature Series Finger Tip Pulse Oximeter 506",
            "Dr Trust USA Professional Finger Tip Pulse Oximeter",
            "Dr Trust Smart Blood Pressure Monitor",
        ],
        "standard_license": "MFG/MD/2021/000123",
        "standard_terms": ["pulse oximeter", "spo2", "pulse rate", "perfusion index", "oled display"]
    },
    "tynor": {
        "canonical_name": "Tynor",
        "canonical_products": [
            "Tynor Knee Support Hinged Neoprene",
            "Tynor Functional Knee Support",
            "Tynor Lumbar Sacro Support Belt",
        ],
        "standard_license": "MFG/MD/2019/000456",
        "standard_terms": ["knee support", "hinged", "neoprene", "orthopedic", "patella"]
    },
    "melactis": {
        "canonical_name": "Melactis",
        "canonical_products": [
            "Melactis Pigment Corrector Serum 30ml",
            "Melactis Serum",
        ],
        "standard_license": None,
        "standard_terms": ["pigment corrector", "serum", "dark spots", "skin tone"]
    },
    "omron": {
        "canonical_name": "Omron",
        "canonical_products": [
            "Omron HEM-7120 Fully Automatic Digital Blood Pressure Monitor",
            "Omron MC-246 Digital Thermometer",
        ],
        "standard_license": "IMP/MD/2020/000789",
        "standard_terms": ["blood pressure monitor", "intellisense", "hypertension"]
    },
    "accu-chek": {
        "canonical_name": "Accu-Chek",
        "canonical_products": [
            "Accu-Chek Active Blood Glucose Glucometer Kit",
            "Accu-Chek Instant Blood Glucose Monitoring System",
        ],
        "standard_license": "IMP/MD/2020/000321",
        "standard_terms": ["blood glucose", "glucometer", "test strips", "mg/dl"]
    }
}


def verify_cdsco_license_format(raw_license: Optional[str]) -> Tuple[Optional[bool], str]:
    """Strictly audit license syntax against CDSCO Medical Device Rules.

    Returns:
        (True, msg) if valid CDSCO syntax.
        (False, msg) if invalid or fabricated license syntax.
        (None, msg) if no license is printed on packaging.
    """
    if not raw_license or not raw_license.strip():
        return None, "No CDSCO/ISO regulatory license printed on packaging (classified as N/A)."

    clean_lic = raw_license.strip().upper()

    if CDSCO_MFG_PATTERN.match(clean_lic):
        return True, f"Verified CDSCO Domestic Manufacturing License format: {clean_lic}"
    if CDSCO_IMP_PATTERN.match(clean_lic):
        return True, f"Verified CDSCO Import Medical Device License format: {clean_lic}"
    if CDSCO_LEGACY_PATTERN.match(clean_lic):
        return True, f"Verified State Licensing Authority format: {clean_lic}"

    return False, f"Malformed regulatory syntax '{clean_lic}'. Does not conform to CDSCO MDR specification (MFG/MD/YYYY/XXXXXX or IMP/MD/YYYY/XXXXXX)."


def fetch_canonical_specs_from_web(brand: str, product_title: str, serpapi_key: Optional[str] = None) -> Optional[str]:
    """Retrieve canonical official product specification via SerpApi Google search if key available."""
    active_key = serpapi_key or config.SERPAPI_KEY
    if not active_key:
        return None

    try:
        from serpapi import GoogleSearch
        q = f'"{brand}" "{product_title}" official site specification'
        params = {
            "engine": "google",
            "q": q,
            "gl": "in",
            "hl": "en",
            "api_key": active_key,
            "num": 3
        }

        def _fetch_specs():
            return GoogleSearch(params).get_dict()

        res = config.execute_with_retry(
            _fetch_specs,
            timeout_seconds=10.0,
            max_retries=1,
            fallback={},
            label="Canonical Specs Search API"
        )
        organic = res.get("organic_results", []) if res else []
        if organic:
            snippets = [r.get("snippet", "") for r in organic if r.get("snippet")]
            return " ".join(snippets)
    except Exception as e:
        logger.warning(f"Error fetching web canonical specs: {e}")

    return None


def audit_barcode_with_serpapi(
    barcode: Optional[str],
    detected_brand: str,
    product_title: str,
    serpapi_key: Optional[str] = None,
    is_demo_mode: bool = False
) -> Tuple[bool, Optional[str], dict]:
    """Audit decoded barcode or QR number against live Google Search (Task T4).
    
    If barcode search results resolve to a completely different brand/product,
    flags as a SOFT warning (is_mismatch = True).
    
    Returns:
        (is_mismatch, warning_message, telemetry)
    """
    if not barcode or not str(barcode).strip():
        return False, None, {
            "status": "SKIPPED",
            "reason": "No barcode detected on packaging",
            "search_query": None,
            "results_count": 0,
            "top_entity_found": "Not provided / Not found"
        }
    
    clean_barcode = str(barcode).strip()
    active_key = serpapi_key or config.SERPAPI_KEY

    # Demo mode handling
    if is_demo_mode:
        if clean_barcode == "8906095170123":
            return False, None, {
                "status": "SUCCESS",
                "is_match": True,
                "reason": "Barcode matches Dr Trust pulse oximeter GTIN",
                "results_count": 5,
                "top_entity_found": "Dr Trust Signature Series Pulse Oximeter 506"
            }
        return False, None, {
            "status": "SUCCESS",
            "is_match": True,
            "reason": "Demo replay mode fixture",
            "results_count": 1,
            "top_entity_found": "Demo Barcode Match"
        }

    if not active_key:
        return False, None, {
            "status": "DEGRADED",
            "reason": "SERPAPI_KEY not configured; live barcode audit skipped",
            "search_query": clean_barcode,
            "results_count": 0,
            "top_entity_found": "Unqueried"
        }

    try:
        print(f"[BARCODE SEARCH] Querying SerpApi for barcode '{clean_barcode}'...")

        params = {
            "engine": "google",
            "q": f'"{clean_barcode}"',
            "gl": "in",
            "hl": "en",
            "num": 5,
            "api_key": active_key
        }

        def _fetch_barcode():
            return GoogleSearch(params).get_dict()

        res = config.execute_with_retry(
            _fetch_barcode,
            timeout_seconds=10.0,
            max_retries=1,
            fallback={},
            label="SerpApi Barcode Search"
        )

        organic = res.get("organic_results", []) if res else []
        if not organic:
            # Try without quotes if quoted query had 0 results
            params["q"] = clean_barcode
            res2 = config.execute_with_retry(
                lambda: GoogleSearch(params).get_dict(),
                timeout_seconds=8.0,
                max_retries=1,
                fallback={},
                label="SerpApi Barcode Search Unquoted"
            )
            organic = res2.get("organic_results", []) if res2 else []

        if not organic:
            return False, None, {
                "status": "SUCCESS",
                "reason": f"No public search results indexed for barcode '{clean_barcode}'",
                "results_count": 0,
                "top_entity_found": "Unindexed Barcode"
            }

        # Check organic result titles and snippets against detected brand and product title
        all_text = " ".join([
            f"{r.get('title', '')} {r.get('snippet', '')}" for r in organic
        ]).lower()

        # Tokenize brand and product
        brand_tokens = [w.lower() for w in re.findall(r"\w+", detected_brand) if len(w) >= 3 and w.lower() not in {"the", "and", "ltd", "pvt", "inc", "brand"}]
        title_tokens = [w.lower() for w in re.findall(r"\w+", product_title) if len(w) >= 4 and w.lower() not in {"series", "smart", "with", "device", "support"}]

        brand_found = any(bt in all_text for bt in brand_tokens) if brand_tokens else False
        title_found = any(tt in all_text for tt in title_tokens) if title_tokens else False

        top_title = organic[0].get("title", "Unknown product")

        if not brand_found and not title_found:
            warning = f"Soft Warning: Barcode '{clean_barcode}' resolves to unrelated online entity ('{top_title[:60]}...') instead of '{detected_brand} {product_title}'."
            print(f"[BARCODE SEARCH] {warning}")
            return True, warning, {
                "status": "MISMATCH_DETECTED",
                "top_entity_found": top_title,
                "results_count": len(organic),
                "matched_brand": False,
                "matched_title": False
            }

        return False, None, {
            "status": "SUCCESS",
            "top_entity_found": top_title,
            "results_count": len(organic),
            "matched_brand": brand_found,
            "matched_title": title_found
        }

    except Exception as e:
        logger.warning(f"Barcode search encountered exception: {e}")
        return False, None, {
            "status": "DEGRADED",
            "reason": str(e),
            "results_count": 0,
            "top_entity_found": "Lookup Error"
        }


def audit_packaging_text_integrity(
    product_title: str,
    detected_brand: str,
    raw_ocr_corpus: str,
    cdsco_license_raw: Optional[str],
    serpapi_key: Optional[str] = None
) -> Tuple[Optional[bool], float, List[str]]:
    """Execute complete regulatory format check and exact token/character fuzzy matching.

    Returns:
        (is_license_format_valid, exact_fuzzy_ratio, evidence_list)
    """
    evidence: List[str] = []

    # 1. License Format Check
    is_lic_valid, lic_msg = verify_cdsco_license_format(cdsco_license_raw)
    evidence.append(lic_msg)

    # 2. Canonical Brand Resolution
    norm_brand = detected_brand.strip().lower()
    canonical_entry = None
    matched_brand_key = None
    for b_key, b_info in CANONICAL_BRAND_DIRECTORY.items():
        # Check direct substring or high character similarity (e.g. 'meloctis' vs 'melactis')
        char_sim = difflib.SequenceMatcher(None, norm_brand, b_key).ratio()
        if b_key in norm_brand or norm_brand in b_key or char_sim >= 0.75:
            canonical_entry = b_info
            matched_brand_key = b_key
            break

    # If brand has slight spelling deviation (e.g. 'meloctis' vs 'melactis'):
    brand_char_ratio = 1.0
    if canonical_entry and matched_brand_key:
        canonical_brand_name = canonical_entry["canonical_name"].lower()
        brand_matcher = difflib.SequenceMatcher(None, norm_brand, canonical_brand_name)
        brand_char_ratio = round(brand_matcher.ratio(), 4)
        if brand_char_ratio < 1.0:
            evidence.append(
                f"Spelling deviation on brand name: '{detected_brand}' vs canonical '{canonical_entry['canonical_name']}' "
                f"({int(brand_char_ratio * 100)}% match - penalty applied)."
            )

    # 3. Product Title / Corpus Fuzzy Match
    if canonical_entry:
        target_canonical_title = canonical_entry["canonical_products"][0]
        title_matcher = difflib.SequenceMatcher(None, product_title.lower(), target_canonical_title.lower())
        title_ratio = round(title_matcher.ratio(), 4)

        # Check for specific word deviations (e.g. 'suport' vs 'support')
        title_words = set(re.findall(r"\w+", product_title.lower()))
        expected_words = set(re.findall(r"\w+", target_canonical_title.lower()))

        for word in title_words:
            if word not in expected_words and len(word) > 4:
                closest = difflib.get_close_matches(word, expected_words, n=1, cutoff=0.70)
                if closest and closest[0] != word:
                    token_ratio = round(difflib.SequenceMatcher(None, word, closest[0]).ratio(), 4)
                    evidence.append(f"Typo/knockoff token detected: '{word}' (canonical: '{closest[0]}', exact token match: {int(token_ratio*100)}%).")

        # Combine brand ratio and title ratio
        exact_fuzzy_ratio = min(brand_char_ratio, title_ratio if title_ratio > 0.4 else brand_char_ratio)
        evidence.append(f"Overall text integrity match: {int(exact_fuzzy_ratio * 100)}% exact confidence.")

    else:
        # Fallback to web search or baseline
        web_specs = fetch_canonical_specs_from_web(detected_brand, product_title, serpapi_key=serpapi_key)
        if web_specs:
            matcher = difflib.SequenceMatcher(None, product_title.lower(), web_specs.lower()[:len(product_title) * 2])
            exact_fuzzy_ratio = min(1.0, max(0.5, round(matcher.ratio() * 1.2, 4)))
            evidence.append(f"Verified against online canonical listings with {int(exact_fuzzy_ratio*100)}% text alignment.")
        else:
            exact_fuzzy_ratio = 0.85
            evidence.append("No canonical catalog entry found; baseline text integrity applied.")

    return is_lic_valid, exact_fuzzy_ratio, evidence
