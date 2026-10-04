"""Deterministic Scoring Engine (Pure Functions without Network Calls).

Task T5:
1. text_integrity (Max 35): Exact token match count between OCR and canonical specs.
   If < 3 meaningful tokens match -> status 'warn'. Exact character similarity (no rounding).
2. license_format (Max 25): Pure regex check. Returns 'na' if absent or cosmetic.
   Output text: "License format looks valid/invalid".
3. price (Max 20): Primary check vs printed MRP (fail if > MRP).
   Secondary check vs online median: penalty = clamp((price - median*1.25) / (median*0.5), 0, 1).
   If price missing -> 'na'.
4. fraud_intel (Max 20): Brand + product + fraud keyword. Zero hits -> status 'ok', "No warning signs found".
5. Gate Logic: trigger_map_remediation = (Score < 70) or (price > MRP) or (fraud_intel fails).
"""

import re
import difflib
from typing import List, Optional, Tuple, Dict, Any

import config
from schemas import (
    PackagingOCRResult,
    ShoppingParityResult,
    SafetyIntelResult,
    NearbyPharmacy,
    SubScores,
    AuditScorecard,
)

# CDSCO Medical Device Rules Regex
CDSCO_REGEX = re.compile(
    r"^(?:MFG|IMP)/MD/\d{4}/\d{5,6}$|^MFG/[A-Z]{2}/\d{2,4}/\d{4,6}$",
    re.IGNORECASE
)

# Common non-distinctive stop words
STOPWORDS = {"the", "and", "for", "with", "ltd", "pvt", "inc", "pack", "series", "brand", "item", "medical"}


def compute_text_integrity_subscore(
    ocr_title: str,
    ocr_brand: str,
    canonical_title: Optional[str] = None,
    canonical_brand: Optional[str] = None,
) -> Tuple[float, float, str, int, float, str]:
    """Task T5.1: Pure text integrity sub-score calculation (Max 35).
    
    Returns:
        (earned_points, max_points, status, token_match_count, real_similarity, reason)
    """
    max_pts = 35.0
    combined_ocr = f"{ocr_brand} {ocr_title}".strip().lower()
    
    # Meaningful OCR tokens (length >= 3 and not in stopwords)
    ocr_tokens = [w for w in re.findall(r"\w+", combined_ocr) if len(w) >= 3 and w not in STOPWORDS]
    
    # Reference canonical text
    target_canonical = canonical_title or ocr_title
    target_brand = canonical_brand or ocr_brand
    combined_canonical = f"{target_brand} {target_canonical}".strip().lower()
    canonical_tokens = [w for w in re.findall(r"\w+", combined_canonical) if len(w) >= 3 and w not in STOPWORDS]

    # Exact token match count
    matching_tokens = [w for w in ocr_tokens if w in canonical_tokens]
    token_match_count = len(set(matching_tokens))

    # Exact brand similarity (e.g. 'meloctis' vs 'melactis' = 0.875)
    brand_sim = 1.0
    if ocr_brand and target_brand:
        brand_sim = difflib.SequenceMatcher(None, ocr_brand.strip().lower(), target_brand.strip().lower()).ratio()

    # Title similarity
    title_sim = 1.0
    if ocr_title and target_canonical:
        raw_title_sim = difflib.SequenceMatcher(None, ocr_title.strip().lower(), target_canonical.strip().lower()).ratio()
        if raw_title_sim < 0.4 and brand_sim < 1.0:
            title_sim = brand_sim
        else:
            title_sim = raw_title_sim

    sim = min(brand_sim, title_sim)

    # Status determination: if < 3 meaningful tokens match -> status 'warn'
    if token_match_count < 3:
        status = "warn"
        earned = round(min(max_pts * sim, 24.5), 2)
        reason = f"Token deviation: Only {token_match_count} meaningful tokens matched canonical specification (Status: WARN, similarity: {sim*100:.1f}%)."
    else:
        status = "ok"
        earned = round(max_pts * sim, 2)
        reason = f"Text integrity confirmed: {token_match_count} tokens matched canonical specification (Confidence: {sim*100:.1f}%)."

    return earned, max_pts, status, token_match_count, round(sim, 4), reason


def compute_license_format_subscore(
    raw_license: Optional[str],
    product_category: str = "otc_device"
) -> Tuple[Optional[float], float, str, str]:
    """Task T5.2 & T6: Pure regex license format check (Max 25).
    
    If cosmetic, skips CDSCO MDR format check and sets to 'na'.
    Returns:
        (earned_points, max_points, status, output_text)
    """
    if product_category == "cosmetic":
        return None, 0.0, "na", "License format not evaluated: Cosmetics are exempt from CDSCO Medical Device Rules"

    if not raw_license or not raw_license.strip():
        return None, 0.0, "na", "License format not provided / not required"

    clean_lic = raw_license.strip().upper()
    if CDSCO_REGEX.match(clean_lic):
        return 25.0, 25.0, "valid", "License format looks valid"
    else:
        return 5.0, 25.0, "invalid", "License format looks invalid"


def compute_pricing_subscore(
    store_price: Optional[float],
    printed_mrp: Optional[float],
    online_median: Optional[float]
) -> Tuple[Optional[float], float, str, bool, str]:
    """Task T5.3: Pure pricing fairness sub-score (Max 20).
    
    Primary: Store price vs Printed MRP (fail if > MRP).
    Secondary: penalty = clamp((price - median*1.25) / (median*0.5), 0, 1).
    Returns:
        (earned_points, max_points, status, is_above_mrp, reason)
    """
    max_pts = 20.0

    if store_price is None or store_price <= 0.0:
        return None, 0.0, "na", False, "Store Asking Price missing (N/A)"

    # Primary check: Store price vs Printed MRP
    if printed_mrp is not None and printed_mrp > 0.0 and store_price > printed_mrp:
        return 0.0, max_pts, "fail", True, f"Illegal overcharge: Store price ₹{store_price:,.2f} exceeds printed MRP ₹{printed_mrp:,.2f} (Legal Metrology Violation)"

    # Secondary check: Store price vs Online Median
    if online_median is not None and online_median > 0.0:
        # Task T5 exact formula: penalty only begins at +25% over median
        raw_penalty = (store_price - (online_median * 1.25)) / (online_median * 0.5)
        penalty = max(0.0, min(1.0, raw_penalty))
        earned = round(max_pts * (1.0 - penalty), 2)

        if penalty == 0.0:
            status = "ok"
            markup_pct = ((store_price - online_median) / online_median) * 100.0
            reason = f"Store price markup ({markup_pct:+.1f}%) within 25% tolerance of online median ₹{online_median:,.2f}"
        elif penalty < 1.0:
            status = "warn"
            markup_pct = ((store_price - online_median) / online_median) * 100.0
            reason = f"High markup detected: Store price {markup_pct:+.1f}% above online median ₹{online_median:,.2f} (Penalty: {penalty:.2f})"
        else:
            status = "fail"
            markup_pct = ((store_price - online_median) / online_median) * 100.0
            reason = f"Extreme price gouging: Store price {markup_pct:+.1f}% exceeds median + 75% limit"

        return earned, max_pts, status, False, reason

    # Fallback to printed MRP if online median unqueried
    if printed_mrp is not None and printed_mrp > 0.0:
        earned = max_pts if store_price <= printed_mrp else 0.0
        status = "ok" if store_price <= printed_mrp else "fail"
        return earned, max_pts, status, False, f"Store price complies with printed MRP ₹{printed_mrp:,.2f}"

    return None, 0.0, "na", False, "Printed MRP and online benchmarks unstated (N/A)"


def compute_fraud_intel_subscore(
    has_active_recall: bool,
    risk_factor: float,
    snippets: List[str]
) -> Tuple[float, float, str, str]:
    """Task T5.4: Pure adversarial fraud intelligence sub-score (Max 20).
    
    Zero hits = 'No warning signs found' (status: 'ok', confidence <= 0.6).
    Returns:
        (earned_points, max_points, status, message)
    """
    max_pts = 20.0

    if has_active_recall or risk_factor > 0.0:
        penalty = min(1.0, max(0.2, risk_factor if risk_factor > 0.0 else 0.5))
        earned = round(max_pts * (1.0 - penalty), 2)
        status = "fail" if penalty >= 0.7 else "warn"
        message = f"Adversarial alert: Threat notifications identified (conviction: {penalty:.2f})"
        return earned, max_pts, status, message

    # Zero hits -> 'No warning signs found' (soft signal, status ok)
    return 20.0, max_pts, "ok", "No warning signs found"


def compute_audit_scorecard(
    ocr_result: PackagingOCRResult,
    canonical_title: Optional[str] = None,
    canonical_brand: Optional[str] = None,
    parity_result: Optional[ShoppingParityResult] = None,
    intel_result: Optional[SafetyIntelResult] = None,
    barcode_mismatch: bool = False,
    fallback_pharmacies: Optional[List[NearbyPharmacy]] = None,
    is_demo_mode: bool = False
) -> AuditScorecard:
    """Master pure scoring function (no network calls, fully unit-testable).
    
    Task T5:
    1. Computes all 4 sub-scores with exact mathematical formulas.
    2. Dynamically recalculates total denominator base.
    3. Evaluates Gate Logic for map router trigger:
       trigger_map_remediation = (Score < 70) or (price > MRP) or (fraud_intel fails).
    """
    evidence: List[str] = []

    # 1. Text Integrity Sub-score (Max 35)
    t_earned, t_max, t_stat, t_match_cnt, t_sim, t_reason = compute_text_integrity_subscore(
        ocr_title=ocr_result.product_title,
        ocr_brand=ocr_result.detected_brand,
        canonical_title=canonical_title,
        canonical_brand=canonical_brand
    )
    evidence.append(t_reason)

    # 2. Regulatory Sub-score (Max 25 or N/A)
    raw_lic = ocr_result.cdsco_license_raw
    cat = getattr(ocr_result, "product_category", "otc_device")
    r_earned, r_max, r_stat, r_msg = compute_license_format_subscore(raw_lic, product_category=cat)
    if cat == "cosmetic":
        evidence.append("Regulatory License: N/A (Cosmetics are exempt from CDSCO Medical Device Rules; excluded from score denominator base).")
    elif r_stat == "na":
        evidence.append("Regulatory License: N/A (Not provided / not required; excluded from score denominator base).")
    else:
        evidence.append(f"Regulatory License: {r_msg} ({r_earned}/{r_max} pts).")

    # 3. Pricing Fairness Sub-score (Max 20 or N/A)
    store_price = parity_result.scanned_purchase_price if parity_result else None
    printed_mrp = ocr_result.printed_mrp_inr
    online_median = parity_result.online_median if parity_result else None

    p_earned, p_max, p_stat, is_above_mrp, p_reason = compute_pricing_subscore(
        store_price=store_price,
        printed_mrp=printed_mrp,
        online_median=online_median
    )
    evidence.append(p_reason)

    # 4. Fraud Intel Sub-score (Max 20)
    has_recall = intel_result.has_active_recall if intel_result else False
    risk_factor = intel_result.risk_factor if intel_result else 0.0
    snippets = intel_result.evidence_snippets if intel_result else []

    f_earned, f_max, f_stat, f_msg = compute_fraud_intel_subscore(
        has_active_recall=has_recall,
        risk_factor=risk_factor,
        snippets=snippets
    )
    evidence.append(f_msg)

    # Barcode mismatch soft warning (Task T4)
    barcode_warning_text = None
    if barcode_mismatch:
        barcode_warning_text = "Soft Warning: Barcode resolution conflicts with packaging text identity."
        evidence.append(barcode_warning_text)

    # Dynamic Denominator Base & Composite Normalization
    max_available_base = t_max + r_max + p_max + f_max
    total_earned = t_earned + (r_earned if r_earned is not None else 0.0) + (p_earned if p_earned is not None else 0.0) + f_earned

    if max_available_base > 0:
        composite_trust_score = int(round((total_earned / max_available_base) * 100.0))
    else:
        composite_trust_score = 0

    composite_trust_score = max(0, min(100, composite_trust_score))

    # Task T5.5: Gate Logic
    fraud_intel_failed = (f_stat == "fail") or (risk_factor >= 0.70)
    should_trigger_map = (composite_trust_score < 70) or is_above_mrp or fraud_intel_failed

    # Defensible Phrasing Verdict
    if is_above_mrp:
        verdict = "Illegal MRP Overcharge Flagged"
    elif composite_trust_score >= 80:
        if r_stat == "na":
            verdict = "No red flags found. We could not explicitly verify the license."
        else:
            verdict = "No red flags found. Formatting consistent with specifications."
    elif composite_trust_score >= 60:
        verdict = "Price Discrepancy Detected"
    else:
        verdict = "Packaging Verification Risk"

    sub_scores = SubScores(
        text_integrity=t_earned,
        text_max=t_max,
        text_status=t_stat,
        text_matching_tokens_count=t_match_cnt,
        regulatory=r_earned,
        regulatory_max=r_max,
        regulatory_status=r_stat,
        regulatory_message=r_msg,
        pricing_fairness=p_earned,
        pricing_max=p_max,
        pricing_status=p_stat,
        safety_intel=f_earned,
        safety_max=f_max,
        safety_status=f_stat,
        safety_message=f_msg,
        max_available_points=max_available_base,
        total_earned_points=round(total_earned, 2)
    )

    telemetry = {
        "is_demo_mode": is_demo_mode,
        "ocr_engine": ocr_result.ocr_engine_used,
        "ocr_confidence": ocr_result.ocr_confidence,
        "rotation_applied": ocr_result.rotation_applied,
        "crop_applied": ocr_result.crop_applied,
        "barcode": ocr_result.barcode,
        "score_base_denominator": max_available_base,
        "score_earned_points": round(total_earned, 2),
        "trigger_map_remediation": should_trigger_map
    }

    return AuditScorecard(
        trust_score=composite_trust_score,
        verdict=verdict,
        sub_scores=sub_scores,
        evidence_list=evidence,
        ocr_result=ocr_result,
        parity_result=parity_result,
        intel_result=intel_result,
        fallback_pharmacies=fallback_pharmacies or [],
        trigger_map_remediation=should_trigger_map,
        barcode_audit_warning=barcode_warning_text,
        is_demo_replay=is_demo_mode,
        pipeline_telemetry=telemetry
    )
