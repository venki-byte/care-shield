"""Master Forensic Pipeline DAG Orchestrator with Rigid Execution Tracing & Lens Visibility.

Coordinates all DAG hops:
- T1.1: Privacy ROI Cropping (isomorphic foreground product isolation)
- T1.2: Dual-Path Vision & Local/Gemini OCR Ingestion
- T1.3: Google Lens Web Entity Visibility & Inspection
- T1.4: Regulatory Syntax & Token Deviation Guard
- T1.5: Financial Metrology & Live Google Shopping Parity
- T1.6: Adversarial Safety & Recall Threat Intelligence
- T1.7: Dynamic Scoring Engine with N/A Base Recalculation
- T1.8: Geolocation Remediation & Verified Pharmacy Routing

Strictly adheres to Hard Rules:
- Rule 1: Never assume/infer values. Missing fields are None and rendered as 'Not provided / Not found'.
- Rule 2: Never print, log, or commit API keys. Mask all credentials.
- Rule 3: Wrap all external calls with timeout and retry, returning graceful degraded fallbacks.
- Rule 4: Rigid Pydantic v2 validation for every inter-node hop.
"""

import time
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

import config
import schemas
from schemas import (
    PackagingOCRResult,
    ShoppingParityResult,
    SafetyIntelResult,
    NearbyPharmacy,
    AuditScorecard,
    TraceStep,
    PipelineExecutionTrace,
)
from core.vision_ingest import (
    isolate_product_roi,
    run_local_image_ocr,
    query_gemini_vision,
    inspect_serpapi_lens,
    ingest_packaging_image,
)
from core.deviation_guard import audit_packaging_text_integrity, audit_barcode_with_serpapi
from core.parity_checker import check_price_parity
from core.safety_intel import audit_safety_intel
from core.map_router import find_nearby_verified_pharmacies
from core.scoring_engine import compute_audit_scorecard


def run_pipeline_with_trace(
    image_bytes: Optional[bytes] = None,
    lens_image_url: Optional[str] = None,
    raw_text_override: Optional[str] = None,
    sample_preset: Optional[str] = None,
    store_asking_price: Optional[float] = None,
    manual_printed_mrp: Optional[float] = None,
    user_lat: float = config.DEFAULT_LAT,
    user_lng: float = config.DEFAULT_LNG,
    city_name: str = config.DEFAULT_CITY,
    serpapi_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    is_demo_mode: bool = False,
    precomputed_ocr_result: Optional[PackagingOCRResult] = None,
    precomputed_lens_report: Optional[Dict[str, Any]] = None,
) -> AuditScorecard:
    """Execute complete Care-Shield DAG and record a rigid, verifiable execution trace."""
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4())
    steps: List[TraceStep] = []
    
    effective_demo_mode = is_demo_mode or bool(sample_preset)
    active_serpapi_key = serpapi_key or config.SERPAPI_KEY
    active_gemini_key = gemini_key or config.GEMINI_API_KEY

    # -------------------------------------------------------------------------
    # STEP 1: Privacy ROI Cropping (T1.1_ROI_CROP)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    crop_applied = False
    process_bytes = image_bytes

    if precomputed_ocr_result:
        crop_applied = precomputed_ocr_result.crop_applied
        steps.append(
            TraceStep(
                step_id="T1.1_ROI_CROP",
                step_name="Privacy ROI Product Isolation",
                status="SUCCESS" if crop_applied else "SKIPPED",
                duration_ms=0.0,
                inputs={"precomputed": True},
                outputs={
                    "crop_applied": crop_applied,
                    "privacy_status": "Background apparel/facial features excluded" if crop_applied else "Full frame evaluated"
                },
                telemetry={"engine": "Pre-extracted Packaging Crop"}
            )
        )
    elif image_bytes and not sample_preset:
        process_bytes, crop_applied = isolate_product_roi(image_bytes)
        dur = round((time.perf_counter() - t0) * 1000, 2)
        steps.append(
            TraceStep(
                step_id="T1.1_ROI_CROP",
                step_name="Privacy ROI Product Isolation",
                status="SUCCESS" if crop_applied else "SKIPPED",
                duration_ms=dur,
                inputs={"original_image_bytes": len(image_bytes)},
                outputs={
                    "crop_applied": crop_applied,
                    "processed_image_bytes": len(process_bytes) if process_bytes else "Not provided / Not found",
                    "privacy_status": "Background apparel/facial features excluded" if crop_applied else "Full frame evaluated"
                },
                telemetry={"engine": "OpenCV Canny/Contour Energy Filter"}
            )
        )
    else:
        dur = round((time.perf_counter() - t0) * 1000, 2)
        steps.append(
            TraceStep(
                step_id="T1.1_ROI_CROP",
                step_name="Privacy ROI Product Isolation",
                status="SKIPPED",
                duration_ms=dur,
                inputs={"sample_preset": sample_preset or "None", "has_image_bytes": bool(image_bytes)},
                outputs={"crop_applied": False, "privacy_status": "Preset fixture or text-only input"},
                telemetry={"engine": "Bypassed for preset/text"}
            )
        )

    # -------------------------------------------------------------------------
    # STEP 2: Vision Ingestion & OCR (T1.2_OCR_INGEST)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    if precomputed_ocr_result:
        ocr_result = precomputed_ocr_result
        lens_report = precomputed_lens_report or {
            "status": "SKIPPED",
            "reason": "Precomputed inspection fixture",
            "lens_title": None,
            "lens_brand": None,
            "visual_matches_count": 0,
            "knowledge_graph_count": 0,
            "raw_json_summary": "Not provided / Not found",
            "duration_ms": 0.0,
            "image_url": lens_image_url
        }
        dur = 0.0
    else:
        ocr_result, lens_report = ingest_packaging_image(
            image_bytes=image_bytes,
            raw_text_override=raw_text_override,
            sample_preset=sample_preset,
            serpapi_key=active_serpapi_key,
            gemini_key=active_gemini_key,
            lens_image_url=lens_image_url,
            return_lens_report=True
        )
        dur = round((time.perf_counter() - t0) * 1000, 2)

    # Apply manual MRP override if user explicitly supplied one; otherwise leave None (Rule 1)
    if manual_printed_mrp is not None and manual_printed_mrp > 0.0:
        ocr_result.printed_mrp_inr = manual_printed_mrp

    steps.append(
        TraceStep(
            step_id="T1.2_OCR_INGEST",
            step_name="Dual-Path Packaging Vision Ingestion",
            status="SUCCESS",
            duration_ms=dur,
            inputs={
                "gemini_authenticated": bool(active_gemini_key),
                "input_type": "precomputed" if precomputed_ocr_result else ("preset" if sample_preset else ("image" if image_bytes else "text_override"))
            },
            outputs={
                "product_title": ocr_result.product_title,
                "detected_brand": ocr_result.detected_brand,
                "printed_mrp": ocr_result.printed_mrp_display,
                "batch_number": ocr_result.batch_number_display,
                "cdsco_license": ocr_result.cdsco_license_display,
                "active_components": ocr_result.active_components_display,
                "mfg_date": ocr_result.mfg_date_display,
                "exp_date": ocr_result.exp_date_display,
                "barcode": ocr_result.barcode_display,
                "rotation_applied": f"{ocr_result.rotation_applied}°"
            },
            telemetry={
                "ocr_engine_used": ocr_result.ocr_engine_used,
                "ocr_confidence": ocr_result.ocr_confidence,
                "raw_corpus_chars": len(ocr_result.raw_ocr_corpus),
                "product_category": ocr_result.product_category
            }
        )
    )

    # -------------------------------------------------------------------------
    # TASK T6: Product-Category Guard (Prescription Medicine Abort)
    # -------------------------------------------------------------------------
    if ocr_result.product_category == "prescription_medicine":
        abort_msg = "Care-Shield doesn't audit prescription medicines. Please ask your pharmacist or doctor."
        steps.append(
            TraceStep(
                step_id="T6_PRESCRIPTION_GUARD",
                step_name="Prescription Medicine Category Guard",
                status="FAILED",
                duration_ms=0.0,
                inputs={"product_category": ocr_result.product_category, "product_title": ocr_result.product_title},
                outputs={"action": "ABORT_AUDIT", "reason": abort_msg},
                telemetry={"guard_triggered": True, "bypassed_serpapi": True},
                error=abort_msg
            )
        )
        total_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace = PipelineExecutionTrace(
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(),
            total_duration_ms=total_duration_ms,
            overall_status="ABORTED",
            lens_visibility=lens_report,
            steps=steps
        )
        sub_scores = schemas.SubScores(
            text_integrity=0.0,
            text_max=0.0,
            text_status="na",
            regulatory=None,
            regulatory_max=0.0,
            regulatory_status="na",
            pricing_fairness=None,
            pricing_max=0.0,
            pricing_status="na",
            safety_intel=0.0,
            safety_max=0.0,
            safety_status="na",
            max_available_points=0.0,
            total_earned_points=0.0
        )
        return AuditScorecard(
            trust_score=0,
            verdict="Audit Aborted: Prescription Medicine",
            sub_scores=sub_scores,
            evidence_list=[abort_msg],
            ocr_result=ocr_result,
            is_aborted=True,
            abort_reason=abort_msg,
            trigger_map_remediation=False,
            execution_trace=trace
        )

    # -------------------------------------------------------------------------
    # STEP 3: Google Lens Web Entity Visibility (T1.3_LENS_INSPECTION)
    # -------------------------------------------------------------------------
    steps.append(
        TraceStep(
            step_id="T1.3_LENS_INSPECTION",
            step_name="Google Lens Entity Inspector",
            status=lens_report.get("status", "SKIPPED"),
            duration_ms=lens_report.get("duration_ms", 0.0),
            inputs={
                "image_url": lens_image_url or "Not provided / Not found",
                "serpapi_authenticated": bool(active_serpapi_key)
            },
            outputs={
                "lens_title": lens_report.get("lens_title") or "Not provided / Not found",
                "lens_brand": lens_report.get("lens_brand") or "Not provided / Not found",
                "visual_matches_count": lens_report.get("visual_matches_count", 0),
                "knowledge_graph_count": lens_report.get("knowledge_graph_count", 0),
                "execution_reason": lens_report.get("reason") or "Live Google Lens visual matches evaluated"
            },
            telemetry={
                "raw_response_preview": lens_report.get("raw_json_summary", "Not provided / Not found")[:300]
            },
            error=lens_report.get("reason") if lens_report.get("status") in ("DEGRADED", "FAILED") else None
        )
    )

    # -------------------------------------------------------------------------
    # STEP 4: Barcode / GTIN Entity Resolution (T1.4_BARCODE_SEARCH)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    barcode_mismatch, barcode_warning, barcode_telemetry = audit_barcode_with_serpapi(
        barcode=ocr_result.barcode,
        detected_brand=ocr_result.detected_brand,
        product_title=ocr_result.product_title,
        serpapi_key=active_serpapi_key,
        is_demo_mode=effective_demo_mode
    )
    dur = round((time.perf_counter() - t0) * 1000, 2)
    steps.append(
        TraceStep(
            step_id="T1.4_BARCODE_SEARCH",
            step_name="Barcode & GTIN Entity Resolution",
            status=barcode_telemetry.get("status", "SKIPPED"),
            duration_ms=dur,
            inputs={
                "barcode": ocr_result.barcode_display,
                "barcode_type": ocr_result.barcode_type or "Not provided / Not found",
                "serpapi_authenticated": bool(active_serpapi_key)
            },
            outputs={
                "barcode_mismatch": barcode_mismatch,
                "barcode_warning": barcode_warning or "Barcode verified / unflagged",
                "top_entity_found": barcode_telemetry.get("top_entity_found") or "Not provided / Not found"
            },
            telemetry=barcode_telemetry
        )
    )

    # -------------------------------------------------------------------------
    # STEP 5: Regulatory & Text Deviation Guard (T1.5_REGULATORY_GUARD)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    is_lic_valid, exact_fuzzy_ratio, text_evidence = audit_packaging_text_integrity(
        product_title=ocr_result.product_title,
        detected_brand=ocr_result.detected_brand,
        raw_ocr_corpus=ocr_result.raw_ocr_corpus,
        cdsco_license_raw=ocr_result.cdsco_license_raw,
        serpapi_key=active_serpapi_key
    )
    ocr_result.is_license_format_valid = is_lic_valid
    dur = round((time.perf_counter() - t0) * 1000, 2)
    steps.append(
        TraceStep(
            step_id="T1.5_REGULATORY_GUARD",
            step_name="CDSCO Regulatory & Token Deviation Guard",
            status="SUCCESS",
            duration_ms=dur,
            inputs={
                "cdsco_license_raw": ocr_result.cdsco_license_display,
                "detected_brand": ocr_result.detected_brand,
                "product_title": ocr_result.product_title
            },
            outputs={
                "is_license_format_valid": "Compliant" if is_lic_valid is True else ("Malformed" if is_lic_valid is False else "Not provided / Not found"),
                "token_confidence_pct": f"{int(exact_fuzzy_ratio * 100)}%",
                "findings": text_evidence
            },
            telemetry={"fuzzy_matching_ratio": exact_fuzzy_ratio}
        )
    )

    # -------------------------------------------------------------------------
    # STEP 6: Financial Metrology & Live Price Parity (T1.6_PARITY_CHECKER)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    effective_asking_price = store_asking_price if (store_asking_price is not None and store_asking_price > 0) else None
    parity_result = check_price_parity(
        product_title=ocr_result.product_title,
        detected_brand=ocr_result.detected_brand,
        scanned_price=effective_asking_price,
        printed_mrp_inr=ocr_result.printed_mrp_inr,
        serpapi_key=active_serpapi_key,
        is_demo_mode=effective_demo_mode
    )
    dur = round((time.perf_counter() - t0) * 1000, 2)
    steps.append(
        TraceStep(
            step_id="T1.6_PARITY_CHECKER",
            step_name="Financial Metrology & Google Shopping Parity",
            status="SUCCESS" if parity_result.is_live_query or effective_demo_mode else "DEGRADED",
            duration_ms=dur,
            inputs={
                "store_asking_price": parity_result.scanned_price_display,
                "printed_mrp": ocr_result.printed_mrp_display,
                "serpapi_authenticated": bool(active_serpapi_key)
            },
            outputs={
                "online_median": parity_result.online_median_display,
                "markup_percent": f"{parity_result.markup_percent:+.1f}%",
                "is_above_printed_mrp": parity_result.is_above_printed_mrp,
                "mrp_audit_skipped": parity_result.mrp_audit_skipped,
                "benchmark_merchants_count": len(parity_result.benchmark_merchants)
            },
            telemetry={
                "is_live_query": parity_result.is_live_query,
                "merchants_sampled": [f"{m.name}: ₹{m.price:,.2f}" for m in parity_result.benchmark_merchants[:4]]
            },
            error=parity_result.api_warning
        )
    )

    # -------------------------------------------------------------------------
    # STEP 7: Adversarial Safety Intelligence (T1.7_SAFETY_INTEL)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    intel_result = audit_safety_intel(
        product_title=ocr_result.product_title,
        detected_brand=ocr_result.detected_brand,
        serpapi_key=active_serpapi_key,
        is_demo_mode=effective_demo_mode
    )
    dur = round((time.perf_counter() - t0) * 1000, 2)
    steps.append(
        TraceStep(
            step_id="T1.7_SAFETY_INTEL",
            step_name="Adversarial Safety & Recall Threat Intelligence",
            status="SUCCESS" if intel_result.is_live_query or effective_demo_mode else "DEGRADED",
            duration_ms=dur,
            inputs={
                "brand": ocr_result.detected_brand,
                "product_title": ocr_result.product_title,
                "serpapi_authenticated": bool(active_serpapi_key)
            },
            outputs={
                "has_active_recall": intel_result.has_active_recall,
                "threat_risk_factor": intel_result.risk_factor,
                "matched_snippets_count": len(intel_result.evidence_snippets)
            },
            telemetry={"is_live_query": intel_result.is_live_query},
            error=intel_result.api_warning
        )
    )

    # -------------------------------------------------------------------------
    # STEP 8: Scoring Engine & Dynamic Normalization (T1.8_SCORING_ENGINE)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    scorecard = compute_audit_scorecard(
        ocr_result=ocr_result,
        parity_result=parity_result,
        intel_result=intel_result,
        barcode_mismatch=barcode_mismatch,
        is_demo_mode=effective_demo_mode
    )
    dur = round((time.perf_counter() - t0) * 1000, 2)
    steps.append(
        TraceStep(
            step_id="T1.8_SCORING_ENGINE",
            step_name="Deterministic Dynamic Trust Scoring Engine",
            status="SUCCESS",
            duration_ms=dur,
            inputs={
                "text_integrity_score": f"{scorecard.sub_scores.text_integrity}/{scorecard.sub_scores.text_max}",
                "regulatory_score": f"{scorecard.sub_scores.regulatory if scorecard.sub_scores.regulatory is not None else 'N/A'}/{scorecard.sub_scores.regulatory_max}",
                "pricing_score": f"{scorecard.sub_scores.pricing_fairness if scorecard.sub_scores.pricing_fairness is not None else 'N/A'}/{scorecard.sub_scores.pricing_max}",
                "safety_score": f"{scorecard.sub_scores.safety_intel}/{scorecard.sub_scores.safety_max}"
            },
            outputs={
                "composite_trust_score": f"{scorecard.trust_score} / 100",
                "categorical_verdict": scorecard.verdict,
                "dynamic_base_denominator": scorecard.sub_scores.max_available_points,
                "total_earned_points": scorecard.sub_scores.total_earned_points,
                "trigger_map_remediation": scorecard.trigger_map_remediation
            },
            telemetry={"defensible_verdict_applied": True}
        )
    )

    # -------------------------------------------------------------------------
    # STEP 9: Geolocation Remediation Router (T1.9_MAP_ROUTER)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    if scorecard.trigger_map_remediation:
        pharmacies = find_nearby_verified_pharmacies(
            lat=user_lat,
            lng=user_lng,
            city_name=city_name,
            serpapi_key=active_serpapi_key,
            is_demo_mode=effective_demo_mode
        )
        scorecard.fallback_pharmacies = pharmacies
        maps_status = "SUCCESS" if pharmacies else "DEGRADED"
    else:
        maps_status = "SKIPPED"
    dur = round((time.perf_counter() - t0) * 1000, 2)

    steps.append(
        TraceStep(
            step_id="T1.9_MAP_ROUTER",
            step_name="Geolocation Remediation & Google Maps Router",
            status=maps_status,
            duration_ms=dur,
            inputs={
                "city": city_name,
                "coordinates": f"{user_lat}, {user_lng}",
                "serpapi_authenticated": bool(active_serpapi_key),
                "trigger_map_remediation": scorecard.trigger_map_remediation
            },
            outputs={
                "verified_pharmacies_returned": len(scorecard.fallback_pharmacies),
                "top_pharmacies": [
                    f"{p.name} ({p.rating}★, {p.review_count} revs)" for p in scorecard.fallback_pharmacies[:3]
                ] if scorecard.fallback_pharmacies else "Not triggered or not found"
            },
            telemetry={
                "filter_rating_min": 4.0,
                "filter_reviews_min": 10,
                "routing_triggered": scorecard.trigger_map_remediation
            }
        )
    )

    # -------------------------------------------------------------------------
    # Assemble Master Pipeline Execution Trace
    # -------------------------------------------------------------------------
    total_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    pipeline_trace = PipelineExecutionTrace(
        trace_id=trace_id,
        timestamp=datetime.now().isoformat(),
        total_duration_ms=total_duration_ms,
        overall_status="SUCCESS",
        lens_visibility=lens_report,
        steps=steps
    )

    scorecard.execution_trace = pipeline_trace
    return scorecard
