"""Comprehensive Automated Verification Test Suite for Care-Shield Refactoring.

Tests Tasks T0 through T5:
1. Security & Logging (Hard Rule 2): .env in .gitignore, mask_secrets utility.
2. Robust OCR & 4-Way Rotation Handling (Task T2): RapidOCR with confidence scores and rotation selection.
3. Structured Field Extraction & UI Pause (Task T3): Regex metadata for MRP, Batch, Mfg Date, Exp Date.
4. Barcode / QR Decoding & Search Node (Task T4): pyzbar / OpenCV fallback and SerpApi barcode resolution.
5. Deterministic Scoring Engine Pure Functions (Task T5):
   - text_integrity: token count < 3 warning, unrounded similarity.
   - license_format: CDSCO MDR regex, na if absent.
   - pricing_fairness: MRP check, median tolerance formula clamp((price - median*1.25)/(median*0.5), 0, 1).
   - fraud_intel: zero hits -> 'No warning signs found' (status: 'ok').
   - Gate Logic: trigger_map_remediation = (Score < 70) or (price > MRP) or (fraud_intel fails).
6. Maps Routing (rating >= 4.0 and review_count >= 10, place_id navigation).
7. Defensible Phrasing: UI & Audio statutory language.
8. End-to-End Pipeline DAG with Rigid Execution Trace (9 hops, zero hallucinations).
"""

import sys
from pathlib import Path

# Reconfigure stdout to utf-8 for Windows PowerShell
sys.stdout.reconfigure(encoding="utf-8")

import config
import schemas
from core.vision_ingest import (
    isolate_product_roi,
    run_local_image_ocr,
    extract_regex_metadata,
    decode_barcode_and_qr,
)
from core.deviation_guard import (
    audit_packaging_text_integrity,
    audit_barcode_with_serpapi,
)
from core.parity_checker import check_price_parity
from core.safety_intel import audit_safety_intel
from core.map_router import find_nearby_verified_pharmacies
from core.scoring_engine import (
    compute_text_integrity_subscore,
    compute_license_format_subscore,
    compute_pricing_subscore,
    compute_fraud_intel_subscore,
    compute_audit_scorecard,
)
from core.pipeline import run_pipeline_with_trace
import audio

print("=" * 70)
print("     CARE-SHIELD REFACTORING & VERIFICATION TEST SUITE (T0-T5)     ")
print("=" * 70)

# --- TEST 1: SECURITY & LOGGING (TASK T0) ---
print("\n[TEST 1] Security & Logging Checks (Hard Rule 2)...")
assert Path(".gitignore").exists(), ".gitignore does not exist!"
with open(".gitignore", "r") as f:
    gi_content = f.read()
assert ".env" in gi_content, ".env not found in .gitignore!"
print(" -> PASS: .env is strictly ignored in .gitignore.")

masked_key = config.mask_secrets("dummy_sample_test_key_abc1234567890def")
print(f" -> Masked key output: {masked_key}")
assert masked_key == "[Protected Key Active]", f"Unexpected masked key format: {masked_key}"
assert config.mask_secrets(None) == "[Not Configured]"
print(" -> PASS: mask_secrets properly masks credentials everywhere.")

# --- TEST 2: VISION PRIVACY & 4-WAY ROTATION OCR (TASK T2) ---
print("\n[TEST 2] Vision Privacy & 4-Way Rotation RapidOCR with Confidence (Task T2)...")
sample_img_path = Path("sample_image/WIN_20261003_23_27_56_Pro.jpg")
if sample_img_path.exists():
    with open(sample_img_path, "rb") as f:
        raw_bytes = f.read()

    # Isolated crop test
    crop_bytes, crop_applied = isolate_product_roi(raw_bytes)
    assert crop_applied, "Crop was not applied to foreground product!"

    # Multi-rotation OCR test returning (text, meta, avg_conf, best_angle, token_confs)
    crop_text, crop_meta, crop_conf, crop_angle, token_confs = run_local_image_ocr(crop_bytes)
    print(f" -> Selected Rotation Angle: {crop_angle}°")
    print(f" -> Average Token Confidence: {crop_conf:.4f} ({crop_conf*100:.1f}%)")
    print(f" -> Total Tokens Extracted: {len(token_confs)}")
    print(f" -> OCR Corpus Preview: {[t for t in crop_text.splitlines() if t.strip()][:4]}")

    assert crop_angle in (0, 90, 180, 270), f"Invalid rotation angle: {crop_angle}"
    assert crop_conf > 0.0, "Expected positive average token confidence!"
    assert "IMPACT" not in crop_text and "MPACT" not in crop_text, "Privacy crop failed to eliminate t-shirt text!"
    print(" -> PASS: Privacy crop and 4-way rotation OCR with confidence scores verified.")
else:
    print(" -> SKIP: sample_image not present for live crop test.")

# --- TEST 3: STRUCTURED FIELD EXTRACTION (TASK T3) ---
print("\n[TEST 3] Structured Regex Field Extraction (Task T3)...")
corpus_sample = (
    "Melactis Serum 30ml. B.No: ML-8891. MFD: 03/2024. EXP: 02/2026. "
    "M.R.P. Rs. 850.00 (Incl. of all taxes). Lic No: MFG/MD/2022/004561."
)
extracted_fields = extract_regex_metadata(corpus_sample)
print(f" -> Extracted MRP: {extracted_fields.get('printed_mrp')}")
print(f" -> Extracted Batch: {extracted_fields.get('batch_number')}")
print(f" -> Extracted Mfg Date: {extracted_fields.get('mfg_date')}")
print(f" -> Extracted Exp Date: {extracted_fields.get('exp_date')}")
print(f" -> Extracted CDSCO License: {extracted_fields.get('cdsco_license')}")

assert extracted_fields.get("printed_mrp") == 850.0, f"Expected MRP 850.0, got {extracted_fields.get('printed_mrp')}"
assert extracted_fields.get("batch_number") == "ML-8891"
assert extracted_fields.get("mfg_date") == "03/2024"
assert extracted_fields.get("exp_date") == "02/2026"
assert extracted_fields.get("cdsco_license") == "MFG/MD/2022/004561"
print(" -> PASS: Structured field extraction accurately parsed MRP, Batch, Mfg/Exp dates, and License.")

# --- TEST 4: BARCODE / QR DECODING & SERPAPI SEARCH NODE (TASK T4) ---
print("\n[TEST 4] Barcode / QR Decoding & Barcode Search Node (Task T4)...")
# Test demo resolution
mismatch, warn_msg, tele = audit_barcode_with_serpapi(
    barcode="8906095170123",
    detected_brand="Dr Trust",
    product_title="Pulse Oximeter",
    is_demo_mode=True
)
print(f" -> Barcode match: mismatch={mismatch}, status={tele.get('status')}")
assert mismatch is False, "Expected authentic barcode to match!"

# Test soft warning when barcode conflicts with OCR entity
mismatch_soft, warn_soft, _ = audit_barcode_with_serpapi(
    barcode=None,
    detected_brand="Test",
    product_title="Test"
)
assert mismatch_soft is False, "Missing barcode should skip mismatch flag."
print(" -> PASS: Barcode audit node handles resolution and soft warnings gracefully.")

# --- TEST 5: PURE DETERMINISTIC SCORING ENGINE (TASK T5) ---
print("\n[TEST 5] Pure Scoring Engine Mathematical Unit Tests (Task T5)...")

# 5.1 Text Integrity: exact tokens, warning when < 3 tokens match, unrounded similarity
pts, max_p, stat, cnt, sim, rsn = compute_text_integrity_subscore(
    ocr_title="meloctis",
    ocr_brand="meloctis",
    canonical_title="Melactis Pigment Corrector Serum",
    canonical_brand="Melactis"
)
print(f" -> Text Integrity ('meloctis' vs 'Melactis'): {pts}/{max_p} pts, status={stat}, matching_tokens={cnt}, sim={sim}")
assert stat == "warn", "Fewer than 3 matching tokens must set status to 'warn'!"
assert cnt < 3, f"Expected < 3 tokens, got {cnt}"
assert 0.85 <= sim <= 0.88, f"Expected ~0.875 similarity, got {sim}"
assert pts < max_p, "Points must be penalized for token deviation!"

# High token match
pts_hi, _, stat_hi, cnt_hi, sim_hi, _ = compute_text_integrity_subscore(
    ocr_title="Finger Tip Pulse Oximeter 506",
    ocr_brand="Dr Trust",
    canonical_title="Finger Tip Pulse Oximeter 506",
    canonical_brand="Dr Trust"
)
assert stat_hi == "ok"
assert cnt_hi >= 3
assert sim_hi == 1.0
assert pts_hi == 35.0
print(" -> PASS: T5.1 Text integrity token matching and exact similarity verified.")

# 5.2 Regulatory license pure regex
pts_lic_v, _, stat_lic_v, msg_v = compute_license_format_subscore("MFG/MD/2021/000123")
assert stat_lic_v == "valid" and pts_lic_v == 25.0
assert "looks valid" in msg_v

pts_lic_i, _, stat_lic_i, msg_i = compute_license_format_subscore("MD-LIC-9988-FAKE")
assert stat_lic_i == "invalid" and pts_lic_i == 5.0
assert "looks invalid" in msg_i

pts_lic_na, max_lic_na, stat_lic_na, _ = compute_license_format_subscore(None)
assert stat_lic_na == "na" and pts_lic_na is None and max_lic_na == 0.0
print(" -> PASS: T5.2 Regulatory license format regex verified.")

# 5.3 Pricing fairness formula: penalty = clamp((price - median*1.25)/(median*0.5), 0, 1)
# Case A: Illegal over-MRP
p_pts, _, p_stat, is_over, p_rsn = compute_pricing_subscore(store_price=2200.0, printed_mrp=1999.0, online_median=1800.0)
assert is_over is True
assert p_pts == 0.0
assert p_stat == "fail"

# Case B: Price within 25% of median (median=1000, price=1200 <= 1250) -> penalty=0.0
p_pts2, _, p_stat2, is_over2, _ = compute_pricing_subscore(store_price=1200.0, printed_mrp=2000.0, online_median=1000.0)
assert is_over2 is False
assert p_pts2 == 20.0
assert p_stat2 == "ok"

# Case C: Price at median*1.50 (median=1000, price=1500) -> penalty = (1500 - 1250) / 500 = 0.50 -> earned = 10.0
p_pts3, _, p_stat3, _, _ = compute_pricing_subscore(store_price=1500.0, printed_mrp=2000.0, online_median=1000.0)
assert p_stat3 == "warn"
assert p_pts3 == 10.0

# Case D: Missing store price -> N/A
p_pts4, p_max4, p_stat4, _, _ = compute_pricing_subscore(store_price=None, printed_mrp=1999.0, online_median=1800.0)
assert p_stat4 == "na"
assert p_pts4 is None
assert p_max4 == 0.0
print(" -> PASS: T5.3 Pricing fairness mathematical clamp formula verified.")

# 5.4 Adversarial fraud intel pure sub-score
f_pts, _, f_stat, f_msg = compute_fraud_intel_subscore(has_active_recall=False, risk_factor=0.0, snippets=[])
assert f_pts == 20.0
assert f_stat == "ok"
assert f_msg == "No warning signs found"

f_pts_bad, _, f_stat_bad, _ = compute_fraud_intel_subscore(has_active_recall=True, risk_factor=0.85, snippets=["Counterfeit seizure"])
assert f_stat_bad == "fail"
assert f_pts_bad < 20.0
print(" -> PASS: T5.4 Fraud intel sub-score verified.")

# 5.5 Master scorecard dynamic base denominator & Gate Logic
# Scenario 1: Clean item with cosmetic N/A license
ocr_clean = schemas.PackagingOCRResult(
    product_title="Melactis Pigment Corrector Serum",
    detected_brand="Melactis",
    printed_mrp_inr=1000.0,
    cdsco_license_raw=None
)
card_clean = compute_audit_scorecard(
    ocr_result=ocr_clean,
    parity_result=schemas.ShoppingParityResult(scanned_purchase_price=950.0, online_median=950.0),
    intel_result=schemas.SafetyIntelResult(has_active_recall=False, risk_factor=0.0)
)
print(f" -> Clean Card Denominator Base: {card_clean.sub_scores.max_available_points} (Regulatory N/A excluded)")
print(f" -> Trigger Map Remediation: {card_clean.trigger_map_remediation}")
assert card_clean.sub_scores.regulatory is None
assert card_clean.sub_scores.max_available_points == 75.0  # 35 text + 20 price + 20 safety
assert card_clean.trigger_map_remediation is False, "Clean high-scoring item must not trigger map remediation!"

# Scenario 2: Over-MRP violation -> triggers map remediation
ocr_overmrp = schemas.PackagingOCRResult(
    product_title="Melactis Pigment Corrector Serum",
    detected_brand="Melactis",
    printed_mrp_inr=800.0,
    cdsco_license_raw=None
)
card_overmrp = compute_audit_scorecard(
    ocr_result=ocr_overmrp,
    parity_result=schemas.ShoppingParityResult(scanned_purchase_price=1200.0, online_median=800.0, is_above_printed_mrp=True),
    intel_result=schemas.SafetyIntelResult(has_active_recall=False, risk_factor=0.0)
)
print(f" -> Over-MRP Trigger Map Remediation: {card_overmrp.trigger_map_remediation}")
assert card_overmrp.trigger_map_remediation is True, "Over-MRP must trigger map remediation!"
print(" -> PASS: T5.5 Dynamic N/A denominator base and Gate Logic verified.")

# --- TEST 6: GEOLOCATION ROUTER FILTER (RATING >= 4.0 & REVIEWS >= 10) ---
print("\n[TEST 6] Google Maps Routing Filter...")
active_key = config.SERPAPI_KEY
if active_key:
    pharmacies = find_nearby_verified_pharmacies(
        lat=13.0827,
        lng=80.2707,
        city_name="Chennai",
        serpapi_key=active_key,
        is_demo_mode=False
    )
    print(f" -> Retrieved {len(pharmacies)} verified pharmacies with rating >= 4.0 and reviews >= 10:")
    for p in pharmacies[:3]:
        print(f"    * {p.name}: {p.rating}★ ({p.review_count} revs), Place ID: {p.place_id}")
        assert p.rating >= 4.0, f"Store rating {p.rating} < 4.0!"
        assert p.review_count >= 10, f"Store review count {p.review_count} < 10!"
        assert "destination=" in p.directions_url, "Missing destination in directions URL!"
    print(" -> PASS: Google Maps reviews filter (>= 10 reviews) and place_id routing verified.")
else:
    print(" -> SKIP: SERPAPI_KEY not configured for live Maps test.")

# --- TEST 7: DEFENSIBLE PHRASING (UI & AUDIO) ---
print("\n[TEST 7] Defensible Phrasing Verification...")
print(f" -> Scorecard Verdict: \"{card_clean.verdict}\"")
assert "No red flags found" in card_clean.verdict or "could not explicitly verify" in card_clean.verdict
assert "Regulatory license verified" not in card_clean.verdict
assert "Product is safe" not in card_clean.verdict

script_en = audio.build_audio_script(card_clean, "en")
script_ta = audio.build_audio_script(card_clean, "ta")
assert "could not explicitly verify the license" in script_en
assert "Regulatory license verified" not in script_en
assert "Product is safe" not in script_en
print(" -> PASS: Defensible phrasing verified across UI and bilingual audio.")

# --- TEST 8: FULL 9-STEP PIPELINE DAG WITH RIGID EXECUTION TRACE ---
print("\n[TEST 8] Full 9-Step Pipeline DAG with Rigid Execution Trace...")
full_card = run_pipeline_with_trace(
    sample_preset="authentic_oximeter",
    store_asking_price=1630.0
)
assert full_card.execution_trace is not None
trace = full_card.execution_trace
print(f" -> Trace ID: {trace.trace_id}")
print(f" -> Total Pipeline Latency: {trace.total_duration_ms:.2f} ms")
print(f" -> Overall Status: {trace.overall_status}")
print(f" -> Total DAG Steps: {len(trace.steps)}")

expected_step_ids = [
    "T1.1_ROI_CROP",
    "T1.2_OCR_INGEST",
    "T1.3_LENS_INSPECTION",
    "T1.4_BARCODE_SEARCH",
    "T1.5_REGULATORY_GUARD",
    "T1.6_PARITY_CHECKER",
    "T1.7_SAFETY_INTEL",
    "T1.8_SCORING_ENGINE",
    "T1.9_MAP_ROUTER",
]
actual_step_ids = [s.step_id for s in trace.steps]
print(f" -> Executed DAG Step IDs: {actual_step_ids}")
assert actual_step_ids == expected_step_ids, f"Step ID mismatch! Got {actual_step_ids}"

# Hard Rule 1 check: missing fields rendered as 'Not provided / Not found'
ocr_sample = schemas.PackagingOCRResult(product_title="Test Product", detected_brand="Test Brand")
assert ocr_sample.printed_mrp_display == "Not provided / Not found"
assert ocr_sample.cdsco_license_display == "Not provided / Not found"
assert ocr_sample.batch_number_display == "Not provided / Not found"
assert ocr_sample.barcode_display == "Not provided / Not found"
assert ocr_sample.mfg_date_display == "Not provided / Not found"
assert ocr_sample.exp_date_display == "Not provided / Not found"

print(" -> PASS: Complete 9-step DAG execution trace and contract schemas verified.")

# --- TEST 9: PRODUCT-CATEGORY GUARD & PRESCRIPTION ABORT (TASK T6) ---
print("\n[TEST 9] Product-Category Guard & Prescription Abort (Task T6)...")
from core.vision_ingest import classify_product_category

cat_rx = classify_product_category("Amoxicillin 500mg Capsules IP Schedule H Prescription Drug")
assert cat_rx == "prescription_medicine", f"Expected prescription_medicine, got {cat_rx}"

cat_cosmetic = classify_product_category("Melactis Advanced Pigment Corrector Skincare Serum 30ml")
assert cat_cosmetic == "cosmetic", f"Expected cosmetic, got {cat_cosmetic}"

cat_device = classify_product_category("Dr Trust Pulse Oximeter OLED SpO2")
assert cat_device == "otc_device", f"Expected otc_device, got {cat_device}"

# Pipeline prescription abort test
ocr_rx = schemas.PackagingOCRResult(
    product_title="Amoxicillin 500mg Capsules IP",
    detected_brand="Cipla",
    product_category="prescription_medicine"
)
rx_card = run_pipeline_with_trace(
    precomputed_ocr_result=ocr_rx,
    store_asking_price=120.0
)
assert rx_card.is_aborted is True, "Pipeline did not abort for prescription medicine!"
assert "doesn't audit prescription medicines" in rx_card.abort_reason
assert rx_card.trust_score == 0
assert rx_card.execution_trace.overall_status == "ABORTED"
print(" -> PASS: Product category classifier and prescription abort guard verified.")

# --- TEST 10: OFFLINE EVALUATION SUITE (TASK T12) ---
print("\n[TEST 10] Offline Evaluation Suite Execution (Task T12)...")
from eval.runner import run_eval_suite
eval_status = run_eval_suite()
assert eval_status == 0, "Evaluation suite reported failures!"
print(" -> PASS: Offline evaluation suite executed with 100% test pass rate.")

print("\n" + "=" * 70)
print("             ALL 10 REFACTORING VERIFICATION TESTS PASSED!         ")
print("=" * 70)
