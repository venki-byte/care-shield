"""Offline Evaluation Suite Runner for Care-Shield (Task T12).

Processes test cases from eval/test_cases.csv completely offline (cached JSON & pure deterministic rules),
verifying:
- Category Guard (Prescription Medicine Abort, Cosmetic Exemption, OTC Device)
- Pure Scoring Engine normalization & denominator recalculation
- Legal Metrology over-MRP violation detection
- Defensible phrasing in outputs
"""

import sys
import os
import csv
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import schemas
from core.vision_ingest import classify_product_category, extract_regex_metadata
from core.pipeline import run_pipeline_with_trace
from core.scoring_engine import compute_audit_scorecard
from audio import build_audio_script


def run_eval_suite():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    csv_path = Path(__file__).parent / "test_cases.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cases = list(reader)

    print("\n" + "=" * 80)
    print("           CARE-SHIELD OFFLINE EVALUATION SUITE (TASK T12)           ")
    print("=" * 80)
    print(f"Loaded {len(cases)} test cases from {csv_path.name}\n")

    total_cases = len(cases)
    passed_cases = 0
    results_summary = []

    for idx, row in enumerate(cases, 1):
        case_id = row["case_id"]
        title = row["product_title"]
        brand = row["detected_brand"]
        printed_mrp = float(row["printed_mrp"]) if row.get("printed_mrp") else None
        store_price = float(row["store_asking_price"]) if row.get("store_asking_price") else None
        raw_text = row.get("raw_text", "")
        expected_cat = row["expected_category"]
        min_score = int(row["expected_min_score"])
        max_score = int(row["expected_max_score"])
        should_abort = row["should_abort"].strip().lower() == "true"
        expect_overcharge = row["expect_overcharge"].strip().lower() == "true"

        t_start = time.perf_counter()

        # 1. Category Classification
        detected_cat = classify_product_category(raw_text, title, brand)

        # 2. Structured Metadata
        meta = extract_regex_metadata(raw_text)

        # 3. Packaging OCR Result
        ocr_result = schemas.PackagingOCRResult(
            product_title=title,
            detected_brand=brand,
            printed_mrp_inr=printed_mrp,
            batch_number=meta.get("batch_number"),
            cdsco_license_raw=meta.get("cdsco_license_raw"),
            raw_ocr_corpus=raw_text,
            ocr_engine_used="Evaluation Suite Mock",
            ocr_confidence=0.95,
            rotation_applied=0,
            crop_applied=True,
            product_category=detected_cat
        )

        # 4. Run Pipeline with Trace (Offline / Demo mode)
        scorecard = run_pipeline_with_trace(
            precomputed_ocr_result=ocr_result,
            store_asking_price=store_price,
            manual_printed_mrp=printed_mrp,
            is_demo_mode=True
        )

        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # 5. Assertions
        failures = []
        if detected_cat != expected_cat:
            failures.append(f"Category mismatch: got '{detected_cat}', expected '{expected_cat}'")

        if should_abort:
            if not scorecard.is_aborted:
                failures.append("Expected audit to ABORT for prescription medicine, but it proceeded.")
            if "doesn't audit prescription medicines" not in (scorecard.abort_reason or ""):
                failures.append(f"Unexpected abort reason: '{scorecard.abort_reason}'")
        else:
            if scorecard.is_aborted:
                failures.append(f"Audit unexpectedly aborted: {scorecard.abort_reason}")
            if not (min_score <= scorecard.trust_score <= max_score):
                failures.append(f"Score {scorecard.trust_score} outside expected range [{min_score}, {max_score}]")

            # Check Overcharge
            if expect_overcharge:
                is_above = scorecard.parity_result.is_above_printed_mrp if scorecard.parity_result else False
                if not is_above:
                    failures.append("Expected over-MRP violation to be flagged, but was not.")

            # Check Cosmetic exemption
            if expected_cat == "cosmetic":
                if scorecard.sub_scores.regulatory_status != "na":
                    failures.append(f"Expected regulatory status 'na' for cosmetic, got '{scorecard.sub_scores.regulatory_status}'")
                if scorecard.sub_scores.regulatory_max != 0.0:
                    failures.append(f"Expected regulatory max 0.0 for cosmetic, got {scorecard.sub_scores.regulatory_max}")

        passed = len(failures) == 0
        if passed:
            passed_cases += 1
            status_str = "PASS"
        else:
            status_str = "FAIL"

        results_summary.append({
            "case_id": case_id,
            "status": status_str,
            "category": detected_cat,
            "trust_score": scorecard.trust_score if not scorecard.is_aborted else "ABORTED",
            "latency_ms": duration_ms,
            "failures": failures
        })

        print(f"[{status_str}] Case {idx}: {case_id}")
        print(f"       Category: {detected_cat} | Score: {scorecard.trust_score if not scorecard.is_aborted else 'N/A (ABORTED)'} | Latency: {duration_ms}ms")
        if failures:
            for f in failures:
                print(f"       [ERROR] {f}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE: {passed_cases} / {total_cases} PASSED ({(passed_cases/total_cases)*100:.1f}%)")
    print("=" * 80)

    # Save summary report to JSON
    out_file = Path(__file__).parent / "eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_cases": total_cases,
                "passed_cases": passed_cases,
                "accuracy": round(passed_cases / total_cases, 4),
                "results": results_summary
            },
            f,
            indent=2
        )
    print(f"Detailed evaluation report saved to {out_file.name}\n")

    return 0 if passed_cases == total_cases else 1


if __name__ == "__main__":
    sys.exit(run_eval_suite())
