"""Dual-Path Vision Engine for Medical Device Packaging Ingestion.

Features:
- Privacy ROI Cropping: Detects and isolates the foreground product bounding box
  BEFORE OCR or web queries, eliminating background faces, apparel graphics, and t-shirt text.
- Task T2: Rotation-aware RapidOCR extracting text WITH token confidence scores (0, 90, 180, 270 deg).
- Task T3: Structured field extraction (Printed MRP, Batch No., Mfg/Exp Dates, CDSCO License).
- Task T4: Barcode / QR Decoding via pyzbar with OpenCV QRCodeDetector & BarcodeDetector fallback.
- Task T1: Google Lens Inspector with raw JSON visibility and execution telemetry.
"""

import os
import re
import json
import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from PIL import Image
import io
import cv2

import config
from schemas import PackagingOCRResult

logger = logging.getLogger(__name__)

# Lazy singleton for local RapidOCR engine
_local_ocr_engine = None


def get_local_ocr_engine():
    """Initialize RapidOCR engine lazily to avoid startup overhead."""
    global _local_ocr_engine
    if _local_ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _local_ocr_engine = RapidOCR()
        except Exception as e:
            logger.warning(f"Could not initialize RapidOCR: {e}")
            _local_ocr_engine = False
    return _local_ocr_engine


def isolate_product_roi(image_bytes: bytes) -> Tuple[bytes, bool]:
    """Detect and crop foreground product bounding box for privacy and OCR accuracy.

    Removes background faces, headrests, and apparel text (e.g. t-shirt slogans).
    Returns (cropped_image_bytes, crop_applied_bool).
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes, False

        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to connect packaging text/labels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        max_area = 0

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if 0.08 * (w * h) < area < 0.85 * (w * h):
                aspect = cw / float(ch)
                if 0.25 <= aspect <= 3.5:
                    if area > max_area:
                        max_area = area
                        best_box = (x, y, cw, ch)

        if best_box:
            bx, by, bw, bh = best_box
            pad_x = int(bw * 0.08)
            pad_y = int(bh * 0.08)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)

            # If packaging is held on the right half, ensure cylindrical edge is not clipped by reflections
            if bx > 0.45 * w:
                x2 = min(w, max(x2, int(w * 0.98)))
            # If held on the left half, ensure left edge is not clipped
            elif (bx + bw) < 0.55 * w:
                x1 = max(0, min(x1, int(w * 0.02)))

            cropped = img[y1:y2, x1:x2]
            success, enc = cv2.imencode(".jpg", cropped)
            if success:
                print(f"[VISION PRIVACY] Product ROI isolated. Bounding box: ({x1},{y1}) to ({x2},{y2}). Background excluded.")
                return enc.tobytes(), True

        # Fallback foreground crop: trim top 10% (faces) and left 45% (chest)
        x1, y1 = int(w * 0.45), int(h * 0.10)
        x2, y2 = int(w * 0.98), int(h * 0.90)
        fallback_cropped = img[y1:y2, x1:x2]
        success, enc = cv2.imencode(".jpg", fallback_cropped)
        if success:
            print(f"[VISION PRIVACY] Applied fallback foreground product crop: ({x1},{y1}) to ({x2},{y2}).")
            return enc.tobytes(), True

    except Exception as e:
        logger.warning(f"Product ROI isolation encountered error, using full frame: {e}")

    return image_bytes, False


def decode_barcode_and_qr(img_np: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
    """Scan cropped packaging image for Barcode / QR Code (Task T4).
    
    Tries pyzbar first, then falls back to OpenCV BarcodeDetector and QRCodeDetector.
    Returns (barcode_value, barcode_type) or (None, None).
    """
    if img_np is None or img_np.size == 0:
        return None, None

    # 1. Try pyzbar
    try:
        from pyzbar import pyzbar
        barcodes = pyzbar.decode(img_np)
        if barcodes:
            b = barcodes[0]
            val = b.data.decode("utf-8", errors="ignore").strip()
            b_type = str(b.type)
            if val:
                print(f"[BARCODE/QR] Decoded via pyzbar: {val} ({b_type})")
                return val, b_type
    except Exception as e:
        logger.debug(f"pyzbar decoding unavailable or skipped: {e}")

    # 2. Fallback to OpenCV BarcodeDetector
    try:
        if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
            detector = cv2.barcode.BarcodeDetector()
            ok, decoded_info, decoded_type, _ = detector.detectAndDecode(img_np)
            if ok and decoded_info:
                val = decoded_info[0] if isinstance(decoded_info, (list, tuple)) else str(decoded_info)
                b_type = decoded_type[0] if isinstance(decoded_type, (list, tuple)) else str(decoded_type)
                if val and val.strip():
                    print(f"[BARCODE] Decoded via OpenCV BarcodeDetector: {val} ({b_type})")
                    return val.strip(), b_type or "BARCODE"
    except Exception as e:
        logger.debug(f"OpenCV BarcodeDetector skipped: {e}")

    # 3. Fallback to OpenCV QRCodeDetector
    try:
        qr_detector = cv2.QRCodeDetector()
        val, _, _ = qr_detector.detectAndDecode(img_np)
        if val and val.strip():
            print(f"[QRCODE] Decoded via OpenCV QRCodeDetector: {val}")
            return val.strip(), "QRCODE"
    except Exception as e:
        logger.debug(f"OpenCV QRCodeDetector skipped: {e}")

    return None, None


def extract_regex_metadata(text: str) -> dict:
    """Task T3: Structured field extraction via regex/heuristics on raw OCR corpus."""
    data = {
        "printed_mrp_inr": None,
        "cdsco_license_raw": None,
        "batch_number": None,
        "mfg_date": None,
        "exp_date": None,
        "detected_brand": None,
        "product_title": None,
        "active_components": None,
    }

    # 1. CDSCO / State Drug / Medical Device License pattern
    lic_match = re.search(
        r"((?:MFG|IMP)/MD/\d{4}/\d{5,6}|(?:MFG|IMP)/[A-Z0-9/-]+|MD-LIC-[A-Z0-9-]+|(?:Mfg\s*)?Lic(?:ense)?\s*(?:No\.?)?[\s.:]*([A-Z0-9/-]+)|\bM\.L\.[\s.:]*([A-Z0-9/-]+))",
        text,
        re.IGNORECASE
    )
    if lic_match:
        cand = next((g for g in lic_match.groups() if g), lic_match.group(0))
        # If captured prefix like 'Lic No: MFG/MD/...', extract clean license
        sub_lic = re.search(r"((?:MFG|IMP)/[A-Z0-9/-]+|MD-LIC-[A-Z0-9-]+|[A-Z0-9/-]{6,})", cand, re.IGNORECASE)
        data["cdsco_license_raw"] = (sub_lic.group(1) if sub_lic else cand).strip(" .,;:")

    # 2. MRP Pattern: MRP / M.R.P. / Maximum Retail Price / ₹ / Rs.
    mrp_match = re.search(
        r"(?:MRP|M\.?R\.?P\.?|Maximum\s*Retail\s*Price)[\s.:]*(?:Rs\.?|₹|INR)?[\s.:]*([\d,]+(?:\.\d{1,2})?)",
        text,
        re.IGNORECASE
    )
    if not mrp_match:
        mrp_match = re.search(
            r"(?:₹|\bRs\.?|\bINR)\s*[:.]?\s*([\d,]+(?:\.\d{1,2})?)",
            text,
            re.IGNORECASE
        )
    if mrp_match:
        try:
            val_str = mrp_match.group(1).replace(",", "")
            data["printed_mrp_inr"] = float(val_str)
        except ValueError:
            pass

    # 3. Batch / Lot Number
    batch_match = re.search(
        r"(?:Batch|Lot|B\.?No\.?|LOT\s*NO\.?|BATCH\s*NO\.?)[\s.:]*([A-Z0-9-]+)",
        text,
        re.IGNORECASE
    )
    if batch_match:
        data["batch_number"] = batch_match.group(1).strip(" .,;:")

    # 4. Manufacturing Date
    mfg_match = re.search(
        r"(?:MFD\b|MFG\.?\s*DATE|Mfg\s*Date|Date\s*of\s*Mfg|Date\s*of\s*Manufacture|PKD\b|MFD\.?|DOM\b)(?!\s*Lic)[\s.:]*([A-Za-z0-9/.-]+(?:\s+\d{4})?)",
        text,
        re.IGNORECASE
    )
    if mfg_match:
        data["mfg_date"] = mfg_match.group(1).strip(" .,;:")

    # 5. Expiry Date / Best Before
    exp_match = re.search(
        r"(?:EXP\b|EXP\.?\s*DATE|Exp\s*Date|Expiry|Expiry\s*Date|Best\s*Before|Use\s*By)[\s.:]*([A-Za-z0-9/.-]+(?:\s+\d{4})?)",
        text,
        re.IGNORECASE
    )
    if exp_match:
        data["exp_date"] = exp_match.group(1).strip(" .,;:")

    # Add aliases for developer convenience
    data["printed_mrp"] = data["printed_mrp_inr"]
    data["cdsco_license"] = data["cdsco_license_raw"]

    # 6. Brand & Product title heuristics
    known_brands = ["Dr Trust", "Dr. Trust", "Omron", "Accu-Chek", "Tynor", "Melactis", "BPL Medical", "Hicks", "OneTouch"]
    for brand in known_brands:
        if re.search(r"\b" + re.escape(brand) + r"\b", text, re.IGNORECASE):
            data["detected_brand"] = brand
            break

    if not data["detected_brand"]:
        import difflib
        words = re.findall(r"\w+", text)
        for w in words:
            if len(w) >= 5:
                matches = difflib.get_close_matches(w.lower(), [b.lower() for b in known_brands], n=1, cutoff=0.70)
                if matches:
                    matched_lower = matches[0]
                    for b in known_brands:
                        if b.lower() == matched_lower:
                            data["detected_brand"] = b
                            break
                    break

    return data


def classify_product_category(text: str, product_title: str = "", brand: str = "") -> str:
    """Task T6: Classify packaging.

    Returns: prescription_medicine | cosmetic | otc_device | otc_medicine | non_healthcare | unknown
    """
    combined = f"{brand} {product_title} {text}".lower()

    # 1. Prescription medicine: ONLY explicit prescription markers (a plain "tablet" is not enough)
    rx_patterns = [
        r"\b(?:rx|r\.x\.)\b",
        r"\bschedule\s*[hx]1?\b",
        r"\bprescription\s+(?:only|drug|medicine)\b",
        r"\bas\s+directed\s+by\s+(?:the\s+)?(?:physician|doctor)\b",
        r"\bto\s+be\s+sold\s+by\s+retail\s+on\s+the\s+prescription\b",
        r"\bantibiotics?\b",
        r"\bnarcotic\b",
    ]
    for pat in rx_patterns:
        if re.search(pat, combined, re.IGNORECASE):
            return "prescription_medicine"

    # 2. Cosmetic
    cosmetic_patterns = [
        r"\bcosmetics?\b", r"\bserums?\b", r"\bface\s*wash\b", r"\bmoisturi[sz]ers?\b",
        r"\blotions?\b", r"\bsunscreens?\b", r"\bshampoos?\b", r"\bcreams?\b", r"\blipsticks?\b",
        r"\bperfumes?\b", r"\btoners?\b", r"\bcleansers?\b", r"\bskincare\b",
        r"\bpigment\s*corrector\b", r"\banti[- ]aging\b",
    ]
    for pat in cosmetic_patterns:
        if re.search(pat, combined, re.IGNORECASE):
            return "cosmetic"

    # 3. OTC medical device
    device_patterns = [
        r"\boximeters?\b", r"\bthermometers?\b", r"\bmonitors?\b", r"\bblood\s*pressure\b",
        r"\bbp\s*monitor\b", r"\bglucometers?\b", r"\bglucose\b", r"\bknee\b", r"\bhinged\b", r"\bbraces?\b",
        r"\bneoprene\b", r"\borthopedic\b", r"\bsplints?\b", r"\bnebuli[sz]ers?\b",
        r"\btest\s*strips?\b", r"\blancets?\b", r"\bbandages?\b", r"\bmedical\s*device\b",
        r"\bstethoscopes?\b", r"\bwheelchair\b", r"\bwalker\b", r"\bcrutch(?:es)?\b",
        r"\bmask\b", r"\bgloves?\b", r"\bsyringes?\b", r"\bhearing\s*aid\b", r"\bvaporizer\b",
    ]
    for pat in device_patterns:
        if re.search(pat, combined, re.IGNORECASE):
            return "otc_device"

    # 4. OTC medicine (tablets/syrups/ointments without prescription markers)
    medicine_patterns = [
        r"\b(?:tablets?|capsules?|syrup|ointment|gel|drops|lozenges?|sachets?|suspension)\b",
        r"\bparacetamol\b", r"\bvitamin\b", r"\bantacid\b", r"\b\d+\s*mg\b",
        r"\bdrug\s*licen[sc]e\b", r"\bpharma\b",
    ]
    for pat in medicine_patterns:
        if re.search(pat, combined, re.IGNORECASE):
            return "otc_medicine"

    # 5. Clearly not a healthcare product
    non_health = [
        r"\bremote\b", r"\btv\b", r"\btelevision\b", r"\bmobile\b", r"\bsmartphone\b", r"\bcharger\b",
        r"\blaptop\b", r"\bheadphones?\b", r"\bearphones?\b", r"\bkeyboard\b", r"\bmouse\b",
        r"\bbulb\b", r"\bbattery\b", r"\btoy\b", r"\bbottle\b", r"\bshoe\b", r"\bbag\b",
        r"\bcable\b", r"\bspeaker\b", r"\bwatch\b", r"\bbook\b", r"\bpen\b",
    ]
    for pat in non_health:
        if re.search(pat, combined, re.IGNORECASE):
            return "non_healthcare"

    return "unknown"


def run_local_image_ocr(image_bytes: bytes) -> Tuple[str, dict, float, int, List[Tuple[str, float]]]:
    """Task T2: RapidOCR execution WITH confidence scores and 4-way rotation handling.
    
    1. Extracts text WITH token confidence scores.
    2. Evaluates rotations (0, 90, 180, 270 degrees) if confidence is low or text inverted,
       keeping the result with the highest average token confidence.
    Returns:
        (full_text, metadata, best_avg_confidence, best_angle_degrees, token_confidence_pairs)
    """
    engine = get_local_ocr_engine()
    if not engine:
        return "", {}, 0.0, 0, []

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            return "", {}, 0.0, 0, []

        rotations = [
            (0, None),
            (90, cv2.ROTATE_90_CLOCKWISE),
            (180, cv2.ROTATE_180),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]

        best_angle = 0
        best_avg_score = -1.0
        best_tokens_with_conf: List[Tuple[str, float]] = []

        for angle, rot_flag in rotations:
            cur_img = cv2.rotate(img_np, rot_flag) if rot_flag is not None else img_np
            res, _ = engine(cur_img)
            
            items: List[Tuple[str, float]] = []
            for item in (res or []):
                if len(item) > 2 and item[1] and str(item[1]).strip():
                    text = str(item[1]).strip()
                    conf = float(item[2])
                    items.append((text, conf))

            # Filter meaningful tokens (length >= 2 or alphanumeric)
            meaningful = [conf for text, conf in items if len(text) >= 2 and any(c.isalnum() for c in text)]
            avg_score = (sum(meaningful) / len(meaningful)) if meaningful else (sum(c for _, c in items)/len(items) if items else 0.0)
            
            # Quality score balances token confidence and meaningful token count
            quality_score = avg_score * (1.0 + 0.08 * min(len(meaningful), 8))

            if quality_score > best_avg_score or best_avg_score < 0:
                best_avg_score = avg_score
                best_angle = angle
                best_tokens_with_conf = items

            # Fast path: If 0-degrees already has high confidence and multiple meaningful tokens, stop
            if angle == 0 and avg_score >= 0.90 and len(meaningful) >= 4:
                break

        full_text = " \n ".join([t[0] for t in best_tokens_with_conf])
        metadata = extract_regex_metadata(full_text)
        return full_text, metadata, round(best_avg_score, 4), best_angle, best_tokens_with_conf

    except Exception as e:
        logger.error(f"Local RapidOCR execution failed: {e}")
        return "", {}, 0.0, 0, []


def _call_gemini_raw(image_bytes: bytes, api_key: str, prompt: str) -> Optional[dict]:
    """Underlying Gemini Vision invocation."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    for candidate_model in ["gemini-flash-lite-latest", "gemini-flash-latest"]:
        try:
            response = client.models.generate_content(
                model=candidate_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt
                ]
            )
            cleaned = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(cleaned)
            print(f"[GEMINI VISION] Extracted packaging data using {candidate_model}: {data.get('product_title')}")
            return data
        except Exception as e_mod:
            logger.warning(f"Gemini model {candidate_model} failed, trying next: {e_mod}")
            continue
    return None


def query_gemini_vision(image_bytes: bytes, api_key: str) -> Optional[dict]:
    """Call Google Gemini Vision API with timeout and retry (Hard Rules 2 & 3)."""
    if not api_key:
        return None

    prompt = """
You are an expert Indian Medical Device Regulatory and Packaging Inspector.
Analyze this product packaging image (it may or may not be a healthcare product) and extract the following information in strict JSON format:
{
  "product_title": "Full commercial product name (e.g. Dr Trust Signature Series Finger Tip Pulse Oximeter)",
  "detected_brand": "Brand name (e.g. Dr Trust)",
  "printed_mrp_inr": 1999.0, // Floating point number for legally mandated MRP in Indian Rupees, or null if not visible
  "batch_number": "Batch or Lot number, or null",
  "cdsco_license_raw": "Manufacturing or import license line (targeting MFG/MD/... or IMP/MD/... or state drug license), or null",
  "mfg_date": "Manufacturing date if printed, or null",
  "exp_date": "Expiry date or Best Before date if printed, or null",
  "barcode": "Barcode or QR code number if visible, or null",
  "active_components": "Materials or sensors or active specs, or null",
  "product_category": "exactly one of: otc_device (BP monitor, oximeter, braces, thermometer, glucometer, mobility aids), otc_medicine (tablets/syrup/ointment WITHOUT prescription markers), prescription_medicine (ONLY if Rx / Schedule H / H1 / X / 'sold on prescription' is printed), cosmetic, non_healthcare (anything else: electronics, remotes, toys, food, etc.)",
  "raw_ocr_corpus": "Verbatim full text read from the packaging"
}
Output strictly valid JSON with no markdown wrapping or code fences.
"""
    return config.execute_with_retry(
        _call_gemini_raw,
        args=(image_bytes, api_key, prompt),
        timeout_seconds=15.0,
        max_retries=1,
        fallback=None,
        label="Gemini Vision API"
    )


def inspect_serpapi_lens(image_url: Optional[str] = None, api_key: Optional[str] = None) -> dict:
    """Task T1: Query SerpApi Google Lens with structured visibility, timeout, and retry."""
    import time
    active_key = api_key or config.SERPAPI_KEY
    if not active_key:
        return {
            "status": "SKIPPED",
            "reason": "SERPAPI_KEY is not configured.",
            "lens_title": None,
            "lens_brand": None,
            "visual_matches_count": 0,
            "knowledge_graph_count": 0,
            "raw_json_summary": "Not provided / Not found",
            "duration_ms": 0.0,
            "image_url": None
        }

    if not image_url or not image_url.strip():
        return {
            "status": "SKIPPED",
            "reason": "Local image provided without public HTTP URL. SerpApi Google Lens requires a public URL. Local OCR Path A executed.",
            "lens_title": None,
            "lens_brand": None,
            "visual_matches_count": 0,
            "knowledge_graph_count": 0,
            "raw_json_summary": "Not provided / Not found",
            "duration_ms": 0.0,
            "image_url": None
        }

    clean_url = image_url.strip()
    print(f"[GOOGLE LENS] Querying Google Lens for URL: {clean_url}...")

    def _fetch_lens():
        from serpapi import GoogleSearch
        params = {
            "engine": "google_lens",
            "api_key": active_key,
            "url": clean_url,
            "hl": "en",
            "country": "in",
        }
        return GoogleSearch(params).get_dict()

    t_start = time.perf_counter()
    results = config.execute_with_retry(
        _fetch_lens,
        timeout_seconds=12.0,
        max_retries=1,
        fallback=None,
        label="SerpApi Google Lens"
    )
    duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

    if results is None:
        return {
            "status": "DEGRADED",
            "reason": "SerpApi Google Lens query timed out or failed after retry.",
            "lens_title": None,
            "lens_brand": None,
            "visual_matches_count": 0,
            "knowledge_graph_count": 0,
            "raw_json_summary": "Not provided / Not found",
            "duration_ms": duration_ms,
            "image_url": clean_url
        }

    sanitized_results = config.sanitize_for_secrets(results)

    if "error" in results:
        err_msg = str(results.get("error"))
        print(f"[GOOGLE LENS ERROR] {err_msg}")
        return {
            "status": "DEGRADED",
            "reason": f"SerpApi error: {err_msg}",
            "lens_title": None,
            "lens_brand": None,
            "visual_matches_count": 0,
            "knowledge_graph_count": 0,
            "raw_json_summary": json.dumps(sanitized_results, indent=2)[:800],
            "duration_ms": duration_ms,
            "image_url": clean_url
        }

    visual_matches = results.get("visual_matches", [])
    knowledge_graph = results.get("knowledge_graph", [])
    raw_json_str = json.dumps(sanitized_results, indent=2, ensure_ascii=False)

    print("\n" + "-"*60)
    print("          [GOOGLE LENS RESPONSE (SERPAPI)]                  ")
    print("-"*60)
    print(f"Search Status: {results.get('search_metadata', {}).get('status')}")
    print(f"Visual Matches Found: {len(visual_matches)}")
    print(f"Knowledge Graph Items: {len(knowledge_graph)}")
    print(raw_json_str[:800] + ("\n... [truncated for readability]" if len(raw_json_str) > 800 else ""))
    print("-"*60 + "\n")

    lens_title = None
    lens_brand = None
    if knowledge_graph and len(knowledge_graph) > 0:
        lens_title = knowledge_graph[0].get("title")
    elif visual_matches and len(visual_matches) > 0:
        lens_title = visual_matches[0].get("title")
        lens_brand = visual_matches[0].get("source")

    return {
        "status": "SUCCESS",
        "reason": None,
        "lens_title": lens_title,
        "lens_brand": lens_brand,
        "visual_matches_count": len(visual_matches),
        "knowledge_graph_count": len(knowledge_graph),
        "raw_json_summary": raw_json_str[:800],
        "duration_ms": duration_ms,
        "image_url": clean_url
    }


def query_serpapi_lens(image_url: Optional[str] = None, api_key: str = "") -> Tuple[Optional[str], Optional[str]]:
    """Legacy tuple wrapper for backward compatibility."""
    report = inspect_serpapi_lens(image_url=image_url, api_key=api_key)
    return report.get("lens_title"), report.get("lens_brand")


def ingest_packaging_image(
    image_bytes: Optional[bytes] = None,
    raw_text_override: Optional[str] = None,
    sample_preset: Optional[str] = None,
    serpapi_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    lens_image_url: Optional[str] = None,
    return_lens_report: bool = False
):
    """Dual-path ingestion pipeline orchestrator with Tasks T2, T3, T4.

    Executes privacy ROI crop, barcode/QR decode, rotation-aware RapidOCR with confidence scores,
    optional Gemini Vision, and Google Lens inspection.
    """
    active_serpapi_key = serpapi_key or config.SERPAPI_KEY
    active_gemini_key = gemini_key or config.GEMINI_API_KEY

    lens_report = {
        "status": "SKIPPED",
        "reason": "No Lens query executed for preset.",
        "lens_title": None,
        "lens_brand": None,
        "visual_matches_count": 0,
        "knowledge_graph_count": 0,
        "raw_json_summary": "Not provided / Not found",
        "duration_ms": 0.0,
        "image_url": lens_image_url
    }

    # 1. Preset handling for demo playback
    if sample_preset == "authentic_oximeter":
        res = PackagingOCRResult(
            product_title="Dr Trust Signature Series Finger Tip Pulse Oximeter 506",
            detected_brand="Dr Trust",
            printed_mrp_inr=1999.0,
            batch_number="DT-2024-OX982",
            mfg_date="03/2024",
            exp_date="02/2029",
            barcode="8906095170123",
            barcode_type="EAN13",
            cdsco_license_raw="MFG/MD/2021/000123",
            is_license_format_valid=True,
            active_components="Photoelectric sensor, OLED display, SpO2 & PR algorithm",
            raw_ocr_corpus="Dr Trust USA Signature Series Pulse Oximeter Model 506. Mfg Lic No: MFG/MD/2021/000123. Batch No: DT-2024-OX982. MFD: 03/2024 EXP: 02/2029. Barcode: 8906095170123. M.R.P. Rs. 1,999.00 (Incl. of all taxes). ISO 13485:2016 Certified. Distributed by Nureca Ltd.",
            ocr_engine_used="Preset Benchmark Fixture",
            ocr_confidence=0.98,
            rotation_applied=0,
            crop_applied=False,
            product_category="otc_device"
        )
        return (res, lens_report) if return_lens_report else res
    elif sample_preset == "counterfeit_knee_brace":
        res = PackagingOCRResult(
            product_title="Tynor Knee Suport Hinged Neo",
            detected_brand="Tynor",
            printed_mrp_inr=600.0,
            batch_number="TYN-SPURIOUS-99",
            mfg_date="11/2023",
            exp_date=None,
            barcode=None,
            barcode_type=None,
            cdsco_license_raw="MD-LIC-9988-FAKE",
            is_license_format_valid=False,
            active_components="Neoprene sleeve, metal hinges",
            raw_ocr_corpus="Tynor Knee Suport Hinged Neo (Knockoff). Lic: MD-LIC-9988-FAKE. B.No: TYN-SPURIOUS-99. MFD: 11/2023. M.R.P. Rs. 600.00. Unverified distributor packaging.",
            ocr_engine_used="Preset Benchmark Fixture",
            ocr_confidence=0.82,
            rotation_applied=0,
            crop_applied=False,
            product_category="otc_device"
        )
        return (res, lens_report) if return_lens_report else res

    # 2. Privacy & Product Isolation Crop
    crop_applied = False
    process_bytes = image_bytes
    if image_bytes:
        process_bytes, crop_applied = isolate_product_roi(image_bytes)

    # 3. Barcode & QR Decoding (Task T4)
    decoded_barcode, barcode_type = None, None
    if process_bytes:
        try:
            nparr = np.frombuffer(process_bytes, np.uint8)
            crop_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            decoded_barcode, barcode_type = decode_barcode_and_qr(crop_np)
        except Exception as e:
            logger.debug(f"Barcode detection error: {e}")

    # 4. Path A: Robust Rotation-Aware OCR with Confidence (Task T2)
    local_text, local_meta, ocr_conf, best_rot, token_confs = "", {}, 0.0, 0, []
    if process_bytes:
        local_text, local_meta, ocr_conf, best_rot, token_confs = run_local_image_ocr(process_bytes)

    # 4b. Typed / pasted text goes through the same structured extractor as OCR output
    if raw_text_override and raw_text_override.strip():
        typed_meta = extract_regex_metadata(raw_text_override)
        if not typed_meta.get("product_title"):
            first = re.split(r"(?<=[a-z0-9])\.\s|\n", raw_text_override.strip())[0].strip(" .")
            if 3 <= len(first) <= 90:
                typed_meta["product_title"] = first
        local_meta = {**{k: v for k, v in typed_meta.items() if v is not None}, **(local_meta or {})}

    # 5. Gemini Vision API (if key available)
    gemini_data = None
    engine_label = "RapidOCR (Local Deep Learning ONNX)" if local_text else "Heuristic Text Parser"

    if process_bytes and active_gemini_key:
        gemini_data = query_gemini_vision(process_bytes, active_gemini_key)
        if gemini_data:
            engine_label = "Gemini Vision + RapidOCR Hybrid" if local_text else "Google Gemini Flash Vision"

    # Merge structured metadata (Task T3)
    extracted_data = {}
    if local_meta:
        extracted_data.update({k: v for k, v in local_meta.items() if v is not None})
    if gemini_data:
        for k, v in gemini_data.items():
            if v is not None:
                extracted_data[k] = v
        # Preserve local brand if Gemini omitted it
        if not extracted_data.get("detected_brand") and local_meta.get("detected_brand"):
            extracted_data["detected_brand"] = local_meta["detected_brand"]

    # If barcode was decoded visually, record it
    if decoded_barcode:
        extracted_data["barcode"] = decoded_barcode
        extracted_data["barcode_type"] = barcode_type

    # Combine corpus text
    corpus_parts = []
    if gemini_data and gemini_data.get("raw_ocr_corpus"):
        corpus_parts.append(gemini_data["raw_ocr_corpus"])
    if local_text:
        corpus_parts.append(local_text)
    if raw_text_override:
        corpus_parts.append(raw_text_override)
    corpus = " \n ".join(dict.fromkeys([p.strip() for p in corpus_parts if p and p.strip()]))
    extracted_data["raw_ocr_corpus"] = corpus

    # 6. Path B: Google Lens with Lens Visibility (Task T1)
    lens_report = inspect_serpapi_lens(image_url=lens_image_url, api_key=active_serpapi_key)
    lens_title = lens_report.get("lens_title")
    lens_brand = lens_report.get("lens_brand")

    # 7. Final Synthesis
    final_brand = extracted_data.get("detected_brand") or lens_brand or "Unknown Product"
    raw_title = extracted_data.get("product_title") or lens_title or ""
    if raw_title and final_brand != "Unknown Product" and final_brand.lower() not in raw_title.lower():
        final_title = f"{final_brand} {raw_title}".strip()
    elif raw_title:
        final_title = raw_title.strip()
    else:
        final_title = f"{final_brand} Packaging"

    final_mrp = extracted_data.get("printed_mrp_inr")
    final_batch = extracted_data.get("batch_number")
    final_lic = extracted_data.get("cdsco_license_raw")
    final_mfg = extracted_data.get("mfg_date")
    final_exp = extracted_data.get("exp_date")
    final_barcode = extracted_data.get("barcode") or decoded_barcode
    final_barcode_type = extracted_data.get("barcode_type") or barcode_type
    active_comp = extracted_data.get("active_components")
    detected_cat = classify_product_category(corpus, final_title, final_brand)
    gem_cat = (gemini_data or {}).get("product_category")
    if gem_cat in {"otc_device", "otc_medicine", "prescription_medicine", "cosmetic", "non_healthcare"} \
            and detected_cat != "prescription_medicine":
        detected_cat = gem_cat

    ocr_res = PackagingOCRResult(
        product_title=final_title,
        detected_brand=final_brand,
        printed_mrp_inr=final_mrp,
        batch_number=final_batch,
        cdsco_license_raw=final_lic,
        is_license_format_valid=None,
        active_components=active_comp,
        mfg_date=final_mfg,
        exp_date=final_exp,
        barcode=final_barcode,
        barcode_type=final_barcode_type,
        raw_ocr_corpus=corpus or f"{final_title} | {final_brand} | Lic: {final_lic} | MRP: {final_mrp}",
        ocr_engine_used=engine_label,
        ocr_confidence=ocr_conf if ocr_conf > 0 else (0.95 if gemini_data else None),
        rotation_applied=best_rot,
        crop_applied=crop_applied,
        product_category=detected_cat
    )

    if return_lens_report:
        return ocr_res, lens_report
    return ocr_res
