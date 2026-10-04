"""Rigid Pydantic v2 data models for every DAG hop in Care-Shield.

Enforces strict JSON schema validation across Vision Ingestion, Parity Checking,
Regulatory Guards, Safety Intel, Map Routing, Scoring nodes, and Pipeline Execution Tracing.
Strictly adheres to Hard Rules:
- Rule 1: Missing fields must be None and rendered as 'Not provided / Not found'.
- Rule 4: Strict Pydantic v2 validation for all inter-node contracts.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class PackagingOCRResult(BaseModel):
    """Output contract for packaging ingestion, rotation-handling OCR, and barcode decoding."""
    product_title: str = Field(..., description="Canonical or extracted commercial product title")
    detected_brand: str = Field(..., description="Detected brand name")
    printed_mrp_inr: Optional[float] = Field(default=None, description="Legally mandated printed MRP in INR (None if not visible or unstated)")
    batch_number: Optional[str] = Field(default=None, description="Extracted Batch/Lot identifier")
    cdsco_license_raw: Optional[str] = Field(default=None, description="Extracted CDSCO manufacturing/import license string")
    is_license_format_valid: Optional[bool] = Field(default=None, description="True if format strictly matches CDSCO MDR rules, False if malformed, None if absent/NA")
    active_components: Optional[str] = Field(default=None, description="Key active ingredients or materials")
    mfg_date: Optional[str] = Field(default=None, description="Manufacturing date on packaging")
    exp_date: Optional[str] = Field(default=None, description="Expiry date or Best Before date on packaging")
    barcode: Optional[str] = Field(default=None, description="Decoded Barcode or QR code value")
    barcode_type: Optional[str] = Field(default=None, description="Barcode symbology (e.g. EAN13, QRCODE, CODE128)")
    raw_ocr_corpus: str = Field(default="", description="Full raw OCR text for audit trails")
    ocr_engine_used: str = Field(default="Heuristic / Regex", description="Name of OCR engine that extracted the packaging details")
    ocr_confidence: Optional[float] = Field(default=None, description="Average token confidence score (0.0 to 1.0)")
    rotation_applied: int = Field(default=0, description="Orientation angle used for OCR: 0, 90, 180, or 270 degrees")
    crop_applied: bool = Field(default=False, description="True if privacy ROI bounding box crop was applied to isolate product")
    product_category: str = Field(default="otc_device", description="Detected category: cosmetic, otc_device, prescription_medicine, or unknown")

    # Hard Rule 1: Missing fields rendered as 'Not provided / Not found'
    @property
    def printed_mrp_display(self) -> str:
        return f"₹{self.printed_mrp_inr:,.2f}" if self.printed_mrp_inr is not None else "Not provided / Not found"

    @property
    def batch_number_display(self) -> str:
        return self.batch_number if self.batch_number else "Not provided / Not found"

    @property
    def cdsco_license_display(self) -> str:
        return self.cdsco_license_raw if self.cdsco_license_raw else "Not provided / Not found"

    @property
    def active_components_display(self) -> str:
        return self.active_components if self.active_components else "Not provided / Not found"

    @property
    def mfg_date_display(self) -> str:
        return self.mfg_date if self.mfg_date else "Not provided / Not found"

    @property
    def exp_date_display(self) -> str:
        return self.exp_date if self.exp_date else "Not provided / Not found"

    @property
    def barcode_display(self) -> str:
        if self.barcode:
            b_type = f" ({self.barcode_type})" if self.barcode_type else ""
            return f"{self.barcode}{b_type}"
        return "Not provided / Not found"

    @property
    def product_category_display(self) -> str:
        mapping = {
            "cosmetic": "Cosmetic / Skincare (Exempt from CDSCO MDR)",
            "otc_device": "Over-The-Counter Medical Device",
            "prescription_medicine": "Prescription Medicine (Schedule H/Rx)",
            "otc_medicine": "Over-The-Counter Medicine (Tablet / Syrup / Ointment)",
            "non_healthcare": "Not a healthcare product",
            "unknown": "General Healthcare / Unspecified",
        }
        return mapping.get(self.product_category, self.product_category.title())


class BenchmarkMerchant(BaseModel):
    """Representative retail benchmark price point from verified Indian pharmacies."""
    name: str = Field(..., description="Merchant name (e.g. Apollo Pharmacy, Tata 1mg, Netmeds)")
    price: float = Field(..., description="Listed price in INR")
    link: Optional[str] = Field(default=None, description="Product listing URL")
    extracted_title: Optional[str] = Field(default=None, description="Merchant listed product title")


class ShoppingParityResult(BaseModel):
    """Financial metrology and pricing parity audit results."""
    online_prices: List[float] = Field(default_factory=list, description="Array of prices gathered from Indian pharmacy sources")
    online_median: Optional[float] = Field(default=None, description="Cleaned online median price in INR")
    iqr_filtered_min: Optional[float] = Field(default=None, description="IQR minimum threshold after outlier removal")
    iqr_filtered_max: Optional[float] = Field(default=None, description="IQR maximum threshold after outlier removal")
    scanned_purchase_price: Optional[float] = Field(default=None, description="Physical store asking price entered")
    is_above_printed_mrp: bool = Field(default=False, description="Flagged as illegal overcharge if asking price > printed MRP")
    mrp_audit_skipped: bool = Field(default=False, description="True if printed MRP was omitted, skipping over-MRP legal check")
    markup_percent: float = Field(default=0.0, description="Markup percentage relative to cleaned online median")
    benchmark_merchants: List[BenchmarkMerchant] = Field(default_factory=list, description="List of verified merchant price points")
    is_live_query: bool = Field(default=False, description="True if price benchmarks were fetched via live SerpApi Google Shopping call")
    api_warning: Optional[str] = Field(default=None, description="Warning message if live API call could not be made")

    @property
    def scanned_price_display(self) -> str:
        return f"₹{self.scanned_purchase_price:,.2f}" if self.scanned_purchase_price is not None else "Not provided / Not found"

    @property
    def online_median_display(self) -> str:
        return f"₹{self.online_median:,.2f}" if self.online_median is not None else "Not provided / Not found"


class SafetyIntelResult(BaseModel):
    """Adversarial safety search result checking recalls, seizures, and counterfeit alerts."""
    has_active_recall: bool = Field(default=False, description="Whether active recall or spurious drug notices exist")
    risk_factor: float = Field(default=0.0, ge=0.0, le=1.0, description="Threat conviction factor from 0.0 (safe) to 1.0 (high risk)")
    evidence_snippets: List[str] = Field(default_factory=list, description="Matched snippets citing CDSCO warnings, counterfeit raids, etc.")
    is_live_query: bool = Field(default=False, description="True if threat intel was fetched via live SerpApi Google Search")
    api_warning: Optional[str] = Field(default=None, description="Warning message if live API call could not be made")


class NearbyPharmacy(BaseModel):
    """Physical pharmacy alternative retrieved for remediation."""
    name: str = Field(..., description="Pharmacy name")
    rating: Optional[float] = Field(default=None, description="Google Maps user rating (>= 4.0 required)")
    review_count: Optional[int] = Field(default=None, description="Total verified Google user reviews count (>= 10 required)")
    formatted_address: str = Field(..., description="Physical street address")
    is_open: Optional[bool] = Field(default=None, description="Operational open status")
    directions_url: str = Field(..., description="Direct Google Maps navigation URL with place_id / coordinates")
    place_id: Optional[str] = Field(default=None, description="Google Maps Place ID")
    latitude: Optional[float] = Field(default=None, description="Latitude coordinate")
    longitude: Optional[float] = Field(default=None, description="Longitude coordinate")
    is_live_lookup: bool = Field(default=False, description="True if retrieved via live SerpApi Google Maps lookup")


class SubScores(BaseModel):
    """Granular sub-scores supporting dynamic N/A bases and Task T5 exact scoring rules."""
    text_integrity: float = Field(..., ge=0.0, le=35.0, description="Text & spelling integrity sub-score (0-35)")
    text_max: float = Field(default=35.0, description="Maximum possible points for text integrity")
    text_status: str = Field(default="ok", description="'ok', 'warn', or 'fail'")
    text_matching_tokens_count: int = Field(default=0, description="Exact token match count between OCR and canonical specs")
    
    regulatory: Optional[float] = Field(default=None, description="CDSCO MDR format compliance sub-score (None if N/A)")
    regulatory_max: float = Field(default=25.0, description="Maximum possible points for regulatory (0 if N/A)")
    regulatory_status: str = Field(default="na", description="'valid', 'invalid', or 'na'")
    regulatory_message: str = Field(default="License format not evaluated", description="Text e.g. 'License format looks valid/invalid'")
    
    pricing_fairness: Optional[float] = Field(default=None, description="Price fairness sub-score (None if N/A)")
    pricing_max: float = Field(default=20.0, description="Maximum possible points for pricing (0 if N/A)")
    pricing_status: str = Field(default="ok", description="'ok', 'warn', 'fail', or 'na'")
    
    safety_intel: float = Field(default=20.0, ge=0.0, le=20.0, description="Recall & counterfeit adversarial threat sub-score (0-20)")
    safety_max: float = Field(default=20.0, description="Maximum possible points for safety intel")
    safety_status: str = Field(default="ok", description="'ok', 'warn', or 'fail'")
    safety_message: str = Field(default="No warning signs found", description="Human-readable safety notice")
    
    max_available_points: float = Field(default=100.0, description="Dynamic total denominator base (e.g. 75 if regulatory is N/A)")
    total_earned_points: float = Field(..., description="Total points earned across applicable dimensions")


class TraceStep(BaseModel):
    """Single node execution record in the forensic pipeline trace."""
    step_id: str = Field(..., description="Step code e.g. T1.1_ROI_CROP")
    step_name: str = Field(..., description="Human-readable step name")
    status: str = Field(..., description="SUCCESS | SKIPPED | DEGRADED | FAILED")
    duration_ms: float = Field(default=0.0, description="Step execution time in milliseconds")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters (strictly masked)")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output facts (missing = 'Not provided / Not found')")
    telemetry: Dict[str, Any] = Field(default_factory=dict, description="Diagnostics, engine details, HTTP status")
    error: Optional[str] = Field(default=None, description="Error message if degraded or failed")


class PipelineExecutionTrace(BaseModel):
    """Complete end-to-end execution trace for all DAG hops."""
    trace_id: str = Field(..., description="Unique UUID for this pipeline execution trace")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Execution start timestamp")
    total_duration_ms: float = Field(default=0.0, description="Total pipeline latency in ms")
    overall_status: str = Field(default="SUCCESS", description="Overall execution status")
    lens_visibility: Dict[str, Any] = Field(default_factory=dict, description="Dedicated Google Lens visibility & telemetry summary")
    steps: List[TraceStep] = Field(default_factory=list, description="Ordered step traces")


class AuditScorecard(BaseModel):
    """Master audit verdict payload bundling scores, evidence, trace, and remediation."""
    trust_score: int = Field(..., ge=0, le=100, description="Composite Trust Score normalized to 0-100%")
    verdict: str = Field(..., description="Defensible categorical verdict string")
    sub_scores: SubScores = Field(..., description="Granular dynamic sub-scores")
    evidence_list: List[str] = Field(default_factory=list, description="Human-readable forensic findings and reasons")
    ocr_result: PackagingOCRResult = Field(..., description="Ingested packaging data")
    parity_result: Optional[ShoppingParityResult] = Field(default=None, description="Financial audit result")
    intel_result: Optional[SafetyIntelResult] = Field(default=None, description="Safety and recall intelligence result")
    fallback_pharmacies: List[NearbyPharmacy] = Field(default_factory=list, description="Recommended alternative pharmacies")
    trigger_map_remediation: bool = Field(default=False, description="True if score < 70, over-MRP, or fraud intel alert triggered map router")
    barcode_audit_warning: Optional[str] = Field(default=None, description="Soft warning if barcode search conflicts with OCR entity")
    is_aborted: bool = Field(default=False, description="True if audit was aborted (e.g. prescription medicine)")
    abort_reason: Optional[str] = Field(default=None, description="Reason if audit was aborted")
    is_demo_replay: bool = Field(default=False, description="True if loaded from offline recorded fixture")
    pipeline_telemetry: Dict[str, Any] = Field(default_factory=dict, description="Live execution telemetry for audit trails")
    execution_trace: Optional[PipelineExecutionTrace] = Field(default=None, description="Complete DAG execution trace")
