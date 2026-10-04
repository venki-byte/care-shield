"""Care-Shield: Senior-Accessible Streamlit Dashboard — Institutional Dark Terminal Edition.

Algorithmic defense against illegal price gouging, spurious medical packaging,
and counterfeit healthcare devices, adhering strictly to CDSCO Medical Device Rules (2017)
and Legal Metrology (Packaged Commodities) Rules (2011).

Hard Rules:
- Rule 1: Missing fields rendered as 'Not provided / Not found'.
- Rule 2: Never print, log, or commit API keys. Mask all credentials.
- Rule 3: Wrap external calls with timeout and retry, returning graceful fallbacks.
- Rule 4: Rigid Pydantic v2 validation for every inter-node hop.
"""

import json
import urllib.parse
from pathlib import Path
import streamlit as st

import config
import schemas
from core.vision_ingest import ingest_packaging_image
from core.pipeline import run_pipeline_with_trace
from core.extras import find_jan_aushadhi_kendras, fetch_brand_news, compute_savings
from audio import build_audio_script, synthesize_audio_verdict

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Care-Shield | Medical Device Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": (
            "### Care-Shield — Institutional Terminal\n"
            "Algorithmic fraud detection for medical packaging.\n"
            "Compliant with CDSCO MDR 2017 & Legal Metrology Act 2009."
        )
    }
)

# ─── Institutional Dark Terminal CSS (matches ProspectusIQ aesthetic) ─────────
TERMINAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global Reset ── */
.stApp {
    background-color: #0B0F19 !important;
    color: #F8FAFC !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
code, pre, .mono { font-family: 'JetBrains Mono', 'Consolas', monospace; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1440px !important;
}
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #F8FAFC !important; }
section[data-testid="stSidebar"] hr { border-color: #1E293B !important; }
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label span p { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] .stRadio label span p { color: #F8FAFC !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background-color: #151D2E !important;
    border: 1px solid #222F49 !important;
    border-radius: 8px !important;
    padding: 9px 14px !important;
    margin-bottom: 8px !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    border-color: #6366F1 !important;
    background-color: #1A2438 !important;
}
/* Sidebar success/warning/info */
section[data-testid="stSidebar"] .stAlert { font-size: 0.82rem !important; }

/* ── Inputs ── */
div[data-testid="stTextInput"] input {
    background-color: #1E293B !important; color: #F8FAFC !important;
    border: 1px solid #334155 !important; border-radius: 6px !important;
    padding: 8px 12px !important; font-size: 14px !important;
}
div[data-testid="stTextInput"] input:focus { border-color: #6366F1 !important; box-shadow: 0 0 0 1px #6366F1 !important; }
div[data-testid="stTextInput"] input::placeholder { color: #64748B !important; }
div[data-testid="stTextInput"] label { color: #94A3B8 !important; font-size: 13px !important; font-weight: 500 !important; }
div[data-testid="stNumberInput"] input {
    background-color: #1E293B !important; color: #F8FAFC !important;
    border: 1px solid #334155 !important; border-radius: 6px !important;
}
div[data-testid="stNumberInput"] label { color: #94A3B8 !important; font-size: 13px !important; }
div[data-testid="stTextArea"] textarea {
    background-color: #1E293B !important; color: #F8FAFC !important;
    border: 1px solid #334155 !important; border-radius: 6px !important;
}
div[data-testid="stTextArea"] label { color: #94A3B8 !important; font-size: 13px !important; }

/* ── Selectbox ── */
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #1E293B !important; color: #F8FAFC !important;
    border: 1px solid #334155 !important; border-radius: 6px !important;
}
div[data-testid="stSelectbox"] svg { fill: #94A3B8 !important; }
div[data-baseweb="popover"], ul[role="listbox"] {
    background-color: #1E293B !important; border: 1px solid #334155 !important; color: #F8FAFC !important;
}
li[role="option"] { background-color: #1E293B !important; color: #F8FAFC !important; }
li[role="option"]:hover, li[aria-selected="true"] { background-color: #2D3D58 !important; color: #FFFFFF !important; }

div[data-baseweb="select"] > div,
div[data-baseweb="select"] input,
div[data-baseweb="select"] span,
div[data-baseweb="select"] div[value] {
    background-color: #1E293B !important; color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
}
div[data-testid="stSelectbox"] label p,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label div,
div[data-testid="stFileUploader"] label p,
div[data-testid="stToggle"] label p,
div[data-testid="stCheckbox"] label p { color: #E2E8F0 !important; }
div[data-testid="stRadio"] div[role="radiogroup"] label { opacity: 1 !important; }
div[data-testid="stFileUploaderDropzone"] {
    background-color: #151D2E !important; border: 1.5px dashed #475569 !important;
    border-radius: 10px !important;
}
div[data-testid="stFileUploaderDropzone"] * { color: #CBD5E1 !important; }
div[data-testid="stFileUploaderDropzone"] button {
    background-color: #6366F1 !important; color: #FFFFFF !important; border: none !important;
}
div[data-testid="stFileUploaderDropzone"] button * { color: #FFFFFF !important; }
div[data-testid="stCameraInput"] > div { background-color: #151D2E !important; border-radius: 10px !important; }
div[data-testid="stAlert"] p { color: inherit !important; }

/* ── Buttons ── */
button[kind="primary"] {
    background-color: #6366F1 !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; padding: 0.6rem 1.4rem !important;
    font-size: 0.95rem !important; letter-spacing: 0.01em !important;
    transition: all 0.15s ease !important;
}
button[kind="primary"]:hover {
    background-color: #4F46E5 !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.4) !important;
}
button[kind="secondary"] {
    background-color: #1E293B !important; color: #E2E8F0 !important;
    border: 1px solid #334155 !important; border-radius: 8px !important;
    font-weight: 600 !important;
}
button[kind="secondary"]:hover {
    border-color: #6366F1 !important; color: #FFFFFF !important;
    background-color: #24324A !important;
}
.stFormSubmitButton > button {
    background-color: #6366F1 !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; padding: 0.65rem 1.4rem !important;
    width: 100% !important; font-size: 1rem !important;
}
.stFormSubmitButton > button:hover { background-color: #4F46E5 !important; }
.stLinkButton a {
    background-color: #1E293B !important; color: #E2E8F0 !important;
    border: 1px solid #334155 !important; border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background-color: #151D2E !important;
    border: 1px solid #222F49 !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25) !important;
}
[data-testid="metric-container"] label { color: #94A3B8 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #F8FAFC !important; font-size: 1.3rem !important; font-weight: 800 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0F172A !important;
    border-bottom: 1px solid #1E293B !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: #0F172A !important; color: #64748B !important;
    border-radius: 6px 6px 0 0 !important; padding: 8px 18px !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background-color: #151D2E !important; color: #F8FAFC !important;
    border-bottom: 2px solid #6366F1 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background-color: #151D2E !important;
    border: 1px solid #222F49 !important;
    border-radius: 0 8px 8px 8px !important;
    padding: 20px !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background-color: #151D2E !important;
    border: 1px solid #222F49 !important;
    border-radius: 8px !important;
    margin-bottom: 12px !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important; color: #F8FAFC !important;
    padding: 12px 16px !important;
}
div[data-testid="stExpander"] summary:hover { color: #6366F1 !important; }

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; }
div[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    background-color: #151D2E !important;
    border: 1px solid #222F49 !important;
    border-radius: 8px !important;
}

/* ── st.container(border=True) ── */
div[data-testid="stVerticalBlock"] > div[style*="border"] {
    background-color: #151D2E !important;
    border-color: #222F49 !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr { border-color: #1E293B !important; margin: 1.5rem 0 !important; }

/* ── Caption / small text ── */
.stCaption, .stCaptionContainer { color: #64748B !important; }
small { color: #94A3B8 !important; }

/* ───────────── CARE-SHIELD COMPONENTS ───────────── */

/* Hero */
.cs-hero {
    background: linear-gradient(135deg, #1E3A5F 0%, #0B0F19 55%, #12261F 100%);
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 36px 44px;
    margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.cs-hero::before {
    content: ""; position: absolute; top: 0; right: 0;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.cs-hero h1 { margin:0; font-size:2.4rem; font-weight:900; color:#F8FAFC; letter-spacing:-0.5px; }
.cs-hero p  { margin:8px 0 0; font-size:1rem; color:#94A3B8; max-width:600px; line-height:1.6; }
.cs-hero .badge {
    display:inline-block;
    background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.35);
    border-radius:999px; padding:3px 12px;
    font-size:0.75rem; font-weight:600; color:#A5B4FC;
    margin:14px 4px 0 0; letter-spacing:0.04em;
}

@media (max-width: 640px) {
    .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
    .cs-hero { padding: 22px 20px; }
    .cs-hero h1 { font-size: 1.7rem; }
    .cs-score .snum { font-size: 3rem; }
}
/* Step header */
.cs-step {
    display:flex; align-items:center; gap:14px;
    background:#151D2E; border:1px solid #222F49;
    border-left:4px solid #6366F1;
    border-radius:8px; padding:14px 20px;
    margin:28px 0 16px;
}
.cs-step .num {
    background:#6366F1; color:white; border-radius:50%;
    width:30px; height:30px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    font-weight:800; font-size:0.88rem;
}
.cs-step .title { font-size:1.02rem; font-weight:700; color:#F8FAFC; margin:0; }
.cs-step .sub   { font-size:0.78rem; color:#64748B; margin:2px 0 0; }

/* Score card */
.cs-score {
    border-radius:10px; padding:28px 24px; text-align:center;
    border:1px solid #222F49; height:100%;
}
.cs-green { background:rgba(16,185,129,0.08); border-color:rgba(16,185,129,0.4); }
.cs-amber { background:rgba(245,158,11,0.08); border-color:rgba(245,158,11,0.4); }
.cs-red   { background:rgba(244,63,94,0.08);  border-color:rgba(244,63,94,0.4);  }
.cs-grey  { background:#151D2E;               border-color:#334155;              }
.cs-score .snum { font-size:4rem; font-weight:900; line-height:1.05; margin:0; }
.cs-score .slbl { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; opacity:0.55; margin:2px 0 0; }
.cs-score .svrd { font-size:1.2rem; font-weight:700; margin:8px 0 0; }
.cs-score .ssub { font-size:0.8rem; color:#64748B; margin:6px 0 0; line-height:1.4; }
.cs-green .snum { color:#10B981; } .cs-green .svrd { color:#10B981; }
.cs-amber .snum { color:#F59E0B; } .cs-amber .svrd { color:#F59E0B; }
.cs-red   .snum { color:#F43F5E; } .cs-red   .svrd { color:#F43F5E; }
.cs-grey  .snum { color:#94A3B8; } .cs-grey  .svrd { color:#94A3B8; }

/* Mode badges */
.badge-live {
    background:rgba(16,185,129,0.12); color:#10B981;
    border:1px solid rgba(16,185,129,0.4); padding:4px 14px;
    border-radius:999px; font-size:0.78rem; font-weight:700;
}
.badge-demo {
    background:rgba(99,102,241,0.12); color:#A5B4FC;
    border:1px solid rgba(99,102,241,0.35); padding:4px 14px;
    border-radius:999px; font-size:0.78rem; font-weight:700;
}

/* Evidence bullet */
.cs-ev {
    background:#0B0F19; border-left:3px solid #6366F1;
    border-radius:0 6px 6px 0; padding:8px 14px;
    margin:6px 0; font-size:0.86rem; color:#CBD5E1;
    font-family: 'JetBrains Mono', monospace;
}

/* Pharmacy card */
.cs-pharmacy {
    background:#151D2E; border:1px solid #222F49; border-radius:10px;
    padding:16px 18px; height:100%;
}
.cs-pharmacy h4 { margin:6px 0 4px; font-size:0.93rem; color:#F8FAFC; }
.cs-pharmacy .tag { font-size:0.72rem; font-weight:700; color:#6366F1; text-transform:uppercase; letter-spacing:0.05em; }
.cs-pharmacy .addr { font-size:0.78rem; color:#64748B; margin:4px 0 0; }

/* Online alt card */
.cs-alt {
    background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.25);
    border-radius:10px; padding:16px; text-align:center;
}
.cs-alt .price { font-size:1.6rem; font-weight:900; color:#10B981; margin:0; }
.cs-alt .shop  { font-size:0.82rem; font-weight:700; color:#94A3B8; margin:4px 0; }
.cs-alt .title { font-size:0.75rem; color:#64748B; line-height:1.35; }

/* Savings banner */
.cs-savings {
    background:linear-gradient(135deg,rgba(16,185,129,0.14),rgba(99,102,241,0.12));
    border:1px solid rgba(16,185,129,0.45); border-radius:10px;
    padding:16px 20px; margin:18px 0; font-size:1.05rem; line-height:1.7; color:#F8FAFC;
}
.cs-savings b { color:#34D399; }

/* Section header */
.cs-section-hdr {
    font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#6366F1; margin:0 0 12px;
}

/* Sidebar logo area */
.sb-logo  { text-align:center; padding:18px 0 6px; font-size:3rem; line-height:1; }
.sb-title { text-align:center; font-size:1.1rem; font-weight:900; color:#F8FAFC; margin:0; }
.sb-sub   { text-align:center; font-size:0.72rem; color:#64748B; margin-bottom:4px; }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">🛡️</div>', unsafe_allow_html=True)
    st.markdown('<p class="sb-title">Care-Shield</p>', unsafe_allow_html=True)
    st.markdown('<p class="sb-sub">Medical Device Guardian</p>', unsafe_allow_html=True)
    st.divider()

    active_serpapi_key = config.SERPAPI_KEY
    active_gemini_key  = config.GEMINI_API_KEY

    st.markdown('<p class="cs-section-hdr">Inspection Settings</p>', unsafe_allow_html=True)

    demo_mode_toggle = st.toggle(
        "⚡ Demo Mode (Mock Fixture)",
        value=False,
        help="Loads fixtures/mock_counterfeit.json — bypasses all live API calls."
    )
    demo_replay = bool(config.USE_FIXTURES) or demo_mode_toggle
    preset_selection = st.selectbox(
        "Inspection Scenario",
        options=[
            "📷 Live Upload / Camera",
            "✅ Genuine Item (Oximeter Demo)",
            "🚨 Counterfeit Item (Knee Brace Demo)",
        ],
        index=0 if not (demo_replay or demo_mode_toggle) else (2 if demo_mode_toggle else 1)
    )
    selected_city = st.selectbox(
        "📍 Metro City (Geo-Routing)",
        options=list(config.CITY_COORDINATES.keys()),
        index=0
    )
    audio_lang = st.radio(
        "🔊 Audio Language",
        options=["English (en)", "हिन्दी - Hindi (hi)", "தமிழ் - Tamil (ta)"],
        index=0
    )
    lang_code = "ta" if "Tamil" in audio_lang else ("hi" if "Hindi" in audio_lang else "en")

    st.divider()
    st.caption("CDSCO MDR 2017 · Legal Metrology Act 2009 · Zero Hallucination")

# ─── Demo Mode Handler ─────────────────────────────────────────────────────────
if demo_mode_toggle:
    try:
        fixture_path = Path("fixtures/mock_counterfeit.json")
        if fixture_path.exists():
            with open(fixture_path, "r", encoding="utf-8") as f:
                mock_json = json.load(f)
            loaded_sc = schemas.AuditScorecard.model_validate(mock_json)
            st.session_state.scorecard     = loaded_sc
            st.session_state.extracted_ocr = loaded_sc.ocr_result
    except Exception as e:
        st.sidebar.error(f"Fixture load failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cs-hero">
  <h1>🛡️ Care-Shield</h1>
  <p><b>Before you pay:</b> is it the right product, is the price fair, and where is it cheapest?
     Scan any medical device or medicine — live SerpApi prices, counterfeit &amp; recall checks,
     and the legal MRP ceiling. Built for Indian families and senior citizens.</p>
  <span class="badge">CDSCO MDR 2017</span>
  <span class="badge">Legal Metrology Act 2009</span>
  <span class="badge">Zero Hallucination</span>
  <span class="badge">Privacy-Safe ROI Crop</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Vision Ingest
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cs-step">
  <div class="num">1</div>
  <div>
    <div class="title">Ingest Packaging Image</div>
    <div class="sub">Privacy ROI crop · Rotation-aware RapidOCR · Barcode decode · Category guard</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Preset state ──
sample_preset_arg   = None
default_store_price = 0.0
if "Genuine" in preset_selection and not demo_mode_toggle:
    sample_preset_arg   = "authentic_oximeter"
    default_store_price = 1630.0
elif "Counterfeit" in preset_selection or demo_mode_toggle:
    sample_preset_arg   = "counterfeit_knee_brace"
    default_store_price = 1450.0

uploaded_image = None
manual_text    = None
lens_url_input = ""

col_input = st.container()

with col_input:
    if sample_preset_arg == "authentic_oximeter":
        st.info("**Demo Preset:** Dr Trust Signature Series Pulse Oximeter — Genuine, compliant MRP label.", icon="📦")
    elif sample_preset_arg == "counterfeit_knee_brace":
        st.error("**Demo Preset:** Tynor Knee Suport Hinged Neo — Knockoff spelling · malformed license · extreme markup.", icon="🚨")
    else:
        input_type = st.radio(
            "Capture Method",
            ["📷 Take / Upload Photo", "🤳 Live Camera", "⌨️ Direct Text Input"],
            horizontal=True, label_visibility="collapsed"
        )
        if "Upload" in input_type:
            file_img = st.file_uploader(
                "Tap to take a photo (back camera) or choose from gallery", type=["jpg", "jpeg", "png", "webp"]
            )
            st.caption("📱 On phones, tap **Browse files → Camera** to use the rear camera.")
            if file_img:
                uploaded_image = file_img.getvalue()
                st.image(file_img, caption="Uploaded image preview", use_container_width=True)
        elif "Live" in input_type:
            st.caption("Uses your device camera (needs HTTPS + permission). Switch camera with the ⟳ icon in the viewer; if it fails use *Take / Upload Photo*.")
            cam_img = st.camera_input("Point camera at MRP label or fine print")
            if cam_img:
                uploaded_image = cam_img.getvalue()
        else:
            manual_text = st.text_area(
                "Paste or type packaging text",
                placeholder="e.g. Melactis Pigment Corrector Serum 30ml. Batch: ML-2024. Lic: None",
                height=100
            )

st.caption(
    "🔒 **Privacy ROI cropping is always active** — background faces and unrelated text are "
    "filtered out before any image is analysed. Gemini Vision reads the packaging; typed text works too."
)

# ── Scan button ──
scan_btn_label = (
    "📷 Scan & Extract Packaging Details"
    if not sample_preset_arg
    else "📦 Ingest Packaging Data from Preset"
)
scan_packaging = st.button(scan_btn_label, type="primary", use_container_width=True)

needs_auto_ingest = (
    "extracted_ocr" not in st.session_state
    and sample_preset_arg
    and not demo_mode_toggle
)

if scan_packaging or needs_auto_ingest:
    with st.spinner("Running Privacy ROI Crop → Rotation-Aware RapidOCR → Barcode Decode → Category Guard…"):
        ocr_res, lens_rep = ingest_packaging_image(
            image_bytes=uploaded_image,
            raw_text_override=manual_text,
            sample_preset=sample_preset_arg,
            serpapi_key=active_serpapi_key,
            gemini_key=active_gemini_key,
            lens_image_url="",
            return_lens_report=True
        )
        st.session_state.extracted_ocr = ocr_res
        st.session_state.lens_report   = lens_rep
        if "scorecard" in st.session_state and scan_packaging:
            del st.session_state["scorecard"]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Review Form + UI Pause (T3 & T6)
# ─────────────────────────────────────────────────────────────────────────────
if "extracted_ocr" in st.session_state and not demo_mode_toggle:
    ocr_data: schemas.PackagingOCRResult = st.session_state.extracted_ocr
    lens_rep  = st.session_state.get("lens_report", {})

    st.markdown("""
    <div class="cs-step">
      <div class="num">2</div>
      <div>
        <div class="title">Review Extracted Fields &amp; Enter Store Price</div>
        <div class="sub">Pipeline paused — correct OCR errors, enter store price, then run the audit</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # T6: Prescription abort
    if ocr_data.product_category == "prescription_medicine":
        st.error(
            "**Care-Shield doesn't audit prescription medicines.**\n\n"
            "Schedule H / H1 / X prescription drugs require an authorized prescription "
            "and a licensed pharmacist. Please consult your pharmacist or doctor.",
            icon="🚨"
        )
        st.stop()

    if ocr_data.product_category == "non_healthcare":
        st.warning(
            "**This doesn't look like a healthcare product.** Care-Shield audits medical devices "
            "(BP monitors, oximeters, braces, thermometers...) and over-the-counter medicines. "
            "Please scan a medical product's packaging.",
            icon="🩺"
        )
        st.stop()

    if ocr_data.product_category == "otc_medicine":
        st.info(
            "**Over-the-counter medicine detected.** MRP, price and recall checks apply; the CDSCO "
            "device-licence check is skipped (medicines follow the Drugs & Cosmetics Act). "
            "Ask for the cheaper generic at a Jan Aushadhi Kendra.",
            icon="💊"
        )

    if ocr_data.product_category == "cosmetic":
        st.info(
            "**Cosmetic / Skincare detected.** "
            "CDSCO Medical Device license check will be skipped (marked N/A — not applicable to cosmetics).",
            icon="ℹ️"
        )

    st.info(
        "**Pipeline paused.** Vision node has extracted fields below. "
        "Correct any OCR misreadings, enter the **Store Asking Price (₹)**, "
        "then click **Run Forensic Audit**.",
        icon="⏸️"
    )

    with st.form(key="forensic_audit_form"):
        left_col, right_col = st.columns(2, gap="large")

        with left_col:
            st.markdown('<p class="cs-section-hdr">✏️ Editable Identifiers</p>', unsafe_allow_html=True)
            edited_title = st.text_input("Product Title", value=ocr_data.product_title,
                                         help="Correct any OCR character misreadings.")
            edited_brand = st.text_input("Brand / Manufacturer", value=ocr_data.detected_brand)
            default_mrp_val = float(ocr_data.printed_mrp_inr or 0.0)
            edited_mrp = st.number_input(
                "Printed MRP (₹ INR) — Optional",
                min_value=0.0, max_value=500_000.0, value=default_mrp_val, step=50.0,
                help="Set to 0 if absent. Over-MRP statutory check will be skipped."
            )
            if edited_mrp == 0.0:
                st.caption("ℹ️ MRP: Not provided / Not found → Statutory overcharge check skipped")
            else:
                st.caption(f"⚖️ Legally Mandated MRP: ₹{edited_mrp:,.2f}")

        with right_col:
            st.markdown('<p class="cs-section-hdr">💰 Transaction & Metadata</p>', unsafe_allow_html=True)
            form_store_price = st.number_input(
                "Store Asking Price (₹ INR) — *Required*",
                min_value=0.0, max_value=500_000.0,
                value=default_store_price, step=50.0,
                help="Physical retail price quoted by the pharmacy counter."
            )
            st.markdown('<p class="cs-section-hdr" style="margin-top:14px;">Extracted Metadata (read-only)</p>', unsafe_allow_html=True)
            meta = {
                "Category":      ocr_data.product_category_display,
                "Batch / Lot":   ocr_data.batch_number_display,
                "Mfg Date":      ocr_data.mfg_date_display,
                "Exp Date":      ocr_data.exp_date_display,
                "CDSCO License": ocr_data.cdsco_license_display,
                "Barcode / QR":  ocr_data.barcode_display,
                "OCR Engine":    (f"{ocr_data.ocr_engine_used} "
                                  f"({int((ocr_data.ocr_confidence or 0)*100)}% conf, "
                                  f"{ocr_data.rotation_applied}° orient)"),
                "Privacy Crop":  "Foreground ROI isolated" if ocr_data.crop_applied else "Full frame",
            }
            rows = "".join(
                f"<tr><td style='color:#64748B;font-size:0.78rem;font-weight:600;padding:4px 8px 4px 0;'>{k}</td>"
                f"<td style='color:#CBD5E1;font-size:0.78rem;font-family:monospace;padding:4px 0;'>{v}</td></tr>"
                for k, v in meta.items()
            )
            st.markdown(
                f"<table style='border-collapse:collapse;width:100%;'>{rows}</table>",
                unsafe_allow_html=True
            )

        run_forensic_audit = st.form_submit_button(
            "🛡️ Run Forensic Audit", type="primary", use_container_width=True
        )

    if run_forensic_audit:
        if form_store_price <= 0.0:
            st.error(
                "**Store Asking Price is required!** "
                "Enter the pharmacy asking price (> ₹0.00) before running the audit.",
                icon="⚠️"
            )
        else:
            ocr_data.product_title   = edited_title.strip()
            ocr_data.detected_brand  = edited_brand.strip()
            ocr_data.printed_mrp_inr = edited_mrp if edited_mrp > 0.0 else None

            city_info = config.CITY_COORDINATES[selected_city]
            is_demo   = demo_replay or bool(sample_preset_arg)

            with st.spinner("Running SerpApi Google Shopping · Threat Intel · Scoring Engine · Maps Routing…"):
                sc = run_pipeline_with_trace(
                    precomputed_ocr_result=ocr_data,
                    precomputed_lens_report=lens_rep,
                    store_asking_price=form_store_price,
                    manual_printed_mrp=ocr_data.printed_mrp_inr,
                    user_lat=city_info["lat"],
                    user_lng=city_info["lng"],
                    city_name=selected_city,
                    serpapi_key=active_serpapi_key,
                    gemini_key=active_gemini_key,
                    is_demo_mode=is_demo
                )
                st.session_state.scorecard = sc

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Scorecard
# ─────────────────────────────────────────────────────────────────────────────
scorecard: schemas.AuditScorecard | None = st.session_state.get("scorecard")

if scorecard:
    st.markdown(f"""
    <div class="cs-step" style="border-left-color:#10B981;">
      <div class="num" style="background:#10B981;">{3 if not demo_mode_toggle else 2}</div>
      <div>
        <div class="title">Master Forensic Audit Scorecard</div>
        <div class="sub">Algorithmic verdict · evidence-backed · defensible phrasing · zero hallucination</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    is_aborted = getattr(scorecard, "is_aborted", False)

    # Relevance guard: benchmark listings must (1) mention the product noun, (2) not be a cross-border
    # re-seller, and (3) sit in a plausible price band (around the printed MRP, else the median).
    if scorecard.parity_result and not scorecard.is_demo_replay:
        import statistics
        _pr = scorecard.parity_result
        _noun = (scorecard.ocr_result.product_title.split() or [""])[-1].lower()
        _ms = _pr.benchmark_merchants
        if len(_noun) >= 4:
            _ms = [m for m in _ms if _noun[:6] in (m.extracted_title or "").lower()]
        _ms = [m for m in _ms if "ubuy" not in m.name.lower()]
        _ref = scorecard.ocr_result.printed_mrp_inr or (statistics.median([m.price for m in _ms]) if _ms else None)
        if _ref:
            _ms = [m for m in _ms if 0.4 * _ref <= m.price <= 2.0 * _ref]
        _pr.benchmark_merchants = _ms
        if len(_ms) < 2:
            _pr.online_median = None  # not enough validated listings -> show no median rather than a junk one
        if len(_ms) >= 2:
            _pr.online_median = round(statistics.median([m.price for m in _ms]), 2)
            _pr.markup_percent = round(((_pr.scanned_purchase_price or 0) - _pr.online_median) / _pr.online_median * 100, 2)                 if _pr.scanned_purchase_price else _pr.markup_percent

    # ── Mode badge ──
    if scorecard.is_demo_replay:
        st.markdown(
            '<span class="badge-demo">🟣 DEMO REPLAY — Offline Pre-recorded Benchmark</span>',
            unsafe_allow_html=True
        )
    else:
        crop_tag = "ROI Isolated" if scorecard.ocr_result.crop_applied else "Full Frame"
        st.markdown(
            f'<span class="badge-live">🟢 LIVE AUDIT</span> &nbsp;'
            f'<small style="color:#64748B;">'
            f'<b style="color:#94A3B8;">OCR:</b> {scorecard.ocr_result.ocr_engine_used} &nbsp;·&nbsp; '
            f'<b style="color:#94A3B8;">Category:</b> {scorecard.ocr_result.product_category_display} &nbsp;·&nbsp; '
            f'<b style="color:#94A3B8;">Privacy:</b> {crop_tag}'
            f'</small>',
            unsafe_allow_html=True
        )
    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Score badge + sub-score metrics ──
    t_score = scorecard.trust_score
    sub     = scorecard.sub_scores

    if is_aborted:
        cls, icon, disp = "cs-grey",  "🛑", "N/A"
    elif t_score >= 80:
        cls, icon, disp = "cs-green", "✅", f"{t_score}"
    elif t_score >= 60:
        cls, icon, disp = "cs-amber", "⚠️", f"{t_score}"
    else:
        cls, icon, disp = "cs-red",   "🚨", f"{t_score}"

    if (not is_aborted and scorecard.parity_result and scorecard.parity_result.is_above_printed_mrp
            and cls in ("cs-green", "cs-amber")):
        cls, icon = "cs-amber", "🚨"   # a statutory violation must never look green

    col_badge, col_sub = st.columns([2, 3], gap="large")

    with col_badge:
        st.markdown(f"""
        <div class="cs-score {cls}">
          <p class="snum">{disp}</p>
          <p class="slbl">out of 100</p>
          <p class="svrd">{icon} {scorecard.verdict.upper()}</p>
          <p class="ssub">{sub.total_earned_points:.1f} / {sub.max_available_points:.1f} pts<br/>Dynamically normalized</p>
        </div>
        """, unsafe_allow_html=True)

    with col_sub:
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("📝 Text & Brand",
                      f"{sub.text_integrity:.1f} / {sub.text_max:.0f}",
                      help="Packaging spelling & token similarity vs. canonical specifications.")
            reg_val = (f"{sub.regulatory:.1f} / {sub.regulatory_max:.0f}"
                       if sub.regulatory is not None else "N/A")
            st.metric("🏛️ CDSCO Regulatory", reg_val,
                      help="MDR 2017 license format. N/A for cosmetics — excluded from denominator.")
        with mc2:
            price_val = (f"{sub.pricing_fairness:.1f} / {sub.pricing_max:.0f}"
                         if sub.pricing_fairness is not None else "N/A")
            st.metric("⚖️ Legal Metrology", price_val,
                      help="Statutory MRP check + online parity benchmark.")
            st.metric("🛡️ Safety & Threat Intel",
                      f"{sub.safety_intel:.1f} / {sub.safety_max:.0f}",
                      help="CDSCO advisories, spurious batch alerts, recall notices.")

    st.divider()

    # ── Savings & Price Chart ──
    pr_ = scorecard.parity_result
    if pr_ and pr_.scanned_purchase_price:
        cheapest_ = min((m.price for m in pr_.benchmark_merchants if m.price >= 0.4 * (pr_.online_median or 0)), default=None)
        sv = compute_savings(pr_.scanned_purchase_price, scorecard.ocr_result.printed_mrp_inr,
                             pr_.online_median, cheapest_)
        mrp_ = scorecard.ocr_result.printed_mrp_inr
        gap_pct = None
        if mrp_ and pr_.online_median and pr_.online_median < 0.92 * mrp_:
            gap_pct = round(100 * (mrp_ - pr_.online_median) / mrp_)
        lines = []
        if sv["save_vs_cheapest"]:
            lines.append(f"💰 You could save <b>₹{sv['save_vs_cheapest']:,.0f} ({sv['save_pct']}%)</b> by buying from the cheapest relevant seller found online.")
        if gap_pct:
            lines.append(f"🏷️ <b>MRP is a legal ceiling, not the fair price</b> — the online median (₹{pr_.online_median:,.0f}) is <b>{gap_pct}% below</b> the printed MRP (₹{mrp_:,.0f}).")
        if sv["illegal_overcharge"]:
            lines.append(f"🚨 Charged <b>₹{sv['illegal_overcharge']:,.0f} above the legal MRP</b> — you can demand a refund of the excess.")
        elif mrp_ and pr_.scanned_purchase_price <= mrp_ and lines:
            lines.append("✅ The store price is within the legal MRP (legal) — but compare before paying.")
        if lines:
            st.markdown('<div class="cs-savings">' + "<br/>".join(lines) + "</div>", unsafe_allow_html=True)
        if not pr_.benchmark_merchants and not scorecard.is_demo_replay:
            st.info("No reliable online listings matched this product, so no market comparison is shown "
                    "(we never use unrelated products as a benchmark). Try a clearer photo or edit the "
                    "product title / brand above.", icon="ℹ️")
        import pandas as pd
        rows_ = [("Store asking price", pr_.scanned_purchase_price)]
        if scorecard.ocr_result.printed_mrp_inr:
            rows_.append(("Printed MRP (legal cap)", scorecard.ocr_result.printed_mrp_inr))
        if pr_.online_median:
            rows_.append(("Online median", pr_.online_median))
        rows_ += [(m.name, m.price) for m in sorted((x for x in pr_.benchmark_merchants if x.price >= 0.4 * (pr_.online_median or 0)), key=lambda x: x.price)[:3]]
        st.markdown('<p class="cs-section-hdr">📊 Price Comparison (₹)</p>', unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame(rows_, columns=["Source", "Price (INR)"]).set_index("Source"),
                     horizontal=True, color="#6366F1")

    # ── Audio ──
    st.markdown('<p class="cs-section-hdr">🔊 Spoken Audio Verdict</p>', unsafe_allow_html=True)
    script_text = build_audio_script(scorecard, language=lang_code)

    @st.cache_data(show_spinner=False)
    def _fetch_cached_audio(text: str, lang: str):
        return synthesize_audio_verdict(scorecard, language=lang)

    au1, au2 = st.columns([1, 3], gap="large")
    with au1:
        audio_bytes = _fetch_cached_audio(script_text, lang_code)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.info("Voice synthesis unavailable.", icon="ℹ️")
    with au2:
        st.markdown(f"**Voice Script ({audio_lang}):**")
        st.markdown(f"> *\"{script_text}\"*")

    st.divider()

    # ── Evidence Tabs ──
    st.markdown('<p class="cs-section-hdr">🔬 Forensic Evidence Panel</p>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝  Text & Brand",
        "🏛️  Regulatory",
        "⚖️  Pricing",
        "🛡️  Safety Intel"
    ])

    with tab1:
        text_conf = f"{int((scorecard.ocr_result.ocr_confidence or 0.85)*100)}%"
        c1, c2, c3 = st.columns(3)
        c1.metric("Points",      f"{sub.text_integrity:.1f} / {sub.text_max:.0f}")
        c2.metric("OCR Conf",    text_conf)
        c3.metric("Status",      sub.text_status.upper())
        st.caption(f"Token Matches: {sub.text_matching_tokens_count} meaningful tokens matched canonical spec.")
        evs = [ev for ev in scorecard.evidence_list
               if any(k in ev.lower() for k in ["text","token","typo","knockoff","spelling","integrity"])]
        for ev in evs:
            st.markdown(f'<div class="cs-ev">🔎 {ev}</div>', unsafe_allow_html=True)
        st.markdown(f"**Extracted Title:** `{scorecard.ocr_result.product_title}`")
        st.markdown(f"**Detected Brand:** `{scorecard.ocr_result.detected_brand}`")
        if scorecard.ocr_result.raw_ocr_corpus:
            with st.expander("Raw OCR Corpus", expanded=False):
                st.text(scorecard.ocr_result.raw_ocr_corpus)

    with tab2:
        reg_str = (f"{sub.regulatory:.1f} / {sub.regulatory_max:.0f}"
                   if sub.regulatory is not None else "N/A — Excluded from Denominator")
        c1, c2, c3 = st.columns(3)
        c1.metric("Points",  reg_str)
        c2.metric("Method",  "Deterministic Regex" if sub.regulatory_status != "na" else "Exempt")
        c3.metric("Status",  sub.regulatory_status.upper())
        st.markdown(f"**License String:** `{scorecard.ocr_result.cdsco_license_display}`")
        st.markdown(f"**Product Category:** `{scorecard.ocr_result.product_category_display}`")
        evs = [ev for ev in scorecard.evidence_list
               if any(k in ev.lower() for k in ["regulatory","license","cdsco","mdr","exemption"])]
        for ev in evs:
            st.markdown(f'<div class="cs-ev">📜 {ev}</div>', unsafe_allow_html=True)
        if scorecard.ocr_result.product_category == "cosmetic":
            st.info("Cosmetics are not regulated under CDSCO MDR 2017. Regulatory points excluded from denominator.", icon="ℹ️")

    with tab3:
        price_str = (f"{sub.pricing_fairness:.1f} / {sub.pricing_max:.0f}"
                     if sub.pricing_fairness is not None else "N/A — MRP Unstated")
        c1, c2, c3 = st.columns(3)
        c1.metric("Points",  price_str)
        c2.metric("Method",  "Statutory Metrology")
        c3.metric("Status",  sub.pricing_status.upper())
        if scorecard.parity_result:
            pr = scorecard.parity_result
            p1, p2, p3 = st.columns(3)
            p1.metric("Store Price",   pr.scanned_price_display)
            p2.metric("Printed MRP",   scorecard.ocr_result.printed_mrp_display)
            p3.metric("Online Median", f"{pr.online_median_display} ({pr.markup_percent:+.1f}%)" if pr.online_median else "Not enough reliable listings")
            if pr.is_above_printed_mrp:
                st.error("🚨 **Statutory Violation:** Price exceeds printed MRP (Legal Metrology Act 2009, §36).", icon="🚨")
            elif pr.mrp_audit_skipped:
                st.info("Printed MRP absent — overcharge check skipped (no hallucination).", icon="ℹ️")
        if (not scorecard.is_demo_replay and active_gemini_key
                and not (scorecard.parity_result and scorecard.parity_result.benchmark_merchants)):
            from core.gemini_fallback import gemini_price_estimate

            @st.cache_data(show_spinner=False, ttl=3600)
            def _est(q, k):
                return gemini_price_estimate(q, k)

            est = _est(f"{scorecard.ocr_result.detected_brand} {scorecard.ocr_result.product_title}", active_gemini_key)
            if est:
                st.info(f"🤖 **AI estimate (not live data, not used in the score):** {est}", icon="ℹ️")
        evs = [ev for ev in scorecard.evidence_list
               if any(k in ev.lower() for k in ["price","mrp","markup","metrology","gouging","overcharge"])]
        for ev in evs:
            st.markdown(f'<div class="cs-ev">⚖️ {ev}</div>', unsafe_allow_html=True)

    with tab4:
        safety_conf = f"{1.0 - (scorecard.intel_result.risk_factor if scorecard.intel_result else 0.0):.2f}"
        c1, c2, c3 = st.columns(3)
        c1.metric("Points",    f"{sub.safety_intel:.1f} / {sub.safety_max:.0f}")
        c2.metric("Conviction", safety_conf)
        c3.metric("Status",    sub.safety_status.upper())
        st.markdown(f"**Threat Summary:** {sub.safety_message}")
        if scorecard.intel_result and scorecard.intel_result.evidence_snippets:
            for snip in scorecard.intel_result.evidence_snippets:
                st.markdown(f'<div class="cs-ev">🚨 {snip}</div>', unsafe_allow_html=True)
        else:
            st.success("No active CDSCO seizure notices, recall alerts, or counterfeit warnings found.", icon="✅")

    st.divider()

    # ── Top 3 Online Alternatives ──
    if scorecard.parity_result and scorecard.parity_result.benchmark_merchants:
        _med = scorecard.parity_result.online_median or 0
        cheapest3 = sorted((m for m in scorecard.parity_result.benchmark_merchants if m.price >= 0.4 * _med),
                           key=lambda x: x.price)[:3]
        st.markdown('<p class="cs-section-hdr">🛒 Top 3 Cheapest Online Alternatives (Google Shopping)</p>',
                    unsafe_allow_html=True)
        st.caption("Price benchmarks from verified Indian online pharmacies:")
        alt_cols = st.columns(len(cheapest3), gap="medium")
        for idx, alt in enumerate(cheapest3):
            with alt_cols[idx]:
                st.markdown(f"""
                <div class="cs-alt">
                  <p class="price">₹{alt.price:,.2f}</p>
                  <p class="shop">{alt.name}</p>
                  <p class="title">{alt.extracted_title or 'Standard Medical Spec'}</p>
                </div>
                """, unsafe_allow_html=True)
                if alt.link:
                    st.link_button(f"🔗 {alt.name}", alt.link, use_container_width=True)
        st.divider()

    # ── Consumer Protection Checklist ──
    is_risky = (
        (scorecard.trust_score < 70)
        or any(k in scorecard.verdict for k in ["Risk", "Overcharge", "Caution", "Discrepancy"])
        or (scorecard.parity_result and scorecard.parity_result.is_above_printed_mrp)
    )
    if is_risky:
        st.markdown('<p class="cs-section-hdr">⚠️ Consumer Protection Action Checklist</p>',
                    unsafe_allow_html=True)
        st.error(
            "Packaging anomalies or statutory violations detected. "
            "Follow these steps before making payment.",
            icon="⚠️"
        )
        st.markdown("""
- 🧾 **Ask for a GST Invoice** — Insist on a Tax Invoice with GSTIN, batch number, and expiry date. Cash memos prevent legal recourse.
- ⚖️ **Compare with Printed MRP** — Billed price must NOT exceed the printed MRP *(Legal Metrology Act 2009, §36)*.
- 📞 **National Consumer Helpline: 1915** — Report overcharging or spurious packaging at [consumerhelpline.gov.in](https://consumerhelpline.gov.in).
        """)
        st.divider()

    # ── WhatsApp Share ──
    st.markdown('<p class="cs-section-hdr">📲 Share with Family Caregiver</p>', unsafe_allow_html=True)
    price_d  = scorecard.parity_result.scanned_price_display if scorecard.parity_result else "Unstated"
    median_d = scorecard.parity_result.online_median_display  if scorecard.parity_result else "Unstated"
    pharm_d  = (
        f"Top nearby pharmacy: {scorecard.fallback_pharmacies[0].name} "
        f"({scorecard.fallback_pharmacies[0].directions_url})"
        if scorecard.fallback_pharmacies else "No physical remediation needed."
    )
    share_text = (
        f"🛡️ *Care-Shield Forensic Audit Report*\n\n"
        f"📦 *Product:* {scorecard.ocr_result.product_title}\n"
        f"🏷️ *Brand:* {scorecard.ocr_result.detected_brand}\n"
        f"📊 *Trust Score:* {scorecard.trust_score} / 100 ({scorecard.verdict})\n"
        f"💰 *Store Price:* {price_d} (Online Median: {median_d})\n"
        f"🏥 *Nearby Alt:* {pharm_d}\n\n"
        f"_Audited via Care-Shield Medical Device Guardian._"
    )
    st.link_button(
        "📲 Send to family on WhatsApp",
        f"https://wa.me/?text={urllib.parse.quote(share_text)}",
        use_container_width=True
    )

    st.divider()

    # ── Pipeline Trace ──
    if scorecard.execution_trace:
        trace = scorecard.execution_trace
        st.markdown('<p class="cs-section-hdr">🔬 Pipeline Execution Trace</p>', unsafe_allow_html=True)
        tr1, tr2, tr3, tr4 = st.columns(4)
        tr1.metric("Trace ID",      trace.trace_id[:10] + "…")
        tr2.metric("Total Latency", f"{trace.total_duration_ms:.0f} ms")
        tr3.metric("Status",        trace.overall_status)
        tr4.metric("Timestamp",     trace.timestamp[:19].replace("T", " "))

        with st.expander("📋 DAG Step Trace", expanded=False):
            for step in trace.steps:
                s_icon = "🟢" if step.status == "SUCCESS" else ("⚪" if step.status == "SKIPPED" else "🟠")
                with st.container(border=True):
                    sc1, sc2, sc3 = st.columns([3, 1, 1])
                    sc1.markdown(f"{s_icon} **[{step.step_id}]** {step.step_name}")
                    sc2.markdown(f"`{step.duration_ms:.1f} ms`")
                    sc3.markdown(f"`{step.status}`")
                    if step.inputs:
                        st.caption(f"↳ In: {json.dumps(step.inputs, ensure_ascii=False)[:120]}")
                    if step.outputs:
                        st.caption(f"↳ Out: {json.dumps(step.outputs, ensure_ascii=False)[:120]}")
                    if step.error:
                        st.warning(step.error)

    st.divider()

    # ── SerpApi engines + Jan Aushadhi + News ──
    live_ok = bool(active_serpapi_key) and not scorecard.is_demo_replay
    st.markdown('<p class="cs-section-hdr">🔎 Powered by SerpApi — Engines Used</p>', unsafe_allow_html=True)
    engines = ["Google Shopping", "Google Search", "Google Maps", "Google News"]
    _warns = " ".join([
        (scorecard.parity_result.api_warning or "") if scorecard.parity_result else "",
        (scorecard.intel_result.api_warning or "") if scorecard.intel_result else "",
    ])
    used_gemini = "Gemini" in _warns and not scorecard.is_demo_replay
    badges = " ".join(f'<span class="{"badge-live" if live_ok else "badge-demo"}">{e}</span>' for e in engines)
    if used_gemini:
        badges += ' <span class="badge-demo">Gemini grounded search (fallback)</span>'
    st.markdown(badges, unsafe_allow_html=True)
    if used_gemini:
        st.caption("SerpApi was unavailable for part of this audit, so Care-Shield automatically switched to "
                   "Gemini with Google Search grounding (real web sources, no invented data).")
    else:
        st.caption("Live calls this audit." if live_ok else "Demo replay — live SerpApi calls are skipped.")

    if live_ok or (not scorecard.is_demo_replay and active_gemini_key):
        @st.cache_data(show_spinner=False, ttl=3600)
        def _ja(lat, lng, city, key):
            return find_jan_aushadhi_kendras(lat, lng, city, key)

        @st.cache_data(show_spinner=False, ttl=3600)
        def _news(brand, title, key):
            return fetch_brand_news(brand, title, key)

        st.markdown('<p class="cs-section-hdr">🏛️ Cheapest Legit Option — Jan Aushadhi Kendras Near You</p>',
                    unsafe_allow_html=True)
        st.caption("Government (PMBJP) generic-medicine stores — often 50–90% cheaper than branded retail. Ask for the generic equivalent.")
        _c = config.CITY_COORDINATES[selected_city]
        ja = _ja(_c["lat"], _c["lng"], selected_city, active_serpapi_key) if active_serpapi_key else []
        st.link_button("🗺️ Search Jan Aushadhi Kendras on Google Maps",
                       "https://www.google.com/maps/search/Jan+Aushadhi+Kendra+near+" + urllib.parse.quote_plus(selected_city),
                       use_container_width=True)
        if ja:
            jc = st.columns(len(ja), gap="medium")
            for i, k in enumerate(ja):
                with jc[i]:
                    rv = f"⭐ {k['rating']} ({k['reviews']} reviews)" if k.get("rating") else ""
                    st.markdown(f"""<div class="cs-pharmacy"><div class="tag">🟢 Live Google Maps</div>
                    <h4>🏥 {k['name']}</h4><p style="margin:2px 0;font-size:0.85rem;">{rv}</p>
                    <p class="addr">📍 {k['address']}</p></div>""", unsafe_allow_html=True)
                    st.link_button("🗺️ Get Directions", k["directions_url"], use_container_width=True)
        elif active_serpapi_key:
            st.info("No Jan Aushadhi Kendra found in this area right now.", icon="ℹ️")

        news = _news(scorecard.ocr_result.detected_brand, scorecard.ocr_result.product_title, active_serpapi_key)
        st.markdown('<p class="cs-section-hdr">📰 Latest News — Recalls & Spurious Alerts (Google News)</p>',
                    unsafe_allow_html=True)
        if news:
            for n in news:
                st.markdown(f'<div class="cs-ev">📰 <a href="{n["link"]}" target="_blank" style="color:#A5B4FC;">{n["title"]}</a>'
                            f' <small>— {n["source"]} {n["date"]}</small></div>', unsafe_allow_html=True)
        else:
            st.success("No recent recall, spurious-product or overcharging news found for this brand.", icon="✅")
    st.divider()

    # ── Nearby Pharmacies ──
    st.markdown(f'<p class="cs-section-hdr">📍 Highly Rated Nearby Pharmacies — {selected_city}</p>',
                unsafe_allow_html=True)
    if scorecard.fallback_pharmacies:
        st.caption("Filtered: rating ≥ 4.0 AND review count > 10 (Google Maps API):")
        ph_cols = st.columns(min(len(scorecard.fallback_pharmacies), 3), gap="medium")
        for idx, pharm in enumerate(scorecard.fallback_pharmacies[:3]):
            with ph_cols[idx]:
                live_tag = "🟢 Live Google Maps" if pharm.is_live_lookup else "🟣 Demo Reference"
                rev_str  = f"({pharm.review_count} reviews)" if pharm.review_count else ""
                open_str = "Open Now" if pharm.is_open else "Status Unknown"
                st.markdown(f"""
                <div class="cs-pharmacy">
                  <div class="tag">{live_tag}</div>
                  <h4>🏥 {pharm.name}</h4>
                  <p style="margin:4px 0;font-size:0.88rem;">⭐ <b style="color:#F59E0B;">{pharm.rating} / 5.0</b> <small>{rev_str}</small></p>
                  <p style="margin:2px 0;font-size:0.8rem;color:#94A3B8;">🕐 {open_str}</p>
                  <p class="addr">📍 {pharm.formatted_address}</p>
                </div>
                """, unsafe_allow_html=True)
                st.link_button("🗺️ Get Directions", pharm.directions_url, use_container_width=True)
    else:
        if active_serpapi_key:
            st.success("No remediation required — product is compliant with pricing standards.", icon="✅")
        else:
            st.info("Add SERPAPI_KEY to .env to enable live Google Maps pharmacy lookups.", icon="💡")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
st.caption(
    "Care-Shield · Open Deterministic Legal Metrology Engine · "
    "CDSCO MDR 2017 · Legal Metrology Act 2009 · "
    "Zero Hallucination"
)
