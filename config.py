"""Care-Shield Configuration & System Constants.

Loads environment variables and secrets, defines scoring weights,
sets legal metrology thresholds, and provides resilience utilities.
Strictly adheres to Hard Rules:
- Rule 2: Never print, log, or commit API keys. Load from env or st.secrets only.
- Rule 3: Wrap external calls with timeout, retry once, and return graceful fallbacks.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load .env if present
load_dotenv(BASE_DIR / ".env", override=True)


def _get_secret(key_name: str, fallback_key: Optional[str] = None) -> str:
    """Retrieve secret strictly from os.environ or st.secrets without hardcoded defaults."""
    # 1. Check os.environ
    val = os.getenv(key_name)
    if not val and fallback_key:
        val = os.getenv(fallback_key)
    if val and val.strip():
        return val.strip()

    # 2. Check streamlit secrets if available
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets is not None:
            if key_name in st.secrets:
                return str(st.secrets[key_name]).strip()
            if fallback_key and fallback_key in st.secrets:
                return str(st.secrets[fallback_key]).strip()
    except Exception:
        pass

    return ""


# Credentials loaded strictly from env or st.secrets (Hard Rule 2)
SERPAPI_KEY: str = _get_secret("SERPAPI_KEY", fallback_key="SERPAPI_API_KEY")
SERPAPI_API_KEY: str = SERPAPI_KEY  # Alias for backward compatibility
GEMINI_API_KEY: str = _get_secret("GEMINI_API_KEY")

# Execution Flag for offline demo playback
USE_FIXTURES: bool = os.getenv("USE_FIXTURES", "false").lower() in ("true", "1", "yes")


def has_serpapi_key() -> bool:
    """Check if a non-empty SerpApi key is currently loaded."""
    return bool(SERPAPI_KEY and not SERPAPI_KEY.startswith("your_"))


def has_gemini_key() -> bool:
    """Check if a non-empty Gemini key is currently loaded."""
    return bool(GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_"))


def mask_secrets(val: Optional[Any]) -> str:
    """Mask credentials and secrets completely for safe display in logs, UI, and traces.
    
    Hard Rule 2: Never print, log, or commit API keys.
    Returns:
        '[Not Configured]' if None, empty, or not a string.
        '[Placeholder Unset]' if placeholder string like 'your_...'.
        '[Protected Key Active]' for active keys - NEVER reveals any characters.
    """
    if val is None or not isinstance(val, str) or not val.strip():
        return "[Not Configured]"
    clean = val.strip()
    if clean.startswith("your_"):
        return "[Placeholder Unset]"
    return "[Protected Key Active]"


# Backward compatibility alias
mask_api_key = mask_secrets


def sanitize_for_secrets(data: Any) -> Any:
    """Recursively scrub any dict, list, string, or json of API keys and URLs containing api_key."""
    import re
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(term in k_lower for term in ["api_key", "secret", "token", "password"]):
                cleaned[k] = "[REDACTED]"
            elif k_lower in ["json_endpoint", "raw_html_file"]:
                cleaned[k] = "[REDACTED_ENDPOINT]"
            else:
                cleaned[k] = sanitize_for_secrets(v)
        return cleaned
    elif isinstance(data, list):
        return [sanitize_for_secrets(item) for item in data]
    elif isinstance(data, str):
        s = re.sub(r"([?&]api_key=)[a-zA-Z0-9_-]+", r"\1[REDACTED]", data)
        for secret in [SERPAPI_KEY, GEMINI_API_KEY]:
            if secret and len(secret) > 8 and secret in s:
                s = s.replace(secret, "[REDACTED]")
        return s
    return data


def execute_with_retry(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout_seconds: float = 15.0,
    max_retries: int = 1,
    fallback: Any = None,
    label: str = "External Call"
) -> Any:
    """Execute an external network call with strict timeout and single retry on failure.
    
    Hard Rule 3: Every external call must be wrapped in try/except with a timeout,
    retry once on network error, and return a graceful degraded result without crashing the DAG.
    """
    if kwargs is None:
        kwargs = {}

    attempts = 0
    last_err: Optional[Exception] = None

    while attempts <= max_retries:
        attempts += 1
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            executor.shutdown(wait=False)
            return result
        except FuturesTimeoutError:
            executor.shutdown(wait=False)
            last_err = TimeoutError(f"Request timed out after {timeout_seconds}s")
            logger.warning(f"[{label}] Attempt {attempts} timed out after {timeout_seconds}s.")
            if attempts <= max_retries:
                time.sleep(1.0)
                continue
        except Exception as e:
            executor.shutdown(wait=False)
            last_err = e
            logger.warning(f"[{label}] Attempt {attempts} failed with error: {e}")
            if attempts <= max_retries:
                time.sleep(1.0)
                continue

    logger.error(f"[{label}] All {attempts} attempts failed. Returning degraded fallback. Cause: {last_err}")
    return fallback


# Mathematical scoring weights (Sum to 100)
WEIGHT_TEXT: int = 35
WEIGHT_REGULATORY: int = 25
WEIGHT_PRICING: int = 20
WEIGHT_INTEL: int = 20

# Financial thresholds & Metrology guards
MARKUP_TOLERANCE_PERCENT: float = 25.0  # Prices up to 25% above online median incur 0 penalty
TRUST_THRESHOLD: int = 70               # Scores below 70 trigger the Google Maps routing node

# City Coordinate Presets for Geolocation Remediation
CITY_COORDINATES: dict[str, dict[str, float]] = {
    "Chennai": {"lat": 13.0827, "lng": 80.2707},
    "Bengaluru": {"lat": 12.9716, "lng": 77.5946},
    "Mumbai": {"lat": 19.0760, "lng": 72.8777},
    "Delhi NCR": {"lat": 28.6139, "lng": 77.2090},
    "Hyderabad": {"lat": 17.3850, "lng": 78.4867},
    "Kolkata": {"lat": 22.5726, "lng": 88.3639},
}

DEFAULT_CITY: str = "Chennai"
DEFAULT_LAT: float = CITY_COORDINATES[DEFAULT_CITY]["lat"]
DEFAULT_LNG: float = CITY_COORDINATES[DEFAULT_CITY]["lng"]
