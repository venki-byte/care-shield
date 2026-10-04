"""Multilingual Text-to-Speech (TTS) Audio Generator for Care-Shield.

Produces accessible spoken audio summaries in English, Hindi and Tamil for senior citizens
and visually impaired users, strictly adhering to defensible, non-assertive legal phrasing.
Never claims 'regulatory license verified' or 'product is safe' without government registry lookup.
"""

import io
import logging
from typing import Optional
from gtts import gTTS

from schemas import AuditScorecard

logger = logging.getLogger(__name__)


def build_audio_script(scorecard: AuditScorecard, language: str = "en") -> str:
    """Compose clear, legally defensible spoken verdict script for senior accessibility."""
    price = scorecard.parity_result.scanned_purchase_price if scorecard.parity_result else 0.0
    mrp = scorecard.ocr_result.printed_mrp_inr
    median = scorecard.parity_result.online_median if scorecard.parity_result else None
    score = scorecard.trust_score

    # Handle aborted audits (e.g. prescription medicines)
    if scorecard.is_aborted:
        if language == "hi":
            return "Care-Shield डॉक्टर की पर्ची वाली दवाओं की जाँच नहीं करता। कृपया अपने फार्मासिस्ट या डॉक्टर से सलाह लें।"
        if language == "ta":
            return "Care-Shield பரிந்துரைக்கப்பட்ட மருந்துகளை சரிபார்க்காது. உங்கள் மருந்தாளர் அல்லது மருத்துவரை அணுகவும்."
        return "Care-Shield does not audit prescription medicines. Please consult your pharmacist or doctor."

    pharmacy_mention_en = ""
    pharmacy_mention_ta = ""
    if scorecard.fallback_pharmacies:
        top_pharmacy = scorecard.fallback_pharmacies[0]
        pharmacy_mention_en = f" Highly rated alternative pharmacies such as {top_pharmacy.name} are available nearby."
        pharmacy_mention_ta = f" அருகில் {top_pharmacy.name} போன்ற மாற்று மருந்தகங்கள் உள்ளன."

    pharmacy_mention_hi = ""
    if scorecard.fallback_pharmacies:
        pharmacy_mention_hi = f" पास में {scorecard.fallback_pharmacies[0].name} जैसी अच्छी रेटिंग वाली फार्मेसी उपलब्ध हैं।"

    if language == "hi":
        if scorecard.parity_result and scorecard.parity_result.is_above_printed_mrp and mrp:
            return (
                f"कानूनी माप-तौल चेतावनी! दुकान का मूल्य {int(price)} रुपये है, जो छपे हुए "
                f"अधिकतम खुदरा मूल्य {int(mrp)} रुपये से अधिक है। यह कानून का उल्लंघन है। "
                f"विश्वसनीयता स्कोर सौ में से {score} है।{pharmacy_mention_hi}"
            )
        elif score < 70 or (scorecard.parity_result and scorecard.parity_result.markup_percent > 25.0):
            median_phrase = f", जो ऑनलाइन औसत मूल्य {int(median)} रुपये से अधिक है" if median else ""
            return (
                f"सावधान! माँगा गया मूल्य {int(price)} रुपये है{median_phrase}। "
                f"पैकेजिंग में गड़बड़ियाँ मिली हैं।{pharmacy_mention_hi}"
            )
        else:
            return (
                "जाँच पूरी हुई। पैकेजिंग या मूल्य में कोई स्पष्ट गड़बड़ी नहीं मिली। "
                "हम सरकारी रजिस्ट्री में लाइसेंस की सीधी पुष्टि नहीं कर सके। "
                f"दुकान का मूल्य {int(price)} रुपये है।"
            )

    if language == "ta":
        if scorecard.parity_result and scorecard.parity_result.is_above_printed_mrp and mrp:
            return (
                f"சட்ட அளவியல் எச்சரிக்கை! கடையின் விற்பனை விலை {int(price)} ரூபாய், "
                f"சட்டப்பூர்வ அச்சிடப்பட்ட MRP விலை {int(mrp)} ரூபாயை விட அதிகம். "
                f"இது சட்ட விதிமீறல் ஆகும். நம்பகத்தன்மை மதிப்பீடு நூற்றுக்கு {score}.{pharmacy_mention_ta}"
            )
        elif score < 70 or (scorecard.parity_result and scorecard.parity_result.markup_percent > 25.0):
            median_phrase = f", ஆன்லைன் சராசரி விலையான {int(median)} ரூபாயை விட அதிகம்" if median else ""
            return (
                f"சரிபார்ப்பு எச்சரிக்கை! கோரப்பட்ட விலை {int(price)} ரூபாய்{median_phrase}. "
                f"பேக்கேஜிங் எழுத்துக்களில் மாறுபாடுகள் கண்டறியப்பட்டுள்ளன.{pharmacy_mention_ta}"
            )
        else:
            # Defensible phrasing (Step 6 Directive)
            return (
                f"சரிபார்ப்பு முடிந்தது. வெளிப்படையான முரண்பாடுகள் இல்லை. "
                f"உரிமத்தை அரசு தரவுத்தளத்தில் நேரடியாக உறுதிப்படுத்த இயலவில்லை. "
                f"கடையின் விற்பனை விலை {int(price)} ரூபாய்."
            )
    else:
        # Default English script (Step 6 Directive)
        if scorecard.parity_result and scorecard.parity_result.is_above_printed_mrp and mrp:
            return (
                f"Legal Metrology Alert. The store asking price of {int(price)} Rupees "
                f"exceeds the legally mandated Maximum Retail Price of {int(mrp)} Rupees. "
                f"Overcharging violates Section 36 of the Legal Metrology Act.{pharmacy_mention_en}"
            )
        elif score < 70 or (scorecard.parity_result and scorecard.parity_result.markup_percent > 25.0):
            median_phrase = f" significantly higher than the online benchmark of {int(median)} Rupees" if median else ""
            return (
                f"Verification Warning. The asking price of {int(price)} Rupees is{median_phrase}. "
                f"Potential discrepancies detected in packaging text.{pharmacy_mention_en}"
            )
        else:
            # Defensible phrasing (Step 6 Directive: Never say 'license verified' or 'product is safe')
            return (
                f"Audit complete. No red flags found in packaging format or pricing. "
                f"Note that we could not explicitly verify the license against an official government registry. "
                f"Store asking price is {int(price)} Rupees."
            )


def synthesize_audio_verdict(scorecard: AuditScorecard, language: str = "en") -> Optional[bytes]:
    """Generate MP3 audio bytes using gTTS with error handling."""
    script = build_audio_script(scorecard, language)
    try:
        lang_code = language if language in ("ta", "hi") else "en"
        tts = gTTS(text=script, lang=lang_code, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        logger.warning(f"gTTS audio synthesis failed (offline or network issue): {e}")
        return None
