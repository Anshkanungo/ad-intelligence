"""
Ad Intelligence Pipeline — Language Detection Module

Detects languages in extracted text and transcripts.
Uses langdetect library.

Usage:
    from pipeline.modules.language_detection import LanguageDetectionModule
    lang_detector = LanguageDetectionModule()
    result = lang_detector.detect_language("Just Do It. Impossible is Nothing.")
"""

from langdetect import detect, detect_langs, LangDetectException

from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)

# ISO 639-1 code → full name mapping (common languages)
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "bn": "Bengali", "ur": "Urdu", "tr": "Turkish", "pl": "Polish",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish",
    "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
    "tl": "Filipino", "he": "Hebrew", "el": "Greek", "cs": "Czech",
    "ro": "Romanian", "hu": "Hungarian", "uk": "Ukrainian",
}


class LanguageDetectionModule:
    """Detects languages in text content."""

    @log_execution_time
    def detect_language(self, text: str) -> dict:
        """
        Detect language(s) in the given text.

        Args:
            text: Text string to analyze (OCR output or transcript)

        Returns:
            dict with keys:
                - primary_language: str — ISO 639-1 code
                - primary_language_name: str — full name
                - secondary_languages: list[str] — other detected languages
                - is_multilingual: bool
                - all_detected: list[dict] — all languages with probabilities
        """
        result = {
            "primary_language": "",
            "primary_language_name": "",
            "secondary_languages": [],
            "is_multilingual": False,
            "all_detected": [],
        }

        if not text or len(text.strip()) < 3:
            logger.warning("Text too short for language detection")
            return result

        try:
            # Get all detected languages with probabilities
            langs = detect_langs(text)

            all_detected = []
            for lang in langs:
                code = str(lang.lang)
                prob = round(float(lang.prob), 3)
                name = LANGUAGE_NAMES.get(code, code.upper())
                all_detected.append({
                    "code": code,
                    "name": name,
                    "probability": prob,
                })

            if all_detected:
                primary = all_detected[0]
                result["primary_language"] = primary["code"]
                result["primary_language_name"] = primary["name"]
                result["all_detected"] = all_detected

                # Secondary languages (probability > 0.1)
                secondary = [
                    d["code"] for d in all_detected[1:]
                    if d["probability"] > 0.1
                ]
                result["secondary_languages"] = secondary
                result["is_multilingual"] = len(secondary) > 0

            logger.info(
                f"Language: {result['primary_language_name']} "
                f"({result['primary_language']})"
                f"{' + ' + ', '.join(result['secondary_languages']) if result['is_multilingual'] else ''}"
            )

        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
        except Exception as e:
            logger.error(f"Language detection error: {e}", exc_info=True)

        return result


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("Language Detection Module — Quick Test")
    print("=" * 50)

    detector = LanguageDetectionModule()
    print("✓ Module initialized")

    test_cases = [
        ("Just Do It. The best shoes for running.", "English"),
        ("Impossible n'est pas français. Achetez maintenant!", "French"),
        ("Das beste Auto der Welt. Jetzt kaufen.", "German"),
        ("最高の品質をお届けします。今すぐ購入。", "Japanese"),
        ("Just Do It. Compra ahora. Limited time offer.", "Mixed EN/ES"),
    ]

    for text, expected in test_cases:
        result = detector.detect_language(text)
        status = "✓" if result["primary_language"] else "✗"
        print(f"\n  {status} Expected: {expected}")
        print(f"    Detected: {result['primary_language_name']} ({result['primary_language']})")
        if result["is_multilingual"]:
            print(f"    Also: {result['secondary_languages']}")

    print(f"\n{'=' * 50}")
    print("Language detection module ready!")
    print(f"{'=' * 50}")