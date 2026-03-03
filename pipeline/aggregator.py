"""
Ad Intelligence Pipeline — Signal Aggregator

Merges outputs from all extraction modules into a single context blob
that gets fed to the LLM reasoning engine.

Usage:
    from pipeline.aggregator import SignalAggregator
    aggregator = SignalAggregator()
    context = aggregator.merge(ocr_result, yolo_result, color_result, ...)
"""

from utils.logger import get_logger

logger = get_logger(__name__)


class SignalAggregator:
    """Merges all module outputs into a single context string for the LLM."""

    def merge(
        self,
        media_type: str,
        ocr_result: dict | None = None,
        yolo_result: dict | None = None,
        color_result: dict | None = None,
        scene_result: dict | None = None,
        language_result: dict | None = None,
        transcription_result: dict | None = None,
        frame_results: list[dict] | None = None,
        video_info: dict | None = None,
    ) -> str:
        """
        Merge all module outputs into a structured context string.

        Returns a formatted string that the LLM can reason over.
        """
        sections = []
        sections.append(f"=== AD ANALYSIS SIGNALS ===")
        sections.append(f"Media Type: {media_type.upper()}")

        # ── OCR Text ──
        if ocr_result and ocr_result.get("text_count", 0) > 0:
            sections.append("\n--- OCR EXTRACTED TEXT ---")
            sections.append(f"Total text fragments: {ocr_result['text_count']}")
            sections.append(f"Average confidence: {ocr_result['avg_confidence']}")
            sections.append(f"All text found (in reading order):")
            for item in ocr_result.get("text_with_positions", []):
                sections.append(f"  - \"{item['text']}\" (confidence: {item['confidence']})")
            sections.append(f"Combined text: {ocr_result.get('full_text', '')}")
        else:
            sections.append("\n--- OCR EXTRACTED TEXT ---")
            sections.append("No text detected by OCR.")

        # ── Object Detection ──
        if yolo_result and yolo_result.get("total_objects", 0) > 0:
            sections.append("\n--- OBJECT DETECTION ---")
            sections.append(f"Objects found: {yolo_result['objects']}")
            sections.append(f"Counts: {yolo_result['object_counts']}")
            sections.append(f"People detected: {yolo_result['people_detected']} (count: {yolo_result['people_count']})")
        else:
            sections.append("\n--- OBJECT DETECTION ---")
            sections.append("No standard objects detected (ad may use stylized/graphic design elements).")

        # ── Color Analysis ──
        if color_result and color_result.get("dominant_colors_hex"):
            sections.append("\n--- COLOR ANALYSIS ---")
            sections.append(f"Dominant colors (hex): {color_result['dominant_colors_hex']}")
            sections.append(f"Color mood: {color_result.get('color_mood', '')}")
            sections.append(f"Brightness: {color_result.get('brightness', '')}")
            sections.append(f"Saturation: {color_result.get('saturation', '')}")
            if color_result.get("color_percentages"):
                for cp in color_result["color_percentages"][:4]:
                    sections.append(f"  {cp['hex']}: {cp['percentage']}%")

        # ── Scene Description (BLIP) ──
        if scene_result and scene_result.get("caption"):
            sections.append("\n--- SCENE DESCRIPTION (AI Vision) ---")
            sections.append(f"Caption: {scene_result['caption']}")
            if scene_result.get("ad_content"):
                sections.append(f"Ad content: {scene_result['ad_content']}")
            if scene_result.get("setting"):
                sections.append(f"Setting: {scene_result['setting']}")
            if scene_result.get("people"):
                sections.append(f"People: {scene_result['people']}")
            if scene_result.get("mood"):
                sections.append(f"Mood: {scene_result['mood']}")

        # ── Language Detection ──
        if language_result and language_result.get("primary_language"):
            sections.append("\n--- LANGUAGE ---")
            sections.append(f"Primary: {language_result['primary_language_name']} ({language_result['primary_language']})")
            if language_result.get("is_multilingual"):
                sections.append(f"Secondary: {language_result['secondary_languages']}")

        # ── Audio/Transcription ──
        if transcription_result and transcription_result.get("has_speech"):
            sections.append("\n--- AUDIO TRANSCRIPTION ---")
            sections.append(f"Language: {transcription_result.get('language', '')}")
            sections.append(f"Full transcript: {transcription_result['transcript']}")
            if transcription_result.get("segments"):
                sections.append("Timestamped segments:")
                for seg in transcription_result["segments"][:20]:
                    sections.append(f"  [{seg['start']}s - {seg['end']}s] {seg['text']}")

        # ── Video Info ──
        if video_info:
            sections.append("\n--- VIDEO INFO ---")
            sections.append(f"Duration: {video_info.get('duration_seconds', 0)}s")
            sections.append(f"Resolution: {video_info.get('resolution', '')}")
            sections.append(f"FPS: {video_info.get('fps', 0)}")
            sections.append(f"Scenes/frames analyzed: {video_info.get('frame_count', 0)}")

        # ── Per-frame analysis for video ──
        if frame_results:
            sections.append("\n--- PER-FRAME ANALYSIS ---")
            for i, fr in enumerate(frame_results):
                sections.append(f"\n  Frame {i+1} (t={fr.get('timestamp_sec', '?')}s):")
                if fr.get("ocr") and fr["ocr"].get("text_count", 0) > 0:
                    sections.append(f"    OCR text: {fr['ocr']['full_text']}")
                if fr.get("scene") and fr["scene"].get("caption"):
                    sections.append(f"    Scene: {fr['scene']['caption']}")
                if fr.get("yolo") and fr["yolo"].get("total_objects", 0) > 0:
                    sections.append(f"    Objects: {fr['yolo']['objects']}")

        sections.append("\n=== END OF SIGNALS ===")

        context = "\n".join(sections)
        logger.info(f"Aggregated context: {len(context)} chars from {media_type} ad")
        return context


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("Signal Aggregator — Quick Test")
    print("=" * 50)

    agg = SignalAggregator()

    # Simulate module outputs
    context = agg.merge(
        media_type="image",
        ocr_result={
            "raw_texts": ["BUY NOW", "+123-456-789"],
            "text_with_positions": [
                {"text": "BUY NOW", "bbox": [], "confidence": 0.95},
                {"text": "+123-456-789", "bbox": [], "confidence": 0.88},
            ],
            "full_text": "BUY NOW | +123-456-789",
            "avg_confidence": 0.915,
            "text_count": 2,
        },
        color_result={
            "dominant_colors_hex": ["#0c0b0a", "#ff5722"],
            "color_percentages": [
                {"hex": "#0c0b0a", "percentage": 60.0},
                {"hex": "#ff5722", "percentage": 25.0},
            ],
            "color_mood": "Dark/Dramatic",
            "brightness": "Dark",
            "saturation": "Muted",
        },
        scene_result={
            "caption": "a flyer for a shoe sale",
            "ad_content": "promoting nike shoes",
            "setting": "dark background studio",
            "people": "",
            "mood": "bold and energetic",
        },
        language_result={
            "primary_language": "en",
            "primary_language_name": "English",
            "is_multilingual": False,
            "secondary_languages": [],
        },
    )

    print(context)
    print(f"\n{'=' * 50}")
    print(f"Context length: {len(context)} chars")
    print("Aggregator ready!")
    print(f"{'=' * 50}")