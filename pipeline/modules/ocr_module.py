"""
Ad Intelligence Pipeline — OCR Module (EasyOCR)

Extracts ALL visible text from an image.
Returns raw text fragments with positions and confidence scores.

Uses EasyOCR — reliable, multilingual, works on CPU.
(Switched from PaddleOCR due to PaddlePaddle v3 compatibility issues on Windows)

Usage:
    from pipeline.modules.ocr_module import OCRModule
    ocr = OCRModule()
    result = ocr.extract(image_array)
"""

import numpy as np

from utils.config import config
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class OCRModule:
    """Extracts text from images using EasyOCR."""

    def __init__(self):
        self._reader = None

    def _get_reader(self):
        """Lazy-load EasyOCR reader (downloads model on first use)."""
        if self._reader is None:
            logger.info("Loading EasyOCR model...")
            import easyocr
            self._reader = easyocr.Reader(
                ["en"],         # Languages
                gpu=False,      # CPU only for free hosting
                verbose=False,
            )
            logger.info("EasyOCR model loaded")
        return self._reader

    @log_execution_time
    def extract(self, image: np.ndarray) -> dict:
        """
        Extract all text from an image.

        Args:
            image: OpenCV image array (BGR format)

        Returns:
            dict with keys:
                - raw_texts: list[str] — all text fragments found
                - text_with_positions: list[dict] — text + bbox + confidence
                - full_text: str — all text joined
                - avg_confidence: float — mean OCR confidence
                - text_count: int — number of text fragments
        """
        result = {
            "raw_texts": [],
            "text_with_positions": [],
            "full_text": "",
            "avg_confidence": 0.0,
            "text_count": 0,
        }

        try:
            reader = self._get_reader()

            # EasyOCR returns list of (bbox, text, confidence)
            # bbox format: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            ocr_output = reader.readtext(image)

            if not ocr_output:
                logger.warning("No text detected in image")
                return result

            positions = []
            confidences = []

            for (bbox, text, conf) in ocr_output:
                # Skip low confidence
                if conf < config.ocr_confidence_threshold:
                    continue

                # Skip empty text
                if not text.strip():
                    continue

                # Flatten bbox for storage
                flat_bbox = [coord for point in bbox for coord in point]

                # Top-y for sorting (reading order)
                sort_y = bbox[0][1] if bbox else 0

                confidences.append(conf)
                positions.append({
                    "text": text.strip(),
                    "bbox": [round(c, 1) for c in flat_bbox],
                    "confidence": round(conf, 3),
                    "_sort_y": sort_y,
                })

            # Sort by vertical position (top to bottom)
            positions.sort(key=lambda p: p["_sort_y"])

            # Remove sort key
            for p in positions:
                del p["_sort_y"]

            result["raw_texts"] = [p["text"] for p in positions]
            result["text_with_positions"] = positions
            result["full_text"] = " | ".join(result["raw_texts"])
            result["avg_confidence"] = round(
                sum(confidences) / len(confidences), 3
            ) if confidences else 0.0
            result["text_count"] = len(result["raw_texts"])

            logger.info(
                f"OCR found {result['text_count']} text fragments "
                f"(avg confidence: {result['avg_confidence']})"
            )

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}", exc_info=True)

        return result


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import cv2
    from pathlib import Path

    print("=" * 50)
    print("OCR Module (EasyOCR) — Quick Test")
    print("=" * 50)

    ocr = OCRModule()
    print("✓ OCR module initialized")

    test_paths = [
        Path("tests/sample_ads/test.jpg"),
        Path("tests/sample_ads/test.png"),
    ]

    test_image = None
    for p in test_paths:
        if p.exists():
            test_image = p
            break

    if test_image:
        print(f"\nProcessing: {test_image}")
        img = cv2.imread(str(test_image))
        result = ocr.extract(img)

        print(f"  Texts found: {result['text_count']}")
        print(f"  Avg confidence: {result['avg_confidence']}")
        print(f"  Raw texts:")
        for t in result["raw_texts"]:
            print(f"    - {t}")
        print(f"  Full text: {result['full_text'][:200]}")
    else:
        print("\n⚠ No test image. Drop an ad into tests/sample_ads/")

    print(f"\n{'=' * 50}")
    print("OCR module ready!")
    print(f"{'=' * 50}")