"""
Ad Intelligence Pipeline — Object Detection Module (YOLOv8-nano)

Detects objects, people, and products in ad images.
Uses YOLOv8-nano pretrained on COCO dataset (80 classes).

Usage:
    from pipeline.modules.object_detection import ObjectDetectionModule
    detector = ObjectDetectionModule()
    result = detector.detect(image_array)
"""

import numpy as np
from ultralytics import YOLO

from utils.config import config
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class ObjectDetectionModule:
    """Detects objects in images using YOLOv8-nano."""

    def __init__(self):
        self._model = None

    def _get_model(self) -> YOLO:
        """Lazy-load YOLOv8 model (downloads on first use)."""
        if self._model is None:
            logger.info(f"Loading YOLO model: {config.yolo_model}")
            self._model = YOLO(config.yolo_model)
            logger.info("YOLO model loaded")
        return self._model

    @log_execution_time
    def detect(self, image: np.ndarray) -> dict:
        """
        Detect objects in an image.

        Args:
            image: OpenCV image array (BGR format)

        Returns:
            dict with keys:
                - objects: list[str] — unique object labels found
                - object_counts: dict — label → count
                - detections: list[dict] — full detection details
                - people_detected: bool
                - people_count: int
                - total_objects: int
        """
        result = {
            "objects": [],
            "object_counts": {},
            "detections": [],
            "people_detected": False,
            "people_count": 0,
            "total_objects": 0,
        }

        try:
            model = self._get_model()
            predictions = model(image, conf=config.yolo_confidence, verbose=False)

            if not predictions or len(predictions) == 0:
                logger.warning("No objects detected")
                return result

            detections = []
            label_counts = {}

            for pred in predictions:
                boxes = pred.boxes
                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    label = model.names[cls_id]
                    conf = float(boxes.conf[i])
                    bbox = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]

                    label_counts[label] = label_counts.get(label, 0) + 1
                    detections.append({
                        "label": label,
                        "confidence": round(conf, 3),
                        "bbox": [round(c, 1) for c in bbox],
                    })

            # Sort by confidence
            detections.sort(key=lambda d: d["confidence"], reverse=True)

            result["objects"] = list(label_counts.keys())
            result["object_counts"] = label_counts
            result["detections"] = detections
            result["total_objects"] = len(detections)
            result["people_detected"] = "person" in label_counts
            result["people_count"] = label_counts.get("person", 0)

            logger.info(
                f"Detected {result['total_objects']} objects: "
                f"{', '.join(f'{v}x {k}' for k, v in label_counts.items())}"
            )

        except Exception as e:
            logger.error(f"Object detection failed: {e}", exc_info=True)

        return result


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import cv2
    from pathlib import Path

    print("=" * 50)
    print("Object Detection Module — Quick Test")
    print("=" * 50)

    detector = ObjectDetectionModule()
    print("✓ Detector initialized")

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
        result = detector.detect(img)

        print(f"  Total objects: {result['total_objects']}")
        print(f"  Labels: {result['objects']}")
        print(f"  Counts: {result['object_counts']}")
        print(f"  People: {result['people_count']}")
    else:
        print("\n⚠ No test image. Drop an ad image into tests/sample_ads/")
        print("  Testing with blank image...")
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect(blank)
        print(f"  Objects on blank: {result['total_objects']} (expected: 0)")

    print(f"\n{'=' * 50}")
    print("Object detection module ready!")
    print(f"{'=' * 50}")