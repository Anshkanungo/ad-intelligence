"""
Ad Intelligence Pipeline — Color Analysis Module

Extracts dominant colors from images using OpenCV + KMeans clustering.
Returns hex codes, percentages, and a basic color mood.

Usage:
    from pipeline.modules.color_analysis import ColorAnalysisModule
    analyzer = ColorAnalysisModule()
    result = analyzer.analyze(image_array)
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans

from utils.config import config
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class ColorAnalysisModule:
    """Extracts dominant colors using KMeans clustering."""

    @log_execution_time
    def analyze(self, image: np.ndarray) -> dict:
        """
        Analyze dominant colors in an image.

        Args:
            image: OpenCV image array (BGR format)

        Returns:
            dict with keys:
                - dominant_colors_hex: list[str] — hex codes sorted by dominance
                - color_percentages: list[dict] — hex + percentage
                - color_mood: str — inferred mood (Warm, Cool, etc.)
                - brightness: str — Dark, Medium, Bright
                - saturation: str — Muted, Moderate, Vibrant
        """
        result = {
            "dominant_colors_hex": [],
            "color_percentages": [],
            "color_mood": "",
            "brightness": "",
            "saturation": "",
        }

        try:
            # Resize for speed
            size = config.color_resize
            resized = cv2.resize(image, (size, size))

            # Convert BGR → RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Reshape to flat pixel array: (n_pixels, 3)
            pixels = rgb.reshape(-1, 3).astype(np.float32)

            # KMeans clustering
            n_clusters = config.color_clusters
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            kmeans.fit(pixels)

            # Get cluster sizes (percentage of image each color covers)
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            total_pixels = len(kmeans.labels_)

            # Build color list sorted by dominance
            colors = []
            for i in range(len(labels)):
                center = kmeans.cluster_centers_[labels[i]]
                r, g, b = int(center[0]), int(center[1]), int(center[2])
                hex_code = f"#{r:02x}{g:02x}{b:02x}"
                pct = round((counts[i] / total_pixels) * 100, 1)
                colors.append({
                    "hex": hex_code,
                    "rgb": [r, g, b],
                    "percentage": pct,
                })

            # Sort by percentage (most dominant first)
            colors.sort(key=lambda c: c["percentage"], reverse=True)

            result["dominant_colors_hex"] = [c["hex"] for c in colors]
            result["color_percentages"] = [
                {"hex": c["hex"], "percentage": c["percentage"]}
                for c in colors
            ]

            # Analyze mood based on HSV values
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            avg_h = float(np.mean(hsv[:, :, 0]))  # Hue 0-179
            avg_s = float(np.mean(hsv[:, :, 1]))  # Saturation 0-255
            avg_v = float(np.mean(hsv[:, :, 2]))  # Value/Brightness 0-255

            # Brightness
            if avg_v < 85:
                result["brightness"] = "Dark"
            elif avg_v < 170:
                result["brightness"] = "Medium"
            else:
                result["brightness"] = "Bright"

            # Saturation
            if avg_s < 50:
                result["saturation"] = "Muted"
            elif avg_s < 130:
                result["saturation"] = "Moderate"
            else:
                result["saturation"] = "Vibrant"

            # Color mood (simplified rule-based)
            result["color_mood"] = self._infer_mood(avg_h, avg_s, avg_v)

            logger.info(
                f"Colors: {result['dominant_colors_hex'][:3]} | "
                f"Mood: {result['color_mood']} | "
                f"{result['brightness']}, {result['saturation']}"
            )

        except Exception as e:
            logger.error(f"Color analysis failed: {e}", exc_info=True)

        return result

    def _infer_mood(self, hue: float, sat: float, val: float) -> str:
        """Infer color mood from average HSV values."""
        if sat < 30:
            if val < 85:
                return "Dark/Moody"
            elif val > 200:
                return "Clean/Minimal"
            return "Neutral/Monochrome"

        if val < 70:
            return "Dark/Dramatic"

        # Hue-based mood (OpenCV hue is 0-179)
        if hue < 15 or hue > 165:
            return "Warm/Energetic"       # Red tones
        elif hue < 35:
            return "Warm/Inviting"        # Orange/yellow
        elif hue < 80:
            return "Fresh/Natural"        # Green tones
        elif hue < 130:
            return "Cool/Professional"    # Blue tones
        elif hue < 165:
            return "Bold/Creative"        # Purple/magenta

        return "Mixed/Balanced"


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path

    print("=" * 50)
    print("Color Analysis Module — Quick Test")
    print("=" * 50)

    analyzer = ColorAnalysisModule()
    print("✓ Color analyzer initialized")

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
        result = analyzer.analyze(img)

        print(f"  Top colors: {result['dominant_colors_hex'][:4]}")
        print(f"  Mood: {result['color_mood']}")
        print(f"  Brightness: {result['brightness']}")
        print(f"  Saturation: {result['saturation']}")
        for cp in result["color_percentages"][:4]:
            print(f"    {cp['hex']} — {cp['percentage']}%")
    else:
        print("\n⚠ No test image. Testing with synthetic red image...")
        red_img = np.zeros((200, 200, 3), dtype=np.uint8)
        red_img[:, :] = (0, 0, 255)  # BGR red
        result = analyzer.analyze(red_img)
        print(f"  Dominant: {result['dominant_colors_hex']}")
        print(f"  Mood: {result['color_mood']}")

    print(f"\n{'=' * 50}")
    print("Color analysis module ready!")
    print(f"{'=' * 50}")