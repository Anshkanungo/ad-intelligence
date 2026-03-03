"""
Ad Intelligence Pipeline — Scene Description Module

Generates natural language descriptions of ad images.
Strategy: Use BLIP-1 locally (lightweight, ~1GB) for captioning.
The LLM reasoning engine handles deep ad analysis from the caption + other signals.

Falls back to HuggingFace Inference API if local loading fails (RAM constraints).

Usage:
    from pipeline.modules.scene_description import SceneDescriptionModule
    describer = SceneDescriptionModule()
    result = describer.describe(pil_image)
"""

import io
import base64
import requests as http_requests
from PIL import Image

from utils.config import config
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class SceneDescriptionModule:
    """Describes ad images using BLIP (local) or HuggingFace API (fallback)."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._use_local = True       # Try local first
        self._loaded = False

    def _load_local_model(self):
        """Load BLIP captioning model locally (~1GB download on first use)."""
        if self._loaded:
            return

        try:
            logger.info("Loading BLIP captioning model locally...")
            from transformers import BlipProcessor, BlipForConditionalGeneration

            model_id = "Salesforce/blip-image-captioning-base"
            self._processor = BlipProcessor.from_pretrained(model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(model_id)
            self._loaded = True
            self._use_local = True
            logger.info("BLIP model loaded locally")

        except Exception as e:
            logger.warning(f"Local BLIP load failed: {e}. Will use API fallback.")
            self._use_local = False

    def _caption_local(self, image: Image.Image, prompt: str = None) -> str:
        """Generate caption using local BLIP model."""
        try:
            if prompt:
                inputs = self._processor(image, prompt, return_tensors="pt")
            else:
                inputs = self._processor(image, return_tensors="pt")

            output = self._model.generate(**inputs, max_new_tokens=100)
            caption = self._processor.decode(output[0], skip_special_tokens=True)
            return caption.strip()

        except Exception as e:
            logger.error(f"Local caption failed: {e}")
            return ""

    def _caption_api(self, image: Image.Image, model_id: str = None) -> str:
        """Fallback: Use HuggingFace Inference API."""
        if not config.has_hf:
            return ""

        model_id = model_id or "Salesforce/blip-image-captioning-base"
        api_url = f"{config.hf_api_url}{model_id}"
        headers = {"Authorization": f"Bearer {config.hf_api_token}"}

        try:
            # Convert image to bytes
            buffer = io.BytesIO()
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()

            response = http_requests.post(
                api_url, headers=headers,
                data=image_bytes, timeout=30
            )

            if response.status_code == 503:
                logger.warning("HF model loading, retrying in 20s...")
                import time
                time.sleep(20)
                response = http_requests.post(
                    api_url, headers=headers,
                    data=image_bytes, timeout=60
                )

            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    return result[0].get("generated_text", "")
                return str(result[0])
            return str(result)

        except Exception as e:
            logger.error(f"HF API error: {e}")
            return ""

    @log_execution_time
    def describe(self, image: Image.Image) -> dict:
        """
        Generate scene descriptions for an ad image.

        Args:
            image: PIL Image (RGB)

        Returns:
            dict with keys:
                - caption: str — general image description
                - ad_content: str — conditional caption about ad content
                - setting: str — conditional caption about setting
                - people: str — conditional caption about people
                - mood: str — conditional caption about mood
                - method: str — "local" or "api"
        """
        result = {
            "caption": "",
            "ad_content": "",
            "setting": "",
            "people": "",
            "mood": "",
            "method": "",
        }

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Try local model first
        self._load_local_model()

        if self._use_local and self._loaded:
            result["method"] = "local"

            # 1. Unconditional caption
            result["caption"] = self._caption_local(image)
            logger.info(f"Caption: {result['caption'][:100]}")

            # 2. Conditional captions with prompts
            prompts = {
                "ad_content": "this advertisement is promoting",
                "setting": "the setting of this image is",
                "people": "the people in this image are",
                "mood": "the mood of this image is",
            }

            for key, prompt in prompts.items():
                answer = self._caption_local(image, prompt=prompt)
                result[key] = answer
                if answer:
                    logger.debug(f"  {key}: {answer[:80]}")

        else:
            # Fallback to API
            result["method"] = "api"
            result["caption"] = self._caption_api(image)
            logger.info(f"Caption (API): {result['caption'][:100]}")

            # API doesn't support conditional captioning easily,
            # so we only get the basic caption. LLM will do the rest.

        return result


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path

    print("=" * 50)
    print("Scene Description Module — Quick Test")
    print("=" * 50)

    describer = SceneDescriptionModule()
    print("✓ Module initialized")

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
        img = Image.open(test_image).convert("RGB")
        result = describer.describe(img)

        print(f"\n  Method: {result['method']}")
        for key, val in result.items():
            if key != "method":
                display = val[:100] if val else "(empty)"
                print(f"  {key}: {display}")
    else:
        print("\n⚠ No test image. Drop an ad image into tests/sample_ads/")

    print(f"\n{'=' * 50}")
    print("Scene description module ready!")
    print(f"{'=' * 50}")