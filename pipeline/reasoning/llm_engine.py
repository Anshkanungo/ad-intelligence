"""
Ad Intelligence Pipeline — LLM Reasoning Engine (Local Only)

Pure local inference via Qwen2.5-VL 3B (4-bit) on GPU.
No cloud APIs, no rate limits, no data leaving the machine.

Fallback chain:
  1. Local VLM direct (≤4 images) — send images + prompt, get JSON
  2. Local VLM multi-frame (>4 images) — describe each frame, then JSON
  3. Local VLM text-only — no images, just text reasoning

Usage:
    from pipeline.reasoning.llm_engine import LLMEngine
    engine = LLMEngine()
    json_str = engine.reason(system_prompt, user_prompt, images=[img1, img2])
"""

import json
import torch
from PIL import Image

from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class LLMEngine:
    """Local-only LLM engine: Qwen2.5-VL 3B on GPU."""

    def __init__(self):
        self._local_vlm = None
        self._has_gpu = torch.cuda.is_available()

        if self._has_gpu:
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.error("No GPU available — local VLM requires CUDA GPU")

    def _get_vlm(self):
        """Lazy-load the local VLM."""
        if self._local_vlm is None:
            from pipeline.reasoning.local_vlm import LocalVLM
            self._local_vlm = LocalVLM()
        return self._local_vlm

    @log_execution_time
    def reason(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Image.Image | None = None,
        images: list[Image.Image] | None = None,
    ) -> str:
        """
        Main reasoning method.

        Args:
            system_prompt: Schema + instructions
            user_prompt: Aggregated signals + context
            image: Single PIL Image (optional)
            images: List of PIL Images (optional, for video frames)

        Returns:
            Raw JSON string
        """
        if not self._has_gpu:
            logger.error("Cannot reason without GPU")
            return ""

        all_images = []
        if images:
            all_images = images
        elif image:
            all_images = [image]

        vlm = self._get_vlm()

        # ≤4 images: send directly to generate_json
        if len(all_images) <= 4:
            try:
                logger.info(f"Local VLM: direct JSON generation with {len(all_images)} image(s)...")
                result = vlm.generate_json(
                    system_prompt, user_prompt,
                    images=all_images if all_images else None,
                )
                if result:
                    return result
                logger.warning("Local VLM returned empty JSON, retrying...")
            except Exception as e:
                logger.warning(f"Local VLM direct failed: {e}")

        # >4 images: describe each frame, then combine for JSON
        if len(all_images) > 4:
            try:
                return self._multiframe_reason(vlm, system_prompt, user_prompt, all_images)
            except Exception as e:
                logger.warning(f"Local VLM multi-frame failed: {e}")

        # Last resort: text-only (no images)
        if all_images:
            logger.info("Retrying as text-only (dropping images)...")
            try:
                result = vlm.generate_json(system_prompt, user_prompt, images=None)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Local VLM text-only failed: {e}")

        logger.error("All local reasoning attempts failed")
        return ""

    def _multiframe_reason(
        self,
        vlm,
        system_prompt: str,
        user_prompt: str,
        images: list[Image.Image],
    ) -> str:
        """
        For videos with many frames:
        1. Describe each frame individually (~13s each)
        2. Combine all descriptions into one prompt
        3. Generate final JSON with reference images
        """
        logger.info(f"Multi-frame analysis: {len(images)} frames...")

        # Step 1: describe every frame
        descriptions = vlm.describe_frames_batch(images)

        # Step 2: build enhanced prompt with all frame descriptions
        frame_text = "\n\n".join(
            f"Frame {i+1}/{len(images)}: {desc}"
            for i, desc in enumerate(descriptions)
        )

        enhanced_prompt = (
            user_prompt
            + f"\n\n--- FRAME-BY-FRAME ANALYSIS ({len(images)} frames) ---\n"
            + frame_text
            + "\n--- END FRAME ANALYSIS ---"
        )

        # Step 3: generate JSON with first + middle + last frames as visual reference
        ref_indices = list(dict.fromkeys([0, len(images) // 2, len(images) - 1]))
        ref_images = [images[i] for i in ref_indices]

        logger.info(f"Generating final JSON with {len(ref_indices)} reference frames...")
        return vlm.generate_json(system_prompt, enhanced_prompt, images=ref_images)

    def describe_frame(self, image: Image.Image, prompt: str = None) -> str:
        """
        Convenience method: describe a single frame.
        Used by orchestrator for individual frame analysis.
        """
        if not self._has_gpu:
            return ""
        vlm = self._get_vlm()
        return vlm.describe_frame(image, prompt)

    def describe_frames(self, images: list[Image.Image]) -> list[str]:
        """
        Convenience method: describe multiple frames.
        Used by orchestrator for video frame analysis.
        """
        if not self._has_gpu:
            return []
        vlm = self._get_vlm()
        return vlm.describe_frames_batch(images)

    def unload(self):
        """Free GPU memory."""
        if self._local_vlm:
            self._local_vlm.unload()
            self._local_vlm = None


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path

    print("=" * 50)
    print("LLM Engine (Local Only) — Quick Test")
    print("=" * 50)

    engine = LLMEngine()
    print(f"  GPU: {'✓ ' + torch.cuda.get_device_name(0) if engine._has_gpu else '✗ No GPU'}")

    test_img = Path("tests/sample_ads/test.jpg")
    if test_img.exists() and engine._has_gpu:
        img = Image.open(test_img).convert("RGB")

        # Test frame description
        print(f"\n1. Frame description...")
        desc = engine.describe_frame(img)
        print(f"   {desc[:200]}")

        # Test JSON generation
        print(f"\n2. JSON generation...")
        result = engine.reason(
            system_prompt="Return JSON with keys: brand, product, headline. Fill based on the image.",
            user_prompt="Analyze this ad image.",
            image=img,
        )
        if result:
            print(f"   {json.dumps(json.loads(result), indent=2)}")
            print("   ✓ Works!")

        engine.unload()
    else:
        print("\n⚠ Need GPU + tests/sample_ads/test.jpg")

    print(f"\n{'=' * 50}")