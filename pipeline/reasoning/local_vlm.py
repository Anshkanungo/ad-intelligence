"""
Ad Intelligence Pipeline — Local VLM Engine (Qwen2.5-VL 3B, 4-bit quantized)

Runs Qwen2.5-VL locally on GPU with NF4 quantization for fast inference.
~2GB VRAM instead of ~6GB. ~3-5s per frame instead of 313s.

Requires: RTX 4060+ (8GB VRAM), CUDA-enabled PyTorch, bitsandbytes

Usage:
    from pipeline.reasoning.local_vlm import LocalVLM
    vlm = LocalVLM()
    description = vlm.describe_frame(pil_image, "Describe this ad frame in detail")
    json_str = vlm.generate_json(system_prompt, user_prompt, images=[img1, img2])
"""

import json
import torch
from PIL import Image

from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class LocalVLM:
    """Local Qwen2.5-VL 3B inference on GPU with 4-bit quantization."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self._model = None
        self._processor = None
        self._model_id = model_id
        self._loaded = False

    def _load(self):
        """Lazy-load model to GPU with 4-bit NF4 quantization."""
        if self._loaded:
            return

        logger.info(f"Loading {self._model_id} with 4-bit quantization...")

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # Extra compression, minimal quality loss
        )

        # Cap vision tokens: max_pixels controls how many patches the image becomes
        # 256*28*28 = 200,704 pixels → ~256 vision tokens (default is 1280*28*28 = ~1M)
        # This is the #1 lever for inference speed on this model
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            min_pixels=128 * 28 * 28,   # ~128 vision tokens minimum
            max_pixels=256 * 28 * 28,   # ~256 vision tokens maximum
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )

        self._loaded = True

        # Log actual VRAM usage
        vram_used = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"Qwen2.5-VL 3B loaded (4-bit) — VRAM: {vram_used:.1f} GB")

    def _build_messages(self, prompt: str, images: list[Image.Image] | None = None) -> list:
        """Build chat messages with optional images."""
        content = []

        if images:
            for img in images:
                # Resize to control vision token count — 512px is sweet spot
                # for 3B model: captures all text, much fewer tokens than 768
                max_dim = 512
                if max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def _generate(self, messages: list, max_tokens: int = 2048) -> str:
        """Run inference and return text output."""
        self._load()

        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        # Trim input tokens from output
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        response = self._processor.decode(generated[0], skip_special_tokens=True)
        return response.strip()

    @log_execution_time
    def describe_frame(self, image: Image.Image, prompt: str = None) -> str:
        """
        Describe a single frame/image.

        Args:
            image: PIL Image
            prompt: Custom prompt (default: detailed ad description)

        Returns:
            Text description
        """
        if not prompt:
            prompt = (
                "List what you see in this ad. Be brief. Use this format:\n"
                "TEXT: [all visible text, verbatim, separated by |]\n"
                "BRAND: [brand name if visible]\n"
                "PRODUCT: [what is being advertised]\n"
                "PEOPLE: [describe any people briefly]\n"
                "OBJECTS: [key objects]\n"
                "SETTING: [background/scene]\n"
                "COLORS: [dominant colors]\n"
                "Do NOT elaborate. Just list the facts."
            )

        messages = self._build_messages(prompt, images=[image])
        return self._generate(messages, max_tokens=150)

    @log_execution_time
    def describe_frames_batch(self, images: list[Image.Image]) -> list[str]:
        """
        Describe multiple frames efficiently.
        Sends each frame individually but keeps model warm.

        Args:
            images: List of PIL Images

        Returns:
            List of descriptions
        """
        self._load()  # Pre-load model

        descriptions = []
        for i, img in enumerate(images):
            logger.info(f"  Analyzing frame {i+1}/{len(images)}...")
            desc = self.describe_frame(img)
            descriptions.append(desc)
            logger.info(f"  Frame {i+1}: {desc[:80]}...")

        return descriptions

    @log_execution_time
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[Image.Image] | None = None,
    ) -> str:
        """
        Generate structured JSON output from images + prompt.
        Used for the final JSON generation step.

        Args:
            system_prompt: Schema + instructions
            user_prompt: Signals + context
            images: Optional images to include

        Returns:
            JSON string
        """
        full_prompt = system_prompt + "\n\n" + user_prompt
        full_prompt += "\n\nRespond with ONLY valid JSON. No markdown, no explanation."

        messages = self._build_messages(full_prompt, images=images)
        response = self._generate(messages, max_tokens=4096)

        # Clean JSON
        return self._clean_json(response)

    def _clean_json(self, raw: str) -> str:
        """Extract valid JSON from model output."""
        if not raw:
            return ""

        text = raw.strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                candidate = text[start:end]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass

        logger.error("Could not extract valid JSON from local VLM response")
        return ""

    def unload(self):
        """Free GPU memory."""
        if self._model:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False
            torch.cuda.empty_cache()
            logger.info("Local VLM unloaded, GPU memory freed")


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path

    print("=" * 50)
    print("Local VLM (Qwen2.5-VL 3B, 4-bit) — Quick Test")
    print("=" * 50)

    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram:.1f} GB")
    else:
        print("⚠ No CUDA GPU — this will be very slow on CPU")

    vlm = LocalVLM()
    print("✓ Module initialized (model loads on first call)")

    # Test with sample image
    test_image = Path("tests/sample_ads/test.jpg")
    if test_image.exists():
        print(f"\nAnalyzing: {test_image}")
        img = Image.open(test_image).convert("RGB")

        # First call — includes model loading time
        print("\n--- First call (includes model load) ---")
        desc = vlm.describe_frame(img)
        print(f"\nDescription:\n{desc}")

        # Second call — pure inference (should be fast)
        print("\n--- Second call (model warm, pure inference) ---")
        desc2 = vlm.describe_frame(img)
        print(f"\nDescription:\n{desc2}")

        # Log VRAM
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\nVRAM allocated: {vram_used:.2f} GB")
            print(f"VRAM reserved:  {vram_reserved:.2f} GB")

        # Test JSON generation
        print(f"\nTesting JSON generation...")
        json_out = vlm.generate_json(
            system_prompt='Return a JSON with keys: brand, product, headline. Fill based on the image.',
            user_prompt='Analyze this ad.',
            images=[img],
        )
        if json_out:
            print(f"JSON: {json_out[:500]}")
            print("✓ JSON generation works!")
    else:
        print("\n⚠ No test image at tests/sample_ads/test.jpg")
        print("  Put an ad image there and re-run")

    # Cleanup
    vlm.unload()

    print(f"\n{'=' * 50}")
    print("Local VLM ready! 4-bit quantization active.")
    print(f"{'=' * 50}")