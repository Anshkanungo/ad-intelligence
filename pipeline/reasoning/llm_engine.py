"""
Ad Intelligence Pipeline — LLM Reasoning Engine

Sends ad image (or signals) to an LLM and gets back structured JSON.

Strategy:
  1. PRIMARY: Send actual image to VLM (Groq Llama 4 Scout) + schema + supplementary signals
  2. FALLBACK: Send actual image to Gemini Flash (also multimodal)
  3. LAST RESORT: Send text-only signals to Groq/Gemini (no image)

Usage:
    from pipeline.reasoning.llm_engine import LLMEngine
    engine = LLMEngine()
    json_str = engine.reason(system_prompt, user_prompt, image=pil_image)
"""

import io
import json
import base64
from PIL import Image
from groq import Groq
from google import genai

from utils.config import config
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class LLMEngine:
    """Calls VLM/LLM APIs to reason over ad signals and produce JSON."""

    def __init__(self):
        self._groq_client = None
        self._gemini_client = None

    def _get_groq(self) -> Groq:
        if self._groq_client is None and config.has_groq:
            self._groq_client = Groq(api_key=config.groq_api_key)
        return self._groq_client

    def _get_gemini(self):
        if self._gemini_client is None and config.has_gemini:
            self._gemini_client = genai.Client(api_key=config.gemini_api_key)
        return self._gemini_client

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 JPEG string."""
        if image.mode == "RGBA":
            image = image.convert("RGB")
        # Resize if very large (VLM APIs have limits)
        max_dim = 1024
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL image to JPEG bytes."""
        if image.mode == "RGBA":
            image = image.convert("RGB")
        max_dim = 1024
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    @log_execution_time
    def reason(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Image.Image | None = None,
        images: list[Image.Image] | None = None,
    ) -> str:
        """
        Send prompts (and optionally images) to LLM and get JSON response.

        Args:
            system_prompt: System prompt with schema
            user_prompt: User prompt with signals
            image: Optional single PIL Image for VLM mode
            images: Optional list of PIL Images (for video frames)

        Returns:
            Raw JSON string from LLM
        """
        # Normalize: if single image, put in list
        all_images = []
        if images:
            all_images = images
        elif image:
            all_images = [image]

        has_images = len(all_images) > 0

        # Strategy 1: Gemini VLM (preferred for multi-image / video)
        if has_images and len(all_images) > 1 and config.has_gemini:
            for attempt in range(config.llm_max_retries + 1):
                try:
                    result = self._call_gemini_vision(system_prompt, user_prompt, all_images)
                    if result:
                        return result
                except Exception as e:
                    logger.warning(f"Gemini VLM attempt {attempt + 1} failed: {e}")

        # Strategy 2: Groq VLM (single image — Llama 4 Scout)
        if has_images and config.has_groq:
            if len(all_images) == 1:
                # Single image — send directly
                for attempt in range(config.llm_max_retries + 1):
                    try:
                        result = self._call_groq_vision(system_prompt, user_prompt, all_images[0])
                        if result:
                            return result
                    except Exception as e:
                        logger.warning(f"Groq VLM attempt {attempt + 1} failed: {e}")
            else:
                # Multiple images (video) — describe each frame, then reason
                for attempt in range(config.llm_max_retries + 1):
                    try:
                        result = self._call_groq_multi_frame(system_prompt, user_prompt, all_images)
                        if result:
                            return result
                    except Exception as e:
                        logger.warning(f"Groq multi-frame attempt {attempt + 1} failed: {e}")

        # Strategy 3: Gemini VLM single image fallback
        if has_images and config.has_gemini:
            best_image = all_images[len(all_images) // 2] if len(all_images) > 1 else all_images[0]
            for attempt in range(config.llm_max_retries + 1):
                try:
                    result = self._call_gemini_vision(system_prompt, user_prompt, [best_image])
                    if result:
                        return result
                except Exception as e:
                    logger.warning(f"Gemini VLM single attempt {attempt + 1} failed: {e}")

        # Strategy 4: Text-only Groq
        if config.has_groq:
            logger.info("Falling back to text-only Groq...")
            for attempt in range(config.llm_max_retries + 1):
                try:
                    result = self._call_groq_text(system_prompt, user_prompt)
                    if result:
                        return result
                except Exception as e:
                    logger.warning(f"Groq text attempt {attempt + 1} failed: {e}")

        # Strategy 5: Text-only Gemini
        if config.has_gemini:
            logger.info("Falling back to text-only Gemini...")
            for attempt in range(config.llm_max_retries + 1):
                try:
                    result = self._call_gemini_text(system_prompt, user_prompt)
                    if result:
                        return result
                except Exception as e:
                    logger.warning(f"Gemini text attempt {attempt + 1} failed: {e}")

        logger.error("All LLM providers failed!")
        return ""

    def _call_groq_multi_frame(self, system_prompt: str, user_prompt: str, images: list[Image.Image]) -> str:
        """
        For video: describe each frame via Groq VLM, then combine all
        descriptions into one final reasoning call with the schema.
        """
        client = self._get_groq()
        if not client:
            return ""

        model = config.groq_vision_model
        logger.info(f"Groq multi-frame: analyzing {len(images)} frames individually...")

        frame_descriptions = []
        for i, img in enumerate(images):
            try:
                b64_image = self._image_to_base64(img)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                f"This is frame {i+1} of {len(images)} from a video advertisement. "
                                "Describe EVERYTHING you see in detail: brand names, product names, "
                                "text on screen, logos, people, objects, colors, actions. "
                                "Read ALL text visible including on products/packaging."
                            )},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                        ],
                    }],
                    temperature=0.1,
                    max_tokens=500,
                )
                desc = response.choices[0].message.content.strip()
                frame_descriptions.append(f"Frame {i+1}/{len(images)}: {desc}")
                logger.info(f"  Frame {i+1}: {desc[:80]}...")
            except Exception as e:
                logger.warning(f"  Frame {i+1} failed: {e}")
                frame_descriptions.append(f"Frame {i+1}/{len(images)}: (analysis failed)")

        # Now do final reasoning call with all frame descriptions
        combined_frames = "\n\n".join(frame_descriptions)
        enhanced_user_prompt = (
            user_prompt +
            f"\n\n--- VLM FRAME-BY-FRAME ANALYSIS ---\n{combined_frames}\n--- END FRAME ANALYSIS ---"
        )

        logger.info("Groq multi-frame: final reasoning call with all frame descriptions...")
        return self._call_groq_text(system_prompt, enhanced_user_prompt)

    # ══════════════════════════════════════════
    # GROQ CALLS
    # ══════════════════════════════════════════

    def _call_groq_vision(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        """Call Groq VLM (Llama 4 Scout) with image."""
        client = self._get_groq()
        if not client:
            return ""

        model = config.groq_vision_model
        logger.info(f"Calling Groq VLM ({model}) with image...")

        b64_image = self._image_to_base64(image)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        },
                    ],
                },
            ],
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        logger.info(f"Groq VLM response: {len(content)} chars")
        return self._clean_json(content)

    def _call_groq_text(self, system_prompt: str, user_prompt: str) -> str:
        """Call Groq text-only (Llama 3.3 70B)."""
        client = self._get_groq()
        if not client:
            return ""

        logger.info(f"Calling Groq text ({config.groq_model})...")

        response = client.chat.completions.create(
            model=config.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        logger.info(f"Groq text response: {len(content)} chars")
        return self._clean_json(content)

    # ══════════════════════════════════════════
    # GEMINI CALLS
    # ══════════════════════════════════════════

    def _call_gemini_vision(self, system_prompt: str, user_prompt: str, images: list[Image.Image]) -> str:
        """Call Gemini Flash with one or more images."""
        client = self._get_gemini()
        if not client:
            return ""

        logger.info(f"Calling Gemini VLM ({config.gemini_model}) with {len(images)} image(s)...")

        # Build parts list: text prompt + all images
        parts = [system_prompt + "\n\n" + user_prompt]

        for img in images:
            img_bytes = self._image_to_bytes(img)
            parts.append(genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        response = client.models.generate_content(
            model=config.gemini_model,
            contents=parts,
            config=genai.types.GenerateContentConfig(
                temperature=config.llm_temperature,
                max_output_tokens=config.llm_max_tokens,
                response_mime_type="application/json",
            ),
        )

        content = response.text
        logger.info(f"Gemini VLM response: {len(content)} chars")
        return self._clean_json(content)

    def _call_gemini_text(self, system_prompt: str, user_prompt: str) -> str:
        """Call Gemini text-only."""
        client = self._get_gemini()
        if not client:
            return ""

        logger.info(f"Calling Gemini text ({config.gemini_model})...")

        response = client.models.generate_content(
            model=config.gemini_model,
            contents=[system_prompt + "\n\n" + user_prompt],
            config=genai.types.GenerateContentConfig(
                temperature=config.llm_temperature,
                max_output_tokens=config.llm_max_tokens,
                response_mime_type="application/json",
            ),
        )

        content = response.text
        logger.info(f"Gemini text response: {len(content)} chars")
        return self._clean_json(content)

    # ══════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════

    def _clean_json(self, raw: str) -> str:
        """Clean LLM output to get valid JSON."""
        if not raw:
            return ""

        text = raw.strip()

        # Remove markdown code fences if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Validate it's parseable JSON
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}")
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                candidate = text[start:end]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass
            logger.error("Could not extract valid JSON from LLM response")
            return ""


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path

    print("=" * 50)
    print("LLM Engine — Quick Test")
    print("=" * 50)

    engine = LLMEngine()
    print("✓ Engine initialized")

    keys = config.validate_keys()
    print(f"  Groq:   {'✓' if keys['groq'] else '✗'}")
    print(f"  Gemini: {'✓' if keys['gemini'] else '✗'}")

    # Quick text test
    if keys["groq"] or keys["gemini"]:
        print("\nTesting text-only call...")
        result = engine.reason(
            system_prompt="Return a JSON object with a single key 'status' set to 'ok'.",
            user_prompt="Test.",
        )
        if result:
            print(f"  Text response: {json.loads(result)}")

        # VLM test with image
        test_img = Path("tests/sample_ads/test.jpg")
        if test_img.exists():
            print(f"\nTesting VLM call with image: {test_img}")
            img = Image.open(test_img).convert("RGB")
            result = engine.reason(
                system_prompt="Describe this ad image. Return JSON with keys: brand, product, headline.",
                user_prompt="Analyze this advertisement.",
                image=img,
            )
            if result:
                parsed = json.loads(result)
                print(f"  VLM response: {json.dumps(parsed, indent=2)[:500]}")
                print("  ✓ VLM connection works!")

    print(f"\n{'=' * 50}")
    print("LLM engine ready!")
    print(f"{'=' * 50}")