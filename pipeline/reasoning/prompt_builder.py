"""
Ad Intelligence Pipeline — Prompt Builder

Builds the mega-prompt that gets sent to the LLM with all signals + schema.
The LLM's job: reason over signals, fill every JSON field, return valid JSON.

Usage:
    from pipeline.reasoning.prompt_builder import PromptBuilder
    builder = PromptBuilder()
    system_prompt, user_prompt = builder.build(aggregated_context)
"""

from schema.ad_schema import get_schema_json_template
from utils.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are an expert Ad Intelligence Extraction Engine. Your job is to analyze advertisement signals and produce a comprehensive, structured JSON analysis.

You will receive raw signals extracted from an advertisement by multiple AI modules:
- OCR (text extraction)
- Object Detection (YOLO)
- Color Analysis (dominant colors, mood)
- Scene Description (AI vision captioning)
- Language Detection
- Audio Transcription (for video/audio ads)
- Video frame analysis (for video ads)

YOUR TASK:
1. Analyze ALL signals together — cross-reference, deduplicate, resolve conflicts
2. Fill EVERY field in the JSON schema below
3. Use your knowledge to INFER fields that signals don't directly provide (industry, target audience, tone, themes, etc.)
4. If information is genuinely NOT available and cannot be inferred, use: "" for strings, [] for lists, false for booleans, 0 for numbers

CRITICAL RULES:
- Output ONLY valid JSON. No markdown, no backticks, no explanation before or after
- NEVER omit a field. The schema shape must be EXACTLY as shown
- NEVER add extra fields not in the schema
- NEVER invent or guess brand names. Only identify a brand if you see a clear logo, brand name text, or unmistakable trademark in the image. If placeholder text like "YOURSTORENAME" or "YOURWEBSITE" appears, the brand is UNKNOWN — set company_name to ""
- NEVER invent prices, URLs, or specific product names unless they are clearly visible in the ad
- Be skeptical of AI vision signals — they may hallucinate brand names. Trust what you can directly SEE in the image over supplementary signals
- DO use your reasoning to infer classification fields (industry, tone, audience, themes) — these are analytical, not factual
- Cross-reference ALL signals: if OCR text, vision description, and the image itself all agree, confidence is high. If they conflict, trust the image
- Distinguish between template/placeholder content (e.g. "YOURSTORENAME", "yourwebsite.com", "Lorem ipsum") and real ad content
- For the _meta section: leave processed_at, processing_time_sec, modules_used, and errors empty — they are filled by the pipeline

JSON SCHEMA (fill every field):
"""

USER_PROMPT_TEMPLATE = """Here are the extracted signals from an advertisement. Analyze them and return the complete JSON.

{context}

IMPORTANT: If images/frames are attached, examine them carefully. They are the primary source of truth.
- Read ALL text visible in the images (brand names, product names, taglines, URLs, etc.)
- Identify the brand from logos and text ON the product/packaging
- For video: the attached images are key frames from the video — analyze them as a sequence telling a story
- Fill in video_analysis.scenes with descriptions of what each frame shows

Return ONLY the complete JSON object following the exact schema. No other text."""


class PromptBuilder:
    """Builds system and user prompts for LLM reasoning."""

    def __init__(self):
        self._schema_template = get_schema_json_template()

    def build(self, aggregated_context: str) -> tuple[str, str]:
        """
        Build the complete prompts for LLM.

        Args:
            aggregated_context: Merged output from SignalAggregator

        Returns:
            tuple of (system_prompt, user_prompt)
        """
        system = SYSTEM_PROMPT + "\n" + self._schema_template
        user = USER_PROMPT_TEMPLATE.format(context=aggregated_context)

        logger.info(
            f"Prompt built: system={len(system)} chars, "
            f"user={len(user)} chars"
        )

        return system, user


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("Prompt Builder — Quick Test")
    print("=" * 50)

    builder = PromptBuilder()

    fake_context = """=== AD ANALYSIS SIGNALS ===
Media Type: IMAGE

--- OCR EXTRACTED TEXT ---
Total text fragments: 2
Combined text: BUY NOW | SALE 70% OFF

--- SCENE DESCRIPTION (AI Vision) ---
Caption: a shoe advertisement with dark background

=== END OF SIGNALS ==="""

    system, user = builder.build(fake_context)

    print(f"  System prompt: {len(system)} chars")
    print(f"  User prompt: {len(user)} chars")
    print(f"\n  System (first 300 chars):")
    print(f"  {system[:300]}...")
    print(f"\n  User (first 300 chars):")
    print(f"  {user[:300]}...")

    print(f"\n{'=' * 50}")
    print("Prompt builder ready!")
    print(f"{'=' * 50}")