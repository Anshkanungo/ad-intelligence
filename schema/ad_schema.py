"""
Ad Intelligence Pipeline — Fixed Pydantic Schema v1.0.0

Every field is ALWAYS present. Missing data = null / "" / [] / false.
Schema shape NEVER changes regardless of ad type (image, video, audio).
13 top-level sections. This is the single output contract.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ──────────────────────────────────────────────
# Sub-models for nested structures
# ──────────────────────────────────────────────

class VideoScene(BaseModel):
    """A single scene detected in a video ad."""
    timestamp_start: str = ""
    timestamp_end: str = ""
    description: str = ""

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v):
        if isinstance(v, str):
            return cls(description=v)
        return v

    model_config = {"extra": "ignore"}


class AudioSegment(BaseModel):
    """A single segment of transcribed audio."""
    start: float = 0.0
    end: float = 0.0
    text: str = ""


# ──────────────────────────────────────────────
# Section 1: _meta — Pipeline metadata
# ──────────────────────────────────────────────

class Meta(BaseModel):
    schema_version: str = "1.0.0"
    pipeline_version: str = "1.0.0"
    processed_at: str = ""
    processing_time_sec: float = 0.0
    input_media_type: str = ""
    input_filename: str = ""
    input_resolution: str = ""
    confidence_score: float = 0.0
    modules_used: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Section 2: source — Ad origin/medium
# ──────────────────────────────────────────────

class Source(BaseModel):
    medium: str = ""
    platform: str = ""
    format: str = ""
    estimated_era: str = ""


# ──────────────────────────────────────────────
# Section 3: brand — Company/brand identification
# ──────────────────────────────────────────────

class Brand(BaseModel):
    company_name: str = ""
    parent_company: str = ""
    sub_brand: str = ""
    logo_detected: bool = False
    logo_description: str = ""
    logo_position: str = ""
    brand_colors_hex: list[str] = Field(default_factory=list)
    brand_url: str = ""
    brand_social_handles: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Section 4: product — What's being advertised
# ──────────────────────────────────────────────

class Product(BaseModel):
    product_name: str = ""
    product_category: str = ""
    product_subcategory: str = ""
    product_description: str = ""
    product_features: list[str] = Field(default_factory=list)
    price_mentioned: str = ""
    offer_or_discount: str = ""
    model_or_variant: str = ""


# ──────────────────────────────────────────────
# Section 5: text_content — All extracted text
# ──────────────────────────────────────────────

class TextContent(BaseModel):
    headline: str = ""
    tagline: str = ""
    subheadline: str = ""
    body_copy: str = ""
    call_to_action: str = ""
    fine_print: str = ""
    contact_info: str = ""
    hashtags: list[str] = Field(default_factory=list)
    all_raw_text: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Section 6: language — Language detection
# ──────────────────────────────────────────────

class Language(BaseModel):
    primary_language: str = ""
    primary_language_name: str = ""
    secondary_languages: list[str] = Field(default_factory=list)
    is_multilingual: bool = False


# ──────────────────────────────────────────────
# Section 7: audio — Audio analysis
# ──────────────────────────────────────────────

class Audio(BaseModel):
    has_speech: bool = False
    transcript: str = ""
    speaker_count: int = 0
    has_music: bool = False
    music_mood: str = ""
    music_genre: str = ""
    sound_effects: list[str] = Field(default_factory=list)
    voiceover_tone: str = ""
    jingle_detected: bool = False


# ──────────────────────────────────────────────
# Section 8: visual_analysis — Visual composition
# ──────────────────────────────────────────────

class VisualAnalysis(BaseModel):
    scene_description: str = ""
    setting: str = ""
    dominant_colors_hex: list[str] = Field(default_factory=list)
    color_mood: str = ""
    layout_style: str = ""
    people_detected: bool = False
    people_count: int = 0
    people_description: str = ""
    celebrity_or_figure: str = ""
    objects_detected: list[str] = Field(default_factory=list)
    image_quality: str = ""
    aspect_ratio: str = ""


# ──────────────────────────────────────────────
# Section 9: video_analysis — Video-specific
# ──────────────────────────────────────────────

class VideoAnalysis(BaseModel):
    duration_seconds: float = 0.0
    scene_count: int = 0
    scenes: list[VideoScene] = Field(default_factory=list)
    has_subtitles: bool = False
    transition_styles: list[str] = Field(default_factory=list)
    pacing: str = ""


# ──────────────────────────────────────────────
# Section 9: video_analysis — Video-specific
# ──────────────────────────────────────────────

class VideoAnalysis(BaseModel):
    duration_seconds: float = 0.0
    scene_count: int = 0
    scenes: list[VideoScene] = Field(default_factory=list)
    has_subtitles: bool = False
    transition_styles: list[str] = Field(default_factory=list)
    pacing: str = ""


# ──────────────────────────────────────────────
# Section 10: classification — Ad categorization
# ──────────────────────────────────────────────

class Classification(BaseModel):
    industry: str = ""
    ad_objective: str = ""
    ad_style: str = ""
    target_audience: str = ""
    target_gender: str = ""
    target_age_range: str = ""
    emotional_appeal: str = ""
    tone: str = ""
    themes: list[str] = Field(default_factory=list)
    seasonal_relevance: str = ""


# ──────────────────────────────────────────────
# Section 11: compliance_and_legal
# ──────────────────────────────────────────────

class ComplianceAndLegal(BaseModel):
    has_disclaimer: bool = False
    disclaimer_text: str = ""
    has_age_restriction: bool = False
    age_restriction_note: str = ""
    has_trademark_symbols: bool = False
    copyright_notice: str = ""
    regulatory_body_mentioned: str = ""


# ──────────────────────────────────────────────
# Section 12: engagement_elements
# ──────────────────────────────────────────────

class EngagementElements(BaseModel):
    has_qr_code: bool = False
    qr_code_content: str = ""
    has_coupon_code: bool = False
    coupon_code: str = ""
    has_social_proof: bool = False
    social_proof_type: str = ""
    urgency_elements: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Section 13: ad_description — Summary
# ──────────────────────────────────────────────

class AdDescription(BaseModel):
    short_summary: str = ""
    detailed_description: str = ""
    creative_strategy: str = ""


# ══════════════════════════════════════════════
# ROOT MODEL — The single fixed output contract
# ══════════════════════════════════════════════

class AdIntelligenceOutput(BaseModel):
    """
    Universal Ad Intelligence Schema v1.0.0

    This is the ONLY output shape. Every field always present.
    Null/empty for missing data. Never omit, never add extra fields.

    Works for: TV, radio, YouTube, print, brochure, billboard,
    digital, magazine, newspaper, flyer, social media, podcast — everything.
    """
    meta: Meta = Field(default_factory=Meta, alias="_meta")
    source: Source = Field(default_factory=Source)
    brand: Brand = Field(default_factory=Brand)
    product: Product = Field(default_factory=Product)
    text_content: TextContent = Field(default_factory=TextContent)
    language: Language = Field(default_factory=Language)
    audio: Audio = Field(default_factory=Audio)
    visual_analysis: VisualAnalysis = Field(default_factory=VisualAnalysis)
    video_analysis: VideoAnalysis = Field(default_factory=VideoAnalysis)
    classification: Classification = Field(default_factory=Classification)
    compliance_and_legal: ComplianceAndLegal = Field(default_factory=ComplianceAndLegal)
    engagement_elements: EngagementElements = Field(default_factory=EngagementElements)
    ad_description: AdDescription = Field(default_factory=AdDescription)

    model_config = {"populate_by_name": True}

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string with _meta alias."""
        return self.model_dump_json(indent=indent, by_alias=True)

    def to_dict(self) -> dict:
        """Serialize to dict with _meta alias."""
        return self.model_dump(by_alias=True)


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def get_empty_schema() -> AdIntelligenceOutput:
    """Returns a fully populated schema with all default/empty values."""
    return AdIntelligenceOutput()


def get_schema_json_template() -> str:
    """Returns the JSON template string — used in LLM prompts."""
    return get_empty_schema().to_json(indent=2)


def validate_output(data: dict) -> tuple[bool, AdIntelligenceOutput | None, str]:
    """
    Validate a dict against the schema.
    Pre-cleans common LLM mistakes before validation.
    Returns (is_valid, parsed_model_or_None, error_message).
    """
    try:
        # Pre-clean: fix scenes returned as strings
        if "video_analysis" in data and "scenes" in data["video_analysis"]:
            cleaned_scenes = []
            for s in data["video_analysis"]["scenes"]:
                if isinstance(s, str):
                    cleaned_scenes.append({"description": s, "timestamp_start": "", "timestamp_end": ""})
                elif isinstance(s, dict):
                    cleaned_scenes.append(s)
                else:
                    cleaned_scenes.append({"description": str(s), "timestamp_start": "", "timestamp_end": ""})
            data["video_analysis"]["scenes"] = cleaned_scenes

        # Pre-clean: fix people_description returned as list
        if "visual_analysis" in data:
            va = data["visual_analysis"]
            if isinstance(va.get("people_description"), list):
                va["people_description"] = ", ".join(str(p) for p in va["people_description"])
            if isinstance(va.get("objects_detected"), str):
                va["objects_detected"] = [va["objects_detected"]]

        # Pre-clean: fix any list fields that got strings
        for section_name in ["text_content", "brand", "classification", "engagement_elements"]:
            if section_name in data:
                section = data[section_name]
                for key, val in section.items():
                    if isinstance(val, str) and key in [
                        "hashtags", "all_raw_text", "brand_colors_hex", "brand_social_handles",
                        "product_features", "themes", "urgency_elements", "transition_styles",
                        "sound_effects", "secondary_languages", "dominant_colors_hex", "objects_detected"
                    ]:
                        section[key] = [v.strip() for v in val.split(",") if v.strip()]

        model = AdIntelligenceOutput.model_validate(data)
        return True, model, ""
    except Exception as e:
        return False, None, str(e)


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("Ad Intelligence Schema — Quick Test")
    print("=" * 50)

    # Test 1: Create empty schema
    empty = get_empty_schema()
    print(f"\n✓ Empty schema created successfully")
    print(f"  Sections: {len(AdIntelligenceOutput.model_fields)}")

    # Test 2: Serialize to JSON
    json_str = empty.to_json()
    print(f"  JSON size: {len(json_str)} chars")

    # Test 3: Serialize to dict
    d = empty.to_dict()
    assert "_meta" in d, "Alias _meta not working!"
    print(f"  Dict keys: {list(d.keys())}")

    # Test 4: Validate good data
    ok, model, err = validate_output(d)
    assert ok, f"Validation failed on empty schema: {err}"
    print(f"  Validation: PASS")

    # Test 5: Validate bad data
    ok, model, err = validate_output({"_meta": {"confidence_score": "not_a_number"}})
    print(f"  Bad data validation: {'correctly rejected' if not ok else 'ERROR — should have failed!'}")

    # Test 6: Print the template (what we send to LLM)
    template = get_schema_json_template()
    print(f"\n✓ JSON Template for LLM prompts ({len(template)} chars):")
    # Print first 500 chars
    print(template[:500] + "\n  ... (truncated)")

    print(f"\n{'=' * 50}")
    print("All tests passed! Schema is ready.")
    print(f"{'=' * 50}")