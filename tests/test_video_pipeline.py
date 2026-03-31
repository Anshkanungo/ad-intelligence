"""
Video Pipeline v3 — Fast: CLIP + Groq (no local VLM)

Flow:
  Phase 0: Whisper transcript + CLIP preview encoding (~15s)
  Phase 1: Smart 4-frame selection via CLIP embeddings (~25s)
  Phase 2: 1 preview + 4 frames + transcript → Groq VLM → JSON (~5s)

Total: ~45 seconds (down from 413s in v1, 129s in v2)

Groq limit: 5 images max per request = 1 preview + 4 video frames

Run: python test_video_pipeline.py
"""

import io
import json
import time
import base64
import torch
from pathlib import Path
from PIL import Image
from groq import Groq

from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)

VIDEO_PATH = Path("tests/sample_ads/test.mp4")
PREVIEW_PATH = Path("tests/sample_ads/preview.jpg")


def load_preview_images(preview_paths: list[Path]) -> list[Image.Image]:
    """Load product preview images."""
    previews = []
    for p in preview_paths:
        if p.exists():
            img = Image.open(p).convert("RGB")
            previews.append(img)
            print(f"  Loaded preview: {p.name} ({img.size[0]}x{img.size[1]})")
        else:
            print(f"  Preview not found: {p}")
    return previews


def phase0_prebriefing(video_path: str) -> dict:
    """
    Phase 0: Transcribe audio only. CLIP is handled in Phase 1.
    ~10s total.
    """
    print("\n" + "=" * 60)
    print("PHASE 0: PRE-BRIEFING (Whisper)")
    print("=" * 60)

    t0 = time.perf_counter()

    # ── Transcribe audio ──
    from pipeline.modules.audio_extraction import AudioExtractionModule
    audio_mod = AudioExtractionModule()
    audio_result = audio_mod.extract(video_path)

    transcript = ""
    whisper_result = None

    if audio_result and audio_result.get("has_audio"):
        print(f"  Audio extracted: {audio_result['audio_path']}")

        from pipeline.modules.transcription import TranscriptionModule
        whisper_mod = TranscriptionModule()
        whisper_result = whisper_mod.transcribe(audio_result["audio_path"])

        if whisper_result and whisper_result.get("transcript"):
            transcript = whisper_result["transcript"]
            print(f"  Transcript ({len(transcript)} chars):")
            print(f"    {transcript[:300]}...")

            if whisper_result.get("segments"):
                print(f"  Segments ({len(whisper_result['segments'])}):")
                for seg in whisper_result["segments"][:10]:
                    print(f"    [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
        else:
            print("  No speech detected")

        audio_mod.cleanup(audio_result["audio_path"])
    else:
        print("  No audio track found")

    elapsed = time.perf_counter() - t0
    print(f"\n  Phase 0 complete: {elapsed:.1f}s")

    return {
        "transcript": transcript,
        "whisper_result": whisper_result,
    }


def phase1_frame_selection(video_path: str, preview_images: list[Image.Image]) -> list[dict]:
    """
    Phase 1: Smart CLIP frame selection — picks exactly 4 frames.
    CLIP loads once here, encodes preview + all frames, selects 4, unloads.
    ~20s total.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: SMART FRAME SELECTION (4 frames)")
    print("=" * 60)

    t0 = time.perf_counter()

    from pipeline.modules.frame_extraction import FrameExtractionModule
    frame_mod = FrameExtractionModule()

    # Pass raw preview images — frame_mod encodes them with its own CLIP
    result = frame_mod.extract(video_path, preview_images=preview_images or None)

    if not result or not result.get("frames"):
        print("  Frame extraction failed!")
        return []

    frames = result["frames"]
    method = result.get("method", "unknown")
    print(f"\n  {method} selected {len(frames)} frames:")
    for i, f in enumerate(frames):
        bucket = f.get("bucket", "?")
        psim = f.get("preview_similarity", 0)
        ts = f.get("timestamp_sec", "?")
        print(f"    Frame {i+1} (t={ts}s) [{bucket}]: preview_sim={psim:.3f}")

    frame_mod.unload()
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    print(f"\n  Phase 1 complete: {elapsed:.1f}s, {len(frames)} frames")

    return frames


def phase2_groq_reasoning(
    frames: list[dict],
    briefing: dict,
    preview_images: list[Image.Image],
) -> str:
    """
    Phase 2: Send 1 preview + 4 frames + transcript → Groq VLM → JSON.
    Single API call, ~5s.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: GROQ VLM REASONING")
    print("=" * 60)

    t0 = time.perf_counter()

    transcript = briefing.get("transcript", "")
    segments = []
    if briefing.get("whisper_result") and briefing["whisper_result"].get("segments"):
        segments = briefing["whisper_result"]["segments"]

    # Build frame info text (what CLIP tells us about each frame)
    frame_descriptions = []
    for i, f in enumerate(frames):
        ts = f.get("timestamp_sec", 0)
        bucket = f.get("bucket", "unknown")
        psim = f.get("preview_similarity", 0)

        # Find transcript near this timestamp
        nearby_text = ""
        for seg in segments:
            if abs(seg["start"] - ts) < 3.0:
                nearby_text += seg["text"] + " "

        desc = f"Frame {i+1} (t={ts}s, type={bucket}, product_match={psim:.2f})"
        if nearby_text.strip():
            desc += f" — Audio at this moment: \"{nearby_text.strip()[:150]}\""
        frame_descriptions.append(desc)

    frame_context = "\n".join(frame_descriptions)

    # Build context for Groq
    context = f"""=== AD ANALYSIS SIGNALS ===
Media Type: VIDEO

--- FRAME SELECTION INFO ---
{frame_context}

--- AUDIO TRANSCRIPTION ---
{transcript if transcript else "No speech detected."}

--- TIMESTAMPS ---
{chr(10).join(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}" for seg in segments[:25]) if segments else "No segments."}

=== END OF SIGNALS ==="""

    # Build Groq message
    from schema.ad_schema import get_schema_json_template
    from pipeline.reasoning.prompt_builder import SYSTEM_PROMPT

    schema_template = get_schema_json_template()
    system_prompt = SYSTEM_PROMPT + "\n" + schema_template

    def encode_image(pil_img: Image.Image) -> str:
        if max(pil_img.size) > 1024:
            pil_img = pil_img.copy()
            pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Build image descriptions for the prompt
    image_list = []
    if preview_images:
        image_list.append(f"Image 1: Product preview (what the advertised product looks like)")
    for i in range(len(frames)):
        bucket = frames[i].get("bucket", "")
        ts = frames[i].get("timestamp_sec", "?")
        img_num = (len(preview_images) if preview_images else 0) + i + 1
        image_list.append(f"Image {img_num}: Video frame at t={ts}s ({bucket} shot)")

    image_guide = "\n".join(image_list)

    user_content = [
        {"type": "text", "text": f"""Analyze this video advertisement and return the complete JSON.

The attached images are:
{image_guide}

{context}

INSTRUCTIONS:
1. The preview image shows the ACTUAL PRODUCT being advertised — use it to identify the product in video frames
2. Examine ALL video frames for text, brand logos, taglines, people, settings, and story
3. Use the audio transcript to understand the ad's narrative and emotional tone
4. The ad might be about a SERVICE (not just a physical product) — look at context clues
5. Fill the video_analysis.scenes field with descriptions of what happens in each frame
6. Infer industry, target audience, tone, and themes from the complete picture

Return ONLY the complete JSON object following the exact schema."""},
    ]

    # Attach images: preview first, then video frames
    if preview_images:
        b64 = encode_image(preview_images[0])
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
        print(f"  Attached 1 preview image")

    for i, f in enumerate(frames):
        pil_img = Image.fromarray(f["image"][:, :, ::-1])  # BGR → RGB
        b64 = encode_image(pil_img)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    total_images = (1 if preview_images else 0) + len(frames)
    print(f"  Attached {len(frames)} video frames")
    print(f"  Total images to Groq: {total_images} (max 5)")

    if total_images > 5:
        logger.warning(f"Groq image limit is 5, sending {total_images} — may fail!")

    # Call Groq
    print(f"\n  Calling Groq VLM ({config.groq_vision_model})...")
    client = Groq(api_key=config.groq_api_key)

    response = client.chat.completions.create(
        model=config.groq_vision_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    raw_json = response.choices[0].message.content
    print(f"  Groq response: {len(raw_json)} chars")

    elapsed = time.perf_counter() - t0
    print(f"\n  Phase 2 complete: {elapsed:.1f}s")

    return raw_json


def validate_and_display(raw_json: str):
    """Validate JSON and display key results."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse error: {e}")
        print(f"  Raw: {raw_json[:500]}")
        return

    from schema.ad_schema import validate_output

    is_valid, model, err = validate_output(data)
    if is_valid and model:
        print("  ✓ Schema validation PASSED\n")
        print(f"  Brand:       {model.brand.company_name}")
        print(f"  Product:     {model.product.product_name}")
        print(f"  Category:    {model.product.product_category}")
        print(f"  Headline:    {model.text_content.headline}")
        print(f"  Tagline:     {model.text_content.tagline}")
        print(f"  CTA:         {model.text_content.call_to_action}")
        print(f"  Industry:    {model.classification.industry}")
        print(f"  Objective:   {model.classification.ad_objective}")
        print(f"  Audience:    {model.classification.target_audience}")
        print(f"  Tone:        {model.classification.tone}")
        print(f"  Themes:      {model.classification.themes}")
        print(f"  Summary:     {model.ad_description.short_summary}")

        if model.video_analysis and model.video_analysis.scenes:
            print(f"\n  Scenes ({len(model.video_analysis.scenes)}):")
            for s in model.video_analysis.scenes[:5]:
                print(f"    - {s.description[:100] if s.description else '(no desc)'}...")

        output_path = Path("tests/sample_ads/test_video_output.json")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"\n  Full JSON saved to: {output_path}")
    else:
        print(f"  ✗ Validation failed: {err}")
        output_path = Path("tests/sample_ads/test_video_raw.json")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  Raw JSON saved to: {output_path}")


def main():
    print("=" * 60)
    print("VIDEO PIPELINE v3 — FAST (CLIP → Groq, no local VLM)")
    print("=" * 60)

    if not VIDEO_PATH.exists():
        print(f"  ✗ Video not found: {VIDEO_PATH}")
        return

    print(f"  Video:   {VIDEO_PATH}")
    print(f"  Preview: {PREVIEW_PATH if PREVIEW_PATH.exists() else 'None'}")
    print(f"  GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    print(f"  Groq:    {'✓' if config.has_groq else '✗ MISSING'}")

    if not config.has_groq:
        print("\n  Set GROQ_API_KEY in .env to run pipeline")
        return

    if not PREVIEW_PATH.exists():
        print("\n  ⚠ No preview image — frame selection will use scene-change only")

    total_t0 = time.perf_counter()

    # Load preview
    preview_images = []
    if PREVIEW_PATH.exists():
        preview_images = load_preview_images([PREVIEW_PATH])

    # Phase 0: Whisper only (no CLIP here)
    briefing = phase0_prebriefing(str(VIDEO_PATH))

    # Phase 1: Smart 4-frame selection (CLIP loads once here)
    frames = phase1_frame_selection(str(VIDEO_PATH), preview_images)
    if not frames:
        print("\n  ✗ No frames extracted — aborting")
        return

    # Phase 2: Groq reasoning (single API call)
    raw_json = phase2_groq_reasoning(frames, briefing, preview_images)

    # Validate and display
    validate_and_display(raw_json)

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n{'=' * 60}")
    print(f"TOTAL PIPELINE TIME: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()