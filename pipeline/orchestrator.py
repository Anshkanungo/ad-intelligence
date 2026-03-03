"""
Ad Intelligence Pipeline — Orchestrator

The main controller that runs the full pipeline:
  Input → Media Detection → Module Execution → Aggregation → LLM → Validation → Output

Usage:
    from pipeline.orchestrator import AdIntelligencePipeline
    pipeline = AdIntelligencePipeline()
    result = pipeline.run(uploaded_file)
"""

import json
import time
from datetime import datetime, timezone
from PIL import Image
import cv2
import numpy as np

from schema.ad_schema import AdIntelligenceOutput, validate_output
from utils.config import config
from utils.media_handler import MediaHandler, MediaInfo
from utils.logger import get_logger, PipelineProgress

from pipeline.modules.ocr_module import OCRModule
from pipeline.modules.object_detection import ObjectDetectionModule
from pipeline.modules.color_analysis import ColorAnalysisModule
from pipeline.modules.scene_description import SceneDescriptionModule
from pipeline.modules.language_detection import LanguageDetectionModule
from pipeline.modules.transcription import TranscriptionModule
from pipeline.modules.frame_extraction import FrameExtractionModule
from pipeline.modules.audio_extraction import AudioExtractionModule

from pipeline.aggregator import SignalAggregator
from pipeline.reasoning.prompt_builder import PromptBuilder
from pipeline.reasoning.llm_engine import LLMEngine

logger = get_logger(__name__)


class AdIntelligencePipeline:
    """Full end-to-end ad intelligence extraction pipeline."""

    def __init__(self):
        # Media handling
        self.media_handler = MediaHandler()

        # Extraction modules (lazy-loaded internally)
        self.ocr = OCRModule()
        self.object_detector = ObjectDetectionModule()
        self.color_analyzer = ColorAnalysisModule()
        self.scene_describer = SceneDescriptionModule()
        self.language_detector = LanguageDetectionModule()
        self.transcriber = TranscriptionModule()
        self.frame_extractor = FrameExtractionModule()
        self.audio_extractor = AudioExtractionModule()

        # Reasoning
        self.aggregator = SignalAggregator()
        self.prompt_builder = PromptBuilder()
        self.llm_engine = LLMEngine()

    def run(self, uploaded_file, progress_callback=None) -> AdIntelligenceOutput:
        """
        Run the full pipeline on an uploaded file.

        Args:
            uploaded_file: Streamlit UploadedFile or file-like object
            progress_callback: Optional callable(step, total, message)
                for UI progress updates

        Returns:
            AdIntelligenceOutput — the complete fixed-schema result
        """
        start_time = time.perf_counter()
        errors = []
        modules_used = []

        def update_progress(step, total, msg):
            if progress_callback:
                progress_callback(step, total, msg)
            logger.info(f"[{step}/{total}] {msg}")

        # ─── Step 1: Process Upload ───
        update_progress(1, 8, "Processing upload...")
        media_info = self.media_handler.process_upload(uploaded_file)

        if not media_info.is_valid:
            return self._error_result(
                media_info, start_time, f"Invalid file: {media_info.error}"
            )

        try:
            # Route by media type
            if media_info.media_type == "image":
                return self._run_image_pipeline(
                    media_info, start_time, errors, modules_used, update_progress
                )
            elif media_info.media_type == "video":
                return self._run_video_pipeline(
                    media_info, start_time, errors, modules_used, update_progress
                )
            elif media_info.media_type == "audio":
                return self._run_audio_pipeline(
                    media_info, start_time, errors, modules_used, update_progress
                )
            else:
                return self._error_result(
                    media_info, start_time, f"Unsupported type: {media_info.media_type}"
                )
        finally:
            # Always cleanup temp files
            self.media_handler.cleanup(media_info)

    # ══════════════════════════════════════════
    # IMAGE PIPELINE
    # ══════════════════════════════════════════

    def _run_image_pipeline(self, media_info, start_time, errors, modules_used, progress):
        """Run all image analysis modules → aggregate → VLM with image → validate."""

        # Step 2: OCR
        progress(2, 8, "Extracting text (OCR)...")
        ocr_result = self._safe_run("ocr", lambda: self.ocr.extract(media_info.image_array), errors)
        if ocr_result: modules_used.append("ocr")

        # Step 3: Object Detection
        progress(3, 8, "Detecting objects (YOLO)...")
        yolo_result = self._safe_run("yolo", lambda: self.object_detector.detect(media_info.image_array), errors)
        if yolo_result: modules_used.append("yolo")

        # Step 4: Color Analysis
        progress(4, 8, "Analyzing colors...")
        color_result = self._safe_run("color", lambda: self.color_analyzer.analyze(media_info.image_array), errors)
        if color_result: modules_used.append("color")

        # Step 5: Scene Description
        progress(5, 8, "Describing scene (BLIP)...")
        scene_result = self._safe_run("blip", lambda: self.scene_describer.describe(media_info.pil_image), errors)
        if scene_result: modules_used.append("blip")

        # Step 6: Language Detection (from OCR text)
        progress(6, 8, "Detecting language...")
        lang_result = None
        if ocr_result and ocr_result.get("full_text"):
            lang_result = self._safe_run(
                "langdetect",
                lambda: self.language_detector.detect_language(ocr_result["full_text"]),
                errors
            )
            if lang_result: modules_used.append("langdetect")

        # Step 7: LLM Reasoning (with image for VLM)
        progress(7, 8, "AI reasoning over signals...")
        context = self.aggregator.merge(
            media_type="image",
            ocr_result=ocr_result,
            yolo_result=yolo_result,
            color_result=color_result,
            scene_result=scene_result,
            language_result=lang_result,
        )

        output = self._run_llm(context, errors, image=media_info.pil_image)
        modules_used.append("llm")

        # Step 8: Finalize
        progress(8, 8, "Finalizing results...")
        return self._finalize(output, media_info, start_time, modules_used, errors)

    # ══════════════════════════════════════════
    # VIDEO PIPELINE (v2 — VLM with frames)
    # ══════════════════════════════════════════

    def _run_video_pipeline(self, media_info, start_time, errors, modules_used, progress):
        """
        Video pipeline v2:
        Extract frames + audio → OCR all frames → Whisper → Send frames to VLM → JSON
        """

        # Step 2: Extract key frames
        progress(2, 8, "Extracting key frames...")
        frame_result = self._safe_run(
            "frames", lambda: self.frame_extractor.extract(media_info.temp_path), errors
        )
        if frame_result: modules_used.append("frames")

        # Step 3: Extract + transcribe audio
        progress(3, 8, "Extracting and transcribing audio...")
        transcription_result = None
        audio_result = self._safe_run(
            "audio_extract", lambda: self.audio_extractor.extract(media_info.temp_path), errors
        )

        if audio_result and audio_result.get("has_audio"):
            modules_used.append("audio_extract")
            transcription_result = self._safe_run(
                "whisper",
                lambda: self.transcriber.transcribe(audio_result["audio_path"]),
                errors
            )
            if transcription_result: modules_used.append("whisper")
            self.audio_extractor.cleanup(audio_result.get("audio_path", ""))

        # Step 4: Smart OCR — only on the most text-dense frame (not all frames)
        # The VLM will read text from all frames directly, so per-frame OCR is redundant
        progress(4, 8, "Reading text from key frames (OCR)...")
        frame_pil_images = []
        merged_ocr = {
            "raw_texts": [], "text_with_positions": [],
            "full_text": "", "avg_confidence": 0.0, "text_count": 0,
        }

        if frame_result and frame_result.get("frames"):
            frames = frame_result["frames"]

            # Collect PIL images for VLM
            for f in frames:
                pil_img = Image.fromarray(f["image"][:, :, ::-1])  # BGR→RGB
                frame_pil_images.append(pil_img)

            # Find the most text-dense frame using edge detection (fast)
            edge_scores = []
            for f in frames:
                gray = cv2.cvtColor(f["image"], cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_scores.append(int(np.sum(edges > 0)))

            # OCR only the top 3 most text-dense frames (fast, not all 14)
            top_indices = sorted(range(len(edge_scores)), key=lambda i: edge_scores[i], reverse=True)[:3]

            all_ocr_texts = []
            for idx in top_indices:
                # Resize to 640px width for faster OCR
                frame_img = frames[idx]["image"]
                h, w = frame_img.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame_img = cv2.resize(frame_img, (640, int(h * scale)))

                ocr_res = self._safe_run(
                    "ocr", lambda fi=frame_img: self.ocr.extract(fi), errors
                )
                if ocr_res and ocr_res.get("raw_texts"):
                    all_ocr_texts.extend(ocr_res["raw_texts"])
                    if "ocr" not in modules_used:
                        modules_used.append("ocr")

            # Deduplicate
            seen = set()
            unique_ocr = []
            for t in all_ocr_texts:
                t_lower = t.strip().lower()
                if t_lower and t_lower not in seen:
                    seen.add(t_lower)
                    unique_ocr.append(t.strip())

            merged_ocr = {
                "raw_texts": unique_ocr,
                "text_with_positions": [{"text": t, "bbox": [], "confidence": 0.0} for t in unique_ocr],
                "full_text": " | ".join(unique_ocr),
                "avg_confidence": 0.0,
                "text_count": len(unique_ocr),
            }

        # Step 5: Color analysis on first frame
        progress(5, 8, "Analyzing colors...")
        primary_color = None
        if frame_result and frame_result.get("frames"):
            first_frame = frame_result["frames"][0]["image"]
            primary_color = self._safe_run(
                "color", lambda: self.color_analyzer.analyze(first_frame), errors
            )
            if primary_color: modules_used.append("color")

        # Step 6: Language detection from OCR or transcript
        progress(6, 8, "Detecting language...")
        lang_result = None
        lang_source = ""
        if transcription_result and transcription_result.get("transcript"):
            lang_source = transcription_result["transcript"]
        elif merged_ocr.get("full_text"):
            lang_source = merged_ocr["full_text"]

        if lang_source:
            lang_result = self._safe_run(
                "langdetect",
                lambda: self.language_detector.detect_language(lang_source),
                errors
            )
            if lang_result: modules_used.append("langdetect")

        # Step 7: Send frames + signals to VLM
        progress(7, 8, "AI analyzing video frames...")
        video_info = {
            "duration_seconds": media_info.duration_seconds,
            "resolution": media_info.resolution,
            "fps": media_info.fps,
            "frame_count": len(frame_pil_images),
        }

        context = self.aggregator.merge(
            media_type="video",
            ocr_result=merged_ocr,
            color_result=primary_color,
            language_result=lang_result,
            transcription_result=transcription_result,
            video_info=video_info,
        )

        # Send ALL CLIP-selected frames to VLM (no cap — CLIP already filtered)
        vlm_images = frame_pil_images

        output = self._run_llm(context, errors, images=vlm_images)
        modules_used.append("llm")

        # Step 8: Finalize
        progress(8, 8, "Finalizing results...")
        return self._finalize(output, media_info, start_time, modules_used, errors)

    # ══════════════════════════════════════════
    # AUDIO PIPELINE
    # ══════════════════════════════════════════

    def _run_audio_pipeline(self, media_info, start_time, errors, modules_used, progress):
        """Transcribe audio → language detect → LLM reasoning."""

        # Step 2: Transcribe
        progress(2, 8, "Transcribing audio (Whisper)...")
        transcription_result = self._safe_run(
            "whisper",
            lambda: self.transcriber.transcribe(media_info.temp_path),
            errors
        )
        if transcription_result: modules_used.append("whisper")

        # Step 3: Language detection
        progress(3, 8, "Detecting language...")
        lang_result = None
        if transcription_result and transcription_result.get("transcript"):
            lang_result = self._safe_run(
                "langdetect",
                lambda: self.language_detector.detect_language(transcription_result["transcript"]),
                errors
            )
            if lang_result: modules_used.append("langdetect")

        # Step 4-6: Skip visual modules (audio only)
        progress(4, 8, "Skipping visual analysis (audio only)...")

        # Step 7: LLM Reasoning
        progress(7, 8, "AI reasoning over signals...")
        context = self.aggregator.merge(
            media_type="audio",
            transcription_result=transcription_result,
            language_result=lang_result,
        )

        output = self._run_llm(context, errors)
        modules_used.append("llm")

        # Step 8: Finalize
        progress(8, 8, "Finalizing results...")
        return self._finalize(output, media_info, start_time, modules_used, errors)

    # ══════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════

    def _safe_run(self, name: str, func, errors: list) -> dict | None:
        """Run a module safely, catching exceptions."""
        try:
            return func()
        except Exception as e:
            msg = f"{name} failed: {str(e)}"
            logger.error(msg)
            errors.append(msg)
            return None

    def _run_llm(self, context: str, errors: list, image=None, images=None) -> AdIntelligenceOutput:
        """Run LLM reasoning and validate output. Passes image(s) for VLM mode."""
        system_prompt, user_prompt = self.prompt_builder.build(context)

        for attempt in range(config.llm_max_retries + 1):
            raw_json = self.llm_engine.reason(
                system_prompt, user_prompt, image=image, images=images
            )

            if not raw_json:
                errors.append(f"LLM returned empty response (attempt {attempt + 1})")
                continue

            try:
                data = json.loads(raw_json)
                is_valid, model, err = validate_output(data)

                if is_valid and model:
                    logger.info("LLM output validated successfully")
                    return model
                else:
                    logger.warning(f"Validation failed (attempt {attempt + 1}): {err}")
                    errors.append(f"Validation attempt {attempt + 1}: {err[:200]}")
                    # Add error context to next attempt
                    user_prompt += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {err[:500]}\nPlease fix and try again."

            except json.JSONDecodeError as e:
                errors.append(f"JSON parse error (attempt {attempt + 1}): {str(e)[:200]}")

        # All attempts failed — return empty schema
        logger.error("All LLM attempts failed, returning empty schema")
        return AdIntelligenceOutput()

    def _finalize(
        self,
        output: AdIntelligenceOutput,
        media_info: MediaInfo,
        start_time: float,
        modules_used: list,
        errors: list,
    ) -> AdIntelligenceOutput:
        """Fill in _meta fields and return final output."""
        elapsed = round(time.perf_counter() - start_time, 2)

        output.meta.schema_version = config.schema_version
        output.meta.pipeline_version = config.pipeline_version
        output.meta.processed_at = datetime.now(timezone.utc).isoformat()
        output.meta.processing_time_sec = elapsed
        output.meta.input_media_type = media_info.media_type
        output.meta.input_filename = media_info.original_filename
        output.meta.input_resolution = media_info.resolution
        output.meta.modules_used = list(set(modules_used))
        output.meta.errors = errors

        # Confidence: rough estimate based on how many modules succeeded
        expected = {"image": 5, "video": 7, "audio": 2}
        expected_count = expected.get(media_info.media_type, 5)
        success_count = len([m for m in modules_used if m != "llm"])
        output.meta.confidence_score = round(
            min(1.0, success_count / expected_count), 2
        )

        logger.info(
            f"Pipeline complete: {media_info.media_type} | "
            f"{elapsed}s | confidence={output.meta.confidence_score} | "
            f"modules={output.meta.modules_used}"
        )

        return output

    def _error_result(
        self, media_info: MediaInfo, start_time: float, error: str
    ) -> AdIntelligenceOutput:
        """Return an empty schema with error info."""
        output = AdIntelligenceOutput()
        output.meta.processed_at = datetime.now(timezone.utc).isoformat()
        output.meta.processing_time_sec = round(time.perf_counter() - start_time, 2)
        output.meta.input_filename = media_info.original_filename
        output.meta.input_media_type = media_info.media_type
        output.meta.errors = [error]
        output.meta.confidence_score = 0.0
        return output


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path

    print("=" * 50)
    print("Pipeline Orchestrator — Quick Test")
    print("=" * 50)

    pipeline = AdIntelligencePipeline()
    print("✓ Pipeline initialized with all modules")

    # Test with sample image
    test_image = Path("tests/sample_ads/test.jpg")
    if test_image.exists():
        print(f"\nRunning full pipeline on: {test_image}")

        class FakeUpload:
            name = str(test_image)
            def read(self):
                return test_image.read_bytes()
            def seek(self, n):
                pass

        result = pipeline.run(FakeUpload())

        print(f"\n  Type: {result.meta.input_media_type}")
        print(f"  Time: {result.meta.processing_time_sec}s")
        print(f"  Confidence: {result.meta.confidence_score}")
        print(f"  Modules: {result.meta.modules_used}")
        print(f"  Errors: {result.meta.errors}")
        print(f"  Brand: {result.brand.company_name}")
        print(f"  Product: {result.product.product_name}")
        print(f"  Headline: {result.text_content.headline}")
        print(f"  Summary: {result.ad_description.short_summary[:200]}")

        # Save JSON
        output_path = Path("tests/sample_ads/test_output.json")
        output_path.write_text(result.to_json())
        print(f"\n  Full JSON saved to: {output_path}")
    else:
        print("\n⚠ No test image at tests/sample_ads/test.jpg")

    print(f"\n{'=' * 50}")
    print("Orchestrator ready!")
    print(f"{'=' * 50}")