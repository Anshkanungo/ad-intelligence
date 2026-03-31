"""
Ad Intelligence Pipeline — Orchestrator (Local GPU Only)

VLM-first design: Qwen2.5-VL 3B owns the GPU exclusively.
Lightweight CPU modules provide supplementary signals.
No BLIP, no EasyOCR — VLM handles all vision + text reading.

Image: Color(CPU) → YOLO(CPU) → VLM(GPU) → JSON
Video: CLIP frames → Whisper → Color(CPU) → Language(CPU) → VLM(GPU) → JSON
Audio: Whisper → Language → VLM → JSON
"""

import json
import time
from datetime import datetime, timezone
from PIL import Image

from schema.ad_schema import AdIntelligenceOutput, validate_output
from utils.config import config
from utils.media_handler import MediaHandler, MediaInfo
from utils.logger import get_logger

logger = get_logger(__name__)


class AdIntelligencePipeline:
    """Ad intelligence extraction pipeline — VLM-first, local GPU."""

    def __init__(self):
        self.media_handler = MediaHandler()
        self._modules_loaded = {}

        from pipeline.aggregator import SignalAggregator
        from pipeline.reasoning.prompt_builder import PromptBuilder
        from pipeline.reasoning.llm_engine import LLMEngine

        self.aggregator = SignalAggregator()
        self.prompt_builder = PromptBuilder()
        self.llm_engine = LLMEngine()

        # Pre-load VLM so it's warm for first request
        logger.info("Pre-loading local VLM...")
        self.llm_engine._get_vlm()

    def _get_module(self, name: str):
        """Lazy-load CPU-only extraction modules."""
        if name in self._modules_loaded:
            return self._modules_loaded[name]

        try:
            if name == "yolo":
                from pipeline.modules.object_detection import ObjectDetectionModule
                self._modules_loaded[name] = ObjectDetectionModule()
            elif name == "color":
                from pipeline.modules.color_analysis import ColorAnalysisModule
                self._modules_loaded[name] = ColorAnalysisModule()
            elif name == "langdetect":
                from pipeline.modules.language_detection import LanguageDetectionModule
                self._modules_loaded[name] = LanguageDetectionModule()
            elif name == "whisper":
                from pipeline.modules.transcription import TranscriptionModule
                self._modules_loaded[name] = TranscriptionModule()
            elif name == "frames":
                from pipeline.modules.frame_extraction import FrameExtractionModule
                self._modules_loaded[name] = FrameExtractionModule()
            elif name == "audio_extract":
                from pipeline.modules.audio_extraction import AudioExtractionModule
                self._modules_loaded[name] = AudioExtractionModule()
        except Exception as e:
            logger.warning(f"Module {name} not available: {e}")
            return None

        return self._modules_loaded.get(name)

    def run(self, uploaded_file, progress_callback=None) -> AdIntelligenceOutput:
        """Run the full pipeline on an uploaded file."""
        start_time = time.perf_counter()
        errors = []
        modules_used = []

        def progress(step, total, msg):
            if progress_callback:
                progress_callback(step, total, msg)
            logger.info(f"[{step}/{total}] {msg}")

        # Step 1: Process upload
        progress(1, 6, "Processing upload...")
        media_info = self.media_handler.process_upload(uploaded_file)

        if not media_info.is_valid:
            return self._error_result(media_info, start_time, f"Invalid file: {media_info.error}")

        logger.info(f"Media type: {media_info.media_type}")

        try:
            if media_info.media_type == "image":
                return self._run_image(media_info, start_time, errors, modules_used, progress)
            elif media_info.media_type == "video":
                return self._run_video(media_info, start_time, errors, modules_used, progress)
            elif media_info.media_type == "audio":
                return self._run_audio(media_info, start_time, errors, modules_used, progress)
            else:
                return self._error_result(media_info, start_time, f"Unsupported: {media_info.media_type}")
        finally:
            self.media_handler.cleanup(media_info)

    # ══════════════════════════════════════════
    # IMAGE PIPELINE
    # ══════════════════════════════════════════

    def _run_image(self, media_info, start_time, errors, modules_used, progress):
        """Image: Color(CPU) → YOLO(CPU) → VLM(GPU) → JSON"""

        # Step 2: CPU-only modules (no VRAM needed)
        progress(2, 6, "Analyzing colors + detecting objects...")

        color_mod = self._get_module("color")
        color_result = self._safe_run("color", lambda: color_mod.analyze(media_info.image_array), errors) if color_mod else None
        if color_result:
            modules_used.append("color")

        yolo_mod = self._get_module("yolo")
        yolo_result = self._safe_run("yolo", lambda: yolo_mod.detect(media_info.image_array), errors) if yolo_mod else None
        if yolo_result:
            modules_used.append("yolo")

        # Step 3: VLM describes the image (replaces OCR + BLIP)
        progress(3, 6, "VLM reading image (text, scene, brand)...")
        vlm_description = self.llm_engine.describe_frame(media_info.pil_image)
        if vlm_description:
            modules_used.append("vlm_describe")
            logger.info(f"VLM description: {vlm_description[:150]}...")

        # Step 4: Language detection from VLM-extracted text
        progress(4, 6, "Detecting language...")
        lang_result = None
        if vlm_description:
            # Extract the TEXT line from structured VLM output
            text_line = ""
            for line in vlm_description.split("\n"):
                if line.startswith("TEXT:"):
                    text_line = line[5:].strip()
                    break
            if text_line:
                lang_mod = self._get_module("langdetect")
                lang_result = self._safe_run("langdetect", lambda: lang_mod.detect_language(text_line), errors) if lang_mod else None
                if lang_result:
                    modules_used.append("langdetect")

        # Step 5: VLM generates final JSON
        progress(5, 6, "AI generating structured JSON...")
        context = self.aggregator.merge(
            media_type="image",
            yolo_result=yolo_result,
            color_result=color_result,
            language_result=lang_result,
            vlm_description=vlm_description,
        )

        output = self._run_llm(context, errors, image=media_info.pil_image)
        modules_used.append("llm")

        progress(6, 6, "Finalizing...")
        return self._finalize(output, media_info, start_time, modules_used, errors)

    # ══════════════════════════════════════════
    # VIDEO PIPELINE
    # ══════════════════════════════════════════

    def _run_video(self, media_info, start_time, errors, modules_used, progress):
        """Video: CLIP frames → Whisper → Color → VLM describe frames → VLM JSON"""

        # Step 2: Extract key frames + audio
        # Unload VLM first to free VRAM for CLIP + Whisper
        progress(2, 6, "Extracting key frames + audio...")
        self.llm_engine.unload()

        frame_mod = self._get_module("frames")
        frame_result = self._safe_run("frames", lambda: frame_mod.extract(media_info.temp_path), errors) if frame_mod else None
        if frame_result:
            modules_used.append("frames")

        transcription_result = None
        audio_mod = self._get_module("audio_extract")
        audio_result = self._safe_run("audio_extract", lambda: audio_mod.extract(media_info.temp_path), errors) if audio_mod else None

        if audio_result and audio_result.get("has_audio"):
            modules_used.append("audio_extract")
            whisper_mod = self._get_module("whisper")
            if whisper_mod:
                transcription_result = self._safe_run("whisper", lambda: whisper_mod.transcribe(audio_result["audio_path"]), errors)
                if transcription_result:
                    modules_used.append("whisper")
            if audio_result.get("audio_path"):
                audio_mod.cleanup(audio_result["audio_path"])

        # Free CLIP + Whisper VRAM, VLM will reload and stay loaded for rest of pipeline
        self._unload_gpu_modules()

        # Collect PIL frames
        frame_pil_images = []
        if frame_result and frame_result.get("frames"):
            for f in frame_result["frames"]:
                pil_img = Image.fromarray(f["image"][:, :, ::-1])  # BGR → RGB
                frame_pil_images.append(pil_img)

        # Step 3: CPU modules
        progress(3, 6, "Analyzing colors + language...")
        color_result = None
        if frame_result and frame_result.get("frames"):
            color_mod = self._get_module("color")
            if color_mod:
                color_result = self._safe_run("color", lambda: color_mod.analyze(frame_result["frames"][0]["image"]), errors)
                if color_result:
                    modules_used.append("color")

        lang_result = None
        lang_source = transcription_result.get("transcript", "") if transcription_result else ""
        if lang_source:
            lang_mod = self._get_module("langdetect")
            if lang_mod:
                lang_result = self._safe_run("langdetect", lambda: lang_mod.detect_language(lang_source), errors)
                if lang_result:
                    modules_used.append("langdetect")

        # Step 4: VLM describes each frame
        progress(4, 6, f"VLM analyzing {len(frame_pil_images)} frames...")
        frame_descriptions = []
        if frame_pil_images:
            frame_descriptions = self.llm_engine.describe_frames(frame_pil_images)
            if frame_descriptions:
                modules_used.append("vlm_describe")

        # Step 5: VLM generates final JSON
        progress(5, 6, "AI generating structured JSON...")
        video_info = {
            "duration_seconds": media_info.duration_seconds,
            "resolution": media_info.resolution,
            "fps": media_info.fps,
            "frame_count": len(frame_pil_images),
        }

        # Build frame descriptions text for context
        vlm_description = ""
        if frame_descriptions:
            parts = []
            for i, desc in enumerate(frame_descriptions):
                ts = frame_result["frames"][i].get("timestamp_sec", "?") if frame_result else "?"
                parts.append(f"Frame {i+1} (t={ts}s): {desc}")
            vlm_description = "\n\n".join(parts)

        context = self.aggregator.merge(
            media_type="video",
            color_result=color_result,
            language_result=lang_result,
            transcription_result=transcription_result,
            video_info=video_info,
            vlm_description=vlm_description,
        )

        # Send reference frames (first + middle + last) with JSON prompt
        ref_indices = list(dict.fromkeys([0, len(frame_pil_images) // 2, max(0, len(frame_pil_images) - 1)]))
        ref_images = [frame_pil_images[i] for i in ref_indices] if frame_pil_images else None

        output = self._run_llm(context, errors, images=ref_images)
        modules_used.append("llm")

        progress(6, 6, "Finalizing...")
        return self._finalize(output, media_info, start_time, modules_used, errors)

    # ══════════════════════════════════════════
    # AUDIO PIPELINE
    # ══════════════════════════════════════════

    def _run_audio(self, media_info, start_time, errors, modules_used, progress):
        """Audio: Whisper → Language → VLM → JSON"""

        # Unload VLM temporarily for Whisper
        self.llm_engine.unload()

        progress(2, 6, "Transcribing audio (Whisper)...")
        whisper_mod = self._get_module("whisper")
        transcription_result = self._safe_run("whisper", lambda: whisper_mod.transcribe(media_info.temp_path), errors) if whisper_mod else None
        if transcription_result:
            modules_used.append("whisper")

        # Free Whisper, reload VLM
        self._unload_gpu_modules()

        progress(3, 6, "Detecting language...")
        lang_result = None
        if transcription_result and transcription_result.get("transcript"):
            lang_mod = self._get_module("langdetect")
            if lang_mod:
                lang_result = self._safe_run("langdetect", lambda: lang_mod.detect_language(transcription_result["transcript"]), errors)
                if lang_result:
                    modules_used.append("langdetect")

        progress(5, 6, "AI reasoning (local VLM)...")
        context = self.aggregator.merge(
            media_type="audio",
            transcription_result=transcription_result,
            language_result=lang_result,
        )

        output = self._run_llm(context, errors)
        modules_used.append("llm")

        progress(6, 6, "Finalizing...")
        return self._finalize(output, media_info, start_time, modules_used, errors)

    # ══════════════════════════════════════════
    # SHARED HELPERS
    # ══════════════════════════════════════════

    def _safe_run(self, name, func, errors):
        """Run a module safely, catch errors."""
        try:
            return func()
        except Exception as e:
            msg = f"{name} failed: {str(e)}"
            logger.error(msg)
            errors.append(msg)
            return None

    def _unload_gpu_modules(self):
        """Unload any GPU-heavy extraction modules to free VRAM for VLM."""
        import torch

        # Unload CLIP if loaded
        if "frames" in self._modules_loaded:
            try:
                mod = self._modules_loaded["frames"]
                if hasattr(mod, "unload"):
                    mod.unload()
            except Exception:
                pass

        # Unload Whisper if loaded
        if "whisper" in self._modules_loaded:
            try:
                mod = self._modules_loaded["whisper"]
                if hasattr(mod, "unload"):
                    mod.unload()
            except Exception:
                pass

        torch.cuda.empty_cache()
        logger.info("GPU modules unloaded, VRAM freed for VLM")

    def _run_llm(self, context, errors, image=None, images=None):
        """Build prompts, call VLM, validate JSON, retry on failure."""
        system_prompt, user_prompt = self.prompt_builder.build(context)

        for attempt in range(config.llm_max_retries + 1):
            raw_json = self.llm_engine.reason(system_prompt, user_prompt, image=image, images=images)

            if not raw_json:
                errors.append(f"VLM empty response (attempt {attempt + 1})")
                continue

            try:
                data = json.loads(raw_json)
                is_valid, model, err = validate_output(data)
                if is_valid and model:
                    logger.info("VLM output validated successfully")
                    return model
                else:
                    logger.warning(f"Validation failed (attempt {attempt + 1}): {err}")
                    errors.append(f"Validation attempt {attempt + 1}: {err[:200]}")
                    user_prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {err[:500]}\nFix and retry."
            except json.JSONDecodeError as e:
                errors.append(f"JSON parse error (attempt {attempt + 1}): {str(e)[:200]}")

        logger.error("All VLM attempts failed")
        return AdIntelligenceOutput()

    def _finalize(self, output, media_info, start_time, modules_used, errors):
        """Fill metadata and return final output."""
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

        # Confidence: how many modules succeeded vs expected
        expected = {"image": 3, "video": 4, "audio": 2}  # color+yolo+vlm, frames+whisper+color+vlm, whisper+vlm
        expected_count = expected.get(media_info.media_type, 3)
        success_count = len([m for m in modules_used if m not in ("llm", "vlm_describe")])
        output.meta.confidence_score = round(min(1.0, success_count / expected_count), 2)

        logger.info(f"Pipeline complete: {media_info.media_type} | {elapsed}s | modules={modules_used} | confidence={output.meta.confidence_score}")
        return output

    def _error_result(self, media_info, start_time, error):
        """Return empty result with error."""
        output = AdIntelligenceOutput()
        output.meta.processed_at = datetime.now(timezone.utc).isoformat()
        output.meta.processing_time_sec = round(time.perf_counter() - start_time, 2)
        output.meta.input_filename = media_info.original_filename
        output.meta.input_media_type = media_info.media_type
        output.meta.errors = [error]
        output.meta.confidence_score = 0.0
        return output


if __name__ == "__main__":
    from pathlib import Path

    print("=" * 50)
    print("Pipeline Orchestrator (Local GPU) — Quick Test")
    print("=" * 50)

    pipeline = AdIntelligencePipeline()
    print("✓ Pipeline initialized (VLM pre-loaded)")

    test_image = Path("tests/sample_ads/test.jpg")
    if test_image.exists():
        print(f"\nRunning on: {test_image}")

        class FakeUpload:
            name = str(test_image)
            def read(self):
                return test_image.read_bytes()
            def seek(self, n):
                pass

        result = pipeline.run(FakeUpload())
        print(f"\n  Time:       {result.meta.processing_time_sec}s")
        print(f"  Brand:      {result.brand.company_name}")
        print(f"  Product:    {result.product.product_name}")
        print(f"  Headline:   {result.text_content.headline}")
        print(f"  Modules:    {result.meta.modules_used}")
        print(f"  Confidence: {result.meta.confidence_score}")
        if result.meta.errors:
            print(f"  Errors:     {result.meta.errors}")
    else:
        print(f"\n⚠ No test image at {test_image}")

    print(f"\n{'=' * 50}")