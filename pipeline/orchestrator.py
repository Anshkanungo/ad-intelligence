"""
Ad Intelligence Pipeline — Orchestrator (Lightweight for Cloud Deploy)

Stripped-down pipeline for 512MB RAM hosting:
  - Image: Send directly to Groq VLM → JSON
  - Video: CLIP frame selection → Groq multi-frame VLM → JSON
  - Audio: Whisper transcription → Groq text LLM → JSON

No EasyOCR, YOLO, BLIP loaded at startup. VLM handles everything.
"""

import json
import time
import os
from datetime import datetime, timezone
from PIL import Image

from schema.ad_schema import AdIntelligenceOutput, validate_output
from utils.config import config
from utils.media_handler import MediaHandler, MediaInfo
from utils.logger import get_logger

logger = get_logger(__name__)

# Check if we're in lightweight mode (cloud) or full mode (local)
LIGHTWEIGHT_MODE = os.getenv("LIGHTWEIGHT_MODE", "auto")


class AdIntelligencePipeline:
    """Ad intelligence extraction pipeline. Auto-detects lightweight vs full mode."""

    def __init__(self):
        self.media_handler = MediaHandler()
        self._modules_loaded = {}

        # Always import these (lightweight)
        from pipeline.aggregator import SignalAggregator
        from pipeline.reasoning.prompt_builder import PromptBuilder
        from pipeline.reasoning.llm_engine import LLMEngine

        self.aggregator = SignalAggregator()
        self.prompt_builder = PromptBuilder()
        self.llm_engine = LLMEngine()

    def _get_module(self, name: str):
        """Lazy-load modules only when needed."""
        if name in self._modules_loaded:
            return self._modules_loaded[name]

        try:
            if name == "ocr":
                from pipeline.modules.ocr_module import OCRModule
                self._modules_loaded[name] = OCRModule()
            elif name == "yolo":
                from pipeline.modules.object_detection import ObjectDetectionModule
                self._modules_loaded[name] = ObjectDetectionModule()
            elif name == "color":
                from pipeline.modules.color_analysis import ColorAnalysisModule
                self._modules_loaded[name] = ColorAnalysisModule()
            elif name == "blip":
                from pipeline.modules.scene_description import SceneDescriptionModule
                self._modules_loaded[name] = SceneDescriptionModule()
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

    def _is_lightweight(self) -> bool:
        """Check if we should run in lightweight mode."""
        if LIGHTWEIGHT_MODE == "full":
            return False
        if LIGHTWEIGHT_MODE == "light":
            return True
        # Auto-detect: check available RAM
        try:
            import psutil
            ram_mb = psutil.virtual_memory().total / (1024 * 1024)
            return ram_mb < 1500  # Less than 1.5GB = lightweight
        except ImportError:
            # No psutil, check environment hints
            if os.getenv("RENDER"):
                return True
            return False

    def run(self, uploaded_file, progress_callback=None) -> AdIntelligenceOutput:
        """Run the full pipeline on an uploaded file."""
        start_time = time.perf_counter()
        errors = []
        modules_used = []

        def progress(step, total, msg):
            if progress_callback:
                progress_callback(step, total, msg)
            logger.info(f"[{step}/{total}] {msg}")

        # Step 1: Process Upload
        progress(1, 8, "Processing upload...")
        media_info = self.media_handler.process_upload(uploaded_file)

        if not media_info.is_valid:
            return self._error_result(media_info, start_time, f"Invalid file: {media_info.error}")

        lightweight = self._is_lightweight()
        logger.info(f"Running in {'LIGHTWEIGHT' if lightweight else 'FULL'} mode")

        try:
            if media_info.media_type == "image":
                if lightweight:
                    return self._run_image_light(media_info, start_time, errors, modules_used, progress)
                else:
                    return self._run_image_full(media_info, start_time, errors, modules_used, progress)
            elif media_info.media_type == "video":
                return self._run_video(media_info, start_time, errors, modules_used, progress, lightweight)
            elif media_info.media_type == "audio":
                return self._run_audio(media_info, start_time, errors, modules_used, progress)
            else:
                return self._error_result(media_info, start_time, f"Unsupported: {media_info.media_type}")
        finally:
            self.media_handler.cleanup(media_info)

    # ══════════════════════════════════════════
    # IMAGE — LIGHTWEIGHT (VLM only)
    # ══════════════════════════════════════════

    def _run_image_light(self, media_info, start_time, errors, modules_used, progress):
        """Image pipeline: just send the image to VLM. Simple and fast."""

        # Step 2-6: Skip heavy modules
        progress(2, 8, "Preparing image for AI analysis...")

        # Quick color analysis (very lightweight, no model needed)
        color_result = None
        color_mod = self._get_module("color")
        if color_mod:
            color_result = self._safe_run("color", lambda: color_mod.analyze(media_info.image_array), errors)
            if color_result: modules_used.append("color")

        progress(5, 8, "Skipping heavy modules (cloud mode)...")

        # Step 7: Send image directly to VLM
        progress(7, 8, "AI analyzing image...")
        context = self.aggregator.merge(
            media_type="image",
            color_result=color_result,
        )

        output = self._run_llm(context, errors, image=media_info.pil_image)
        modules_used.append("llm")

        progress(8, 8, "Finalizing...")
        return self._finalize(output, media_info, start_time, modules_used, errors)

    # ══════════════════════════════════════════
    # IMAGE — FULL (all modules + VLM)
    # ══════════════════════════════════════════

    def _run_image_full(self, media_info, start_time, errors, modules_used, progress):
        """Full image pipeline with all modules."""

        progress(2, 8, "Extracting text (OCR)...")
        ocr_mod = self._get_module("ocr")
        ocr_result = self._safe_run("ocr", lambda: ocr_mod.extract(media_info.image_array), errors) if ocr_mod else None
        if ocr_result: modules_used.append("ocr")

        progress(3, 8, "Detecting objects (YOLO)...")
        yolo_mod = self._get_module("yolo")
        yolo_result = self._safe_run("yolo", lambda: yolo_mod.detect(media_info.image_array), errors) if yolo_mod else None
        if yolo_result: modules_used.append("yolo")

        progress(4, 8, "Analyzing colors...")
        color_mod = self._get_module("color")
        color_result = self._safe_run("color", lambda: color_mod.analyze(media_info.image_array), errors) if color_mod else None
        if color_result: modules_used.append("color")

        progress(5, 8, "Describing scene (BLIP)...")
        blip_mod = self._get_module("blip")
        scene_result = self._safe_run("blip", lambda: blip_mod.describe(media_info.pil_image), errors) if blip_mod else None
        if scene_result: modules_used.append("blip")

        progress(6, 8, "Detecting language...")
        lang_result = None
        if ocr_result and ocr_result.get("full_text"):
            lang_mod = self._get_module("langdetect")
            lang_result = self._safe_run("langdetect", lambda: lang_mod.detect_language(ocr_result["full_text"]), errors) if lang_mod else None
            if lang_result: modules_used.append("langdetect")

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

        progress(8, 8, "Finalizing...")
        return self._finalize(output, media_info, start_time, modules_used, errors)

    # ══════════════════════════════════════════
    # VIDEO (works in both modes)
    # ══════════════════════════════════════════

    def _run_video(self, media_info, start_time, errors, modules_used, progress, lightweight):
        """Video pipeline: CLIP frames → optional OCR → Whisper → VLM."""

        # Step 2: Extract key frames
        if lightweight:
            # Skip CLIP — use simple interval sampling (no model needed)
            progress(2, 8, "Extracting key frames (interval sampling)...")
            frame_result = self._extract_frames_simple(media_info.temp_path, errors)
        else:
            # Full mode — use CLIP embedding-based selection
            progress(2, 8, "Extracting key frames (CLIP)...")
            frame_mod = self._get_module("frames")
            frame_result = self._safe_run("frames", lambda: frame_mod.extract(media_info.temp_path), errors) if frame_mod else None

        if frame_result: modules_used.append("frames")

        # Step 3: Audio extraction + transcription
        progress(3, 8, "Transcribing audio...")
        transcription_result = None
        audio_mod = self._get_module("audio_extract")
        audio_result = self._safe_run("audio_extract", lambda: audio_mod.extract(media_info.temp_path), errors) if audio_mod else None

        if audio_result and audio_result.get("has_audio"):
            modules_used.append("audio_extract")
            whisper_mod = self._get_module("whisper")
            if whisper_mod:
                transcription_result = self._safe_run("whisper", lambda: whisper_mod.transcribe(audio_result["audio_path"]), errors)
                if transcription_result: modules_used.append("whisper")
            if audio_result.get("audio_path"):
                audio_mod.cleanup(audio_result["audio_path"])

        # Step 4: Collect frames for VLM
        progress(4, 8, "Preparing frames for AI analysis...")
        frame_pil_images = []
        if frame_result and frame_result.get("frames"):
            for f in frame_result["frames"]:
                pil_img = Image.fromarray(f["image"][:, :, ::-1])
                frame_pil_images.append(pil_img)

        # Step 5: OCR only in full mode, and only top 3 text-dense frames
        merged_ocr = {"raw_texts": [], "full_text": "", "text_count": 0}

        if not lightweight and frame_result and frame_result.get("frames"):
            progress(5, 8, "Reading text from key frames...")
            import cv2
            import numpy as np

            frames = frame_result["frames"]
            edge_scores = []
            for f in frames:
                gray = cv2.cvtColor(f["image"], cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_scores.append(int(np.sum(edges > 0)))

            top_indices = sorted(range(len(edge_scores)), key=lambda i: edge_scores[i], reverse=True)[:3]
            all_texts = []
            ocr_mod = self._get_module("ocr")

            if ocr_mod:
                for idx in top_indices:
                    frame_img = frames[idx]["image"]
                    h, w = frame_img.shape[:2]
                    if w > 640:
                        scale = 640 / w
                        frame_img = cv2.resize(frame_img, (640, int(h * scale)))
                    ocr_res = self._safe_run("ocr", lambda fi=frame_img: ocr_mod.extract(fi), errors)
                    if ocr_res and ocr_res.get("raw_texts"):
                        all_texts.extend(ocr_res["raw_texts"])
                        if "ocr" not in modules_used:
                            modules_used.append("ocr")

            seen = set()
            unique = [t.strip() for t in all_texts if t.strip().lower() not in seen and not seen.add(t.strip().lower())]
            merged_ocr = {
                "raw_texts": unique,
                "text_with_positions": [{"text": t, "bbox": [], "confidence": 0.0} for t in unique],
                "full_text": " | ".join(unique),
                "avg_confidence": 0.0,
                "text_count": len(unique),
            }
        else:
            progress(5, 8, "Skipping OCR (cloud mode — VLM reads text directly)...")

        # Step 6: Color + language
        progress(6, 8, "Analyzing colors and language...")
        color_result = None
        if frame_result and frame_result.get("frames"):
            color_mod = self._get_module("color")
            if color_mod:
                color_result = self._safe_run("color", lambda: color_mod.analyze(frame_result["frames"][0]["image"]), errors)
                if color_result: modules_used.append("color")

        lang_result = None
        lang_source = ""
        if transcription_result and transcription_result.get("transcript"):
            lang_source = transcription_result["transcript"]
        elif merged_ocr.get("full_text"):
            lang_source = merged_ocr["full_text"]

        if lang_source:
            lang_mod = self._get_module("langdetect")
            if lang_mod:
                lang_result = self._safe_run("langdetect", lambda: lang_mod.detect_language(lang_source), errors)
                if lang_result: modules_used.append("langdetect")

        # Step 7: Send frames to VLM
        progress(7, 8, "AI analyzing video frames...")
        video_info = {
            "duration_seconds": media_info.duration_seconds,
            "resolution": media_info.resolution,
            "fps": media_info.fps,
            "frame_count": len(frame_pil_images),
        }

        context = self.aggregator.merge(
            media_type="video",
            ocr_result=merged_ocr if merged_ocr["text_count"] > 0 else None,
            color_result=color_result,
            language_result=lang_result,
            transcription_result=transcription_result,
            video_info=video_info,
        )

        output = self._run_llm(context, errors, images=frame_pil_images)
        modules_used.append("llm")

        progress(8, 8, "Finalizing...")
        return self._finalize(output, media_info, start_time, modules_used, errors)

    # ══════════════════════════════════════════
    # AUDIO
    # ══════════════════════════════════════════

    def _run_audio(self, media_info, start_time, errors, modules_used, progress):
        """Audio pipeline: Whisper → LLM reasoning."""

        progress(2, 8, "Transcribing audio (Whisper)...")
        whisper_mod = self._get_module("whisper")
        transcription_result = self._safe_run("whisper", lambda: whisper_mod.transcribe(media_info.temp_path), errors) if whisper_mod else None
        if transcription_result: modules_used.append("whisper")

        progress(3, 8, "Detecting language...")
        lang_result = None
        if transcription_result and transcription_result.get("transcript"):
            lang_mod = self._get_module("langdetect")
            if lang_mod:
                lang_result = self._safe_run("langdetect", lambda: lang_mod.detect_language(transcription_result["transcript"]), errors)
                if lang_result: modules_used.append("langdetect")

        progress(4, 8, "Skipping visual analysis (audio only)...")

        progress(7, 8, "AI reasoning over signals...")
        context = self.aggregator.merge(
            media_type="audio",
            transcription_result=transcription_result,
            language_result=lang_result,
        )

        output = self._run_llm(context, errors)
        modules_used.append("llm")

        progress(8, 8, "Finalizing...")
        return self._finalize(output, media_info, start_time, modules_used, errors)

    # ══════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════

    def _safe_run(self, name, func, errors):
        try:
            return func()
        except Exception as e:
            msg = f"{name} failed: {str(e)}"
            logger.error(msg)
            errors.append(msg)
            return None

    def _extract_frames_simple(self, video_path: str, errors: list) -> dict | None:
        """
        Lightweight frame extraction — no ML models.
        Uses multi-signal fingerprinting: color histogram + structural hash + edge density.
        Catches scene changes like CLIP but with zero RAM overhead.
        """
        try:
            import cv2
            import numpy as np

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = round(total_frames / fps, 2) if fps > 0 else 0

            # Sample at 1 FPS
            sample_interval = max(1, int(fps))

            def get_fingerprint(frame):
                """Compute lightweight multi-signal fingerprint for a frame."""
                small = cv2.resize(frame, (160, 90))  # Tiny for speed

                # Signal 1: Color histogram (HSV, 8 bins per channel)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
                hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
                hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
                color_hist = np.concatenate([hist_h, hist_s, hist_v])
                color_hist = color_hist / (color_hist.sum() + 1e-8)  # Normalize

                # Signal 2: Structural hash (4x4 grid mean brightness)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                grid = []
                gh, gw = h // 4, w // 4
                for r in range(4):
                    for c in range(4):
                        cell = gray[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
                        grid.append(float(np.mean(cell)) / 255.0)
                structure = np.array(grid)

                # Signal 3: Edge density
                edges = cv2.Canny(gray, 100, 200)
                edge_density = float(np.sum(edges > 0)) / (h * w)

                return color_hist, structure, edge_density

            def compute_change(fp1, fp2):
                """Compute weighted change score between two fingerprints."""
                hist1, struct1, edge1 = fp1
                hist2, struct2, edge2 = fp2

                # Color histogram similarity (cosine)
                dot = np.dot(hist1, hist2)
                norm = (np.linalg.norm(hist1) * np.linalg.norm(hist2)) + 1e-8
                color_sim = dot / norm

                # Structural similarity (1 - normalized euclidean)
                struct_dist = np.linalg.norm(struct1 - struct2)
                struct_sim = max(0, 1.0 - struct_dist / 2.0)

                # Edge delta
                edge_delta = abs(edge1 - edge2)

                # Weighted combined score (lower = more different)
                similarity = (0.6 * color_sim) + (0.3 * struct_sim) + (0.1 * (1.0 - min(edge_delta * 10, 1.0)))

                return similarity

            # Pass 1: Sample frames at 1 FPS and compute fingerprints
            sampled = []  # (frame_num, frame, fingerprint)
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % sample_interval == 0:
                    fp = get_fingerprint(frame)
                    sampled.append((frame_num, frame.copy(), fp))
                frame_num += 1

            if not sampled:
                cap.release()
                return None

            logger.info(f"Sampled {len(sampled)} frames at 1 FPS")

            # Pass 2: Select frames with significant change
            threshold = 0.88  # Below this similarity = new scene
            selected_indices = [0]  # Always include first frame
            prev_fp = sampled[0][2]

            for i in range(1, len(sampled)):
                sim = compute_change(prev_fp, sampled[i][2])
                if sim < threshold:
                    selected_indices.append(i)
                    prev_fp = sampled[i][2]

            # Always include last frame (brand/logo reveal)
            last_idx = len(sampled) - 1
            if last_idx not in selected_indices:
                selected_indices.append(last_idx)

            # Also include frame with highest edge density (most text)
            edge_densities = [s[2][2] for s in sampled]  # edge_density from fingerprint
            max_edge_idx = int(np.argmax(edge_densities))
            if max_edge_idx not in selected_indices:
                selected_indices.append(max_edge_idx)

            selected_indices = sorted(set(selected_indices))

            # Cap at 10 frames max to keep VLM costs reasonable
            if len(selected_indices) > 10:
                step = len(selected_indices) // 10
                keep = [selected_indices[i] for i in range(0, len(selected_indices), step)][:10]
                # Ensure first and last are always included
                if selected_indices[0] not in keep:
                    keep[0] = selected_indices[0]
                if selected_indices[-1] not in keep:
                    keep[-1] = selected_indices[-1]
                selected_indices = sorted(set(keep))

            # Build output
            frames = []
            for idx in selected_indices:
                fn, frame, fp = sampled[idx]
                frames.append({
                    "image": frame,
                    "timestamp_sec": round(fn / fps, 2),
                    "scene_index": len(frames),
                })

            cap.release()

            logger.info(
                f"Fingerprint selection: {len(frames)} key frames from "
                f"{len(sampled)} sampled (threshold={threshold})"
            )

            return {
                "frames": frames,
                "frame_count": len(frames),
                "scene_count": len(frames),
                "duration_seconds": duration,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "method": "fingerprint",
            }

        except Exception as e:
            errors.append(f"Frame extraction failed: {e}")
            logger.error(f"Fingerprint frame extraction failed: {e}")
            return None

    def _run_llm(self, context, errors, image=None, images=None):
        system_prompt, user_prompt = self.prompt_builder.build(context)

        for attempt in range(config.llm_max_retries + 1):
            raw_json = self.llm_engine.reason(system_prompt, user_prompt, image=image, images=images)

            if not raw_json:
                errors.append(f"LLM empty response (attempt {attempt + 1})")
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
                    user_prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {err[:500]}\nFix and retry."
            except json.JSONDecodeError as e:
                errors.append(f"JSON parse error (attempt {attempt + 1}): {str(e)[:200]}")

        logger.error("All LLM attempts failed")
        return AdIntelligenceOutput()

    def _finalize(self, output, media_info, start_time, modules_used, errors):
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

        expected = {"image": 2, "video": 4, "audio": 2}
        expected_count = expected.get(media_info.media_type, 3)
        success_count = len([m for m in modules_used if m != "llm"])
        output.meta.confidence_score = round(min(1.0, success_count / expected_count), 2)

        logger.info(f"Pipeline complete: {media_info.media_type} | {elapsed}s | confidence={output.meta.confidence_score}")
        return output

    def _error_result(self, media_info, start_time, error):
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
    print("Pipeline Orchestrator — Quick Test")
    print("=" * 50)

    pipeline = AdIntelligencePipeline()
    print(f"✓ Pipeline initialized (lightweight={pipeline._is_lightweight()})")

    test_image = Path("tests/sample_ads/test.jpg")
    if test_image.exists():
        print(f"\nRunning on: {test_image}")

        class FakeUpload:
            name = str(test_image)
            def read(self): return test_image.read_bytes()
            def seek(self, n): pass

        result = pipeline.run(FakeUpload())
        print(f"  Time: {result.meta.processing_time_sec}s")
        print(f"  Brand: {result.brand.company_name}")
        print(f"  Product: {result.product.product_name}")
        print(f"  Modules: {result.meta.modules_used}")

    print(f"\n{'=' * 50}")