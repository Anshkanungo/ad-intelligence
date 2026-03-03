"""
Ad Intelligence Pipeline — Frame Extraction Module (CLIP + OpenCV)

Extracts key frames from video using CLIP embedding similarity.
Strategy:
  1. Sample frames at 1 FPS
  2. Encode each frame with CLIP
  3. Compare consecutive embeddings — keep frames with significant visual change
  4. Always include first + last frame
  5. Return ALL significant frames (no arbitrary cap)

Usage:
    from pipeline.modules.frame_extraction import FrameExtractionModule
    extractor = FrameExtractionModule()
    result = extractor.extract("video.mp4")
"""

import cv2
import numpy as np
from pathlib import Path

from utils.config import config
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class FrameExtractionModule:
    """Extracts key frames using CLIP embedding-based scene change detection."""

    def __init__(self):
        self._clip_model = None
        self._clip_preprocess = None
        self._tokenizer = None
        self._clip_loaded = False

    def _load_clip(self):
        """Lazy-load CLIP model."""
        if self._clip_loaded:
            return

        try:
            import open_clip
            import torch

            logger.info("Loading CLIP model (ViT-B-32)...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess
            self._clip_loaded = True
            logger.info("CLIP model loaded")

        except ImportError:
            logger.warning("open-clip-torch not installed. Falling back to interval sampling.")
            self._clip_loaded = False
        except Exception as e:
            logger.warning(f"CLIP load failed: {e}. Falling back to interval sampling.")
            self._clip_loaded = False

    def _get_clip_embedding(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Get CLIP embedding for a single frame."""
        if not self._clip_loaded or self._clip_model is None:
            return None

        try:
            import torch
            from PIL import Image

            # BGR → RGB → PIL
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Preprocess and encode
            img_tensor = self._clip_preprocess(pil_img).unsqueeze(0)
            with torch.no_grad():
                embedding = self._clip_model.encode_image(img_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize

            return embedding.squeeze().numpy()

        except Exception as e:
            logger.warning(f"CLIP embedding failed: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @log_execution_time
    def extract(self, video_path: str) -> dict:
        """
        Extract key frames from a video using CLIP similarity.

        Args:
            video_path: Path to video file

        Returns:
            dict with keys:
                - frames: list[dict] — each with 'image' (np.ndarray),
                    'timestamp_sec', 'scene_index'
                - frame_count: int
                - scene_count: int
                - duration_seconds: float
                - fps: float
                - resolution: str
                - method: str — "clip" or "interval"
                - total_sampled: int — frames sampled at 1fps
                - similarity_threshold: float
        """
        result = {
            "frames": [],
            "frame_count": 0,
            "scene_count": 0,
            "duration_seconds": 0.0,
            "fps": 0.0,
            "resolution": "",
            "method": "",
            "total_sampled": 0,
            "similarity_threshold": 0.0,
        }

        try:
            # Get video info
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return result

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result["fps"] = fps
            result["resolution"] = f"{width}x{height}"
            result["duration_seconds"] = round(
                total_frames / fps, 2
            ) if fps > 0 else 0.0
            cap.release()

            # Try CLIP-based extraction
            self._load_clip()

            if self._clip_loaded:
                frames = self._clip_extract(video_path, fps, total_frames)
                result["method"] = "clip"
            else:
                logger.info("CLIP not available, using interval + edge detection fallback")
                frames = self._smart_interval_sample(video_path, fps, total_frames)
                result["method"] = "interval_smart"

            result["frames"] = frames
            result["frame_count"] = len(frames)
            result["scene_count"] = len(frames)

            logger.info(
                f"Extracted {result['frame_count']} key frames "
                f"({result['method']}) from {result['duration_seconds']}s video"
            )

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}", exc_info=True)

        return result

    def _clip_extract(self, video_path: str, fps: float, total_frames: int) -> list[dict]:
        """
        CLIP-based frame extraction:
        1. Sample at 1 FPS
        2. Compute CLIP embeddings
        3. Keep frames where similarity to previous drops below threshold
        4. Always include first + last frame
        """
        similarity_threshold = config.clip_similarity_threshold
        sample_interval = max(1, int(fps))  # 1 FPS

        cap = cv2.VideoCapture(video_path)
        all_sampled = []  # (frame_num, frame_image)

        # Step 1: Sample at 1 FPS
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % sample_interval == 0:
                all_sampled.append((frame_num, frame.copy()))
            frame_num += 1
        cap.release()

        if not all_sampled:
            return []

        logger.info(f"Sampled {len(all_sampled)} frames at 1 FPS")

        # Step 2: Compute CLIP embeddings for all sampled frames
        embeddings = []
        for fn, frame in all_sampled:
            emb = self._get_clip_embedding(frame)
            embeddings.append(emb)

        # Step 3: Select frames with significant change
        selected = []

        # Always include first frame
        selected.append(0)

        prev_emb = embeddings[0]
        for i in range(1, len(all_sampled)):
            curr_emb = embeddings[i]

            if prev_emb is not None and curr_emb is not None:
                sim = self._cosine_similarity(prev_emb, curr_emb)
                if sim < similarity_threshold:
                    selected.append(i)
                    prev_emb = curr_emb
                    logger.debug(f"  Frame {i}: similarity={sim:.3f} — SELECTED (change detected)")
                else:
                    logger.debug(f"  Frame {i}: similarity={sim:.3f} — skipped (too similar)")
            else:
                # If embedding failed, include the frame to be safe
                selected.append(i)
                prev_emb = curr_emb

        # Always include last frame (often has brand/logo reveal)
        last_idx = len(all_sampled) - 1
        if last_idx not in selected:
            selected.append(last_idx)

        # Also include the frame with the most text (OCR-heavy frame)
        # We approximate this by checking which frames have the most edge content
        # (text-heavy frames tend to have more edges)
        edge_scores = []
        for fn, frame in all_sampled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_scores.append(np.sum(edges > 0))

        max_edge_idx = int(np.argmax(edge_scores))
        if max_edge_idx not in selected:
            selected.append(max_edge_idx)
            logger.debug(f"  Frame {max_edge_idx}: added (highest text/edge density)")

        # Sort and deduplicate
        selected = sorted(set(selected))

        # Build output
        frames = []
        for idx in selected:
            fn, frame = all_sampled[idx]
            timestamp = round(fn / fps, 2)
            frames.append({
                "image": frame,
                "timestamp_sec": timestamp,
                "scene_index": len(frames),
            })

        logger.info(
            f"CLIP selected {len(frames)} key frames from {len(all_sampled)} sampled "
            f"(threshold={similarity_threshold})"
        )

        return frames

    def _interval_sample(self, video_path: str, fps: float, total_frames: int) -> list[dict]:
        """Fallback: sample frames at fixed intervals."""
        frames = []
        max_frames = config.max_keyframes
        cap = cv2.VideoCapture(video_path)

        interval = max(1, total_frames // max_frames)

        for i in range(0, total_frames, interval):
            if len(frames) >= max_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret and frame is not None:
                timestamp = round(i / fps, 2)
                frames.append({
                    "image": frame,
                    "timestamp_sec": timestamp,
                    "scene_index": len(frames),
                })

        cap.release()
        return frames

    def _smart_interval_sample(self, video_path: str, fps: float, total_frames: int) -> list[dict]:
        """
        Lightweight fallback: sample at 1 FPS, use histogram comparison
        to detect scene changes. No ML model needed — pure OpenCV.
        Always includes first frame, last frame, and most text-dense frame.
        """
        sample_interval = max(1, int(fps))  # 1 FPS
        cap = cv2.VideoCapture(video_path)

        all_sampled = []
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % sample_interval == 0:
                all_sampled.append((frame_num, frame.copy()))
            frame_num += 1
        cap.release()

        if not all_sampled:
            return []

        logger.info(f"Sampled {len(all_sampled)} frames at 1 FPS (histogram method)")

        # Compute histogram for each frame
        histograms = []
        for fn, frame in all_sampled:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            histograms.append(hist)

        # Select frames with significant histogram change
        selected = [0]  # Always first frame
        threshold = 0.4  # Lower = more sensitive (correlation: 1.0 = identical)

        for i in range(1, len(all_sampled)):
            similarity = cv2.compareHist(histograms[i - 1], histograms[i], cv2.HISTCMP_CORREL)
            if similarity < threshold:
                selected.append(i)

        # Always include last frame
        last_idx = len(all_sampled) - 1
        if last_idx not in selected:
            selected.append(last_idx)

        # Add most text-dense frame (edge detection proxy)
        edge_scores = []
        for fn, frame in all_sampled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_scores.append(int(np.sum(edges > 0)))

        max_edge_idx = int(np.argmax(edge_scores))
        if max_edge_idx not in selected:
            selected.append(max_edge_idx)

        selected = sorted(set(selected))

        # Build output
        frames = []
        for idx in selected:
            fn, frame = all_sampled[idx]
            timestamp = round(fn / fps, 2)
            frames.append({
                "image": frame,
                "timestamp_sec": timestamp,
                "scene_index": len(frames),
            })

        logger.info(f"Histogram selected {len(frames)} key frames from {len(all_sampled)} sampled")
        return frames


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("Frame Extraction Module (CLIP) — Quick Test")
    print("=" * 50)

    extractor = FrameExtractionModule()
    print("✓ Module initialized")

    # Look for test video
    test_videos = []
    for ext in ["*.mp4", "*.mov", "*.avi", "*.webm"]:
        test_videos += list(Path("tests/sample_ads").glob(ext))

    if test_videos:
        video = test_videos[0]
        print(f"\nProcessing: {video}")
        result = extractor.extract(str(video))

        print(f"  Method: {result['method']}")
        print(f"  Frames extracted: {result['frame_count']}")
        print(f"  Duration: {result['duration_seconds']}s")
        print(f"  FPS: {result['fps']}")
        print(f"  Resolution: {result['resolution']}")

        for f in result["frames"]:
            shape = f["image"].shape if f["image"] is not None else "N/A"
            print(f"    Scene {f['scene_index']}: {f['timestamp_sec']}s — {shape}")
    else:
        print("\n⚠ No test video. Drop a .mp4 into tests/sample_ads/ to test")

    print(f"\n{'=' * 50}")
    print("Frame extraction module ready!")
    print(f"{'=' * 50}")