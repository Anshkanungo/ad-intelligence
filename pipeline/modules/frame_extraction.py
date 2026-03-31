"""
Ad Intelligence Pipeline — Frame Extraction Module (CLIP + OpenCV)

Smart frame selection: picks exactly MAX_GROQ_FRAMES frames optimized for
a single Groq VLM call (max 5 images = 1 preview + 4 frames).

Strategy:
  1. Sample frames at 1 FPS
  2. Encode each frame with CLIP
  3. Score every frame against preview embedding (product relevance)
  4. Select frames from 3 buckets:
     - PRODUCT: highest preview similarity (the advertised product/service)
     - STORY: most visually diverse frames NOT similar to product (narrative)
     - BRANDING: last frame (end card — often has logo/tagline)
  5. Deduplicate: no two selected frames with CLIP similarity > 0.85
  6. Return exactly MAX_GROQ_FRAMES frames

Usage:
    from pipeline.modules.frame_extraction import FrameExtractionModule
    extractor = FrameExtractionModule()
    result = extractor.extract("video.mp4", preview_embeddings=[emb])
"""

import cv2
import numpy as np
from pathlib import Path

from utils.config import config
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)

# Groq limit: 5 images total = 1 preview + 4 video frames
MAX_GROQ_FRAMES = 4
# Two selected frames above this similarity are considered duplicates
DEDUP_THRESHOLD = 0.85


class FrameExtractionModule:
    """Extracts key frames using CLIP embedding-based selection."""

    def __init__(self):
        self._clip_model = None
        self._clip_preprocess = None
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

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            img_tensor = self._clip_preprocess(pil_img).unsqueeze(0)
            with torch.no_grad():
                embedding = self._clip_model.encode_image(img_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.squeeze().numpy()

        except Exception as e:
            logger.warning(f"CLIP embedding failed: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _preview_similarity(self, embedding: np.ndarray, preview_embeddings: list) -> float:
        """Max cosine similarity between frame embedding and all preview embeddings."""
        max_sim = 0.0
        for prev_emb in preview_embeddings:
            if hasattr(prev_emb, 'cpu'):
                prev_np = prev_emb.cpu().squeeze().numpy()
            elif hasattr(prev_emb, 'numpy'):
                prev_np = prev_emb.squeeze().numpy()
            else:
                prev_np = np.array(prev_emb).squeeze()
            sim = self._cosine_similarity(embedding, prev_np)
            max_sim = max(max_sim, sim)
        return max_sim

    def _is_duplicate(self, emb: np.ndarray, selected_embs: list[np.ndarray]) -> bool:
        """Check if a frame is too similar to any already-selected frame."""
        for sel_emb in selected_embs:
            if self._cosine_similarity(emb, sel_emb) > DEDUP_THRESHOLD:
                return True
        return False

    @log_execution_time
    def extract(self, video_path: str, preview_embeddings: list = None, preview_images: list = None) -> dict:
        """
        Extract key frames from video.

        Args:
            video_path: Path to video file
            preview_embeddings: Pre-computed CLIP embeddings (optional)
            preview_images: Raw PIL images to encode with CLIP (optional, used if no embeddings)
                Avoids loading CLIP twice — this method encodes previews with its own CLIP.

        Returns:
            dict with frames list, metadata, and per-frame preview_similarity scores
        """
        result = {
            "frames": [],
            "frame_count": 0,
            "duration_seconds": 0.0,
            "fps": 0.0,
            "resolution": "",
            "method": "",
            "total_sampled": 0,
        }

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return result

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = round(total_frames / fps, 2) if fps > 0 else 0.0

            result["fps"] = fps
            result["resolution"] = f"{width}x{height}"
            result["duration_seconds"] = duration
            cap.release()

            self._load_clip()

            # If raw preview images provided, encode them now (single CLIP load)
            if not preview_embeddings and preview_images and self._clip_loaded:
                import torch
                preview_embeddings = []
                for img in preview_images:
                    img_tensor = self._clip_preprocess(img).unsqueeze(0)
                    with torch.no_grad():
                        emb = self._clip_model.encode_image(img_tensor)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                        preview_embeddings.append(emb.squeeze().numpy())
                logger.info(f"Encoded {len(preview_embeddings)} preview image(s) with CLIP")

            if self._clip_loaded and preview_embeddings:
                frames = self._smart_select(video_path, fps, total_frames, duration, preview_embeddings)
                result["method"] = "clip_smart"
            elif self._clip_loaded:
                frames = self._scene_change_select(video_path, fps, total_frames)
                result["method"] = "clip_scene"
            else:
                frames = self._interval_sample(video_path, fps, total_frames)
                result["method"] = "interval"

            result["frames"] = frames
            result["frame_count"] = len(frames)
            result["total_sampled"] = int(duration)  # ~1 per second

            logger.info(
                f"Extracted {len(frames)} key frames ({result['method']}) "
                f"from {duration}s video"
            )

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}", exc_info=True)

        return result

    def _sample_all_frames(self, video_path: str, fps: float) -> list[tuple]:
        """Sample frames at 1 FPS, return list of (frame_num, frame_image)."""
        sample_interval = max(1, int(fps))
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

        logger.info(f"Sampled {len(all_sampled)} frames at 1 FPS")
        return all_sampled

    def _compute_all_embeddings(self, all_sampled: list[tuple], preview_embeddings: list = None):
        """Compute CLIP embeddings and preview similarities for all sampled frames."""
        embeddings = []
        preview_sims = []

        for fn, frame in all_sampled:
            emb = self._get_clip_embedding(frame)
            embeddings.append(emb)

            if emb is not None and preview_embeddings:
                sim = self._preview_similarity(emb, preview_embeddings)
            else:
                sim = 0.0
            preview_sims.append(sim)

        return embeddings, preview_sims

    def _smart_select(
        self, video_path: str, fps: float, total_frames: int,
        duration: float, preview_embeddings: list
    ) -> list[dict]:
        """
        Smart 4-frame selection for Groq:
          Bucket 1 — PRODUCT: top frames by preview similarity
          Bucket 2 — STORY: most diverse frames (low similarity to each other AND to product frames)
          Bucket 3 — BRANDING: last frame (end cards with logos/taglines)
        Deduplicate across all buckets.
        """
        all_sampled = self._sample_all_frames(video_path, fps)
        if not all_sampled:
            return []

        embeddings, preview_sims = self._compute_all_embeddings(all_sampled, preview_embeddings)

        # Rank all frames by preview similarity
        scored = [(i, preview_sims[i]) for i in range(len(all_sampled)) if embeddings[i] is not None]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected_indices = []
        selected_embs = []

        # ── Bucket 1: PRODUCT (top 2 by preview similarity, deduplicated) ──
        product_count = 0
        for idx, sim in scored:
            if product_count >= 2:
                break
            if not self._is_duplicate(embeddings[idx], selected_embs):
                selected_indices.append(idx)
                selected_embs.append(embeddings[idx])
                product_count += 1
                logger.info(f"  PRODUCT frame {idx} (t={round(all_sampled[idx][0]/fps, 1)}s): preview_sim={sim:.3f}")

        # ── Bucket 3: BRANDING (last frame — check before story so it gets priority) ──
        last_idx = len(all_sampled) - 1
        if embeddings[last_idx] is not None and not self._is_duplicate(embeddings[last_idx], selected_embs):
            selected_indices.append(last_idx)
            selected_embs.append(embeddings[last_idx])
            logger.info(f"  BRANDING frame {last_idx} (t={round(all_sampled[last_idx][0]/fps, 1)}s): preview_sim={preview_sims[last_idx]:.3f}")

        # ── Bucket 2: STORY (most diverse frames not yet selected) ──
        remaining_slots = MAX_GROQ_FRAMES - len(selected_indices)

        if remaining_slots > 0:
            # Find frames that are:
            # 1. Not already selected
            # 2. Not duplicates of selected frames
            # 3. Maximally diverse from each other
            # Strategy: pick frame with lowest max-similarity to already selected set

            candidates = [
                i for i in range(len(all_sampled))
                if i not in selected_indices and embeddings[i] is not None
            ]

            for _ in range(remaining_slots):
                if not candidates:
                    break

                # For each candidate, compute its max similarity to any selected frame
                best_idx = None
                best_diversity = float('inf')

                for c in candidates:
                    max_sim_to_selected = max(
                        self._cosine_similarity(embeddings[c], se) for se in selected_embs
                    ) if selected_embs else 0.0

                    # Lower max similarity = more diverse
                    if max_sim_to_selected < best_diversity:
                        best_diversity = max_sim_to_selected
                        best_idx = c

                if best_idx is not None and best_diversity < DEDUP_THRESHOLD:
                    selected_indices.append(best_idx)
                    selected_embs.append(embeddings[best_idx])
                    candidates.remove(best_idx)
                    logger.info(
                        f"  STORY frame {best_idx} "
                        f"(t={round(all_sampled[best_idx][0]/fps, 1)}s): "
                        f"preview_sim={preview_sims[best_idx]:.3f}, "
                        f"diversity={1-best_diversity:.3f}"
                    )
                else:
                    # All remaining candidates are too similar to what we have
                    break

        # If we still have empty slots (unlikely), fill with next best preview-sim frames
        remaining_slots = MAX_GROQ_FRAMES - len(selected_indices)
        if remaining_slots > 0:
            for idx, sim in scored:
                if remaining_slots <= 0:
                    break
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    remaining_slots -= 1
                    logger.info(f"  FILL frame {idx} (t={round(all_sampled[idx][0]/fps, 1)}s): preview_sim={sim:.3f}")

        # Sort by timestamp for chronological order
        selected_indices.sort(key=lambda i: all_sampled[i][0])

        # Build output
        frames = []
        for idx in selected_indices:
            fn, frame = all_sampled[idx]
            timestamp = round(fn / fps, 2)
            frames.append({
                "image": frame,
                "timestamp_sec": timestamp,
                "scene_index": len(frames),
                "preview_similarity": round(preview_sims[idx], 3),
                "bucket": self._get_bucket_label(idx, selected_indices, last_idx, scored),
            })

        logger.info(
            f"Smart selected {len(frames)} frames from {len(all_sampled)} sampled "
            f"(product={product_count}, branding={'yes' if last_idx in selected_indices else 'no'}, "
            f"story={len(frames) - product_count - (1 if last_idx in selected_indices else 0)})"
        )

        return frames

    def _get_bucket_label(self, idx, selected_indices, last_idx, scored) -> str:
        """Label which bucket a frame came from (for logging/debugging)."""
        if idx == last_idx:
            return "branding"
        # Check if it was in top 2 by preview sim
        top_product = [s[0] for s in scored[:10]]  # generous check
        if idx in top_product[:2]:
            return "product"
        return "story"

    def _scene_change_select(self, video_path: str, fps: float, total_frames: int) -> list[dict]:
        """Fallback: scene-change detection without preview (no product awareness)."""
        all_sampled = self._sample_all_frames(video_path, fps)
        if not all_sampled:
            return []

        embeddings, _ = self._compute_all_embeddings(all_sampled)

        # Pick frames with biggest visual change
        selected = [0]  # always first
        prev_emb = embeddings[0]
        scene_threshold = config.clip_similarity_threshold

        for i in range(1, len(all_sampled)):
            if embeddings[i] is not None and prev_emb is not None:
                sim = self._cosine_similarity(prev_emb, embeddings[i])
                if sim < scene_threshold:
                    selected.append(i)
                    prev_emb = embeddings[i]

        # Always last
        last_idx = len(all_sampled) - 1
        if last_idx not in selected:
            selected.append(last_idx)

        # Trim to MAX_GROQ_FRAMES by keeping most evenly spaced
        if len(selected) > MAX_GROQ_FRAMES:
            step = len(selected) / MAX_GROQ_FRAMES
            trimmed = [selected[int(i * step)] for i in range(MAX_GROQ_FRAMES)]
            if last_idx not in trimmed:
                trimmed[-1] = last_idx
            selected = trimmed

        frames = []
        for idx in selected:
            fn, frame = all_sampled[idx]
            frames.append({
                "image": frame,
                "timestamp_sec": round(fn / fps, 2),
                "scene_index": len(frames),
                "preview_similarity": 0.0,
                "bucket": "scene_change",
            })

        return frames

    def _interval_sample(self, video_path: str, fps: float, total_frames: int) -> list[dict]:
        """Last resort fallback: evenly spaced frames."""
        cap = cv2.VideoCapture(video_path)
        interval = max(1, total_frames // MAX_GROQ_FRAMES)
        frames = []

        for i in range(0, total_frames, interval):
            if len(frames) >= MAX_GROQ_FRAMES:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append({
                    "image": frame,
                    "timestamp_sec": round(i / fps, 2),
                    "scene_index": len(frames),
                    "preview_similarity": 0.0,
                    "bucket": "interval",
                })

        cap.release()
        return frames

    def unload(self):
        """Free CLIP model from memory."""
        import torch
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_loaded = False
        torch.cuda.empty_cache()
        logger.info("CLIP model unloaded")


if __name__ == "__main__":
    print("=" * 50)
    print("Frame Extraction Module — Quick Test")
    print("=" * 50)

    extractor = FrameExtractionModule()

    test_videos = []
    for ext in ["*.mp4", "*.mov", "*.avi", "*.webm"]:
        test_videos += list(Path("tests/sample_ads").glob(ext))

    if test_videos:
        video = test_videos[0]
        print(f"\nProcessing: {video}")
        result = extractor.extract(str(video))

        print(f"  Method: {result['method']}")
        print(f"  Frames: {result['frame_count']}")
        print(f"  Duration: {result['duration_seconds']}s")

        for f in result["frames"]:
            print(f"    t={f['timestamp_sec']}s | bucket={f.get('bucket','?')} | preview_sim={f['preview_similarity']:.3f}")
    else:
        print("No test videos found")

    print(f"\n{'=' * 50}")