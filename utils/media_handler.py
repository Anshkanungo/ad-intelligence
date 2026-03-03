"""
Ad Intelligence Pipeline — Media Handler

Handles file type detection, validation, and preprocessing.
Supports image, video, and audio files.

Usage:
    from utils.media_handler import MediaHandler
    handler = MediaHandler()
    media_info = handler.process_upload(uploaded_file)
"""

import os
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image

from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════

@dataclass
class MediaInfo:
    """Information about a processed media file."""
    media_type: str = ""                # "image", "video", "audio"
    original_filename: str = ""
    file_extension: str = ""
    file_size_mb: float = 0.0
    temp_path: str = ""                 # Path to temp file on disk
    is_valid: bool = False
    error: str = ""

    # Image-specific
    width: int = 0
    height: int = 0
    resolution: str = ""                # "1920x1080"
    image_array: np.ndarray | None = None   # OpenCV BGR array
    pil_image: Image.Image | None = None    # PIL RGB image

    # Video-specific
    duration_seconds: float = 0.0
    fps: float = 0.0
    frame_count: int = 0

    # Audio-specific (also populated for video)
    has_audio: bool = False
    audio_path: str = ""                # Path to extracted audio file


# ══════════════════════════════════════════════
# MEDIA HANDLER
# ══════════════════════════════════════════════

class MediaHandler:
    """Processes uploaded files — validates, saves to temp, extracts metadata."""

    def __init__(self):
        self.config = config

    def process_upload(self, uploaded_file) -> MediaInfo:
        """
        Process an uploaded file (Streamlit UploadedFile or file-like object).

        Args:
            uploaded_file: Streamlit UploadedFile or any object with
                          .name, .read(), .size attributes

        Returns:
            MediaInfo with all metadata populated
        """
        info = MediaInfo()

        try:
            # Step 1: Basic file info
            filename = getattr(uploaded_file, "name", "unknown")
            info.original_filename = filename
            info.file_extension = Path(filename).suffix.lower()
            info.media_type = self.config.get_media_type(filename)

            if info.media_type == "unknown":
                info.error = f"Unsupported file type: {info.file_extension}"
                logger.error(info.error)
                return info

            # Step 2: Read file bytes
            file_bytes = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)  # Reset for potential re-read

            info.file_size_mb = round(len(file_bytes) / (1024 * 1024), 2)

            # Step 3: Check file size limits
            size_limit = {
                "image": self.config.max_image_size_mb,
                "video": self.config.max_video_size_mb,
                "audio": self.config.max_audio_size_mb,
            }.get(info.media_type, 50)

            if info.file_size_mb > size_limit:
                info.error = f"File too large: {info.file_size_mb}MB (limit: {size_limit}MB)"
                logger.error(info.error)
                return info

            # Step 4: Save to temp file
            suffix = info.file_extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                info.temp_path = tmp.name

            logger.info(f"Saved temp file: {info.temp_path} ({info.file_size_mb}MB)")

            # Step 5: Type-specific processing
            if info.media_type == "image":
                self._process_image(info)
            elif info.media_type == "video":
                self._process_video(info)
            elif info.media_type == "audio":
                self._process_audio(info)

            if not info.error:
                info.is_valid = True
                logger.info(
                    f"Media ready: {info.media_type} | "
                    f"{info.original_filename} | "
                    f"{info.resolution or 'N/A'} | "
                    f"{info.file_size_mb}MB"
                )

        except Exception as e:
            info.error = f"Processing error: {str(e)}"
            logger.error(info.error, exc_info=True)

        return info

    def _process_image(self, info: MediaInfo):
        """Load and validate image, populate metadata."""
        try:
            # Load with PIL (RGB)
            pil_img = Image.open(info.temp_path).convert("RGB")
            info.pil_image = pil_img
            info.width, info.height = pil_img.size
            info.resolution = f"{info.width}x{info.height}"

            # Load with OpenCV (BGR) for modules that need numpy arrays
            cv_img = cv2.imread(info.temp_path)
            if cv_img is None:
                info.error = "OpenCV could not read the image"
                return
            info.image_array = cv_img

        except Exception as e:
            info.error = f"Image processing error: {str(e)}"

    def _process_video(self, info: MediaInfo):
        """Extract video metadata using OpenCV."""
        try:
            cap = cv2.VideoCapture(info.temp_path)
            if not cap.isOpened():
                info.error = "Could not open video file"
                return

            info.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            info.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info.resolution = f"{info.width}x{info.height}"

            if info.fps > 0:
                info.duration_seconds = round(info.frame_count / info.fps, 2)

            cap.release()

            # Check duration limit
            if info.duration_seconds > self.config.max_video_duration:
                info.error = (
                    f"Video too long: {info.duration_seconds}s "
                    f"(limit: {self.config.max_video_duration}s)"
                )
                return

            # Check for audio track using FFprobe
            info.has_audio = self._check_audio_track(info.temp_path)

        except Exception as e:
            info.error = f"Video processing error: {str(e)}"

    def _process_audio(self, info: MediaInfo):
        """Validate audio file and get duration."""
        try:
            info.has_audio = True
            info.audio_path = info.temp_path

            # Get duration using FFprobe
            duration = self._get_audio_duration(info.temp_path)
            if duration is not None:
                info.duration_seconds = duration

        except Exception as e:
            info.error = f"Audio processing error: {str(e)}"

    def _check_audio_track(self, video_path: str) -> bool:
        """Check if a video file has an audio track."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    video_path
                ],
                capture_output=True, text=True, timeout=10
            )
            return "audio" in result.stdout.lower()
        except Exception:
            logger.warning("FFprobe not available — assuming video has audio")
            return True

    def _get_audio_duration(self, audio_path: str) -> float | None:
        """Get audio duration in seconds using FFprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    audio_path
                ],
                capture_output=True, text=True, timeout=10
            )
            return round(float(result.stdout.strip()), 2)
        except Exception:
            logger.warning("Could not determine audio duration")
            return None

    @staticmethod
    def cleanup(info: MediaInfo):
        """Delete temp files after processing is complete."""
        for path in [info.temp_path, info.audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.debug(f"Cleaned up: {path}")
                except Exception as e:
                    logger.warning(f"Cleanup failed for {path}: {e}")


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("Media Handler — Quick Test")
    print("=" * 50)

    handler = MediaHandler()

    # Test media type detection
    test_files = [
        "nike_ad.jpg", "coca_cola.png", "brochure.webp",
        "tv_spot.mp4", "youtube_ad.mov",
        "radio_ad.mp3", "podcast.wav",
        "document.pdf", "notes.txt"
    ]

    print("\nMedia type detection:")
    for f in test_files:
        media_type = config.get_media_type(f)
        status = "✓" if media_type != "unknown" else "✗"
        print(f"  {status} {f:<25} → {media_type}")

    # Test with a real image (if exists)
    test_image = Path("tests/sample_ads/test.jpg")
    if test_image.exists():
        print(f"\nProcessing real file: {test_image}")

        class FakeUpload:
            name = str(test_image)
            def read(self): return test_image.read_bytes()
            def seek(self, n): pass

        info = handler.process_upload(FakeUpload())
        print(f"  Valid: {info.is_valid}")
        print(f"  Type: {info.media_type}")
        print(f"  Resolution: {info.resolution}")
        print(f"  Size: {info.file_size_mb}MB")
        if info.error:
            print(f"  Error: {info.error}")
        handler.cleanup(info)
    else:
        print(f"\nNo test image found at {test_image}")
        print("  Drop a test image there to test processing.")

    print(f"\n{'=' * 50}")
    print("Media handler ready!")
    print(f"{'=' * 50}")