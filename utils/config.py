"""
Ad Intelligence Pipeline — Configuration Loader

Loads environment variables and provides pipeline-wide settings.
All config in one place. Import and use: from utils.config import config
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
# Walks up from this file to find .env
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
load_dotenv(_project_root / ".env")


@dataclass
class PipelineConfig:
    """Central configuration for the entire pipeline."""

    # ── API Keys ──
    groq_api_key: str = ""
    gemini_api_key: str = ""
    hf_api_token: str = ""

    # ── LLM Settings (Groq — Primary) ──
    groq_model: str = "llama-3.3-70b-versatile"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_fallback_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_max_retries: int = 2

    # ── LLM Settings (Gemini — Fallback) ──
    gemini_model: str = "gemini-2.5-flash-lite"

    # ── OCR Settings ──
    ocr_languages: list[str] = field(default_factory=lambda: ["en"])
    ocr_confidence_threshold: float = 0.5

    # ── Object Detection Settings ──
    yolo_model: str = "yolov8n.pt"          # nano model — fast, small
    yolo_confidence: float = 0.4

    # ── Scene Description (BLIP via HuggingFace) ──
    blip_model_id: str = "Salesforce/blip-image-captioning-base"
    hf_api_url: str = "https://api-inference.huggingface.co/models/"

    # ── Whisper Settings ──
    whisper_model: str = "base"             # tiny, base, small, medium, large
    whisper_language: str | None = None     # None = auto-detect

    # ── Color Analysis ──
    color_clusters: int = 6                 # KMeans clusters
    color_resize: int = 150                 # Resize image for speed

    # ── Video Processing ──
    max_keyframes: int = 20                 # Max frames fallback (interval mode)
    max_video_duration: int = 300           # 5 min max (seconds)
    clip_similarity_threshold: float = 0.50 # Below this = significant change (0-1)

    # ── File Limits ──
    max_image_size_mb: int = 20
    max_video_size_mb: int = 100
    max_audio_size_mb: int = 50
    supported_image_ext: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")
    supported_video_ext: tuple = (".mp4", ".mov", ".avi", ".webm", ".mkv")
    supported_audio_ext: tuple = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")

    # ── Pipeline Settings ──
    pipeline_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    log_level: str = "INFO"

    def __post_init__(self):
        """Load API keys from environment after init."""
        self.groq_api_key = os.getenv("GROQ_API_KEY", self.groq_api_key)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", self.gemini_api_key)
        self.hf_api_token = os.getenv("HF_API_TOKEN", self.hf_api_token)
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key and self.groq_api_key != "your_groq_api_key_here")

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key != "your_gemini_api_key_here")

    @property
    def has_hf(self) -> bool:
        return bool(self.hf_api_token and self.hf_api_token != "your_huggingface_token_here")

    @property
    def all_supported_ext(self) -> tuple:
        return self.supported_image_ext + self.supported_video_ext + self.supported_audio_ext

    def get_media_type(self, filename: str) -> str:
        """Determine media type from filename extension."""
        ext = Path(filename).suffix.lower()
        if ext in self.supported_image_ext:
            return "image"
        elif ext in self.supported_video_ext:
            return "video"
        elif ext in self.supported_audio_ext:
            return "audio"
        return "unknown"

    def validate_keys(self) -> dict[str, bool]:
        """Check which API keys are configured."""
        return {
            "groq": self.has_groq,
            "gemini": self.has_gemini,
            "huggingface": self.has_hf,
        }

    def print_status(self):
        """Print config status for debugging."""
        keys = self.validate_keys()
        print(f"Pipeline Config Status:")
        print(f"  Groq API:      {'✓ configured' if keys['groq'] else '✗ missing'}")
        print(f"  Gemini API:    {'✓ configured' if keys['gemini'] else '✗ missing (optional)'}")
        print(f"  HuggingFace:   {'✓ configured' if keys['huggingface'] else '✗ missing'}")
        print(f"  LLM Model:     {self.groq_model}")
        print(f"  Whisper Model: {self.whisper_model}")
        print(f"  YOLO Model:    {self.yolo_model}")
        print(f"  Log Level:     {self.log_level}")


# ══════════════════════════════════════════════
# SINGLETON — Import this everywhere
# ══════════════════════════════════════════════

config = PipelineConfig()


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    config.print_status()
    print(f"\n  Media type test:")
    print(f"    'ad.jpg'  → {config.get_media_type('ad.jpg')}")
    print(f"    'ad.mp4'  → {config.get_media_type('ad.mp4')}")
    print(f"    'ad.mp3'  → {config.get_media_type('ad.mp3')}")
    print(f"    'ad.txt'  → {config.get_media_type('ad.txt')}")