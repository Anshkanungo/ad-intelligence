"""
Ad Intelligence Pipeline — Transcription Module (OpenAI Whisper)

Transcribes speech from audio files using Whisper (local, base model).
Works on extracted audio from video or direct audio uploads.

Usage:
    from pipeline.modules.transcription import TranscriptionModule
    transcriber = TranscriptionModule()
    result = transcriber.transcribe("audio.wav")
"""

import whisper

from utils.config import config
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class TranscriptionModule:
    """Transcribes audio using OpenAI Whisper (local model)."""

    def __init__(self):
        self._model = None

    def _get_model(self):
        """Lazy-load Whisper model."""
        if self._model is None:
            model_size = config.whisper_model  # "base" by default
            logger.info(f"Loading Whisper model: {model_size}")
            self._model = whisper.load_model(model_size)
            logger.info("Whisper model loaded")
        return self._model

    @log_execution_time
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe speech from an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)

        Returns:
            dict with keys:
                - transcript: str — full transcript text
                - language: str — detected language code
                - segments: list[dict] — timestamped segments
                - has_speech: bool — whether speech was detected
                - duration_seconds: float — audio duration
        """
        result = {
            "transcript": "",
            "language": "",
            "segments": [],
            "has_speech": False,
            "duration_seconds": 0.0,
        }

        try:
            model = self._get_model()

            # Transcribe with language detection
            output = model.transcribe(
                audio_path,
                language=config.whisper_language,   # None = auto-detect
                verbose=False,
            )

            transcript = output.get("text", "").strip()
            language = output.get("language", "")

            if transcript:
                result["transcript"] = transcript
                result["language"] = language
                result["has_speech"] = True

                # Extract segments with timestamps
                segments = []
                for seg in output.get("segments", []):
                    segments.append({
                        "start": round(seg["start"], 2),
                        "end": round(seg["end"], 2),
                        "text": seg["text"].strip(),
                    })

                result["segments"] = segments

                # Duration from last segment
                if segments:
                    result["duration_seconds"] = segments[-1]["end"]

                logger.info(
                    f"Transcribed {len(segments)} segments, "
                    f"language: {language}, "
                    f"text: {transcript[:100]}..."
                )
            else:
                logger.warning("No speech detected in audio")

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)

        return result


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path

    print("=" * 50)
    print("Transcription Module (Whisper) — Quick Test")
    print("=" * 50)

    transcriber = TranscriptionModule()
    print("✓ Module initialized")

    # Look for test audio
    test_audio = []
    for ext in ["*.mp3", "*.wav", "*.m4a"]:
        test_audio += list(Path("tests/sample_ads").glob(ext))

    if test_audio:
        audio = test_audio[0]
        print(f"\nProcessing: {audio}")
        result = transcriber.transcribe(str(audio))

        print(f"  Has speech: {result['has_speech']}")
        print(f"  Language: {result['language']}")
        print(f"  Duration: {result['duration_seconds']}s")
        print(f"  Segments: {len(result['segments'])}")
        print(f"  Transcript: {result['transcript'][:200]}")
    else:
        print("\n⚠ No test audio. Drop a .mp3 or .wav into tests/sample_ads/")

    print(f"\n{'=' * 50}")
    print("Transcription module ready!")
    print(f"{'=' * 50}")