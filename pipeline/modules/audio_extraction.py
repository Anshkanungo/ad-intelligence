"""
Ad Intelligence Pipeline — Audio Extraction Module (FFmpeg)

Extracts audio track from video files using FFmpeg.
Outputs a WAV file for Whisper transcription.

Usage:
    from pipeline.modules.audio_extraction import AudioExtractionModule
    extractor = AudioExtractionModule()
    result = extractor.extract(video_path)
"""

import os
import tempfile
import subprocess

from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class AudioExtractionModule:
    """Extracts audio from video files using FFmpeg."""

    @log_execution_time
    def extract(self, video_path: str) -> dict:
        """
        Extract audio track from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            dict with keys:
                - audio_path: str — path to extracted WAV file
                - has_audio: bool — whether audio was found
                - duration_seconds: float — audio duration
                - error: str — error message if failed
        """
        result = {
            "audio_path": "",
            "has_audio": False,
            "duration_seconds": 0.0,
            "error": "",
        }

        try:
            if not os.path.exists(video_path):
                result["error"] = f"Video file not found: {video_path}"
                return result

            # Create temp WAV file
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav", prefix="ad_audio_"
            )
            audio_path = tmp.name
            tmp.close()

            # Find FFmpeg executable
            import shutil
            ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"

            # FFmpeg command: extract audio as 16kHz mono WAV (ideal for Whisper)
            cmd = [
                ffmpeg_cmd,
                "-i", video_path,
                "-vn",                  # No video
                "-acodec", "pcm_s16le", # 16-bit PCM
                "-ar", "16000",         # 16kHz sample rate (Whisper optimal)
                "-ac", "1",             # Mono
                "-y",                   # Overwrite
                audio_path,
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if proc.returncode != 0:
                # Check if it's because there's no audio stream
                if "does not contain any stream" in proc.stderr:
                    logger.warning("Video has no audio track")
                    result["has_audio"] = False
                    os.unlink(audio_path)
                    return result
                else:
                    result["error"] = f"FFmpeg error: {proc.stderr[-200:]}"
                    os.unlink(audio_path)
                    return result

            # Verify output file exists and has content
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                result["audio_path"] = audio_path
                result["has_audio"] = True

                # Get duration
                duration = self._get_duration(audio_path)
                if duration:
                    result["duration_seconds"] = duration

                logger.info(
                    f"Audio extracted: {result['duration_seconds']}s → {audio_path}"
                )
            else:
                result["error"] = "Audio extraction produced empty file"
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

        except subprocess.TimeoutExpired:
            result["error"] = "FFmpeg timed out (>120s)"
            logger.error(result["error"])
        except FileNotFoundError:
            result["error"] = "FFmpeg not found. Install FFmpeg first."
            logger.error(result["error"])
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Audio extraction failed: {e}", exc_info=True)

        return result

    def _get_duration(self, audio_path: str) -> float | None:
        """Get audio duration using FFprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    audio_path,
                ],
                capture_output=True, text=True, timeout=10,
            )
            return round(float(result.stdout.strip()), 2)
        except Exception:
            return None

    @staticmethod
    def cleanup(audio_path: str):
        """Delete the extracted audio temp file."""
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
                logger.debug(f"Cleaned up: {audio_path}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import shutil
    from pathlib import Path

    print("=" * 50)
    print("Audio Extraction Module — Quick Test")
    print("=" * 50)

    # Check FFmpeg
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(f"✓ FFmpeg found: {ffmpeg}")
    else:
        print("✗ FFmpeg NOT found — install it first")

    extractor = AudioExtractionModule()
    print("✓ Module initialized")

    # Look for test video
    test_videos = list(Path("tests/sample_ads").glob("*.mp4"))
    test_videos += list(Path("tests/sample_ads").glob("*.mov"))

    if test_videos:
        video = test_videos[0]
        print(f"\nProcessing: {video}")
        result = extractor.extract(str(video))

        print(f"  Has audio: {result['has_audio']}")
        print(f"  Duration: {result['duration_seconds']}s")
        print(f"  Audio path: {result['audio_path']}")
        if result["error"]:
            print(f"  Error: {result['error']}")

        # Cleanup
        if result["audio_path"]:
            extractor.cleanup(result["audio_path"])
            print("  Cleaned up temp file")
    else:
        print("\n⚠ No test video. Drop a .mp4 into tests/sample_ads/ to test")

    print(f"\n{'=' * 50}")
    print("Audio extraction module ready!")
    print(f"{'=' * 50}")