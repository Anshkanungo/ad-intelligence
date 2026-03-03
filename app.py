"""
Ad Intelligence Pipeline — Streamlit UI
Upload any ad (image, video, audio) → Get structured JSON analysis.
"""

import json
import streamlit as st
from pathlib import Path

from pipeline.orchestrator import AdIntelligencePipeline
from utils.config import config

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════

st.set_page_config(
    page_title="Ad Intelligence Pipeline",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-top: 0;
    }
    .metric-card {
        background: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
    }
    .stDownloadButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# VIDEO URL DOWNLOAD HELPER
# ══════════════════════════════════════════════

def _download_video(url: str):
    """Download video from YouTube/URL using yt-dlp. Returns a file-like object."""
    import tempfile
    import subprocess
    import os

    try:
        # Create temp file for download
        tmp_dir = tempfile.mkdtemp(prefix="ad_dl_")
        output_path = os.path.join(tmp_dir, "video.mp4")

        # Find FFmpeg path explicitly
        import shutil
        ffmpeg_path = shutil.which("ffmpeg")
        
        # yt-dlp command: download best video+audio up to 1080p, merge with ffmpeg
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--format", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "--output", output_path,
            "--merge-output-format", "mp4",
            "--socket-timeout", "30",
        ]
        
        # Tell yt-dlp where FFmpeg is
        if ffmpeg_path:
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            cmd.extend(["--ffmpeg-location", ffmpeg_dir])
        
        cmd.append(url)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            st.error(f"Download error: {result.stderr[-300:]}")
            return None

        # Find the actual downloaded file (yt-dlp may change extension)
        downloaded = None
        for f in os.listdir(tmp_dir):
            if f.endswith((".mp4", ".webm", ".mkv")):
                downloaded = os.path.join(tmp_dir, f)
                break

        if not downloaded or not os.path.exists(downloaded):
            return None

        # Create a file-like object that mimics Streamlit's UploadedFile
        class DownloadedFile:
            def __init__(self, path):
                self.name = os.path.basename(path)
                self._path = path
                self._data = open(path, "rb").read()
                self.size = len(self._data)
                self._pos = 0

            def read(self):
                return self._data

            def seek(self, pos):
                self._pos = pos

        return DownloadedFile(downloaded)

    except subprocess.TimeoutExpired:
        st.error("Download timed out (>2 min). Try a shorter video.")
        return None
    except FileNotFoundError:
        st.error("yt-dlp not found. Install it: `pip install yt-dlp`")
        return None
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Pipeline Status")

    keys = config.validate_keys()
    st.markdown(f"**Groq LLM:** {'🟢 Ready' if keys['groq'] else '🔴 Missing'}")
    st.markdown(f"**Gemini Fallback:** {'🟢 Ready' if keys['gemini'] else '🟡 Not set'}")
    st.markdown(f"**HuggingFace:** {'🟢 Ready' if keys['huggingface'] else '🟡 Not set'}")

    st.divider()

    st.markdown("### 📋 Supported Formats")
    st.markdown("""
    **Images:** JPG, PNG, WebP, BMP, TIFF  
    **Videos:** MP4, MOV, AVI, WebM, MKV  
    **Audio:** MP3, WAV, M4A, OGG, FLAC
    """)

    st.divider()

    st.markdown("### 📊 JSON Schema")
    st.markdown("""
    13 fixed sections, always present:  
    `_meta` · `source` · `brand` · `product`  
    `text_content` · `language` · `audio`  
    `visual_analysis` · `video_analysis`  
    `classification` · `compliance_and_legal`  
    `engagement_elements` · `ad_description`
    """)

    st.divider()
    st.caption("v1.0.0 · 100% Open Source")

# ══════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════

st.markdown('<p class="main-header">🎯 Ad Intelligence Pipeline</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload any advertisement → Get comprehensive JSON analysis</p>', unsafe_allow_html=True)

# ── Input Method ──
input_tab1, input_tab2 = st.tabs(["📁 Upload File", "🔗 YouTube / Video URL"])

uploaded_file = None

with input_tab1:
    direct_upload = st.file_uploader(
        "Drop your ad here",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif",
              "mp4", "mov", "avi", "webm", "mkv",
              "mp3", "wav", "m4a", "ogg", "flac", "aac"],
        help="Supports images, videos, and audio files",
    )
    if direct_upload:
        uploaded_file = direct_upload

with input_tab2:
    video_url = st.text_input(
        "Paste YouTube or video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    if video_url and st.button("⬇️ Download Video"):
        with st.spinner("Downloading video..."):
            downloaded = _download_video(video_url)
            if downloaded:
                st.session_state["downloaded_file"] = downloaded
                st.success(f"Downloaded: {downloaded.name} ({round(downloaded.size / (1024*1024), 1)} MB)")
            else:
                st.error("Failed to download video. Check the URL and try again.")

    # Use previously downloaded file
    if "downloaded_file" in st.session_state:
        uploaded_file = st.session_state["downloaded_file"]

if uploaded_file is not None:
    # Show file info
    file_ext = Path(uploaded_file.name).suffix.lower()
    media_type = config.get_media_type(uploaded_file.name)
    file_size = getattr(uploaded_file, 'size', 0)
    file_size_mb = round(file_size / (1024 * 1024), 2) if file_size else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("File", uploaded_file.name)
    col2.metric("Type", media_type.upper())
    col3.metric("Size", f"{file_size_mb} MB")

    # Preview
    if media_type == "image":
        st.image(uploaded_file, caption="Uploaded Ad", width="stretch")
    elif media_type == "video":
        if hasattr(uploaded_file, '_data'):
            st.video(uploaded_file._data)
        else:
            st.video(uploaded_file)
    elif media_type == "audio":
        if hasattr(uploaded_file, '_data'):
            st.audio(uploaded_file._data)
        else:
            st.audio(uploaded_file)

    # Reset file position after preview
    uploaded_file.seek(0)

    # ── Run Pipeline ──
    if st.button("🚀 Analyze Ad", type="primary"):

        # Check API keys
        if not keys["groq"] and not keys["gemini"]:
            st.error("❌ No LLM API key configured. Set GROQ_API_KEY or GEMINI_API_KEY in .env")
            st.stop()

        # Progress bar
        progress_bar = st.progress(0, text="Starting pipeline...")
        status_text = st.empty()

        def update_progress(step, total, message):
            progress_bar.progress(step / total, text=message)
            status_text.text(f"Step {step}/{total}: {message}")

        # Run pipeline
        with st.spinner(""):
            pipeline = AdIntelligencePipeline()
            result = pipeline.run(uploaded_file, progress_callback=update_progress)

        progress_bar.progress(1.0, text="Complete!")
        status_text.empty()

        # ── Results ──
        st.divider()
        st.markdown("## 📊 Results")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⏱️ Time", f"{result.meta.processing_time_sec}s")
        m2.metric("🎯 Confidence", f"{result.meta.confidence_score:.0%}")
        m3.metric("🔧 Modules", len(result.meta.modules_used))
        m4.metric("⚠️ Errors", len(result.meta.errors))

        # Show errors if any
        if result.meta.errors:
            with st.expander("⚠️ Pipeline Errors", expanded=False):
                for err in result.meta.errors:
                    st.warning(err)

        # Key findings cards
        st.markdown("### 🔍 Key Findings")

        k1, k2 = st.columns(2)
        with k1:
            st.markdown("**Brand & Product**")
            st.markdown(f"🏢 **Brand:** {result.brand.company_name or 'Not identified'}")
            st.markdown(f"📦 **Product:** {result.product.product_name or 'Not identified'}")
            st.markdown(f"🏷️ **Category:** {result.product.product_category or 'N/A'}")
            st.markdown(f"💰 **Price:** {result.product.price_mentioned or 'Not mentioned'}")
            st.markdown(f"🎁 **Offer:** {result.product.offer_or_discount or 'None'}")

        with k2:
            st.markdown("**Classification**")
            st.markdown(f"🏭 **Industry:** {result.classification.industry or 'N/A'}")
            st.markdown(f"🎯 **Objective:** {result.classification.ad_objective or 'N/A'}")
            st.markdown(f"👥 **Audience:** {result.classification.target_audience or 'N/A'}")
            st.markdown(f"🎭 **Tone:** {result.classification.tone or 'N/A'}")
            st.markdown(f"💡 **Appeal:** {result.classification.emotional_appeal or 'N/A'}")

        # Text content
        st.markdown("### 📝 Text Content")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown(f"**Headline:** {result.text_content.headline or 'N/A'}")
            st.markdown(f"**Tagline:** {result.text_content.tagline or 'N/A'}")
            st.markdown(f"**CTA:** {result.text_content.call_to_action or 'N/A'}")
        with t2:
            st.markdown(f"**Body:** {result.text_content.body_copy or 'N/A'}")
            st.markdown(f"**Contact:** {result.text_content.contact_info or 'N/A'}")
            if result.text_content.hashtags:
                st.markdown(f"**Hashtags:** {', '.join(result.text_content.hashtags)}")

        # Ad Summary
        st.markdown("### 📄 Ad Summary")
        st.info(result.ad_description.short_summary or "No summary available")
        if result.ad_description.detailed_description:
            with st.expander("Detailed Description"):
                st.write(result.ad_description.detailed_description)
        if result.ad_description.creative_strategy:
            with st.expander("Creative Strategy"):
                st.write(result.ad_description.creative_strategy)

        # Color palette
        if result.visual_analysis.dominant_colors_hex:
            st.markdown("### 🎨 Color Palette")
            color_cols = st.columns(len(result.visual_analysis.dominant_colors_hex))
            for i, color in enumerate(result.visual_analysis.dominant_colors_hex):
                with color_cols[i]:
                    st.markdown(
                        f'<div style="background:{color};height:50px;border-radius:8px;'
                        f'border:1px solid #555;"></div>'
                        f'<p style="text-align:center;font-size:0.8rem;">{color}</p>',
                        unsafe_allow_html=True,
                    )

        # Full JSON
        st.markdown("### 📋 Complete JSON Output")

        json_str = result.to_json(indent=2)

        # Download button
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name=f"ad_intelligence_{Path(uploaded_file.name).stem}.json",
            mime="application/json",
        )

        # JSON viewer
        with st.expander("View Full JSON", expanded=False):
            st.json(json.loads(json_str))

else:
    # Empty state
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; padding:60px 0; color:#666;">
            <h3>👆 Upload an advertisement to get started</h3>
            <p>Supports TV ads, radio spots, YouTube ads, magazine pages, brochures, 
            billboards, social media ads, flyers — any format!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )