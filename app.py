"""
Ad Intelligence — Streamlit App

Tabs:
  1. Analyze — Upload video/image → pipeline → JSON → optionally store in vector DB
  2. Search  — Filter + semantic search across stored ads

Run: streamlit run app.py
"""

import io
import json
import time
import base64
import tempfile
import torch
import streamlit as st
from pathlib import Path
from PIL import Image

from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Ad Intelligence", page_icon="📊", layout="wide")

VIDEO_EXTENSIONS = ["mp4", "mov", "avi", "webm", "mkv"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]


# ═══════════════════════════════════════════
# VECTOR STORE (cached singleton)
# ═══════════════════════════════════════════

@st.cache_resource
def get_vector_store():
    """Load vector store once, reuse across reruns."""
    from pipeline.vector_store import AdVectorStore
    return AdVectorStore()


# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════

def save_temp_video(uploaded_file) -> str:
    """Save uploaded video to temp file, return path."""
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    uploaded_file.seek(0)
    return tmp.name


def encode_image(pil_img: Image.Image) -> str:
    """Encode PIL image to base64 JPEG for Groq."""
    if max(pil_img.size) > 1024:
        pil_img = pil_img.copy()
        pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ═══════════════════════════════════════════
# VIDEO PIPELINE
# ═══════════════════════════════════════════

def run_whisper(video_path: str) -> dict:
    """Extract audio and transcribe."""
    from pipeline.modules.audio_extraction import AudioExtractionModule
    from pipeline.modules.transcription import TranscriptionModule

    audio_mod = AudioExtractionModule()
    audio_result = audio_mod.extract(video_path)

    if not audio_result or not audio_result.get("has_audio"):
        return {"transcript": "", "whisper_result": None}

    whisper_mod = TranscriptionModule()
    whisper_result = whisper_mod.transcribe(audio_result["audio_path"])
    audio_mod.cleanup(audio_result["audio_path"])

    transcript = ""
    if whisper_result and whisper_result.get("transcript"):
        transcript = whisper_result["transcript"]

    return {"transcript": transcript, "whisper_result": whisper_result}


def run_frame_selection(video_path: str, ref_images: list[Image.Image]) -> list[dict]:
    """CLIP smart frame selection."""
    from pipeline.modules.frame_extraction import FrameExtractionModule

    frame_mod = FrameExtractionModule()
    result = frame_mod.extract(video_path, preview_images=ref_images or None)
    frame_mod.unload()
    torch.cuda.empty_cache()

    if not result or not result.get("frames"):
        return []

    return result["frames"]


def run_groq_video(
    frames: list[dict],
    briefing: dict,
    ref_images: list[Image.Image],
) -> str:
    """Send ref image + frames + transcript to Groq for video analysis."""
    from groq import Groq
    from schema.ad_schema import get_schema_json_template
    from pipeline.reasoning.prompt_builder import SYSTEM_PROMPT

    transcript = briefing.get("transcript", "")
    segments = []
    if briefing.get("whisper_result") and briefing["whisper_result"].get("segments"):
        segments = briefing["whisper_result"]["segments"]

    frame_descs = []
    for i, f in enumerate(frames):
        ts = f.get("timestamp_sec", 0)
        bucket = f.get("bucket", "unknown")
        psim = f.get("preview_similarity", 0)

        nearby_text = ""
        for seg in segments:
            if abs(seg["start"] - ts) < 3.0:
                nearby_text += seg["text"] + " "

        desc = f"Frame {i+1} (t={ts}s, type={bucket}, product_match={psim:.2f})"
        if nearby_text.strip():
            desc += f' — Audio: "{nearby_text.strip()[:150]}"'
        frame_descs.append(desc)

    context = f"""=== AD ANALYSIS SIGNALS ===
Media Type: VIDEO

--- FRAME SELECTION INFO ---
{chr(10).join(frame_descs)}

--- AUDIO TRANSCRIPTION ---
{transcript if transcript else "No speech detected."}

--- TIMESTAMPS ---
{chr(10).join(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}" for seg in segments[:25]) if segments else "No segments."}

=== END OF SIGNALS ==="""

    schema_template = get_schema_json_template()
    system_prompt = SYSTEM_PROMPT + "\n" + schema_template

    image_list = []
    if ref_images:
        image_list.append("Image 1: Reference image (product/service being advertised)")
    for i in range(len(frames)):
        bucket = frames[i].get("bucket", "")
        ts = frames[i].get("timestamp_sec", "?")
        img_num = (1 if ref_images else 0) + i + 1
        image_list.append(f"Image {img_num}: Video frame at t={ts}s ({bucket} shot)")

    user_content = [
        {"type": "text", "text": f"""Analyze this video advertisement and return the complete JSON.

The attached images are:
{chr(10).join(image_list)}

{context}

INSTRUCTIONS:
1. {"The reference image shows the ACTUAL PRODUCT/SERVICE being advertised — use it to identify what's being sold." if ref_images else "No reference image provided — infer the product from the video frames and transcript."}
2. Examine ALL video frames for text, brand logos, taglines, people, settings, and story
3. Use the audio transcript to understand the ad's narrative and emotional tone
4. The ad might be about a SERVICE (not just a physical product) — look at context clues
5. Fill the video_analysis.scenes field with descriptions of what each frame shows
6. Infer industry, target audience, tone, and themes from the complete picture

Return ONLY the complete JSON object following the exact schema."""},
    ]

    if ref_images:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(ref_images[0])}"},
        })

    import cv2
    for f in frames:
        pil_img = Image.fromarray(cv2.cvtColor(f["image"], cv2.COLOR_BGR2RGB))
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(pil_img)}"},
        })

    client = Groq(api_key=config.groq_api_key)
    response = client.chat.completions.create(
        model=config.groq_vision_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content


# ═══════════════════════════════════════════
# IMAGE PIPELINE
# ═══════════════════════════════════════════

def run_groq_image(images: list[Image.Image]) -> str:
    """Send up to 5 images directly to Groq for static ad analysis."""
    from groq import Groq
    from schema.ad_schema import get_schema_json_template
    from pipeline.reasoning.prompt_builder import SYSTEM_PROMPT

    schema_template = get_schema_json_template()
    system_prompt = SYSTEM_PROMPT + "\n" + schema_template

    image_guide = "\n".join(f"Image {i+1}: Ad image" for i in range(len(images)))

    user_content = [
        {"type": "text", "text": f"""Analyze this advertisement and return the complete JSON.

The attached images are:
{image_guide}

=== AD ANALYSIS SIGNALS ===
Media Type: IMAGE
Number of images: {len(images)}
=== END OF SIGNALS ===

INSTRUCTIONS:
1. Read ALL visible text (brand names, product names, taglines, URLs, prices, CTAs)
2. Identify the brand from logos and text on the product/packaging
3. Describe the scene, people, objects, and setting
4. Infer industry, target audience, tone, and themes
5. If multiple images are provided, they may show different angles or variations of the same ad

Return ONLY the complete JSON object following the exact schema."""},
    ]

    for img in images[:5]:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img)}"},
        })

    client = Groq(api_key=config.groq_api_key)
    response = client.chat.completions.create(
        model=config.groq_vision_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content


# ═══════════════════════════════════════════
# DISPLAY RESULTS + STORE OPTION
# ═══════════════════════════════════════════

def display_results(raw_json: str, elapsed: float):
    """Parse, validate, display JSON results, and offer to store in vector DB."""
    # Persist results in session state so they survive reruns
    st.session_state["last_raw_json"] = raw_json
    st.session_state["last_elapsed"] = elapsed
    _render_results()


def _render_results():
    """Render results from session state (survives Streamlit reruns)."""
    raw_json = st.session_state.get("last_raw_json")
    elapsed = st.session_state.get("last_elapsed", 0)

    if not raw_json:
        return

    st.divider()
    st.subheader(f"Results ({elapsed:.1f}s total)")

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        st.code(raw_json[:2000], language="json")
        return

    from schema.ad_schema import validate_output
    is_valid, model, err = validate_output(data)

    if is_valid and model:
        st.success("Schema validation passed")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Brand", model.brand.company_name or "—")
            st.metric("Product", model.product.product_name or "—")
            st.metric("Category", model.product.product_category or "—")
            st.metric("Industry", model.classification.industry or "—")
        with col2:
            st.metric("Tone", model.classification.tone or "—")
            st.metric("Objective", model.classification.ad_objective or "—")
            st.metric("Audience", model.classification.target_audience or "—")
            if model.text_content.tagline:
                st.metric("Tagline", model.text_content.tagline)

        if model.classification.themes:
            st.write("**Themes:**", ", ".join(model.classification.themes))

        if model.ad_description.short_summary:
            st.write("**Summary:**", model.ad_description.short_summary)

        # --- Store in Vector DB ---
        st.divider()
        st.subheader("💾 Store in Ad Database")

        video_link = st.text_input(
            "Video/Ad Link (URL or identifier)",
            placeholder="https://example.com/ad_video.mp4",
            help="A link to retrieve this ad later. In production this would be a MongoDB ID.",
            key="store_video_link",
        )

        if st.button("Store Ad", type="secondary", key="store_ad_btn"):
            if not video_link.strip():
                st.warning("Please enter a video link or identifier.")
            else:
                store = get_vector_store()
                ad_id = store.add_ad(data, video_link=video_link.strip())
                st.session_state["last_store_msg"] = f"✅ Stored! ID: `{ad_id}` — Total ads in DB: {store.count()}"

        # Show store confirmation (persists across reruns)
        if st.session_state.get("last_store_msg"):
            st.success(st.session_state["last_store_msg"])

        st.subheader("Full JSON Output")
        st.json(data)
    else:
        st.error(f"Validation failed: {err}")
        st.json(data)


# ═══════════════════════════════════════════
# SEARCH TAB
# ═══════════════════════════════════════════

def render_search_tab():
    """Render the search interface for querying stored ads."""
    store = get_vector_store()
    ad_count = store.count()

    st.header(f"🔍 Search Ad Database ({ad_count} ads)")

    if ad_count == 0:
        st.info("No ads stored yet. Go to the **Analyze** tab, process an ad, and store it.")
        return

    # --- Filters ---
    meta = store.get_all_metadata_values()

    col_search, col_filters = st.columns([2, 1])

    with col_search:
        query = st.text_input(
            "Semantic Search",
            placeholder="e.g., emotional family ad, running shoes, premium tech...",
            help="Describe what you're looking for in natural language.",
        )

    with col_filters:
        # Brand filter
        brands = sorted(meta.get("company_name", set()))
        selected_brand = st.selectbox("Brand", ["All"] + brands)

        # Industry filter
        industries = sorted(meta.get("industry", set()))
        selected_industry = st.selectbox("Industry", ["All"] + industries)

    # Second row of filters
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        styles = sorted(meta.get("ad_style", set()))
        selected_style = st.selectbox("Ad Style", ["All"] + styles)

    with col_f2:
        tones = sorted(meta.get("tone", set()))
        selected_tone = st.selectbox("Tone", ["All"] + tones)

    with col_f3:
        categories = sorted(meta.get("product_category", set()))
        selected_category = st.selectbox("Product Category", ["All"] + categories)

    n_results = st.slider("Max results", min_value=1, max_value=50, value=10)

    # --- Build filters ---
    filters = {}
    if selected_brand != "All":
        filters["company_name"] = selected_brand
    if selected_industry != "All":
        filters["industry"] = selected_industry
    if selected_style != "All":
        filters["ad_style"] = selected_style
    if selected_tone != "All":
        filters["tone"] = selected_tone
    if selected_category != "All":
        filters["product_category"] = selected_category

    # --- Search button ---
    search_clicked = st.button("🔍 Search", type="primary")

    if not search_clicked:
        # Show all ads by default
        st.caption("Click Search or adjust filters to query the database.")
        return

    # --- Execute search ---
    t0 = time.perf_counter()
    results = store.search(
        query=query.strip(),
        filters=filters if filters else None,
        n_results=n_results,
    )
    elapsed = time.perf_counter() - t0

    st.write(f"**{len(results)} results** ({elapsed*1000:.0f}ms)")

    if not results:
        st.warning("No ads match your query and filters.")
        return

    # --- Display results ---
    for i, r in enumerate(results):
        p = r["payload"]
        score = r["score"]

        brand = p.get("company_name", "Unknown")
        product = p.get("product_name", "Unknown")
        industry = p.get("industry", "")
        tone = p.get("tone", "")
        style = p.get("ad_style", "")
        summary = p.get("short_summary", "")
        video_link = p.get("video_link", "")
        tagline = p.get("tagline", "")

        with st.expander(
            f"**{brand}** — {product}  |  Score: {score:.3f}",
            expanded=(i < 3),  # Auto-expand top 3
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Industry:** {industry}")
                st.write(f"**Style:** {style}")
                st.write(f"**Tone:** {tone}")
                if tagline:
                    st.write(f"**Tagline:** {tagline}")
            with col2:
                st.write(f"**Audience:** {p.get('target_audience', '—')}")
                st.write(f"**Themes:** {p.get('themes', '—')}")
                st.write(f"**Emotional Appeal:** {p.get('emotional_appeal', '—')}")
                st.write(f"**Objective:** {p.get('ad_objective', '—')}")

            if summary:
                st.write(f"**Summary:** {summary}")

            if video_link:
                st.write(f"**Link:** [{video_link}]({video_link})")

            # Show full JSON in nested expander
            full_json = p.get("full_json", "")
            if full_json:
                with st.expander("View Full JSON"):
                    try:
                        st.json(json.loads(full_json))
                    except json.JSONDecodeError:
                        st.code(full_json, language="json")


# ═══════════════════════════════════════════
# MAIN APP — TABS
# ═══════════════════════════════════════════

st.title("📊 Ad Intelligence")

tab_analyze, tab_search = st.tabs(["🎬 Analyze", "🔍 Search"])

# ── SEARCH TAB ──
with tab_search:
    render_search_tab()

# ── ANALYZE TAB ──
with tab_analyze:
    st.caption("Upload a video ad, images, or both → get structured JSON analysis")

    # Sidebar
    with st.sidebar:
        st.header("Upload")

        video_file = st.file_uploader(
            "Video Ad (optional)",
            type=VIDEO_EXTENSIONS,
            help="A video advertisement to analyze",
        )

        image_files = st.file_uploader(
            "Image(s)",
            type=IMAGE_EXTENSIONS,
            accept_multiple_files=True,
            help="Ad images, product shots, or reference images. For video: helps find relevant frames.",
        )

        if not config.has_groq:
            st.error("Set GROQ_API_KEY in .env")

        st.divider()
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.caption(f"GPU: {gpu_name}")
        st.caption(f"Model: {config.groq_vision_model}")

        # Vector store stats
        store = get_vector_store()
        st.divider()
        st.caption(f"📦 Ads in DB: {store.count()}")

        has_input = video_file or image_files
        run_button = st.button("🚀 Analyze", type="primary", disabled=not has_input or not config.has_groq)

    # Nothing uploaded
    if not video_file and not image_files:
        st.info("Upload a video, image(s), or both in the sidebar to get started.")
        st.stop()

    # Show uploads
    uploaded_images = []
    if image_files:
        for f in image_files:
            f.seek(0)
            uploaded_images.append(Image.open(f).convert("RGB"))

    if video_file and uploaded_images:
        col_v, col_i = st.columns([2, 1])
        with col_v:
            st.video(video_file)
        with col_i:
            st.image(uploaded_images[0], caption="Reference Image", use_container_width=True)
            if len(uploaded_images) > 1:
                for img in uploaded_images[1:]:
                    st.image(img, use_container_width=True)
    elif video_file:
        st.video(video_file)
        st.caption("No reference image — using scene-change detection for frame selection")
    elif uploaded_images:
        cols = st.columns(min(len(uploaded_images), 3))
        for i, img in enumerate(uploaded_images):
            with cols[i % len(cols)]:
                st.image(img, use_container_width=True)

    if not run_button:
        # Re-render previous results if they exist (survives Store button rerun)
        if st.session_state.get("last_raw_json"):
            _render_results()
        st.stop()

    # Clear previous store message on new analysis
    st.session_state.pop("last_store_msg", None)

    # ═══════════════════════════════════════════
    # PIPELINE EXECUTION
    # ═══════════════════════════════════════════

    total_t0 = time.perf_counter()

    # ── MODE 1: Video ──
    if video_file:
        video_file.seek(0)
        video_path = save_temp_video(video_file)

        with st.status("Phase 0: Transcribing audio...", expanded=True) as status:
            t0 = time.perf_counter()
            briefing = run_whisper(video_path)
            elapsed = time.perf_counter() - t0

            transcript = briefing.get("transcript", "")
            if transcript:
                st.write(f"Transcript ({len(transcript)} chars):")
                st.text(transcript[:500] + ("..." if len(transcript) > 500 else ""))
            else:
                st.write("No speech detected")
            status.update(label=f"Phase 0: Transcription ({elapsed:.1f}s)", state="complete")

        with st.status("Phase 1: Selecting key frames...", expanded=True) as status:
            t0 = time.perf_counter()
            frames = run_frame_selection(video_path, uploaded_images)
            elapsed = time.perf_counter() - t0

            if not frames:
                st.error("No frames extracted — aborting")
                st.stop()

            st.write(f"Selected {len(frames)} frames:")
            frame_cols = st.columns(len(frames))
            import cv2
            for i, f in enumerate(frames):
                pil_img = Image.fromarray(cv2.cvtColor(f["image"], cv2.COLOR_BGR2RGB))
                with frame_cols[i]:
                    st.image(pil_img, use_container_width=True)
                    bucket = f.get("bucket", "?")
                    ts = f.get("timestamp_sec", "?")
                    psim = f.get("preview_similarity", 0)
                    st.caption(f"t={ts}s\n{bucket} | sim={psim:.2f}")

            status.update(label=f"Phase 1: Frame selection ({elapsed:.1f}s)", state="complete")

        with st.status("Phase 2: Groq VLM reasoning...", expanded=True) as status:
            t0 = time.perf_counter()
            raw_json = run_groq_video(frames, briefing, uploaded_images)
            elapsed = time.perf_counter() - t0
            status.update(label=f"Phase 2: Groq reasoning ({elapsed:.1f}s)", state="complete")

        try:
            Path(video_path).unlink()
        except Exception:
            pass

        total_elapsed = time.perf_counter() - total_t0
        display_results(raw_json, total_elapsed)

    # ── MODE 2: Image only ──
    else:
        with st.status("Analyzing image(s) with Groq VLM...", expanded=True) as status:
            t0 = time.perf_counter()
            raw_json = run_groq_image(uploaded_images)
            elapsed = time.perf_counter() - t0
            status.update(label=f"Groq analysis ({elapsed:.1f}s)", state="complete")

        total_elapsed = time.perf_counter() - total_t0
        display_results(raw_json, total_elapsed)