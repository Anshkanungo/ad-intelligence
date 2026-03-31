# 🎯 Ad Intelligence Pipeline

> **Upload any advertisement — image, video, or audio — and get a comprehensive, structured JSON analysis. Then store and search your ad library.**

A fully autonomous, open-source pipeline that extracts brand, product, text, visual, audio, and classification intelligence from any ad format. Powered by Vision-Language Models, CLIP-based frame selection, and a Qdrant-backed vector search layer.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Open Source](https://img.shields.io/badge/100%25-Open%20Source-brightgreen)

---

## 🚀 Live Demo

> **⚠️ Hosted Limitations:** The free tier has 512MB RAM, which is insufficient for CLIP-based video frame selection. **Image analysis works fully.** For video analysis with CLIP smart frame selection, please run locally (see below). YouTube URL download also requires local setup due to authentication restrictions on cloud servers.

---

## 📸 What It Does

Upload any ad and get a **fixed 13-section JSON** with:

| Section | What It Extracts |
|---------|-----------------|
| `_meta` | Processing time, confidence, modules used |
| `source` | Medium (TV/print/digital), platform, format, era |
| `brand` | Company name, logo, colors, URL, social handles |
| `product` | Name, category, features, price, offers |
| `text_content` | Headline, tagline, CTA, body copy, all raw text |
| `language` | Primary/secondary languages, multilingual flag |
| `audio` | Transcript, speech, music, voiceover tone |
| `visual_analysis` | Scene description, colors, people, objects, layout |
| `video_analysis` | Duration, scene-by-scene breakdown, pacing |
| `classification` | Industry, audience, tone, themes, emotional appeal |
| `compliance_and_legal` | Disclaimers, age restrictions, trademarks |
| `engagement_elements` | QR codes, coupons, social proof, urgency |
| `ad_description` | Summary, detailed description, creative strategy |

**Same JSON shape every time.** Missing fields are empty, never omitted.

**Store analyzed ads** in a Qdrant vector database and **search** by semantic query, brand, industry, tone, or any combination of filters.

---

## 📊 Example Output

For a 15-second OCA Energy Drink video ad:

```json
{
  "brand": {
    "company_name": "OCA",
    "logo_detected": true,
    "logo_description": "Stylized 'OCA' logo with tropical leaf motifs"
  },
  "product": {
    "product_name": "Energy Drink",
    "product_category": "Beverages",
    "product_description": "Plant-based energy drink in Berry Açai, Guava Passion Fruit, and Mango",
    "product_features": ["Plant-based", "Organic", "USDA Organic certified"]
  },
  "text_content": {
    "tagline": "TUDO BEM! (it's all good)",
    "subheadline": "PLANT-BASED ENERGY DRINK",
    "body_copy": "Made with real fruit extracts, we deliver natural energy that lasts."
  },
  "classification": {
    "industry": "Beverages",
    "target_audience": "Health-conscious consumers, Young adults",
    "tone": "Positive, Energetic, Refreshing",
    "themes": ["Health", "Natural ingredients", "Energy", "Fruit flavors"]
  }
}
```

*(Truncated — full output has 13 sections with 100+ fields)*

---


## 🏗️ Architecture

```
INPUT (image / video / audio)
  │
  ├─ Media Router (detects type)
  │
  ├─ Extraction Layer:
  │   ├── CLIP ViT-B-32 ───── smart frame selection (video, local only)
  │   ├── OpenCV + KMeans ─── color analysis
  │   ├── Whisper base ────── audio transcription (local only)
  │   ├── FFmpeg ──────────── audio/video extraction
  │   └── langdetect ──────── language detection
  │
  ├─ VLM Reasoning Engine:
  │   ├── Groq Llama 4 Scout ── vision analysis (sees actual images/frames)
  │   ├── Groq Llama 3.3 70B ── text-only fallback / reasoning
  │   └── Google Gemini Flash ── VLM fallback (limited quota)
  │
  ├─ Pydantic v2 Validator (strict fixed schema, retry on failure)
  │
  ├─ Vector Store (Qdrant + sentence-transformers):
  │   ├── Semantic search (all-MiniLM-L6-v2 embeddings)
  │   ├── Indexed metadata filters (brand, industry, tone, etc.)
  │   └── Hybrid search (filters + semantic ranking)
  │
  └─ Streamlit UI:
      ├── 🎬 Analyze tab — upload → pipeline → JSON → store
      └── 🔍 Search tab — filter + semantic search across stored ads
```

### Pipeline Flows

**Image:** Image → Groq VLM (Llama 4 Scout) → JSON

**Video:** Whisper transcription → CLIP smart frame selection (4 frames) → 1 reference image + 4 frames + transcript → Groq VLM → JSON

**Audio:** Whisper transcription → Groq LLM → JSON

### Two Modes

| | Cloud (Render) | Local (Full) |
|--|----------------|-------------|
| **Image** | Image → Groq VLM → JSON | Image → Groq VLM → JSON |
| **Video** | Frames (interval) → Groq VLM → JSON | CLIP frame selection → Groq VLM → JSON |
| **Audio** | Not available (Whisper requires local) | Whisper → Groq LLM → JSON |
| **Search** | Qdrant + semantic search | Qdrant + semantic search |
| **RAM** | ~300MB | ~2GB+ |

---

## 🎬 Working with Video Ads (Local Setup Recommended)

The best video analysis experience requires local setup for two reasons:
1. **CLIP model** (600MB) needs more than 512MB RAM for smart frame selection
2. **YouTube downloads** require browser authentication that cloud servers can't provide

### Download YouTube Ads Locally

**Step 1: Install yt-dlp and Deno**

```bash
pip install yt-dlp
# Windows:
winget install DenoLand.Deno
# macOS:
brew install deno
# Linux:
curl -fsSL https://deno.land/install.sh | sh
```

**Step 2: Download any YouTube video**

```bash
# Best quality up to 1080p (recommended)
yt-dlp -f "bestvideo[vcodec^=avc1][height<=1080]+bestaudio" --merge-output-format mp4 -o test_h264.mp4 "https://www.youtube.com/watch?v=VIDEO_ID"

# Quick download (auto selects best)
yt-dlp -o "my_ad.mp4" "https://www.youtube.com/watch?v=VIDEO_ID"

# List available qualities first
yt-dlp -F "https://www.youtube.com/watch?v=VIDEO_ID"
```

> **Note:** FFmpeg must be installed for yt-dlp to merge video+audio streams. Without it, you'll only get 360p.

**Step 3: Upload the downloaded .mp4 file** to the pipeline (either local or hosted version)

---

## 🛠️ Tech Stack (100% Free / Open Source)

| Component | Tool | Purpose |
|-----------|------|---------|
| **Frontend** | Streamlit | Interactive UI (Analyze + Search tabs) |
| **Hosting** | Render (free) | Docker deployment |
| **Frame Selection** | CLIP ViT-B-32 | Embedding-based smart frame picking (local) |
| **Color Analysis** | OpenCV + KMeans | Dominant color extraction |
| **Transcription** | Whisper (base) | Speech-to-text (local) |
| **Video Download** | yt-dlp + FFmpeg | YouTube URL support (local) |
| **Language Detection** | langdetect | Text language identification |
| **VLM (Primary)** | Groq — Llama 4 Scout | Vision-language model (free tier) |
| **LLM (Text)** | Groq — Llama 3.3 70B | Text-only reasoning (free tier) |
| **VLM (Fallback)** | Google Gemini Flash | Multi-image vision (free tier) |
| **Validation** | Pydantic v2 | Strict JSON schema enforcement |
| **Vector Store** | Qdrant (local file-based) | Ad storage + search |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dim semantic embeddings (CPU) |
| **Embedding Search** | sentence-transformers | Encode queries + documents |

**No paid APIs required.** All LLM inference uses free tiers.

---

## 📁 Project Structure

```
ad-intelligence/
├── app.py                          # Streamlit entry point + pipeline logic
├── Dockerfile                      # Docker deployment
├── render.yaml                     # Render config
├── requirements.txt                # Dependencies (cloud/lightweight)
├── .env.example                    # API keys template
├── README.md
│
├── pipeline/
│   ├── orchestrator.py             # Legacy local-GPU pipeline (Qwen-based, unused)
│   ├── aggregator.py               # Merges module outputs into context
│   ├── vector_store.py             # Qdrant vector store (ingest, search, hybrid)
│   ├── modules/
│   │   ├── color_analysis.py       # OpenCV + KMeans colors
│   │   ├── transcription.py        # Whisper wrapper
│   │   ├── frame_extraction.py     # CLIP-based frame selection
│   │   ├── audio_extraction.py     # FFmpeg wrapper
│   │   └── language_detection.py   # langdetect wrapper
│   └── reasoning/
│       ├── llm_engine.py           # Groq / Gemini API calls
│       └── prompt_builder.py       # LLM prompt construction
│
├── schema/
│   └── ad_schema.py                # Pydantic v2 fixed schema (13 sections)
│
├── data/
│   └── qdrant_db/                  # Local Qdrant persistent storage
│
├── docs/
│   ├── architecture.md             # Architecture documentation
│   └── future_training_plan.md     # Training roadmap
│
├── utils/
│   ├── config.py                   # Environment config loader
│   ├── media_handler.py            # File detection & preprocessing
│   └── logger.py                   # Structured logging
│
└── tests/
    ├── sample_ads/                 # Test media files
    ├── test_pipeline.py            # Pipeline integration tests
    ├── test_schema.py              # Schema validation tests
    ├── test_vector_store.py        # Vector store ingestion + search tests
    ├── test_video_pipeline.py      # Standalone video pipeline test (CLIP → Groq)
    ├── test_ocr.py                 # Legacy OCR tests
    ├── ocr_module.py               # Legacy EasyOCR module (archived)
    └── scene_description.py        # Legacy BLIP module (archived)
```

---

## ⚡ Quick Start (Local — Full Mode)

### 1. Clone & Install

```bash
git clone https://github.com/Anshkanungo/ad-intelligence.git
cd ad-intelligence
pip install -r requirements.txt
```

For local video/audio processing, also install ML dependencies:

```bash
pip install torch torchvision torchaudio
pip install openai-whisper open-clip-torch sentence-transformers qdrant-client
```

### 2. Install System Dependencies

```bash
# FFmpeg (required for video/audio)
# Windows: winget install Gyan.FFmpeg
# macOS:   brew install ffmpeg
# Ubuntu:  sudo apt install ffmpeg

# Deno (required for YouTube downloads)
# Windows: winget install DenoLand.Deno
# macOS:   brew install deno
# Ubuntu:  curl -fsSL https://deno.land/install.sh | sh
```

### 3. Get Free API Keys (no credit card needed)

| Key | Where to Get It |
|-----|----------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `HF_API_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

### 4. Configure

```bash
cp .env.example .env
# Edit .env and paste your API keys
```

### 5. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` — upload any ad image, video, or audio file!

---

## 🔍 Vector Search

The pipeline includes a built-in ad database powered by Qdrant:

**Store:** After analyzing an ad, click "Store Ad" to save it with a link/identifier.

**Search:** Switch to the Search tab to query your ad library:
- **Semantic search** — describe what you're looking for in natural language (e.g., "emotional family ad", "premium tech product")
- **Metadata filters** — filter by brand, industry, ad style, tone, or product category
- **Hybrid** — combine semantic queries with exact filters for precise results

The search layer uses `all-MiniLM-L6-v2` embeddings (384-dim, CPU-friendly) for semantic matching, with indexed payload fields for O(1) exact filtering.

---

## 🔮 Future Roadmap

- [ ] Lighter embedding model for cloud video frame selection (MobileCLIP / SigLIP)
- [ ] V-JEPA 2 for temporal video understanding
- [ ] Company name normalization for reliable brand filtering
- [ ] Qdrant server mode (Docker) for payload index support
- [ ] MongoDB integration for full ad data storage (Qdrant for embeddings only)
- [ ] Batch processing (ZIP upload → CSV output)
- [ ] REST API endpoint alongside Streamlit
- [ ] Multi-page PDF/brochure support
- [ ] Competitive analysis (compare ads side-by-side)
- [ ] Ad effectiveness scoring

---

## 📄 License

MIT License — use it however you want.

---

## 🙏 Acknowledgments

Built with open-source tools: CLIP, Whisper, Streamlit, Groq, Google Gemini, Qdrant, and sentence-transformers.
