# 🎯 Ad Intelligence Pipeline

> **Upload any advertisement — image, video, or audio — and get a comprehensive, structured JSON analysis.**

A fully autonomous, open-source pipeline that extracts brand, product, text, visual, audio, and classification intelligence from any ad format. Powered by Vision-Language Models and a multi-module extraction architecture.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Open Source](https://img.shields.io/badge/100%25-Open%20Source-brightgreen)

---

## 🚀 Live Demo

**[Try it here →](https://ad-intelligence-1.onrender.com/)** *(hosted on Render free tier — may take 30s to cold start)*

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

---

## 🏗️ Architecture

```
INPUT (image / video / audio / YouTube URL)
  │
  ├─ Media Router (detects type)
  │
  ├─ Extraction Layer:
  │   ├── CLIP ViT-B-32 ───── smart frame selection (video, local only)
  │   ├── OpenCV + KMeans ─── color analysis
  │   ├── Whisper base ────── audio transcription
  │   ├── FFmpeg ──────────── audio/video extraction
  │   ├── langdetect ──────── language detection
  │   └── EasyOCR ─────────── text extraction (local full mode)
  │
  ├─ VLM Reasoning Engine:
  │   ├── Groq Llama 4 Scout ── image analysis (sees actual image)
  │   ├── Groq multi-frame ──── video (describes each frame individually)
  │   └── Groq Llama 3.3 70B ── text-only fallback / final reasoning
  │
  ├─ Pydantic Validator (strict fixed schema, retry on failure)
  │
  └─ Streamlit UI (display + download JSON)
```

### Two Modes

| | Cloud (Render) | Local (Full) |
|--|----------------|-------------|
| **Image** | Image → VLM → JSON | Image + OCR + YOLO + BLIP + Color → VLM → JSON |
| **Video** | Frames (interval) → VLM → JSON | CLIP frame selection → VLM → JSON |
| **Audio** | Whisper → LLM → JSON | Whisper → LLM → JSON |
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
| **Frontend** | Streamlit | Interactive UI |
| **Hosting** | Render (free) | Docker deployment |
| **Frame Selection** | CLIP ViT-B-32 | Embedding-based smart frame picking (local) |
| **Color Analysis** | OpenCV + KMeans | Dominant color extraction |
| **Transcription** | Whisper (base) | Speech-to-text |
| **OCR** | EasyOCR | Text extraction (local full mode) |
| **Object Detection** | YOLOv8-nano | Object/people detection (local full mode) |
| **Scene Description** | BLIP-1 (local) | Image captioning (local full mode) |
| **Video Download** | yt-dlp + FFmpeg | YouTube URL support (local) |
| **Language Detection** | langdetect | Text language identification |
| **VLM (Primary)** | Groq — Llama 4 Scout | Vision-language model (free tier) |
| **LLM (Text)** | Groq — Llama 3.3 70B | Text-only reasoning (free tier) |
| **VLM (Fallback)** | Google Gemini Flash | Multi-image vision (free tier) |
| **Validation** | Pydantic v2 | Strict JSON schema enforcement |

**No paid APIs required.** All LLM inference uses free tiers.

---

## 📁 Project Structure

```
ad-intelligence/
├── app.py                          # Streamlit entry point
├── Dockerfile                      # Docker deployment
├── render.yaml                     # Render config
├── requirements.txt                # Dependencies
├── .env.example                    # API keys template
├── README.md
│
├── pipeline/
│   ├── orchestrator.py             # Main pipeline controller (auto lightweight/full)
│   ├── aggregator.py               # Merges all module outputs
│   ├── modules/
│   │   ├── ocr_module.py           # EasyOCR wrapper
│   │   ├── object_detection.py     # YOLOv8-nano wrapper
│   │   ├── scene_description.py    # BLIP-1 local captioning
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
├── utils/
│   ├── config.py                   # Environment config loader
│   ├── media_handler.py            # File detection & preprocessing
│   └── logger.py                   # Structured logging
│
└── tests/
    └── sample_ads/                 # Test files
```

---

## ⚡ Quick Start (Local — Full Mode)

### 1. Clone & Install

```bash
git clone https://github.com/Anshkanungo/ad-intelligence.git
cd ad-intelligence
pip install -r requirements.txt
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

## 🔮 Future Roadmap

- [ ] Lighter embedding model for cloud video frame selection (MobileCLIP / SigLIP)
- [ ] V-JEPA 2 for temporal video understanding
- [ ] Custom logo recognition (fine-tuned YOLOv8 on ad datasets)
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

Built with open-source tools: CLIP, Whisper, EasyOCR, YOLOv8, BLIP, Streamlit, Groq, and Google Gemini.