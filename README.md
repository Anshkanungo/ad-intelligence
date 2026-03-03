# 🎯 Ad Intelligence Pipeline

> **Upload any advertisement — image, video, or audio — and get a comprehensive, structured JSON analysis.**

A fully autonomous, open-source pipeline that extracts brand, product, text, visual, audio, and classification intelligence from any ad format. Powered by Vision-Language Models and a multi-module extraction architecture.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Open Source](https://img.shields.io/badge/100%25-Open%20Source-brightgreen)

---

## 🚀 Live Demo

**[Try it here →](https://ad-intelligence.onrender.com)** *(hosted on Render free tier — may take 30s to cold start)*

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
  ├─ Extraction Layer (parallel modules):
  │   ├── EasyOCR ─────────── text extraction
  │   ├── YOLOv8-nano ─────── object detection
  │   ├── BLIP-1 (local) ──── scene description
  │   ├── OpenCV + KMeans ─── color analysis
  │   ├── Whisper base ────── audio transcription
  │   ├── CLIP ViT-B-32 ───── smart frame selection (video)
  │   ├── FFmpeg ──────────── audio/video extraction
  │   └── langdetect ──────── language detection
  │
  ├─ Signal Aggregator (merges all module outputs)
  │
  ├─ VLM Reasoning Engine:
  │   ├── Groq Llama 4 Scout (image + JSON schema → structured output)
  │   ├── Gemini Flash (multi-image for video)
  │   └── Groq Llama 3.3 70B (text-only fallback)
  │
  ├─ Pydantic Validator (strict fixed schema, retry on failure)
  │
  └─ Streamlit UI (display + download JSON)
```

### Image Pipeline
Upload → OCR + YOLO + Color + BLIP → Send actual image to VLM → JSON

### Video Pipeline
YouTube URL or upload → Download at 1080p → CLIP-based smart frame selection → OCR top 3 text-dense frames → Whisper transcription → Send all key frames to VLM → JSON

### Audio Pipeline
Upload → Whisper transcription → Language detection → LLM reasoning → JSON

---

## 🛠️ Tech Stack (100% Free / Open Source)

| Component | Tool | Purpose |
|-----------|------|---------|
| **Frontend** | Streamlit | Interactive UI |
| **Hosting** | Render (free) | Docker deployment |
| **OCR** | EasyOCR | Text extraction from images |
| **Object Detection** | YOLOv8-nano | Object/people detection |
| **Scene Description** | BLIP-1 (local) | Image captioning |
| **Color Analysis** | OpenCV + KMeans | Dominant color extraction |
| **Frame Selection** | CLIP ViT-B-32 | Embedding-based smart frame picking |
| **Transcription** | Whisper (base) | Speech-to-text |
| **Video Download** | yt-dlp + FFmpeg | YouTube URL support |
| **Language Detection** | langdetect | Text language identification |
| **VLM (Primary)** | Groq — Llama 4 Scout | Vision-language model (free tier) |
| **VLM (Fallback)** | Google Gemini Flash | Multi-image vision (free tier) |
| **LLM (Text)** | Groq — Llama 3.3 70B | Text-only reasoning (free tier) |
| **Validation** | Pydantic v2 | Strict JSON schema enforcement |

**No paid APIs required.** All LLM inference uses free tiers.

---

## 📁 Project Structure

```
ad-intelligence/
├── app.py                          # Streamlit entry point
├── Dockerfile                      # Docker deployment
├── render.yaml                     # Render deployment config
├── requirements.txt                # Python dependencies
├── .env.example                    # API keys template
├── README.md                       # This file
├── PROJECT_CONTEXT.md              # Development context document
│
├── pipeline/
│   ├── orchestrator.py             # Main pipeline controller
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
│   ├── media_handler.py            # File type detection & preprocessing
│   └── logger.py                   # Structured logging
│
└── tests/
    └── sample_ads/                 # Test images/videos
```

---

## ⚡ Quick Start (Local)

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/ad-intelligence.git
cd ad-intelligence
pip install -r requirements.txt
```

### 2. Get API Keys (all free, no credit card)

| Key | Where to Get It |
|-----|----------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `HF_API_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

### 3. Configure

```bash
cp .env.example .env
# Edit .env and paste your API keys
```

### 4. Install System Dependencies

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

### 5. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` and upload an ad!

---

## 🌐 Deploy to Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml`
5. Add environment variables in Render dashboard:
   - `GROQ_API_KEY`
   - `GEMINI_API_KEY`
   - `HF_API_TOKEN`
6. Deploy!

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
    "subheadline": "PLANT-BASED ENERGY DRINK"
  },
  "classification": {
    "industry": "Beverages",
    "target_audience": "Health-conscious consumers, Young adults",
    "tone": "Positive, Energetic, Refreshing",
    "themes": ["Health", "Natural ingredients", "Energy"]
  }
}
```

*(Truncated — full output has 13 sections with 100+ fields)*

---

## 🔮 Future Roadmap

- [ ] Custom logo recognition model (fine-tuned YOLOv8)
- [ ] V-JEPA 2 embeddings for temporal video understanding
- [ ] Batch processing (ZIP upload → CSV output)
- [ ] API endpoint (REST API alongside Streamlit)
- [ ] Real-time YouTube URL processing improvements
- [ ] Multi-page PDF/brochure support
- [ ] Competitive analysis (compare ads side-by-side)
- [ ] Ad effectiveness scoring

---

## 📄 License

MIT License — use it however you want.

---

## 🙏 Acknowledgments

Built with open-source tools: EasyOCR, YOLOv8, BLIP, CLIP, Whisper, Streamlit, Groq, and Google Gemini.