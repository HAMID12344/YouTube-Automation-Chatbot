---
title: YouTube Automation Chatbot — Multimodal Video RAG
emoji: 🎥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# YouTube Automation Chatbot — Multimodal Video RAG

An intelligent Multimodal Video Question-Answering Chatbot built with Streamlit, FAISS, Whisper, Vision-Language Models (VLM), and Retrieval-Augmented Generation (RAG). The system understands **both what is spoken** (via speech transcription) and **what is visually shown** (via real video frame extraction and visual analysis) in uploaded videos and YouTube links.

## System Architecture

```
                 VIDEO / YOUTUBE
                       |
             +---------+---------+
             |                   |
           AUDIO               FRAMES
             |                   |
          WHISPER          QWEN2.5-VL
             |                   |
       TRANSCRIPT          VISUAL DESCRIPTIONS
             |                   |
     Transcript FAISS      Visual FAISS
             |                   |
             +---------+---------+
                       |
               QUESTION ROUTER
                       |
          +------------+------------+
          |            |            |
         TEXT        VISUAL     MULTIMODAL
          |            |            |
       Text RAG     Visual RAG   Both RAG
          |            |            |
          +------------+------------+
                       |
                    QWEN LLM
                       |
                 FINAL ANSWER
```

### Retrieval & Question Routing

User queries are dynamically classified and routed:
- **Audio / Transcript Questions** (*"What did the speaker say?", "Who was mentioned?"*): Retrieved primarily from the Transcript FAISS knowledge base.
- **Visual Questions** (*"What is visible in the video?", "What color is the car?", "What object is on the table?"*): Retrieved primarily from the Visual FAISS knowledge base with exact frame timestamps.
- **Multimodal Questions** (*"What was the person doing while talking about X?"*): Retrieves both transcript segments and visual descriptions to synthesize a correlated response.

### Strict Grounding & Zero Hallucination

The assistant strictly follows grounding rules:
- Answers only using supplied transcript and visual evidence.
- Never guesses or invents external facts.
- Exact fallback when evidence does not support the query:
  > `I couldn't find the answer to that in the video.`

---

## Features

- **Input Support:**
  - Local video file uploads (`MP4`, `MOV`, `MKV`, `AVI`, `WEBM`).
  - YouTube URLs (native caption priority, with audio and transcription fallback).
- **Speech Understanding:**
  - Speech-to-text via `faster-whisper` (CPU INT8 base model, cached) with cloud Whisper (`openai/whisper-large-v3`) support.
  - Semantic chunking with `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200).
- **Visual Understanding:**
  - Real representative frame extraction at configurable intervals (`FRAME_INTERVAL_SECONDS`, default 5s).
  - Vision-Language Model inference via `VISION_MODEL_NAME` (`Qwen/Qwen2.5-VL-3B-Instruct`) with resilient local visual scene fallback.
  - Video timestamps associated with every frame record (`[00:01:23]`).
- **Multimodal Knowledge Base:**
  - Text FAISS vector store for speech transcript chunks.
  - Visual FAISS vector store for visual scene descriptions.
  - Multilingual Sentence Transformers (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
- **Video Isolation & Lifecycle:**
  - Complete state isolation between videos: uploading Video B cleanly purges Video A's transcript, frames, vector stores, and conversation.
  - Ingests and processes each video once; subsequent questions reuse cached vector stores without re-transcription or re-extraction.
  - Temporary files (`.mp4`, `.mp3`, `.wav`, extracted frames) are automatically cleaned up.

---

## Configuration & Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | *None* | Hugging Face API Token (optional for cloud inference). |
| `HF_MODEL_ID` | `Qwen/Qwen2.5-Coder-32B-Instruct` | Primary cloud LLM for grounded multimodal generation. |
| `VISION_MODEL_NAME` | `Qwen/Qwen2.5-VL-3B-Instruct` | Vision-Language model for analyzing extracted video frames. |
| `WHISPER_MODEL` | `openai/whisper-large-v3` | Cloud Whisper model. |
| `FRAME_INTERVAL_SECONDS` | `5` | Sampling interval in seconds for representative frame extraction. |
| `FORCE_LOCAL_LLM` | `0` | Set to `1` to bypass cloud LLM and use local GGUF / extractive fallback. |
| `FORCE_LOCAL_WHISPER` | `0` | Set to `1` to force local faster-whisper. |

---

## Local Installation & Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HAMID12344/YouTube-Automation-Chatbot.git
   cd YouTube-Automation-Chatbot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch Streamlit:**
   ```bash
   streamlit run app.py
   ```

5. **Run Automated Test Suite:**
   ```bash
   python test_multimodal_pipeline.py
   ```

---

## Docker & Hugging Face Spaces Deployment

The application is fully containerized and compatible with Hugging Face Spaces Docker runtime.

- **Base Image:** `python:3.11-slim`
- **Exposed Port:** `7860`
- **Non-Root User:** `user` (UID `1000`)
- **Startup Command:**
  ```bash
  streamlit run app.py --server.port=7860 --server.address=0.0.0.0
  ```

### Build & Run Locally with Docker:
```bash
docker build -t youtube-automation-chatbot .
docker run -p 7860:7860 youtube-automation-chatbot
```
Access the application at `http://localhost:7860`.
