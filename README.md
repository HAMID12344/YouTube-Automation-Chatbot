---
title: YouTube Automation Chatbot
emoji: 🎥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# YouTube Automation Chatbot

An intelligent Video Question-Answering Chatbot built with Streamlit, FAISS, Whisper, and Retrieval-Augmented Generation (RAG). Supports both YouTube URLs and local video file uploads.

## Features

- Upload local videos (MP4, MKV, MOV, WebM, AVI)
- YouTube URL support
- Automatic audio extraction
- Whisper transcription (faster-whisper CPU INT8 with Cloud Whisper support)
- Semantic chunking (RecursiveCharacterTextSplitter 1000/200)
- Sentence Transformer embeddings (paraphrase-multilingual-MiniLM-L12-v2)
- FAISS vector search
- RAG question answering
- Local LLM fallback (Qwen GGUF / Extractive RAG fallback)
- Grounded answers (returns strictly *I couldn't find the answer to that in the video.* for out-of-scope queries)
- Video isolation (clears state, cache, and vector store between videos)

## Architecture

```
Video
  ↓
Audio Extraction
  ↓
Whisper
  ↓
Transcript
  ↓
Chunking
  ↓
Embeddings
  ↓
FAISS
  ↓
Question
  ↓
Retrieval
  ↓
LLM
  ↓
Answer
```

## Local Installation

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

4. **Set up environment variables (Optional for Cloud Features):**
   Create a `.env` file in the root directory:
   ```env
   HF_TOKEN="your_huggingface_token_here"
   HF_MODEL_ID="Qwen/Qwen2.5-Coder-32B-Instruct"
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the application in your browser (`http://localhost:8501` locally or port `7860` in Docker).
2. Choose your input mode from the sidebar:
   - **YouTube URL**: Paste any valid YouTube video link and click "Process Video".
   - **Local Video**: Drag and drop an MP4, MKV, MOV, WEBM, or AVI file.
3. Wait for audio extraction, Whisper transcription, and vector indexing to complete.
4. Ask any question in the chat bar. The system will retrieve the most relevant transcript segments and generate a grounded answer.
5. If you ask a question outside the video's scope, the chatbot responds with:
   > *I couldn't find the answer to that in the video.*

## Hugging Face Deployment

This Space is configured for Hugging Face Spaces using **Docker**:
- Uses `python:3.11-slim` with system `ffmpeg`, `git`, and build tools.
- Runs as non-root user `user` (UID `1000`) per Hugging Face Spaces standards.
- Exposes port `7860` and runs in headless mode (`0.0.0.0:7860`).
- Container startup command:
  ```bash
  streamlit run app.py --server.port=7860 --server.address=0.0.0.0
  ```

### Live Demo
To be updated after deployment.
