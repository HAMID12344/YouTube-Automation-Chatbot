---
title: YouTube Automation Chatbot
emoji: 🎥
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

# LocalMind AI — Video & YouTube Automation Chatbot (RAG)

> **Chat with YouTube videos and local video files using Retrieval-Augmented Generation (RAG)!**

An intelligent **Video Question-Answering Chatbot** built with Streamlit, FAISS, Whisper, and Dual LLM (Cloud + Local GGUF).

The application supports both YouTube links and uploaded video files (MP4, MKV, MOV, WebM, AVI), extracting audio, generating transcripts via Whisper, chunking text, indexing vectors in FAISS, and answering questions strictly grounded in the video's content—even without cloud credits or an internet connection.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 📺 **YouTube Integration** | Extract transcripts from YouTube URLs with yt-dlp and fallback caption extraction |
| 📁 **Local Video Upload** | Upload MP4, MKV, MOV, WEBM, AVI files (up to 200MB) with in-memory audio extraction |
| 🎙️ **Dual Whisper Transcription** | Hugging Face Cloud Whisper with automatic fallback to local `faster-whisper` (CPU INT8) |
| 🧠 **Semantic Chunking** | Splits transcripts using `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` |
| ⚡ **Local Embeddings** | Fast, offline 384-dimensional embeddings via `paraphrase-multilingual-MiniLM-L12-v2` |
| 🔍 **In-Memory FAISS Vector Store** | Real-time similarity retrieval of the most relevant transcript passages |
| 🤖 **Dual LLM Architecture** | Cloud LLM (Qwen2.5-72B via Hugging Face) + In-Process Local GGUF (`qwen2.5-0.5b-instruct` via `llama-cpp-python`) |
| 🛡️ **Grounding Protection** | Returns *"I couldn't find the answer to that in the video."* for out-of-scope trivia; protects valid qualifying phrases |
| 🔄 **Video Isolation** | Clear session state and vector store resets when switching between videos |

---

## 🛠️ Architecture & Pipeline

```
USER UPLOADS VIDEO / YOUTUBE URL
               ↓
     AUDIO EXTRACTION (ffmpeg)
               ↓
   WHISPER TRANSCRIPTION (Local faster-whisper / Cloud)
               ↓
     NORMALIZED TRANSCRIPT
               ↓
  TEXT CHUNKING (1000 / 200 overlap)
               ↓
 LOCAL EMBEDDINGS (SentenceTransformer)
               ↓
       FAISS VECTOR STORE
               ↓
       USER ASKS QUESTION
               ↓
     SIMILARITY RETRIEVAL (Top-k chunks)
               ↓
 DUAL LLM (Cloud Qwen / Local GGUF fallback)
               ↓
        GROUNDED ANSWER
```

---

## 📦 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/HAMID12344/YouTube-Automation-Chatbot.git
cd YouTube-Automation-Chatbot
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional)
Create a `.env` file:
```env
# Optional: Hugging Face token for cloud inference
HF_TOKEN=your_huggingface_token_here

# Optional: Force purely local processing (zero cloud credits)
FORCE_LOCAL_LLM=0
FORCE_LOCAL_WHISPER=0

# Path to local GGUF model (default: models/qwen2.5-0.5b-instruct-q4_k_m.gguf)
LOCAL_GGUF_MODEL_PATH=models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### 5. Run Streamlit Application
```bash
streamlit run app.py
```

---

## 📋 Requirements
- Python 3.10+
- `streamlit`
- `faster-whisper`
- `ctranslate2`
- `llama-cpp-python`
- `faiss-cpu`
- `sentence-transformers`
- `langchain` & `langchain-community`
- `imageio-ffmpeg`
- `yt-dlp`

---

## 👤 Author
**HAMID12344**
- GitHub: [HAMID12344](https://github.com/HAMID12344)
