import hashlib
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import imageio_ffmpeg
import yt_dlp
from huggingface_hub import InferenceClient
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    HuggingFacePipeline,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# APP CONFIG & ENVIRONMENT
# ============================================================

load_dotenv()

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Reads configuration safely from environment variables (.env) or st.secrets."""
    val = os.getenv(name)
    if val:
        return val.strip()
    try:
        val = st.secrets.get(name)
        if val:
            return str(val).strip()
    except Exception:
        pass
    return default

HF_TOKEN = (
    get_secret("HF_TOKEN")
    or get_secret("HUGGINGFACEHUB_ACCESS_TOKEN")
    or get_secret("HUGGINGFACE_API_TOKEN")
)
HF_MODEL_ID = get_secret("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
HF_LOCAL_MODEL_ID = get_secret("HF_LOCAL_MODEL_ID", "sshleifer/tiny-gpt2")
WHISPER_MODEL = get_secret("WHISPER_MODEL", "openai/whisper-large-v3")
YOUTUBE_HTTP_PROXY = get_secret("YOUTUBE_HTTP_PROXY")
LOCAL_MODEL_PATH = get_secret(
    "LOCAL_MODEL_PATH",
    r"c:\Users\ahmad\3D Objects\my local ai chat bot\models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

def render_html(html_str: str) -> None:
    """Render HTML safely without Markdown treating indented lines as code blocks."""
    clean_lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    st.markdown("\n".join(clean_lines), unsafe_allow_html=True)


st.set_page_config(
    page_title="YouTube & Video RAG AI",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark AI Dashboard Theme CSS
st.markdown(
    """
    <style>
        /* Base Background & Typography */
        .stApp {
            background: linear-gradient(180deg, #0B0F19 0%, #111827 100%);
            color: #F3F4F6;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }

        .feature-card {
            background: rgba(31, 41, 55, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .feature-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }

        /* Badges & Status */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.3);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            color: #4ADE80;
        }
        .status-dot {
            width: 7px;
            height: 7px;
            background-color: #22C55E;
            border-radius: 50%;
            box-shadow: 0 0 8px #22C55E;
        }

        .tag-pill {
            display: inline-block;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #A5B4FC;
            font-size: 12px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 6px;
        }

        /* Streamlit Input Enhancements */
        .stTextInput > div > div > input {
            background-color: rgba(31, 41, 55, 0.8) !important;
            color: #F9FAFB !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 8px !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 1px #6366F1 !important;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }

        /* Chat Message Styling */
        .stChatMessage {
            background-color: rgba(17, 24, 39, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
            margin-bottom: 12px !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0A0E17 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():
    defaults = {
        "messages": [],
        "vector_store": None,
        "retriever": None,
        "video_processed": False,
        "source_type": None,              # 'YouTube' or 'Uploaded Video'
        "video_metadata": {},
        "video_playback_source": None,             # title, duration, author, url/filename, etc.
        "transcript_text": "",
        "transcript_source": "",
        "transcript_language": "",
        "chunk_count": 0,
        "processed_identifier": "",       # video_id or sha256
        "processed_cache": {},            # identifier -> cached pipeline dict
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session_state()

# ============================================================
# VALIDATION & SECRETS HELPERS
# ============================================================

def require_hf_token():
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not configured. Add your Hugging Face token to .env."
        )

def extract_video_id(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Please enter a YouTube video URL or ID.")
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/live/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)

    raise ValueError(
        "Invalid YouTube URL or ID. Please check the URL and try again. "
        "Supported formats: youtube.com/watch?v=..., youtu.be/..., and youtube.com/shorts/..."
    )

def compute_file_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

# ============================================================
# HUGGING FACE EMBEDDINGS & LLM
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    require_hf_token()
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"token": HF_TOKEN},
        encode_kwargs={"normalize_embeddings": True},
    )

def is_hf_permission_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in [
        "403", "402", "forbidden", "payment required",
        "insufficient permissions", "inference providers",
        "credits", "credit", "depleted"
    ])

class LocalGGUFLLM:
    """Lightweight in-process GGUF LLM wrapper using llama_cpp for reliable local inference."""
    def __init__(self, model_path: str):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

    def invoke(self, prompt: str) -> str:
        res = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful and strictly factual AI video assistant. "
                        "Answer questions ONLY from the provided video context. "
                        "If the answer is not present in the context, you MUST respond exactly: "
                        "'I couldn't find the answer to that in the video.'"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.1,
        )
        return res["choices"][0]["message"]["content"].strip()

@st.cache_resource(show_spinner=False)
def get_fallback_llm():
    if LOCAL_MODEL_PATH and os.path.exists(LOCAL_MODEL_PATH):
        try:
            return LocalGGUFLLM(LOCAL_MODEL_PATH)
        except Exception:
            pass

    return HuggingFacePipeline.from_model_id(
        model_id=HF_LOCAL_MODEL_ID,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 128, "do_sample": False},
    )

@st.cache_resource(show_spinner=False)
def get_llm():
    require_hf_token()
    try:
        llm = HuggingFaceEndpoint(
            repo_id=HF_MODEL_ID,
            task="text-generation",
            huggingfacehub_api_token=HF_TOKEN,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
        )
        return ChatHuggingFace(llm=llm)
    except Exception as exc:
        if not is_hf_permission_error(exc):
            raise
        return get_fallback_llm()

# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio_from_youtube(url_or_id: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Uses yt-dlp to extract audio from a YouTube video to a temporary MP3 file.
    Returns: (audio_path, temp_dir, metadata_dict)
    """
    video_id = extract_video_id(url_or_id)
    url = f"https://www.youtube.com/watch?v={video_id}"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    temp_dir = tempfile.mkdtemp(prefix="yt_rag_")
    out_template = os.path.join(temp_dir, f"{video_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "ffmpeg_location": ffmpeg_exe,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        safe_err = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", str(exc))
        raise RuntimeError(f"Unable to process YouTube video. Please check the URL and try again. ({safe_err})") from exc

    audio_path = os.path.join(temp_dir, f"{video_id}.mp3")
    if not os.path.exists(audio_path):
        candidates = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(('.mp3', '.m4a', '.wav'))]
        if candidates:
            audio_path = candidates[0]
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError("Audio extraction completed but output audio file was not found.")

    metadata = {
        "id": video_id,
        "title": info.get("title", f"YouTube Video {video_id}"),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader", "Unknown"),
        "url": url,
    }
    return audio_path, temp_dir, metadata


def extract_audio_from_video_file(uploaded_file) -> Tuple[str, str, Dict[str, Any]]:
    """
    Uses bundled imageio-ffmpeg executable to extract audio from an uploaded video.
    Returns: (audio_path, temp_dir, metadata_dict)
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_exe or not os.path.exists(ffmpeg_exe):
        raise RuntimeError("FFmpeg executable bundled with imageio-ffmpeg could not be found.")

    temp_dir = tempfile.mkdtemp(prefix="upload_rag_")
    orig_name = uploaded_file.name
    clean_name = re.sub(r"[^A-Za-z0-9_.-]", "_", orig_name)
    input_path = os.path.join(temp_dir, clean_name)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    audio_path = os.path.join(temp_dir, "audio.mp3")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "2",
        audio_path,
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 or not os.path.exists(audio_path):
        err_msg = res.stderr.decode("utf-8", errors="ignore")[-300:]
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to extract audio using imageio-ffmpeg: {err_msg}")

    metadata = {
        "filename": orig_name,
        "title": orig_name,
        "size_mb": round(uploaded_file.size / (1024 * 1024), 2),
        "type": uploaded_file.type or "video",
    }
    return audio_path, temp_dir, metadata

# ============================================================
# TRANSCRIPTION: WHISPER & FALLBACK
# ============================================================

def transcribe_with_whisper(audio_path: str, max_retries: int = 3) -> str:
    """
    Performs speech-to-text using Hugging Face InferenceClient with openai/whisper-large-v3.
    Includes automated retry for transient network hiccups.
    """
    require_hf_token()
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    client = InferenceClient(token=HF_TOKEN)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.automatic_speech_recognition(
                audio=audio_bytes,
                model=WHISPER_MODEL,
            )
            if hasattr(response, "text"):
                transcript = response.text
            elif isinstance(response, dict):
                transcript = response.get("text", "")
            else:
                transcript = str(response)

            transcript = transcript.strip()
            if not transcript:
                raise ValueError("Whisper returned an empty transcript.")
            return transcript
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(2 * attempt)
                continue

    safe_err = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", str(last_err))
    raise RuntimeError(f"Hugging Face Whisper API failed: {safe_err}") from last_err


def fetch_fallback_youtube_transcript(video_id: str, preferred_languages: Optional[List[str]] = None) -> str:
    """Secondary fallback: fetches native YouTube transcript via YouTubeTranscriptApi."""
    if preferred_languages is None:
        preferred_languages = ["en", "hi"]

    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=preferred_languages, preserve_formatting=False)
        texts = [item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "") for item in fetched]
        transcript = " ".join(t.strip() for t in texts if t.strip())
        if transcript:
            return transcript
    except Exception:
        pass

    try:
        transcript_list = api.list(video_id)
        for t in transcript_list:
            try:
                fetched = t.fetch(preserve_formatting=False)
                texts = [item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "") for item in fetched]
                transcript = " ".join(t.strip() for t in texts if t.strip())
                if transcript:
                    return transcript
            except Exception:
                continue
    except Exception as exc:
        safe_err = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", str(exc))
        raise RuntimeError(f"YouTube transcript fallback also failed: {safe_err}") from exc

    raise RuntimeError("Could not retrieve transcript from YouTube.")

# ============================================================
# RAG PIPELINE
# ============================================================

def split_text(transcript: str) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.create_documents([transcript])

def create_vector_store(chunks: List[Any]):
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)

def create_retriever(vector_store, chunk_count: int = 8):
    if chunk_count >= 5:
        k = min(8, max(5, chunk_count))
    else:
        k = max(1, chunk_count)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

def create_prompt() -> PromptTemplate:
    template = """You are a video question-answering assistant.

Answer the user's question using ONLY the supplied video transcript context.

Do not use outside knowledge.

Do not guess.

Do not invent facts.

Do not assume information that is not present in the transcript.

If the answer is not supported by the supplied transcript context, respond exactly:

I couldn't find the answer to that in the video.

VIDEO TRANSCRIPT CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def generate_answer(context: str, question: str) -> str:
    if not context or not context.strip():
        return "I couldn't find the answer to that in the video."

    prompt = create_prompt()
    formatted_prompt = prompt.format(context=context, question=question)

    try:
        llm = get_llm()
        response = llm.invoke(formatted_prompt)
    except Exception as exc:
        if not is_hf_permission_error(exc):
            raise
        llm = get_fallback_llm()
        response = llm.invoke(formatted_prompt)

    if hasattr(response, "content"):
        answer = response.content
    else:
        answer = str(response)

    answer_clean = answer.strip()

    # Targeted refusal detection: only return fallback when model explicitly gives a short refusal
    lower_ans = answer_clean.lower()
    explicit_refusals = [
        "i couldn't find the answer to that in the video",
        "i could not find the answer to that in the video",
        "i cannot find the answer to that in the video",
        "the provided video context does not mention",
        "the provided transcript does not mention",
        "there is no mention of",
        "not supported by the supplied transcript",
        "not supported by the video",
    ]
    if any(p in lower_ans for p in explicit_refusals) and len(answer_clean) < 180:
        return "I couldn't find the answer to that in the video."

    return answer_clean

def run_unified_rag_pipeline(
    transcript: str,
    source_name: str,
    source_type: str,
    metadata: Dict[str, Any],
    identifier: str,
    playback_source: Optional[Any] = None
):
    """
    Shared RAG processor for both YouTube URLs and Uploaded Video files.
    Clears previous video state cleanly to prevent transcript mixing.
    """
    chunks = split_text(transcript)
    if not chunks:
        raise RuntimeError("Transcript text was empty or could not be chunked.")

    vector_store = create_vector_store(chunks)
    retriever = create_retriever(vector_store, chunk_count=len(chunks))

    # Terminal diagnostics
    print("=" * 50)
    print(f"TRANSCRIPT DIAGNOSTICS ({source_type}):")
    print(f"Title: {metadata.get('title', 'Unknown')}")
    print(f"Transcript characters: {len(transcript):,}")
    print(f"Transcript words: {len(transcript.split()):,}")
    print(f"Number of chunks: {len(chunks)}")
    print("=" * 50)

    # Clean switch to new video
    st.session_state.vector_store = vector_store
    st.session_state.retriever = retriever
    st.session_state.video_processed = True
    st.session_state.source_type = source_type
    st.session_state.video_metadata = metadata
    st.session_state.video_playback_source = playback_source
    st.session_state.transcript_text = transcript
    st.session_state.transcript_source = source_name
    st.session_state.chunk_count = len(chunks)
    st.session_state.processed_identifier = identifier
    st.session_state.messages = []  # Reset conversation for new video

    # Cache for duplicate prevention
    st.session_state.processed_cache[identifier] = {
        "vector_store": vector_store,
        "retriever": retriever,
        "source_type": source_type,
        "video_metadata": metadata,
        "video_playback_source": playback_source,
        "transcript_text": transcript,
        "transcript_source": source_name,
        "chunk_count": len(chunks),
    }

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <span style="font-size: 22px;">⚙️</span>
            <span style="font-size: 20px; font-weight: 700; color: #F3F4F6;">System</span>
        </div>
        """
    )

    st.markdown(
        f"""
        <div class="glass-card" style="padding: 14px 16px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94A3B8; font-size: 13px;">AI Model</span>
                <span style="color: #F8FAFC; font-size: 13px; font-weight: 600;">Whisper Large v3</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94A3B8; font-size: 13px;">Embedding</span>
                <span style="color: #F8FAFC; font-size: 13px; font-weight: 600;">MiniLM (384d)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94A3B8; font-size: 13px;">Vector Store</span>
                <span style="color: #F8FAFC; font-size: 13px; font-weight: 600;">FAISS (In-Memory)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94A3B8; font-size: 13px;">LLM</span>
                <span style="color: #F8FAFC; font-size: 13px; font-weight: 600;">Hugging Face</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.08);">
                <span style="color: #94A3B8; font-size: 13px;">Status</span>
                <span style="color: #4ADE80; font-size: 13px; font-weight: 600;">🟢 Online</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset Video", use_container_width=True):
            for k in [
                "messages", "vector_store", "retriever", "video_processed",
                "video_metadata", "transcript_text", "transcript_source",
                "chunk_count", "processed_identifier", "processed_cache", "video_playback_source"
            ]:
                st.session_state[k] = [] if k in ["messages", "processed_cache"] else (
                    None if k in ["vector_store", "retriever"] else (
                        False if k == "video_processed" else (
                            {} if k in ["video_metadata", "processed_cache"] else (
                                0 if k == "chunk_count" else ""
                            )
                        )
                    )
                )
            st.rerun()

# ============================================================
# PREMIUM HEADER
# ============================================================

render_html(
    """
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px; margin-bottom: 24px;">
        <div>
            <h1 style="margin: 0; font-size: 34px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px;">
                🎥 YouTube & Video RAG AI
            </h1>
            <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 16px;">
                Turn any video into an interactive knowledge base.
            </p>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span>
            <span>● AI System Online</span>
        </div>
    </div>
    """
)

if not HF_TOKEN:
    st.error("HF_TOKEN is not configured. Add your Hugging Face token to .env.")

# ============================================================
# MAIN INPUT SECTION: TWO TABS
# ============================================================

tab_yt, tab_upload = st.tabs(["📺 YouTube Video", "📁 Upload Video"])

# TAB 1: YOUTUBE INGESTION
with tab_yt:
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    yt_url = st.text_input(
        "YouTube Video URL or ID",
        placeholder="https://www.youtube.com/watch?v=...",
        key="yt_input_url",
        help="Supported: youtube.com/watch?v=..., youtu.be/..., and youtube.com/shorts/...",
    )
    st.caption("Supported URL formats: `youtube.com/watch?v=...`, `youtu.be/...`, `youtube.com/shorts/...`")

    process_yt = st.button("🚀 Process YouTube Video", type="primary", key="btn_process_yt")

    if process_yt:
        if not yt_url.strip():
            st.error("Please enter a valid YouTube URL or Video ID.")
        else:
            try:
                vid_id = extract_video_id(yt_url)

                # Duplicate check
                if (
                    st.session_state.video_processed
                    and st.session_state.processed_identifier == vid_id
                    and st.session_state.vector_store is not None
                ):
                    st.info("ℹ️ This YouTube video is already processed and ready for questions below!")
                elif vid_id in st.session_state.processed_cache:
                    cached = st.session_state.processed_cache[vid_id]
                    st.session_state.vector_store = cached["vector_store"]
                    st.session_state.retriever = cached["retriever"]
                    st.session_state.video_processed = True
                    st.session_state.source_type = cached["source_type"]
                    st.session_state.video_metadata = cached["video_metadata"]
                    st.session_state.video_playback_source = cached.get("video_playback_source", f"https://www.youtube.com/watch?v={vid_id}")
                    st.session_state.transcript_text = cached["transcript_text"]
                    st.session_state.transcript_source = cached["transcript_source"]
                    st.session_state.chunk_count = cached["chunk_count"]
                    st.session_state.processed_identifier = vid_id
                    st.session_state.messages = []
                    st.success("✓ Video successfully processed (from session cache).")
                else:
                    with st.status("Processing Video...", expanded=True) as status:
                        st.write("01 Audio Extraction")
                        audio_path, temp_dir, meta = extract_audio_from_youtube(yt_url)

                        transcript = None
                        source_used = None

                        try:
                            st.write("02 Whisper Transcription")
                            transcript = transcribe_with_whisper(audio_path)
                            source_used = f"Whisper ({WHISPER_MODEL})"
                        except Exception as whisper_err:
                            st.warning("Whisper API note: using secondary transcript fallback...")
                            try:
                                transcript = fetch_fallback_youtube_transcript(vid_id)
                                source_used = "youtube_transcript_api (fallback)"
                            except Exception as fb_err:
                                raise RuntimeError(f"Both Whisper and YouTube transcript fallback failed.\nWhisper: {whisper_err}\nFallback: {fb_err}")
                        finally:
                            shutil.rmtree(temp_dir, ignore_errors=True)

                        st.write("03 Text Chunking")
                        st.write("04 Embedding Generation")
                        st.write("05 FAISS Indexing")
                        run_unified_rag_pipeline(transcript, source_used, "YouTube", meta, vid_id, playback_source=f"https://www.youtube.com/watch?v={vid_id}")

                        status.update(label="✓ Video successfully processed", state="complete", expanded=False)
            except Exception as e:
                safe_msg = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", str(e))
                st.error(f"❌ Unable to process video: {safe_msg}")

# TAB 2: UPLOADED VIDEO INGESTION
with tab_upload:
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "mkv", "avi", "webm"],
        key="uploaded_file_input",
        help="Supported formats: MP4, MOV, MKV, AVI, WEBM",
    )
    process_upload = st.button("🚀 Process Uploaded Video", type="primary", key="btn_process_upload")

    if process_upload:
        if not uploaded_file:
            st.error("Please select a video file to upload first.")
        else:
            try:
                file_bytes = uploaded_file.getvalue()
                file_hash = compute_file_sha256(file_bytes)

                # Duplicate check
                if (
                    st.session_state.video_processed
                    and st.session_state.processed_identifier == file_hash
                    and st.session_state.vector_store is not None
                ):
                    st.info("ℹ️ This uploaded video is already processed and ready for questions below!")
                elif file_hash in st.session_state.processed_cache:
                    cached = st.session_state.processed_cache[file_hash]
                    st.session_state.vector_store = cached["vector_store"]
                    st.session_state.retriever = cached["retriever"]
                    st.session_state.video_processed = True
                    st.session_state.source_type = cached["source_type"]
                    st.session_state.video_metadata = cached["video_metadata"]
                    st.session_state.video_playback_source = cached.get("video_playback_source")
                    st.session_state.transcript_text = cached["transcript_text"]
                    st.session_state.transcript_source = cached["transcript_source"]
                    st.session_state.chunk_count = cached["chunk_count"]
                    st.session_state.processed_identifier = file_hash
                    st.session_state.messages = []
                    st.success("✓ Video successfully processed (from session cache).")
                else:
                    with st.status("Processing Video...", expanded=True) as status:
                        st.write("01 Audio Extraction")
                        audio_path, temp_dir, meta = extract_audio_from_video_file(uploaded_file)

                        try:
                            st.write("02 Whisper Transcription")
                            transcript = transcribe_with_whisper(audio_path)
                            source_used = f"Whisper ({WHISPER_MODEL})"
                        finally:
                            shutil.rmtree(temp_dir, ignore_errors=True)

                        st.write("03 Text Chunking")
                        st.write("04 Embedding Generation")
                        st.write("05 FAISS Indexing")
                        run_unified_rag_pipeline(transcript, source_used, "Uploaded Video", meta, file_hash, playback_source=file_bytes)

                        status.update(label="✓ Video successfully processed", state="complete", expanded=False)
            except Exception as e:
                safe_msg = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", str(e))
                st.error(f"❌ Unable to process video: {safe_msg}")

# ============================================================
# VIDEO PREVIEW (AFTER PROCESSING)
# ============================================================

if st.session_state.video_processed:
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.subheader("🎥 Video Preview")
    playback_source = st.session_state.get("video_playback_source")
    try:
        if playback_source:
            st.video(playback_source)
        else:
            st.info("Video preview unavailable. You can still ask questions about the processed transcript.")
    except Exception:
        st.info("Video preview unavailable. You can still ask questions about the processed transcript.")

# ============================================================
# VIDEO INFORMATION CARD (AFTER PROCESSING)
# ============================================================

if st.session_state.video_processed:
    meta = st.session_state.video_metadata or {}
    raw_title = str(meta.get("title", "Video"))
    title_escaped = html.escape(raw_title)
    source_type = st.session_state.source_type

    info_items = [
        f"<div style='color: #94A3B8; font-size: 14px; margin-bottom: 4px;'><b>Source:</b> <span class='tag-pill'>{html.escape(source_type)}</span></div>"
    ]
    if source_type == "YouTube":
        uploader = html.escape(str(meta.get("uploader", "N/A")))
        yt_url = html.escape(str(meta.get("url", "#")))
        info_items.append(f"<div style='color: #94A3B8; font-size: 14px; margin-bottom: 4px;'><b>Channel / Author:</b> {uploader}</div>")
        info_items.append(f"<div style='color: #94A3B8; font-size: 14px; margin-bottom: 4px;'><b>YouTube URL:</b> <a href='{yt_url}' target='_blank' style='color:#818CF8;'>Link</a></div>")
    else:
        filename = html.escape(str(meta.get("filename", "N/A")))
        size_mb = meta.get("size_mb", "N/A")
        info_items.append(f"<div style='color: #94A3B8; font-size: 14px; margin-bottom: 4px;'><b>File Name:</b> {filename} ({size_mb} MB)</div>")

    duration = meta.get("duration", "N/A")
    info_items.append(f"<div style='color: #94A3B8; font-size: 14px;'><b>Duration:</b> {duration} seconds</div>")
    info_html = "".join(info_items)

    card_html = f"""
    <div class="glass-card" style="margin-top: 20px;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px;">
            <div>
                <div style="font-size: 12px; font-weight: 700; color: #94A3B8; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">
                    📹 Video Information
                </div>
                <div style="font-size: 17px; font-weight: 700; color: #F8FAFC; margin-bottom: 8px;">
                    {title_escaped}
                </div>
                {info_html}
            </div>
            <div style="border-left: 1px solid rgba(255, 255, 255, 0.08); padding-left: 20px;">
                <div style="font-size: 12px; font-weight: 700; color: #94A3B8; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">
                    🧠 RAG Architecture & Metrics
                </div>
                <div style="color: #94A3B8; font-size: 14px; margin-bottom: 6px;">
                    <b>Vector Store:</b> <span style="color:#4ADE80; font-weight:600;">FAISS • Active</span>
                </div>
                <div style="color: #94A3B8; font-size: 14px; margin-bottom: 6px;">
                    <b>Embedding:</b> <span style="color:#A5B4FC;">MiniLM • 384 dimensions</span>
                </div>
                <div style="color: #94A3B8; font-size: 14px; margin-bottom: 6px;">
                    <b>Number of Chunks:</b> {st.session_state.chunk_count}
                </div>
                <div style="color: #94A3B8; font-size: 14px;">
                    <b>Transcript Characters:</b> {len(st.session_state.transcript_text):,}
                </div>
            </div>
        </div>
    </div>
    """
    render_html(card_html)

    # 📜 Full Transcript / Preview Expandable
    with st.expander("📜 Full Transcript / Preview", expanded=False):
        t_text = st.session_state.transcript_text
        c_stat1, c_stat2, c_stat3 = st.columns(3)
        with c_stat1:
            st.metric("Characters", f"{len(t_text):,}")
        with c_stat2:
            st.metric("Words", f"{len(t_text.split()):,}")
        with c_stat3:
            st.metric("Chunks in FAISS", st.session_state.chunk_count)

        search_phrase = st.text_input("🔍 Search transcript for test phrase:", key="transcript_search_phrase")
        if search_phrase.strip():
            count = t_text.lower().count(search_phrase.lower())
            if count > 0:
                st.success(f"✓ Found {count} occurrence(s) of '{search_phrase}' in transcript.")
            else:
                st.warning(f"Phrase '{search_phrase}' not found in transcript.")

        st.text_area("Transcript Text", t_text, height=220, disabled=True, label_visibility="collapsed")

# ============================================================
# WELCOME STATE (BEFORE VIDEO IS PROCESSED)
# ============================================================

if not st.session_state.video_processed:
    render_html(
        """
        <div class="glass-card" style="text-align: center; padding: 36px 24px; margin-top: 25px;">
            <h2 style="margin: 0 0 10px 0; color: #F8FAFC; font-size: 24px; font-weight: 700;">
                🎬 Your video knowledge base is waiting
            </h2>
            <p style="margin: 0 0 28px 0; color: #94A3B8; font-size: 15px;">
                Paste a YouTube URL or upload a video to get started.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; text-align: left;">
                <div class="feature-card">
                    <div style="font-size: 24px; margin-bottom: 8px;">🎙️</div>
                    <div style="font-weight: 600; color: #F8FAFC; font-size: 15px; margin-bottom: 4px;">Whisper Transcription</div>
                    <div style="color: #94A3B8; font-size: 13px;">Accurate speech-to-text</div>
                </div>
                <div class="feature-card">
                    <div style="font-size: 24px; margin-bottom: 8px;">🧠</div>
                    <div style="font-weight: 600; color: #F8FAFC; font-size: 15px; margin-bottom: 4px;">AI Retrieval</div>
                    <div style="color: #94A3B8; font-size: 13px;">FAISS-powered semantic search</div>
                </div>
                <div class="feature-card">
                    <div style="font-size: 24px; margin-bottom: 8px;">💬</div>
                    <div style="font-weight: 600; color: #F8FAFC; font-size: 15px; margin-bottom: 4px;">Grounded Answers</div>
                    <div style="color: #94A3B8; font-size: 13px;">Answers based on your video</div>
                </div>
            </div>
        </div>
        """
    )

# ============================================================
# CHAT SECTION (AFTER PROCESSING)
# ============================================================

if st.session_state.video_processed:
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.subheader("💬 Ask anything about this video")

    # Render previous conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources / Retrieved Context"):
                    for idx, s in enumerate(msg["sources"], 1):
                        st.markdown(f"**Source:** {s.get('source', 'Video Transcript')}")
                        st.markdown(f"**Timestamp:** {s.get('timestamp', 'N/A (Full Clip Transcript)')}")
                        st.markdown(f"**Relevant Transcript:**\n> {s.get('text', '')}")
                        if idx < len(msg["sources"]):
                            st.divider()

    # Handle user query
    if user_query := st.chat_input("Ask anything about this video..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching video transcript & generating answer..."):
                retriever = st.session_state.retriever
                vector_store = st.session_state.vector_store
                docs = retriever.invoke(user_query) if retriever else []
                context_chunks = [doc.page_content for doc in docs]
                context_str = "\n\n---\n\n".join(context_chunks)

                print("-" * 50)
                print(f"[RETRIEVAL DIAGNOSTICS]")
                print(f"Question: {user_query}")
                print(f"Retrieved chunks count: {len(docs)}")
                if vector_store and hasattr(vector_store, "similarity_search_with_score"):
                    try:
                        scored_docs = vector_store.similarity_search_with_score(user_query, k=len(docs))
                        print("Chunk similarity distances:", [round(score, 4) for _, score in scored_docs])
                    except Exception:
                        pass
                if docs:
                    print(f"Top retrieved chunk snippet: {docs[0].page_content[:120]}...")
                print("-" * 50)

                answer = generate_answer(context_str, user_query)
                st.write(answer)

                sources_payload = []
                if "couldn't find the answer" not in answer.lower():
                    meta_title = st.session_state.video_metadata.get("title") or "Video Transcript"
                    for doc in docs:
                        sources_payload.append({
                            "source": meta_title,
                            "timestamp": "N/A (Full Clip Transcript)",
                            "text": doc.page_content,
                        })

                    if sources_payload:
                        with st.expander("📚 Sources / Retrieved Context"):
                            for idx, s in enumerate(sources_payload, 1):
                                st.markdown(f"**Source:** {s['source']}")
                                st.markdown(f"**Timestamp:** {s['timestamp']}")
                                st.markdown(f"**Relevant Transcript:**\n> {s['text']}")
                                if idx < len(sources_payload):
                                    st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_payload,
        })
