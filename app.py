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

import base64
import glob
from PIL import Image, ImageStat, ImageFilter

import imageio_ffmpeg
import yt_dlp
from huggingface_hub import InferenceClient
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
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
HF_MODEL_ID = get_secret("HF_MODEL_ID", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_LOCAL_MODEL_ID = get_secret("HF_LOCAL_MODEL_ID", "sshleifer/tiny-gpt2")
WHISPER_MODEL = get_secret("WHISPER_MODEL", "openai/whisper-large-v3")
YOUTUBE_HTTP_PROXY = get_secret("YOUTUBE_HTTP_PROXY")
LOCAL_MODEL_PATH = get_secret(
    "LOCAL_MODEL_PATH",
    os.path.join("models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
)
VISION_MODEL_NAME = get_secret("VISION_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
FRAME_INTERVAL_SECONDS = int(get_secret("FRAME_INTERVAL_SECONDS", "5") or 5)

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
        "current_provider": "Hugging Face",
        "visual_vector_store": None,
        "visual_retriever": None,
        "visual_records": [],
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
    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs=kwargs,
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
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local GGUF model not found at path: {model_path}")
        from llama_cpp import Llama
        self.model_path = model_path
        threads = max(1, (os.cpu_count() or 4) - 1)
        # 4096 context window with multithreading for fast CPU inference
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=threads,
            n_batch=512,
            verbose=False,
        )

    def invoke(self, prompt: str) -> str:
        few_shot_system = """You are a video question-answering assistant.
Answer the user's question using ONLY the supplied video transcript context.
If the user asks for the main points, summary, overview, or what the video is about, summarize the key events, people, and topics discussed in the transcript.

Do not use outside knowledge. Do not guess. Do not invent facts.

CRITICAL RULE: If the question is about outside topics, general trivia, or anything NOT in the transcript, you MUST respond EXACTLY:
I couldn't find the answer to that in the video.

Example 1:
Context: The speaker explains how to train a dog using treats and positive reinforcement.
Question: What is the capital of Spain?
Answer: I couldn't find the answer to that in the video.

Example 2:
Context: The speaker explains how to train a dog using treats and positive reinforcement.
Question: What method is used to train the dog?
Answer: The dog is trained using treats and positive reinforcement."""
        res = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": few_shot_system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.1,
        )
        return res["choices"][0]["message"]["content"].strip()

class LocalExtractiveFallbackLLM:
    """
    Resilient, deterministic extractive RAG fallback used when neither Cloud LLM
    nor local GGUF weights are available. Extracts grounded answers directly from
    retrieved context chunks without hallucination.
    """
    def invoke(self, prompt: str, context: Optional[str] = None, question: Optional[str] = None) -> str:
        if not context and "EVIDENCE CONTEXT:" in prompt and "QUESTION:" in prompt:
            try:
                parts = prompt.split("EVIDENCE CONTEXT:", 1)[1].split("QUESTION:", 1)
                context = parts[0].strip()
                question = parts[1].split("ANSWER:", 1)[0].strip()
            except Exception:
                context = prompt
                question = ""
        elif not context and "Context:" in prompt and "Question:" in prompt:
            try:
                parts = prompt.split("Context:", 1)[1].split("Question:", 1)
                context = parts[0].strip()
                question = parts[1].split("Answer:", 1)[0].strip()
            except Exception:
                context = prompt
                question = ""

        if not context or not context.strip():
            return "I couldn't find the answer to that in the video."

        # Verify subject keywords appear in context
        if question:
            general_terms = {
                "what", "who", "where", "when", "why", "how", "which", "is", "are", "was", "were", 
                "the", "a", "an", "in", "on", "at", "of", "to", "for", "and", "or", "not", "this", 
                "that", "video", "about", "main", "points", "summary", "summarize", "tell", "tell me"
            }
            q_keywords = [w for w in re.findall(r"\b[a-zA-Z0-9]+\b", question.lower()) if w not in general_terms and len(w) >= 3]
            if q_keywords and not any(kw in context.lower() for kw in q_keywords):
                return "I couldn't find the answer to that in the video."
        else:
            q_keywords = []

        lower_q = (question or "").lower()

        # Multimodal handling: when context contains both transcript and visual sections
        if "=== TRANSCRIPT EVIDENCE ===" in context and "=== VISUAL EVIDENCE" in context:
            parts = context.split("=== VISUAL EVIDENCE")
            t_part = parts[0].replace("=== TRANSCRIPT EVIDENCE ===", "").strip()
            v_part = ("=== VISUAL EVIDENCE" + parts[1]).strip()

            t_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t_part) if len(s.strip()) > 15]
            v_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", v_part) if len(s.strip()) > 15]

            t_scored = sorted([(sum(1 for kw in q_keywords if kw in s.lower()), s) for s in t_sentences], key=lambda x: x[0], reverse=True)
            v_scored = sorted([(sum(1 for kw in q_keywords if kw in s.lower()), s) for s in v_sentences], key=lambda x: x[0], reverse=True)

            t_best = [s for count, s in t_scored if count > 0]
            v_best = [s for count, s in v_scored if count > 0]

            is_visual_q = any(k in lower_q for k in ["visible", "visual", "see", "seen", "shown", "look", "scene", "color", "doing", "person doing", "object"])
            is_transcript_q = any(k in lower_q for k in ["say", "said", "speak", "spoke", "mention", "mentioned", "talk", "talked", "discuss", "discussed", "words", "audio"])

            if is_visual_q and is_transcript_q:
                ans_t = t_best[0] if t_best else (t_sentences[0] if t_sentences else "")
                ans_v = v_best[0] if v_best else (v_sentences[0] if v_sentences else "")
                return f"{ans_t} Visually: {ans_v}".strip()
            elif is_visual_q and not is_transcript_q:
                if v_best:
                    return " ".join(v_best[:2])
                elif v_sentences:
                    return v_sentences[0]
            elif is_transcript_q and not is_visual_q:
                if t_best:
                    return " ".join(t_best[:2])
                elif t_sentences:
                    return t_sentences[0]

        if any(term in lower_q for term in ["about", "main points", "summary", "overview", "what is this video"]):
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if len(s.strip()) > 20]
            if sentences:
                return "Here is what the video discusses:\n- " + "\n- ".join(sentences[:5])

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if len(s.strip()) > 15]
        if not sentences:
            sentences = [context[:300]]

        if question and q_keywords:
            scored = []
            for s in sentences:
                s_lower = s.lower()
                matches = sum(1 for kw in q_keywords if kw in s_lower)
                scored.append((matches, s))
            scored.sort(key=lambda x: x[0], reverse=True)
            best_matches = [s for count, s in scored if count > 0]
            if best_matches:
                return " ".join(best_matches[:3])

        return sentences[0] if sentences else "I couldn't find the answer to that in the video."


@st.cache_resource(show_spinner=False)
def get_fallback_llm():
    if LOCAL_MODEL_PATH and os.path.exists(LOCAL_MODEL_PATH):
        try:
            print(f"[LLM] Initializing local GGUF model from: {LOCAL_MODEL_PATH}")
            return LocalGGUFLLM(LOCAL_MODEL_PATH)
        except Exception as e:
            print(f"[LLM] Warning: Failed to load local GGUF model: {e}. Falling back to Extractive RAG.")
            return LocalExtractiveFallbackLLM()
    else:
        print("[LLM] Local GGUF model not found on disk. Initializing resilient Extractive RAG Fallback.")
        return LocalExtractiveFallbackLLM()

@st.cache_resource(show_spinner=False)
def get_llm():
    require_hf_token()
    llm = HuggingFaceEndpoint(
        repo_id=HF_MODEL_ID,
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
    )
    return ChatHuggingFace(llm=llm)

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


def extract_audio_from_video_file(uploaded_file) -> Tuple[str, str, Dict[str, Any], str]:
    """
    Uses bundled imageio-ffmpeg executable to extract audio from an uploaded video.
    Returns: (audio_path, temp_dir, metadata_dict, input_video_path)
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_exe or not os.path.exists(ffmpeg_exe):
        ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"

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
    return audio_path, temp_dir, metadata, input_path


# ============================================================
# VISUAL VIDEO ANALYSIS & FRAME EXTRACTION
# ============================================================

def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS timestamp string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def extract_video_frames(
    video_path: str,
    interval_seconds: Optional[int] = None,
    max_frames: int = 15
) -> List[Tuple[str, float, str]]:
    """
    Extracts representative frames from video at configurable intervals.
    Returns: List of (frame_image_path, timestamp_sec, formatted_timestamp_str)
    """
    if interval_seconds is None:
        interval_seconds = FRAME_INTERVAL_SECONDS or 5

    ffmpeg_exe = None
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if not ffmpeg_exe or not os.path.exists(ffmpeg_exe):
        ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"

    frames_dir = os.path.join(os.path.dirname(video_path), "extracted_frames")
    os.makedirs(frames_dir, exist_ok=True)
    out_pattern = os.path.join(frames_dir, "frame_%04d.jpg")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", video_path,
        "-vf", f"fps=1/{interval_seconds}",
        "-q:v", "2",
        out_pattern
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except Exception as exc:
        print(f"[VISION] Frame extraction error: {exc}")

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_files:
        # Fallback: extract single frame at 1s if fps filter yielded nothing
        single_frame = os.path.join(frames_dir, "frame_single.jpg")
        cmd_single = [ffmpeg_exe, "-y", "-ss", "00:00:01", "-i", video_path, "-vframes", "1", "-q:v", "2", single_frame]
        subprocess.run(cmd_single, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if os.path.exists(single_frame):
            frame_files = [single_frame]

    # Subsample if more than max_frames to preserve CPU responsiveness
    if len(frame_files) > max_frames:
        step = len(frame_files) / max_frames
        selected = [frame_files[int(i * step)] for i in range(max_frames)]
        frame_files = selected

    records = []
    for idx, fpath in enumerate(frame_files):
        ts_sec = idx * interval_seconds
        ts_fmt = format_timestamp(ts_sec)
        records.append((fpath, ts_sec, ts_fmt))

    return records


def analyze_frame_visual(image_path: str, timestamp_str: str) -> str:
    """
    Analyzes an extracted video frame using the configured Vision-Language Model
    (Qwen/Qwen2.5-VL-3B-Instruct) via Hugging Face InferenceClient, with a resilient
    local visual scene analyzer fallback to guarantee 0 crashes.
    """
    # Strategy 1: Cloud Vision-Language Model (VLM)
    if HF_TOKEN:
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{img_b64}"
            client = InferenceClient(token=HF_TOKEN)
            res = client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {
                            "type": "text",
                            "text": (
                                "Provide a concise, factual description of what is visually shown in this video frame. "
                                "Identify visible objects, colors, people, actions, and the environment. "
                                "Do not guess or invent details not present in the frame."
                            )
                        }
                    ]
                }],
                max_tokens=100,
                temperature=0.1,
            )
            vlm_text = res.choices[0].message.content.strip()
            if vlm_text:
                print(f"[VISION] VLM ({VISION_MODEL_NAME}) at [{timestamp_str}]: {vlm_text[:70]}...")
                return f"[{timestamp_str}] Visual Analysis: {vlm_text}"
        except Exception as vlm_err:
            safe_err = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", str(vlm_err))
            print(f"[VISION] Cloud VLM ({VISION_MODEL_NAME}) unavailable: {safe_err}. Using local visual analysis.")

    # Strategy 2: Resilient Local Visual Analyzer (real image statistics, dominant color, texture, layout)
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            width, height = rgb_img.size
            stat = ImageStat.Stat(rgb_img)
            r, g, b = stat.mean[:3]
            brightness = (0.299 * r + 0.587 * g + 0.114 * b)

            # Detect dominant color signature
            color_notes = []
            if r > 150 and r > g * 1.3 and r > b * 1.3:
                color_notes.append("predominantly red visual scene / object")
            elif g > 130 and g > r * 1.2 and g > b * 1.2:
                color_notes.append("predominantly green scenery / foliage")
            elif b > 140 and b > r * 1.2 and b > g * 1.1:
                color_notes.append("predominantly blue scene / sky / display")
            elif r > 180 and g > 180 and b < 100:
                color_notes.append("yellow / warm tone composition")
            elif brightness > 210:
                color_notes.append("bright high-key visual environment")
            elif brightness < 45:
                color_notes.append("dark low-key visual environment")
            else:
                color_notes.append("natural daylight / indoor illumination")

            edges = rgb_img.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            edge_mean = sum(edge_stat.mean[:3]) / 3
            complexity = "featuring visible objects and defined structures" if edge_mean > 20 else "featuring smooth or uniform composition"

            desc = f"Frame at {timestamp_str} displays {', '.join(color_notes)} {complexity} (resolution {width}x{height})."
            return f"[{timestamp_str}] Visual Analysis: {desc}"
    except Exception as e:
        return f"[{timestamp_str}] Visual Analysis: Scene captured at timestamp {timestamp_str}."


def create_visual_vector_store(visual_records: List[Dict[str, Any]]):
    """Creates an in-memory FAISS vector store from analyzed visual frame records."""
    docs = [
        Document(
            page_content=rec["description"],
            metadata={"timestamp": rec["timestamp"], "timestamp_sec": rec.get("timestamp_sec", 0)}
        )
        for rec in visual_records if rec.get("description", "").strip()
    ]
    if not docs:
        return None
    embeddings = get_embeddings()
    return FAISS.from_documents(docs, embeddings)


def create_visual_retriever(visual_vector_store, count: int = 5):
    """Creates similarity retriever for visual knowledge base."""
    if not visual_vector_store:
        return None
    k = min(6, max(1, count))
    return visual_vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def classify_question_intent(question: str) -> str:
    """Routes user question to 'visual', 'text', or 'multimodal'."""
    q = question.lower()
    visual_keywords = {
        "visible", "visual", "visuals", "look", "looks", "see", "seeing", "scene", "shown", "show",
        "showing", "color", "colour", "screen", "appear", "appears", "wearing", "clothes", "outfit",
        "holding", "background", "foreground", "person doing", "people doing", "object", "objects",
        "car", "vehicle", "room", "table", "chair", "board", "picture", "frame", "frames",
        "how many people", "who is visible", "what is on screen", "doing"
    }
    text_keywords = {
        "say", "said", "saying", "speak", "spoke", "spoken", "speaker", "mention", "mentioned",
        "discuss", "discussed", "talk", "talked", "talking", "explain", "explained",
        "quote", "words", "speech", "topic", "lecture", "audio", "listen", "hear", "heard"
    }
    has_vis = any(re.search(rf"\b{re.escape(k)}\b", q) for k in visual_keywords)
    has_txt = any(re.search(rf"\b{re.escape(k)}\b", q) for k in text_keywords)
    if has_vis and has_txt:
        return "multimodal"
    if has_vis:
        return "visual"
    if has_txt:
        return "text"
    return "multimodal"


def retrieve_multimodal_context(
    question: str,
    text_retriever: Optional[Any],
    visual_retriever: Optional[Any]
) -> Tuple[str, List[Any], str]:
    """Retrieves relevant transcript and visual evidence based on question intent."""
    intent = classify_question_intent(question)
    transcript_docs = []
    visual_docs = []

    if intent in ("text", "multimodal") and text_retriever is not None:
        try:
            transcript_docs = text_retriever.invoke(question)
        except Exception as e:
            print(f"[RAG] Text retrieval error: {e}")
            transcript_docs = []

    if intent in ("visual", "multimodal") and visual_retriever is not None:
        try:
            visual_docs = visual_retriever.invoke(question)
        except Exception as e:
            print(f"[RAG] Visual retrieval error: {e}")
            visual_docs = []

    if not visual_docs and not transcript_docs:
        if text_retriever is not None:
            try:
                transcript_docs = text_retriever.invoke(question)
            except Exception:
                pass
        if visual_retriever is not None and not transcript_docs:
            try:
                visual_docs = visual_retriever.invoke(question)
            except Exception:
                pass

    all_docs = transcript_docs + visual_docs
    context_sections = []
    if transcript_docs:
        t_text = "\n\n".join(d.page_content for d in transcript_docs)
        context_sections.append(f"=== TRANSCRIPT EVIDENCE ===\n{t_text}")
    if visual_docs:
        v_text = "\n".join(d.page_content for d in visual_docs)
        context_sections.append(f"=== VISUAL EVIDENCE (FRAME & SCENE ANALYSIS) ===\n{v_text}")

    combined_context = "\n\n".join(context_sections).strip()
    return combined_context, all_docs, intent

# ============================================================
# TRANSCRIPTION: WHISPER & FALLBACK
# ============================================================

# ============================================================
# LOCAL WHISPER TRANSCRIPTION (OFFLINE / ZERO CREDITS)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_local_whisper_model(model_size: str = "base"):
    """Loads and caches the local faster-whisper model on CPU with INT8 quantization."""
    try:
        from faster_whisper import WhisperModel
        return WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        print(f"[TRANSCRIPTION] faster_whisper model initialization note: {e}")
        return None


def transcribe_audio_local(audio_path: str) -> str:
    """
    Transcribes audio using a local faster-whisper model without requiring any Hugging Face credits.
    1. Loads local Whisper model from cache.
    2. Transcribes extracted audio.
    3. Combines segments into one transcript.
    4. Validates transcript is not empty.
    """
    print("[TRANSCRIPTION] Local Whisper started", flush=True)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    model = get_local_whisper_model("base")
    if model is None:
        raise RuntimeError(
            "Local Whisper model is unavailable in this environment (faster-whisper package not found). "
            "Please ensure faster-whisper and ctranslate2 are installed or configure HF_TOKEN."
        )
    segments, info = model.transcribe(audio_path, beam_size=5)
    text_parts = [s.text.strip() for s in segments if s.text.strip()]
    transcript_text = " ".join(text_parts).strip()

    # Normalize whitespace
    transcript_text = re.sub(r"\s+", " ", transcript_text).strip()
    if not transcript_text:
        raise ValueError("Local Whisper returned an empty transcript.")

    print("[TRANSCRIPTION] Local transcription completed", flush=True)
    return transcript_text


transcribe_audio_with_whisper = None # alias defined below
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
    docs = splitter.create_documents([transcript])
    return [d for d in docs if d.page_content and d.page_content.strip()]

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
    template = """You are a grounded multimodal video assistant.

Answer the user's question using ONLY the supplied transcript evidence and visual evidence from the video.
If timestamps are provided, reference them in your answer when relevant.
If the user asks what is visible in the video, answer using the visual evidence.
If the user asks what was spoken, answer using the transcript evidence.
If the user asks what was happening while something was spoken, correlate both the transcript and visual evidence.

CRITICAL RULES:
1. Do not use outside knowledge. Do not guess. Do not invent facts.
2. If the requested information cannot be supported by the supplied transcript or visual evidence, respond EXACTLY:
I couldn't find the answer to that in the video.

EVIDENCE CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def generate_answer(context: str, question: str, docs: Optional[List[Any]] = None) -> str:
    """
    Generates a grounded answer using Hugging Face Cloud inference when available,
    falling back automatically to the local GGUF or extractive fallback model on cloud failure.
    """
    if not context or not context.strip():
        return "I couldn't find the answer to that in the video."

    # Grounding safeguard:
    # If a specific factual question has no subject keywords appearing anywhere in context,
    # prevent hallucination from internal parametric memory.
    general_query_terms = {
        "what", "who", "where", "when", "why", "how", "which", "is", "are", "was", "were", 
        "the", "a", "an", "in", "on", "at", "of", "to", "for", "and", "or", "not", "this", 
        "that", "these", "those", "video", "mentioned", "discussed", "about", "main", "points", 
        "summary", "summarize", "tell", "explain", "describe", "happens", "said", "say", 
        "talk", "talking", "does", "did", "do", "can", "could", "would", "should", "any", "all",
        "visible", "see", "seen", "shown", "show", "showing", "color", "colour", "look", "looks",
        "scene", "screen", "frame", "frames", "objects", "object", "doing", "person", "people"
    }
    q_words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    specific_keywords = [w for w in q_words if w not in general_query_terms and (len(w) >= 3 or w.isdigit())]
    if specific_keywords:
        lower_context = context.lower()
        if not any(kw in lower_context for kw in specific_keywords):
            print(f"[GROUNDING] Specific keywords {specific_keywords} not found in context. Returning refusal.")
            return "I couldn't find the answer to that in the video."

    prompt = create_prompt()
    formatted_prompt = prompt.format(context=context, question=question)

    answer_raw = None
    force_local = os.getenv("FORCE_LOCAL_LLM", "").strip().lower() in ("1", "true", "yes")

    # Step 1: Attempt Cloud LLM if token configured and not bypassed
    if HF_TOKEN and not force_local:
        try:
            print("[LLM] Attempting inference with Cloud Provider: Hugging Face")
            llm = get_llm()
            response = llm.invoke(formatted_prompt)
            if hasattr(response, "content"):
                answer_raw = response.content
            else:
                answer_raw = str(response)
            print("[LLM] Provider: Hugging Face (success)")
            if "current_provider" in st.session_state:
                st.session_state["current_provider"] = "Hugging Face"
        except Exception as cloud_exc:
            safe_reason = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", str(cloud_exc))
            print("[LLM] Cloud inference failed")
            print(f"[LLM] Reason: {safe_reason}")
            print("[LLM] Switching to local LLM")
            answer_raw = None
    else:
        if force_local:
            print("[LLM] Cloud inference bypassed (FORCE_LOCAL_LLM=1)")
        else:
            print("[LLM] Cloud inference unavailable (HF_TOKEN missing)")
        print("[LLM] Switching to local LLM")
        answer_raw = None

    # Step 2: Fall back to Local GGUF LLM if cloud failed or bypassed
    if answer_raw is None:
        print("-" * 50)
        print("[LOCAL RAG DIAGNOSTICS]")
        print(f"Question: {question}")
        retrieved_count = len(docs) if docs is not None else "N/A"
        print(f"Retrieved chunks: {retrieved_count}")
        print(f"Context characters: {len(context)}")
        top_snippet = docs[0].page_content[:150] if docs and len(docs) > 0 else context[:150]
        print(f"Top chunk: {top_snippet}...")
        print("-" * 50)
        local_llm = get_fallback_llm()
        if isinstance(local_llm, LocalGGUFLLM):
            print("[LLM] Provider: Local GGUF")
            if "current_provider" in st.session_state:
                st.session_state["current_provider"] = "Local GGUF (qwen2.5-0.5b)"
            answer_raw = local_llm.invoke(formatted_prompt)
        else:
            print("[LLM] Provider: Local Extractive Fallback")
            if "current_provider" in st.session_state:
                st.session_state["current_provider"] = "Local Extractive Fallback"
            answer_raw = local_llm.invoke(formatted_prompt, context=context, question=question)

    answer_clean = answer_raw.strip()

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
    if any(p in lower_ans for p in explicit_refusals) and len(answer_clean) < 260:
        return "I couldn't find the answer to that in the video."

    return answer_clean

def run_unified_rag_pipeline(
    transcript: str,
    source_name: str,
    source_type: str,
    metadata: Dict[str, Any],
    identifier: str,
    playback_source: Optional[Any] = None,
    visual_records: Optional[List[Dict[str, Any]]] = None
):
    """
    Shared RAG processor for both YouTube URLs and Uploaded Video files.
    Clears previous video state cleanly to prevent transcript mixing.
    Builds both transcript and visual knowledge bases.
    """
    chunks = split_text(transcript)
    if not chunks:
        raise RuntimeError("Transcript text was empty or could not be chunked.")

    vector_store = create_vector_store(chunks)
    retriever = create_retriever(vector_store, chunk_count=len(chunks))

    # Visual Vector Store & Retriever
    visual_records = visual_records or []
    visual_vector_store = None
    visual_retriever = None
    if visual_records:
        visual_vector_store = create_visual_vector_store(visual_records)
        visual_retriever = create_visual_retriever(visual_vector_store, count=len(visual_records))

    # Terminal diagnostics
    print("=" * 50)
    print(f"MULTIMODAL DIAGNOSTICS ({source_type}):")
    print(f"Title: {metadata.get('title', 'Unknown')}")
    print(f"Transcript characters: {len(transcript):,}")
    print(f"Transcript words: {len(transcript.split()):,}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Visual frames analyzed: {len(visual_records)}")
    print("=" * 50)

    # Clean switch to new video
    st.session_state.vector_store = vector_store
    st.session_state.retriever = retriever
    st.session_state.visual_records = visual_records
    st.session_state.visual_vector_store = visual_vector_store
    st.session_state.visual_retriever = visual_retriever
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
        "visual_records": visual_records,
        "visual_vector_store": visual_vector_store,
        "visual_retriever": visual_retriever,
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
                    with st.status("🎥 Processing video...", expanded=True) as status:
                        st.write("🔊 Extracting audio...")
                        audio_path, temp_dir, meta = extract_audio_from_youtube(yt_url)

                        transcript = None
                        source_used = None

                        force_local_whisper = os.getenv("FORCE_LOCAL_WHISPER", "").strip().lower() in ("1", "true", "yes")

                        # Strategy 1: Check YouTube native transcript first
                        try:
                            transcript = fetch_fallback_youtube_transcript(vid_id)
                            if transcript and len(transcript.strip()) > 30:
                                source_used = "YouTube Captions"
                                print(f"[TRANSCRIPTION] Using YouTube native captions for video {vid_id}", flush=True)
                        except Exception:
                            transcript = None

                        # Strategy 2: If native transcript not available, try Cloud Whisper, fallback to Local Whisper
                        if not transcript:
                            if HF_TOKEN and not force_local_whisper:
                                try:
                                    st.write("📝 Transcribing video...")
                                    transcript = transcribe_with_whisper(audio_path)
                                    source_used = f"Cloud Whisper ({WHISPER_MODEL})"
                                except Exception as whisper_err:
                                    safe_err = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", str(whisper_err))
                                    print("[TRANSCRIPTION] Cloud Whisper unavailable", flush=True)
                                    print(f"[TRANSCRIPTION] Reason: {safe_err}", flush=True)
                                    print("[TRANSCRIPTION] Switching to LOCAL Whisper", flush=True)
                                    st.write("📝 Transcribing video locally...")
                                    transcript = transcribe_audio_local(audio_path)
                                    source_used = "Local Whisper (faster-whisper base)"
                            else:
                                if force_local_whisper:
                                    print("[TRANSCRIPTION] Cloud Whisper bypassed (FORCE_LOCAL_WHISPER=1)", flush=True)
                                else:
                                    print("[TRANSCRIPTION] Cloud Whisper unavailable (HF_TOKEN not set)", flush=True)
                                print("[TRANSCRIPTION] Switching to LOCAL Whisper", flush=True)
                                st.write("📝 Transcribing video locally...")
                                transcript = transcribe_audio_local(audio_path)
                                source_used = "Local Whisper (faster-whisper base)"

                        shutil.rmtree(temp_dir, ignore_errors=True)

                        
                        
                        st.write("🧠 Building video knowledge base...")
                        run_unified_rag_pipeline(transcript, source_used, "YouTube", meta, vid_id, playback_source=f"https://www.youtube.com/watch?v={vid_id}")

                        status.update(label="✅ Video ready!", state="complete", expanded=False)
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
                    st.session_state.visual_records = cached.get("visual_records", [])
                    st.session_state.visual_vector_store = cached.get("visual_vector_store")
                    st.session_state.visual_retriever = cached.get("visual_retriever")
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
                    with st.status("🎥 Processing video...", expanded=True) as status:
                        st.write("🔊 Extracting audio...")
                        audio_path, temp_dir, meta, input_path = extract_audio_from_video_file(uploaded_file)

                        force_local_whisper = os.getenv("FORCE_LOCAL_WHISPER", "").strip().lower() in ("1", "true", "yes")
                        try:
                            if HF_TOKEN and not force_local_whisper:
                                try:
                                    st.write("📝 Transcribing audio...")
                                    transcript = transcribe_with_whisper(audio_path)
                                    source_used = f"Cloud Whisper ({WHISPER_MODEL})"
                                except Exception as whisper_err:
                                    safe_err = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", str(whisper_err))
                                    print("[TRANSCRIPTION] Cloud Whisper unavailable", flush=True)
                                    print(f"[TRANSCRIPTION] Reason: {safe_err}", flush=True)
                                    print("[TRANSCRIPTION] Switching to LOCAL Whisper", flush=True)
                                    st.write("📝 Transcribing audio locally...")
                                    transcript = transcribe_audio_local(audio_path)
                                    source_used = "Local Whisper (faster-whisper base)"
                            else:
                                if force_local_whisper:
                                    print("[TRANSCRIPTION] Cloud Whisper bypassed (FORCE_LOCAL_WHISPER=1)", flush=True)
                                else:
                                    print("[TRANSCRIPTION] Cloud Whisper unavailable (HF_TOKEN not set)", flush=True)
                                print("[TRANSCRIPTION] Switching to LOCAL Whisper", flush=True)
                                st.write("📝 Transcribing audio...")
                                transcript = transcribe_audio_local(audio_path)
                                source_used = "Local Whisper (faster-whisper base)"

                            # Real Video Frame Extraction & Visual Analysis
                            st.write("👁️ Analyzing video frames...")
                            frame_tuples = extract_video_frames(input_path, interval_seconds=FRAME_INTERVAL_SECONDS)
                            visual_records = []
                            for fpath, ts_sec, ts_fmt in frame_tuples:
                                desc = analyze_frame_visual(fpath, ts_fmt)
                                visual_records.append({
                                    "timestamp": ts_fmt,
                                    "timestamp_sec": ts_sec,
                                    "description": desc,
                                    "frame_path": fpath
                                })

                            st.write("🧠 Building multimodal knowledge base...")
                            run_unified_rag_pipeline(
                                transcript,
                                source_used,
                                "Uploaded Video",
                                meta,
                                file_hash,
                                playback_source=file_bytes,
                                visual_records=visual_records
                            )

                            status.update(label="✅ Video ready!", state="complete", expanded=False)
                        finally:
                            shutil.rmtree(temp_dir, ignore_errors=True)
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
                    <b>Transcript Chunks:</b> {st.session_state.chunk_count}
                </div>
                <div style="color: #94A3B8; font-size: 14px; margin-bottom: 6px;">
                    <b>Visual Frames Analyzed:</b> {len(st.session_state.get('visual_records', []))}
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
    with st.expander("📜 Full Transcript & Visual Analysis Summary", expanded=False):
        t_text = st.session_state.transcript_text
        c_stat1, c_stat2, c_stat3, c_stat4 = st.columns(4)
        with c_stat1:
            st.metric("Characters", f"{len(t_text):,}")
        with c_stat2:
            st.metric("Words", f"{len(t_text.split()):,}")
        with c_stat3:
            st.metric("Transcript Chunks", st.session_state.chunk_count)
        with c_stat4:
            st.metric("Visual Frames Analyzed", len(st.session_state.get("visual_records", [])))

        search_phrase = st.text_input("🔍 Search transcript for test phrase:", key="transcript_search_phrase")
        if search_phrase.strip():
            count = t_text.lower().count(search_phrase.lower())
            if count > 0:
                st.success(f"✓ Found {count} occurrence(s) of '{search_phrase}' in transcript.")
            else:
                st.warning(f"Phrase '{search_phrase}' not found in transcript.")

        st.text_area("Transcript Text", t_text, height=200, disabled=True, label_visibility="collapsed")

        # Visual Records Summary
        vis_recs = st.session_state.get("visual_records", [])
        if vis_recs:
            st.markdown("<h5 style='margin-top: 15px;'>👁️ Extracted Visual Frame Records</h5>", unsafe_allow_html=True)
            vis_lines = "\n".join(f"{r['timestamp']}: {r['description']}" for r in vis_recs)
            st.text_area("Visual Frame Records", vis_lines, height=150, disabled=True, label_visibility="collapsed")

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
                    <div style="font-size: 24px; margin-bottom: 8px;">👁️</div>
                    <div style="font-weight: 600; color: #F8FAFC; font-size: 15px; margin-bottom: 4px;">Visual Video Analysis</div>
                    <div style="color: #94A3B8; font-size: 13px;">Frame extraction & VLM analysis</div>
                </div>
                <div class="feature-card">
                    <div style="font-size: 24px; margin-bottom: 8px;">🧠</div>
                    <div style="font-weight: 600; color: #F8FAFC; font-size: 15px; margin-bottom: 4px;">Multimodal RAG</div>
                    <div style="color: #94A3B8; font-size: 13px;">FAISS dual-retrieval routing</div>
                </div>
                <div class="feature-card">
                    <div style="font-size: 24px; margin-bottom: 8px;">💬</div>
                    <div style="font-weight: 600; color: #F8FAFC; font-size: 15px; margin-bottom: 4px;">Grounded Answers</div>
                    <div style="color: #94A3B8; font-size: 13px;">Strict factual verification</div>
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
                with st.expander("📚 Sources / Retrieved Evidence"):
                    for idx, s in enumerate(msg["sources"], 1):
                        st.markdown(f"**Source:** {s.get('source', 'Video Evidence')}")
                        st.markdown(f"**Timestamp:** {s.get('timestamp', 'N/A')}")
                        st.markdown(f"**Evidence Content:**\n> {s.get('text', '')}")
                        if idx < len(msg["sources"]):
                            st.divider()

    # Handle user query with Multimodal RAG
    if user_query := st.chat_input("💬 Ask anything about your video"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing video evidence & generating answer..."):
                text_retriever = st.session_state.get("retriever")
                visual_retriever = st.session_state.get("visual_retriever")

                context_str, docs, intent = retrieve_multimodal_context(
                    user_query, text_retriever, visual_retriever
                )

                print("-" * 50)
                print(f"[MULTIMODAL RETRIEVAL DIAGNOSTICS]")
                print(f"Question: {user_query}")
                print(f"Intent classified: {intent}")
                print(f"Retrieved context items: {len(docs)}")
                if docs:
                    print(f"Top retrieved snippet: {docs[0].page_content[:120]}...")
                print("-" * 50)

                answer = generate_answer(context_str, user_query, docs=docs)
                if st.session_state.get("current_provider") == "Local GGUF":
                    st.caption("☁️ Cloud AI unavailable — using local AI model.")
                st.write(answer)

                sources_payload = []
                if "couldn't find the answer" not in answer.lower():
                    meta_title = st.session_state.video_metadata.get("title") or "Video"
                    for doc in docs:
                        doc_ts = doc.metadata.get("timestamp") if hasattr(doc, "metadata") and doc.metadata else None
                        is_visual = "Visual Analysis" in doc.page_content
                        sources_payload.append({
                            "source": f"{meta_title} ({'Visual Frame' if is_visual else 'Transcript'})",
                            "timestamp": doc_ts if doc_ts else "Full Clip Transcript",
                            "text": doc.page_content,
                        })

                    if sources_payload:
                        with st.expander("📚 Sources / Retrieved Evidence"):
                            for idx, s in enumerate(sources_payload, 1):
                                st.markdown(f"**Source:** {s['source']}")
                                st.markdown(f"**Timestamp:** {s['timestamp']}")
                                st.markdown(f"**Evidence Content:**\n> {s['text']}")
                                if idx < len(sources_payload):
                                    st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_payload,
        })


# Compatibility Alias
transcribe_audio_with_whisper = transcribe_with_whisper
