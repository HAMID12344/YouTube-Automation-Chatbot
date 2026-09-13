import os
import re
import shutil
import subprocess
import sys
import tempfile
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
    """
    Reads configuration safely from environment variables (.env) or st.secrets.
    """
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

st.set_page_config(
    page_title="YouTube & Video RAG Chatbot",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 800;
            margin-bottom: 2px;
        }
        .subtitle {
            font-size: 17px;
            color: #777;
            margin-bottom: 25px;
        }
        .meta-box {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 12px 18px;
            border-left: 4px solid #ff4b4b;
            margin-bottom: 20px;
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
        "source_type": None,       # 'YouTube' or 'Uploaded Video'
        "video_metadata": {},      # title, duration, author/filename, etc.
        "transcript_text": "",
        "transcript_source": "",
        "transcript_language": "",
        "chunk_count": 0,
        "processed_identifier": "",
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
            "Hugging Face token is missing. Please configure HF_TOKEN in your .env file."
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
        "Invalid YouTube URL or ID. Supported formats include youtube.com/watch?v=..., youtu.be/..., and youtube.com/shorts/..."
    )

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
        raise RuntimeError(f"Failed to download audio with yt-dlp: {exc}") from exc

    audio_path = os.path.join(temp_dir, f"{video_id}.mp3")
    if not os.path.exists(audio_path):
        candidates = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(('.mp3', '.m4a', '.wav'))]
        if candidates:
            audio_path = candidates[0]
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError("Audio extraction completed but output file not found.")

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
        err_msg = res.stderr.decode("utf-8", errors="ignore")[-400:]
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to extract audio using imageio-ffmpeg: {err_msg}")

    metadata = {
        "filename": orig_name,
        "size_mb": round(uploaded_file.size / (1024 * 1024), 2),
        "type": uploaded_file.type or "video",
    }
    return audio_path, temp_dir, metadata

# ============================================================
# TRANSCRIPTION: WHISPER & FALLBACK
# ============================================================

def transcribe_with_whisper(audio_path: str) -> str:
    """
    Performs real speech-to-text using Hugging Face InferenceClient with openai/whisper-large-v3.
    """
    require_hf_token()
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    client = InferenceClient(token=HF_TOKEN)
    try:
        response = client.automatic_speech_recognition(
            audio=audio_bytes,
            model=WHISPER_MODEL,
        )
    except Exception as exc:
        raise RuntimeError(f"Hugging Face Whisper API failed: {exc}") from exc

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


def fetch_fallback_youtube_transcript(video_id: str, preferred_languages: Optional[List[str]] = None) -> str:
    """
    Secondary fallback: fetches native YouTube transcript via YouTubeTranscriptApi.
    """
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
        raise RuntimeError(f"YouTube transcript fallback also failed: {exc}") from exc

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

def create_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.7},
    )

def create_prompt() -> PromptTemplate:
    template = """You are a helpful and strictly factual AI assistant answering questions about a video.

IMPORTANT RULES:
1. Answer ONLY from the supplied video context.
2. Do not invent, extrapolate, or fill in facts that are not present in the context.
3. If the answer is NOT present or supported by the context, you MUST respond exactly:
   "I couldn't find the answer to that in the video."
4. Keep the answer clear and concise.

VIDEO CONTEXT:
-------------------------
{context}
-------------------------

QUESTION:
{question}

ANSWER:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def generate_answer(context: str, question: str) -> str:
    if not context.strip():
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

    # Grounded answer safety check: normalize negative signals
    negative_indicators = [
        "couldn't find",
        "could not find",
        "cannot find",
        "not found",
        "not mentioned",
        "not provided",
        "not present",
        "not in the context",
        "not in the video",
        "no information",
        "does not mention",
        "doesn't mention",
        "don't have any information",
        "do not have any information",
        "isn't mentioned",
        "is not mentioned",
        "isn't provided",
        "is not provided",
    ]
    if any(ind in answer_clean.lower() for ind in negative_indicators):
        return "I couldn't find the answer to that in the video."

    return answer_clean

def run_unified_rag_pipeline(transcript: str, source_name: str, source_type: str, metadata: Dict[str, Any]):
    """
    Shared RAG processor for both YouTube URLs and Uploaded Video files.
    """
    chunks = split_text(transcript)
    if not chunks:
        raise RuntimeError("Transcript text was empty or could not be chunked.")

    vector_store = create_vector_store(chunks)
    retriever = create_retriever(vector_store)

    st.session_state.vector_store = vector_store
    st.session_state.retriever = retriever
    st.session_state.video_processed = True
    st.session_state.source_type = source_type
    st.session_state.video_metadata = metadata
    st.session_state.transcript_text = transcript
    st.session_state.transcript_source = source_name
    st.session_state.chunk_count = len(chunks)
    st.session_state.messages = []

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings & Info")
    st.markdown(
        f"""
        **Architecture**:
        - **Audio Extractor**: `yt-dlp` / `imageio-ffmpeg`
        - **Speech-to-Text**: Hugging Face `{WHISPER_MODEL}`
        - **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2`
        - **Vector Store**: `FAISS` (MMR Search)
        - **LLM**: Hugging Face `{HF_MODEL_ID}` (with local fallback)
        """
    )
    st.divider()

    if st.session_state.video_processed:
        st.subheader("📹 Active Video Details")
        meta = st.session_state.video_metadata
        if st.session_state.source_type == "YouTube":
            st.write(f"**Title**: {meta.get('title', 'N/A')}")
            st.write(f"**Author**: {meta.get('uploader', 'N/A')}")
            st.write(f"**Duration**: {meta.get('duration', 'N/A')} seconds")
        else:
            st.write(f"**Filename**: {meta.get('filename', 'N/A')}")
            st.write(f"**File Size**: {meta.get('size_mb', 'N/A')} MB")

        st.write(f"**Source**: `{st.session_state.transcript_source}`")
        st.write(f"**Chunks Created**: `{st.session_state.chunk_count}`")
        st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            for k in ["messages", "vector_store", "retriever", "video_processed", "video_metadata", "transcript_text", "transcript_source", "chunk_count", "processed_identifier"]:
                st.session_state[k] = [] if k == "messages" else (None if k in ["vector_store", "retriever"] else False if k == "video_processed" else ({} if k == "video_metadata" else (0 if k == "chunk_count" else "")))
            st.rerun()

# ============================================================
# MAIN UI
# ============================================================

st.markdown('<div class="main-title">🎬 YouTube & Video RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Grounded video Q&A with Whisper large-v3 & FAISS Vector Store</div>', unsafe_allow_html=True)

if not HF_TOKEN:
    st.warning("⚠️ Hugging Face token is missing. Please add `HF_TOKEN` to your `.env` file.")

tab_yt, tab_upload = st.tabs(["📺 YouTube Video", "📁 Upload Video File"])

# TAB 1: YOUTUBE INGESTION
with tab_yt:
    yt_url = st.text_input(
        "Enter YouTube Video URL or ID",
        placeholder="e.g. https://www.youtube.com/watch?v=jNQXAC9IVRw or youtu.be/...",
        key="yt_input_url",
    )
    process_yt = st.button("🚀 Process YouTube Video", type="primary", key="btn_process_yt")

    if process_yt:
        if not yt_url.strip():
            st.error("Please enter a valid YouTube URL or Video ID.")
        else:
            try:
                vid_id = extract_video_id(yt_url)
                with st.status("Processing YouTube Video...", expanded=True) as status:
                    st.write("1️⃣ Extracting audio stream via yt-dlp...")
                    audio_path, temp_dir, meta = extract_audio_from_youtube(yt_url)

                    transcript = None
                    source_used = None

                    try:
                        st.write(f"2️⃣ Transcribing audio via Hugging Face `{WHISPER_MODEL}`...")
                        transcript = transcribe_with_whisper(audio_path)
                        source_used = f"Whisper ({WHISPER_MODEL})"
                    except Exception as whisper_err:
                        st.warning(f"Whisper API note: {whisper_err}. Attempting YouTube subtitle fallback...")
                        try:
                            transcript = fetch_fallback_youtube_transcript(vid_id)
                            source_used = "youtube_transcript_api (fallback)"
                        except Exception as fb_err:
                            raise RuntimeError(f"Both Whisper and YouTube transcript fallback failed.\nWhisper error: {whisper_err}\nFallback error: {fb_err}")
                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)

                    st.write("3️⃣ Splitting transcript into chunks & generating multilingual embeddings...")
                    st.write("4️⃣ Indexing in-memory FAISS vector store...")
                    run_unified_rag_pipeline(transcript, source_used, "YouTube", meta)
                    st.session_state.processed_identifier = vid_id

                    status.update(label="✅ YouTube video processed and indexed!", state="complete")
                st.success(f"Ready! Transcribed via **{source_used}** ({len(transcript)} chars, {st.session_state.chunk_count} chunks).")
            except Exception as e:
                st.error(f"Error processing YouTube video: {e}")

# TAB 2: UPLOADED VIDEO INGESTION
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "mkv", "avi", "webm"],
        key="uploaded_file_input",
    )
    process_upload = st.button("🚀 Process Uploaded Video", type="primary", key="btn_process_upload")

    if process_upload:
        if not uploaded_file:
            st.error("Please upload a video file first.")
        else:
            try:
                with st.status("Processing Uploaded Video...", expanded=True) as status:
                    st.write("1️⃣ Extracting audio using bundled imageio-ffmpeg...")
                    audio_path, temp_dir, meta = extract_audio_from_video_file(uploaded_file)

                    try:
                        st.write(f"2️⃣ Transcribing audio via Hugging Face `{WHISPER_MODEL}`...")
                        transcript = transcribe_with_whisper(audio_path)
                        source_used = f"Whisper ({WHISPER_MODEL})"
                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)

                    st.write("3️⃣ Splitting transcript into chunks & generating multilingual embeddings...")
                    st.write("4️⃣ Indexing in-memory FAISS vector store...")
                    run_unified_rag_pipeline(transcript, source_used, "Uploaded Video", meta)
                    st.session_state.processed_identifier = uploaded_file.name

                    status.update(label="✅ Uploaded video processed and indexed!", state="complete")
                st.success(f"Ready! Transcribed via **{source_used}** ({len(transcript)} chars, {st.session_state.chunk_count} chunks).")
            except Exception as e:
                st.error(f"Error processing uploaded video: {e}")

# ============================================================
# CHAT INTERFACE
# ============================================================

st.divider()
st.subheader("💬 Ask Questions About the Video")

if not st.session_state.video_processed:
    st.info("👈 Please process a YouTube video or upload a video file above to begin chatting!")
else:
    meta = st.session_state.video_metadata
    label = meta.get("title") or meta.get("filename") or "Video"
    st.caption(f"Currently querying: **{label}** (Source: {st.session_state.transcript_source})")

    # Render previous conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 Retrieved Video Context Chunks"):
                    for idx, chunk in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {idx}:**\n> {chunk}")

    # Handle user query
    if user_query := st.chat_input("Ask a question grounded in the video..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching video transcript & generating answer..."):
                retriever = st.session_state.retriever
                docs = retriever.invoke(user_query) if retriever else []
                context_chunks = [doc.page_content for doc in docs]
                context_str = "\n\n---\n\n".join(context_chunks)

                answer = generate_answer(context_str, user_query)
                st.write(answer)

                if context_chunks:
                    with st.expander("🔍 Retrieved Video Context Chunks"):
                        for idx, chunk in enumerate(context_chunks, 1):
                            st.markdown(f"**Chunk {idx}:**\n> {chunk}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": context_chunks if "couldn't find the answer" not in answer.lower() else []
        })
