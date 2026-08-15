#!/usr/bin/env python3
"""
- API: /health /version /meta /warmup /setmodel /getmodel
       /v1/translate /v1/translate/{id} /v1/translate/{id}/image
       /v1/changemodel /v1/listmodels /v1/colorize
       /v1/ai/resolve /v1/ai/prompt/default
       /SetFont /GetFont /GetFonts /SetModelType /GetModelType /SetOpenRouterModel
       /SetInpaintMode /GetInpaintMode /SetOcrMode /GetOcrMode
- Logs: /console endpoint to view all backend logs and errors
"""

import asyncio
import base64
import bisect
import io
import json
import math
import os
import pathlib
import time
import traceback
import urllib.parse
import urllib.request
import uuid
import logging
import threading
import functools
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pydantic import BaseModel, Field

# --- FastAPI ---------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query, Request, Form, Depends
from fastapi.responses import JSONResponse, Response, HTMLResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# --- GPU / Device helpers --------------------------------------------------
import torch

def has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def get_torch_device() -> str:
    return "cuda" if has_cuda() else "cpu"

def get_llm_gpu_layers() -> int:
    return -1 if llama_cpp_gpu_available() else 0


def llama_cpp_gpu_available() -> bool:
    if not has_cuda() or llama_cpp is None:
        return False
    try:
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False


def _log_llama_device(component: str) -> bool:
    use_gpu = llama_cpp_gpu_available()
    if use_gpu:
        device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "CUDA"
        logging.info(f"[{component}] llama.cpp CUDA offload enabled on {device_name}; all model layers and KQV will be offloaded.")
    elif has_cuda():
        logging.warning(
            f"[{component}] CUDA is available to PyTorch, but this llama-cpp-python build has no GPU offload support. "
            "Using CPU fallback. Reinstall a CUDA wheel, for example: "
            "python -m pip install --upgrade --force-reinstall llama-cpp-python "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
        )
    else:
        logging.info(f"[{component}] CUDA is unavailable; using CPU fallback.")
    return use_gpu

logging.info(f"[Device] CUDA available: {has_cuda()} -> device='{get_torch_device()}'")

# --- GLM OCR Config (transformers) ---

_glm_ocr_model = None
_glm_ocr_processor = None
_glm_ocr_lock = threading.Lock()
GLM_OCR_REPO = "zai-org/GLM-OCR"

# --- Optional deps ---------------------------------------------------------
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from simple_lama import SimpleLama
except Exception:
    SimpleLama = None

try:
    import llama_cpp
    from llama_cpp import Llama
except Exception:
    llama_cpp = None
    Llama = None

try:
    from hayai_ocr import HayaiOcr
except Exception:
    HayaiOcr = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from chrome_lens_py import LensAPI
except Exception:
    LensAPI = None

# --- Sanitization ----
import re

_ALLOWED_RANGES = (
    (0x0020, 0x007E),
    (0x00A0, 0x00FF),
    (0x0100, 0x017F),
    (0x0180, 0x024F),
    (0x0400, 0x04FF),
    (0x0500, 0x052F),
    (0x2000, 0x206F),
    (0x3000, 0x303F),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xFF00, 0xFFEF),
)

_ALLOWED_LOWS  = tuple(r[0] for r in _ALLOWED_RANGES)
_ALLOWED_HIGHS = tuple(r[1] for r in _ALLOWED_RANGES)

_PUNCT_MAP = {
    0x2018: "'", 0x2019: "'",
    0x201C: '"', 0x201D: '"',
    0x2013: '-', 0x2014: '-',
    0x2026: '...',
    0x00A0: ' ',
    0x2022: '*',
    0x300C: '[', 0x300D: ']',
    0x300E: '[', 0x300F: ']',
    0x2122: '(TM)', 0x00A9: '(c)', 0x00AE: '(R)',
}

def _is_allowed_cp(cp: int) -> bool:
    idx = bisect.bisect_right(_ALLOWED_LOWS, cp) - 1
    return idx >= 0 and cp <= _ALLOWED_HIGHS[idx]

def clean_text_for_font(text: str) -> str:
    if not text:
        return ""
    if not hasattr(clean_text_for_font, '_trans_table'):
        clean_text_for_font._punct_table = str.maketrans(
            {chr(cp): rep for cp, rep in _PUNCT_MAP.items()}
        )
        clean_text_for_font._re_space = re.compile(r'[ \t]+')
        clean_text_for_font._re_nl   = re.compile(r'\n+')
    out = text.translate(clean_text_for_font._punct_table)
    out = ''.join(
        ch for ch in out
        if (ch in '\t\n') or (0x20 <= ord(ch) and _is_allowed_cp(ord(ch)))
    )
    out = clean_text_for_font._re_space.sub(' ', out)
    out = clean_text_for_font._re_nl.sub(' ', out)
    return out.strip()


# --- Config ----------------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_DIR = ROOT_DIR / "models"
LOCAL_VISION_TEMPLATE_PATH = ROOT_DIR / "jinja.txt"
LOCAL_VISION_CHAT_TEMPLATE = ""
try:
    LOCAL_VISION_CHAT_TEMPLATE = LOCAL_VISION_TEMPLATE_PATH.read_text(encoding="utf-8")
    logging.info(f"[Local Vision OCR] Loaded chat template from {LOCAL_VISION_TEMPLATE_PATH}")
except OSError as exc:
    logging.warning(f"[Local Vision OCR] Could not load {LOCAL_VISION_TEMPLATE_PATH}: {exc}")

MODEL_DIR.mkdir(exist_ok=True)
YOLO_MODEL_PATH = MODEL_DIR / "yolo_manga_textbox.pt"
YOLO_HF_RAW = "https://huggingface.co/Kirogii/Yolo-Manga_Textbox-Region_Detect/resolve/main/model.pt"

Qwen_REPO_ID = "Manojb/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
Qwen_MODEL_FILENAME = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"

INPAINT_RADIUS_CV2 = 7
INPAINT_TELEA_RADIUS = 10
INPAINT_NS_RADIUS = 7
INPAINT_DILATE_PASSES = 2
INPAINT_FEATHER_PX = 3
INPAINT_USE_MULTI_PASS = True
INPAINT_COLOR_MATCH = True
LOCAL_VISION_INPAINT_PADDING = 6
LOCAL_VISION_INPAINT_DILATE_KERNEL = 5

# --- Inpainting Model Config (Low/High) -----------------------------------
LAMA_LARGE_URL = "https://huggingface.co/df1412/anime-big-lama/resolve/main/anime-manga-big-lama.pt"
LAMA_LARGE_PATH = MODEL_DIR / "anime-manga-big-lama.pt"

FONT_DIR = ROOT_DIR / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = FONT_DIR / "NotoCJK.ttc"
FONT_URL = "https://github.com/Kirogii/MangaAMTL/releases/download/Packages/NotoCJK.ttc"

if not FONT_PATH.exists():
    try:
        logging.info(f"Downloading font from {FONT_URL}")
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        logging.info(f"Font downloaded: {FONT_PATH}")
    except Exception as e:
        logging.warning(f"Failed to download font: {e}")
        logging.warning("Falling back to NotoCJK.ttf or PIL default.")
        FONT_PATH = pathlib.Path("NotoCJK.ttf")

if not FONT_PATH.exists():
    logging.warning(f"Fallback font {FONT_PATH} not found. PIL default will be used.")

DEFAULT_LANG       = "en"
BUILD_ID           = "manga-v1-2025.01"

# --- Colorizer Config ------------------------------------------------------
COLORIZER_DIR = MODEL_DIR / "colorizer"
COLORIZER_DIR.mkdir(parents=True, exist_ok=True)
COLORIZER_GENERATOR_PATH = COLORIZER_DIR / "v6_generator.onnx"
COLORIZER_SAM_PATH = COLORIZER_DIR / "v6_sam_encoder.onnx"
COLORIZER_GENERATOR_URL = "https://huggingface.co/sharky172/manga-light-colorizer/resolve/main/models/v6_generator.onnx"
COLORIZER_SAM_URL = "https://huggingface.co/sharky172/manga-light-colorizer/resolve/main/models/v6_sam_encoder.onnx"
COLORIZER_DEFAULT_INFER_SIZE = 768

# --- GGUF Model Config -----------------------------------------------------
GGUF_DIR = MODEL_DIR / "gguf"
GGUF_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = ROOT_DIR / "settings.json"

# --- Logging / Console -----------------------------------------------------
class MemoryLogHandler(logging.Handler):
    def __init__(self, capacity: int = 2000):
        super().__init__()
        self.logs = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        self.logs.append(self.format(record))

    def get_logs(self) -> List[str]:
        return list(self.logs)

log_handler = MemoryLogHandler()
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(log_handler)

logging.getLogger("uvicorn").addHandler(log_handler)
logging.getLogger("uvicorn.access").addHandler(log_handler)

# --- Globals ---------------------------------------------------------------
app = FastAPI(title="Manga Translation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

_simple_lama_model = None       # low mode (default SimpleLama / big-lama.pt)
_simple_lama_high_model = None  # high mode (anime-manga-big-lama.pt)
_global_yolo       = None
_global_qwen       = None
_local_vision_qwen = None
_local_vision_model_path: Optional[pathlib.Path] = None
_local_vision_projector_path: Optional[pathlib.Path] = None
_hayai_ocr_model   = None
_paddle_ocr_model  = None

_current_ocr_model = "ja"
_ocr_model_lock = threading.Lock()

_colorizer_session = None
_colorizer_sam_session = None
_colorizer_lock = threading.Lock()
_colorize_enabled = False

_current_qwen_repo_id = Qwen_REPO_ID
_current_qwen_filename = Qwen_MODEL_FILENAME
_current_qwen_path: Optional[pathlib.Path] = None
_qwen_model_lock = threading.Lock()
_yolo_lock = threading.RLock()
_local_vision_inference_lock = threading.Lock()

_jobs: Dict[str, Dict[str, Any]] = {}
_job_lock = asyncio.Lock()
_job_queue: Optional[asyncio.Queue] = None
_worker_task = None
JOB_HEALTH_TIMEOUT_SECONDS = 75.0

_llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
_llm_lock = threading.Lock()

# Inpainting executor (runs in thread pool since it can be slow)
_inpaint_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inpaint")
_inpaint_lock = threading.Lock()

# --- Inpainting Mode Globals ---
_inpaint_mode = "low"  # "low" or "high"
_inpaint_mode_lock = threading.Lock()

# --- OCR Mode Globals ---
_ocr_mode = "hayai"  # "hayai", "glm", "lens", or "openai_endpoint"
_ocr_mode_lock = threading.Lock()

# --- OpenAI-compatible OCR Endpoint Globals ---
_openai_ocr_endpoint: str = "https://api.openai.com/v1/chat/completions"
_openai_ocr_api_key: Optional[str] = None
_openai_ocr_model: str = "gpt-4o-mini"
_openai_ocr_config_lock = threading.Lock()

# --- Google AI Studio OCR Globals ---
_google_ai_ocr_api_key: Optional[str] = None
_google_ai_ocr_model: str = "gemini-2.5-flash-lite"
_google_ai_ocr_rpm: int = 5
_google_ai_ocr_config_lock = threading.Lock()
_google_ai_ocr_rate_lock = asyncio.Lock()
_google_ai_ocr_last_request: float = 0.0

# --- Cloud Mode Globals ---
# When on, the server avoids loading heavy local models (OCR, inpainting,
# colorizer) and relies on cloud services (Google Lens + OpenRouter) instead.
_cloud_mode = False
_cloud_mode_lock = threading.Lock()

# --- Google Lens Globals ---
_lens_api = None
_lens_lock = threading.Lock()

# --- Font Configuration Globals ---
_current_font_path: pathlib.Path = FONT_PATH
_current_stroke_width: int = 0
_font_config_lock = threading.Lock()

# --- Model Type Configuration Globals ---
_current_model_type: str = "local"
_openrouter_api_key: Optional[str] = None
_openrouter_model: str = "openai/gpt-4o-mini"
_model_type_lock = threading.Lock()

# --- OpenRouter Paid-Mode Global ---
# The UI checkbox is "Paid model". A paid OpenRouter account is not on a
# free-tier per-minute quota, so when this is True the translate functions never
# sleep on HTTP 429 — they retry straight away — and the per-box fallback stops
# throttling itself, which is what made translation look one-at-a-time even
# when nothing was actually rate limited.
# When False (free tier) a 429 is waited out properly: whatever the server tells
# us via Retry-After / X-RateLimit-Reset, else OPENROUTER_RATELIMIT_AVG_WAIT.
#
# The wire field / storage key is `free_openrouter` and the endpoints are named
# `/SetOpenRouterFreeMode`, so the canonical global keeps the "free_mode" name
# even though the checkbox is now labelled "Paid model". True == paid account ==
# skip 429 backoff (retry the batch immediately, no per-box fallback).
_openrouter_free_mode: bool = False
_openrouter_free_mode_lock = threading.Lock()

# Free-tier OpenRouter quotas are per-minute windows, so a 429 clears somewhere
# in the next 0-60s. 30s is the expected wait when the response carries no
# reset hint of its own.
OPENROUTER_RATELIMIT_AVG_WAIT = 30.0

# Never sit on a 429 longer than this even if the server asks us to — a job that
# blocks for ten minutes reads as a hang.
OPENROUTER_RATELIMIT_MAX_WAIT = 300.0

# Spacing between per-box fallback requests on the free tier, to stay under the
# per-minute quota. Paid mode skips it and fans the boxes out instead.
OPENROUTER_FREE_TIER_THROTTLE = 1.0

# How many per-box fallback requests a paid account sends at once.
OPENROUTER_FALLBACK_CONCURRENCY = 6

# --- Context-Aware Mode Global ---
# When True, the translation pipeline:
#   1) Appends an honorific-preservation clause to the system prompt.
#   2) Makes ONE extra LLM call per job to build a names dictionary.
#   3) Post-processes translations to reinsert dropped honorifics.
# Cost: ~1 extra API request per chapter + ~200-600 tokens (stated in UI).
_context_aware_mode: bool = False
_context_aware_lock = threading.Lock()
_settings_lock = threading.Lock()


def _settings_snapshot() -> Dict[str, Any]:
    return {
        "cloud_mode": _cloud_mode,
        "ocr_mode": _ocr_mode,
        "inpaint_mode": _inpaint_mode,
        "model_type": _current_model_type,
        "local_model_repo_id": _current_qwen_repo_id,
        "local_model_filename": _current_qwen_filename,
        "openrouter_model": _openrouter_model,
        "openrouter_api_key": _openrouter_api_key,
        "openai_ocr_endpoint": _openai_ocr_endpoint,
        "openai_ocr_model": _openai_ocr_model,
        "openai_ocr_api_key": _openai_ocr_api_key,
        "google_ai_ocr_api_key": _google_ai_ocr_api_key,
        "google_ai_ocr_model": _google_ai_ocr_model,
        "google_ai_ocr_rpm": _google_ai_ocr_rpm,
        "font_path": str(_current_font_path),
        "stroke_width": _current_stroke_width,
        "free_openrouter": _openrouter_free_mode,
        "context_aware": _context_aware_mode,
    }


def _save_settings() -> None:
    snapshot = _settings_snapshot()
    temp_path = SETTINGS_PATH.with_suffix(".json.tmp")
    try:
        with _settings_lock:
            temp_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            temp_path.replace(SETTINGS_PATH)
    except OSError as exc:
        logging.error(f"[Settings] Could not persist settings: {exc}")


def _load_settings() -> None:
    global _cloud_mode, _ocr_mode, _inpaint_mode, _current_model_type
    global _current_qwen_repo_id, _current_qwen_filename
    global _openrouter_model, _openrouter_api_key
    global _openai_ocr_endpoint, _openai_ocr_model, _openai_ocr_api_key
    global _google_ai_ocr_api_key, _google_ai_ocr_model, _google_ai_ocr_rpm
    global _current_font_path, _current_stroke_width
    global _openrouter_free_mode, _context_aware_mode
    if not SETTINGS_PATH.exists():
        return
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("settings root must be an object")
        if data.get("ocr_mode") in {"hayai", "glm", "lens", "openai_endpoint", "google_ai", "local_vision"}:
            _ocr_mode = data["ocr_mode"]
        if data.get("inpaint_mode") in {"low", "high", "none"}:
            _inpaint_mode = data["inpaint_mode"]
        if data.get("model_type") in {"local", "openrouter"}:
            _current_model_type = data["model_type"]
        _cloud_mode = bool(data.get("cloud_mode", _cloud_mode))
        _current_qwen_repo_id = str(data.get("local_model_repo_id") or _current_qwen_repo_id)
        _current_qwen_filename = str(data.get("local_model_filename") or _current_qwen_filename)
        _openrouter_model = str(data.get("openrouter_model") or _openrouter_model)
        _openrouter_api_key = data.get("openrouter_api_key") or None
        _openai_ocr_endpoint = str(data.get("openai_ocr_endpoint") or _openai_ocr_endpoint)
        _openai_ocr_model = str(data.get("openai_ocr_model") or _openai_ocr_model)
        _openai_ocr_api_key = data.get("openai_ocr_api_key") or None
        _google_ai_ocr_api_key = data.get("google_ai_ocr_api_key") or None
        _google_ai_ocr_model = str(data.get("google_ai_ocr_model") or _google_ai_ocr_model)
        rpm = data.get("google_ai_ocr_rpm")
        if isinstance(rpm, int) and 1 <= rpm <= 15:
            _google_ai_ocr_rpm = rpm
        font_path = pathlib.Path(str(data.get("font_path") or _current_font_path))
        if font_path.exists():
            _current_font_path = font_path
        _current_stroke_width = max(0, min(20, int(data.get("stroke_width", _current_stroke_width))))
        _openrouter_free_mode = bool(data.get("free_openrouter", _openrouter_free_mode))
        _context_aware_mode = bool(data.get("context_aware", _context_aware_mode))
        logging.info(f"[Settings] Restored settings from {SETTINGS_PATH}")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logging.error(f"[Settings] Ignoring invalid persisted settings: {exc}")


_load_settings()

# ===========================================================================
# Download helpers
# ===========================================================================
def download_if_missing(url: str, dest: pathlib.Path) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    logging.info(f"Downloading {url} -> {dest} ...")
    urllib.request.urlretrieve(url, dest)
    return dest

def ensure_yolo():
    if YOLO is None:
        raise RuntimeError("ultralytics not installed: pip install ultralytics")
    if not YOLO_MODEL_PATH.exists():
        download_if_missing(YOLO_HF_RAW, YOLO_MODEL_PATH)
    return YOLO_MODEL_PATH

def ensure_lama_large():
    """Download anime-manga-big-lama.pt if missing."""
    if not LAMA_LARGE_PATH.exists() or LAMA_LARGE_PATH.stat().st_size < 10000:
        logging.info(f"[Lama High] Downloading high-quality inpainting model...")
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id="df1412/anime-big-lama", filename="anime-manga-big-lama.pt")
            shutil.copy(str(p), str(LAMA_LARGE_PATH))
            logging.info(f"[Lama High] Downloaded to {LAMA_LARGE_PATH}")
        except ImportError:
            download_if_missing(LAMA_LARGE_URL, LAMA_LARGE_PATH)
            logging.info(f"[Lama High] Downloaded to {LAMA_LARGE_PATH}")
    return LAMA_LARGE_PATH

# ===========================================================================
# Image utils
# ===========================================================================
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.asarray(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

# ===========================================================================
# Colorizer (ONNX)
# ===========================================================================
def ensure_colorizer_models():
    if not COLORIZER_GENERATOR_PATH.exists() or COLORIZER_GENERATOR_PATH.stat().st_size < 10000:
        logging.info(f"[Colorizer] Downloading generator via HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id="sharky172/manga-light-colorizer", filename="models/v6_generator.onnx")
            shutil.copy(str(p), str(COLORIZER_GENERATOR_PATH))
        except ImportError:
            download_if_missing(COLORIZER_GENERATOR_URL, COLORIZER_GENERATOR_PATH)

    if not COLORIZER_SAM_PATH.exists() or COLORIZER_SAM_PATH.stat().st_size < 10000:
        logging.info(f"[Colorizer] Downloading SAM encoder via HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id="sharky172/manga-light-colorizer", filename="models/v6_sam_encoder.onnx")
            shutil.copy(str(p), str(COLORIZER_SAM_PATH))
        except ImportError:
            download_if_missing(COLORIZER_SAM_URL, COLORIZER_SAM_PATH)

def get_colorizer_sessions():
    global _colorizer_session, _colorizer_sam_session
    if ort is None:
        raise RuntimeError("onnxruntime not installed: pip install onnxruntime")
    with _colorizer_lock:
        if _colorizer_session is None:
            ensure_colorizer_models()
            available = ort.get_available_providers()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in available
                         else ["CPUExecutionProvider"])
            logging.info(f"[Colorizer] Loading generator: {COLORIZER_GENERATOR_PATH}")
            _colorizer_session = ort.InferenceSession(str(COLORIZER_GENERATOR_PATH), providers=providers)
            if COLORIZER_SAM_PATH.exists():
                logging.info(f"[Colorizer] Loading SAM encoder: {COLORIZER_SAM_PATH}")
                _colorizer_sam_session = ort.InferenceSession(str(COLORIZER_SAM_PATH), providers=providers)
            else:
                _colorizer_sam_session = None
            logging.info(f"[Colorizer] Ready. Provider: {_colorizer_session.get_providers()[0]}, "
                         f"SAM: {'on' if _colorizer_sam_session else 'off'}")
    return _colorizer_session, _colorizer_sam_session

def _denormalize_rgb(rgb_norm: np.ndarray) -> np.ndarray:
    return np.clip((rgb_norm + 1.0) * 127.5, 0, 255).astype(np.uint8)

def _extract_sam_features_onnx(sam_session, L_bw_norm: np.ndarray):
    L_01 = (L_bw_norm + 1.0) / 2.0
    L_1024 = cv2.resize(L_01, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    rgb_sam = np.stack([L_1024, L_1024, L_1024], axis=0)[np.newaxis].astype(np.float32)
    sam_out = sam_session.run(None, {"rgb_input": rgb_sam})
    sam_level0 = sam_out[0]
    sam_level1 = sam_out[1]
    wd14_embedding = np.zeros((1, 1024), dtype=np.float32)
    return sam_level0, sam_level1, wd14_embedding

def _colorize_onnx(session, L_bw, sam_level0, sam_level1, wd14_embedding) -> np.ndarray:
    L_norm = (L_bw.astype(np.float32) / 127.5) - 1.0
    L_tensor = L_norm[np.newaxis, np.newaxis, :, :]
    ort_inputs = {
        "L_bw": L_tensor,
        "sam_level0": sam_level0,
        "sam_level1": sam_level1,
        "wd14_embedding": wd14_embedding,
    }
    rgb_pred = session.run(None, ort_inputs)[0]
    rgb_pred = rgb_pred[0].transpose(1, 2, 0)
    return _denormalize_rgb(rgb_pred)

def colorize_pil(pil_img: Image.Image,
                 infer_size: int = COLORIZER_DEFAULT_INFER_SIZE) -> Image.Image:
    session, sam_session = get_colorizer_sessions()
    gray = np.array(pil_img.convert("L"))
    orig_H, orig_W = gray.shape
    L_bw = cv2.resize(gray, (infer_size, infer_size), interpolation=cv2.INTER_AREA)
    H_in, W_in = L_bw.shape
    L_norm = (L_bw.astype(np.float32) / 127.5) - 1.0

    if sam_session is not None:
        sam_level0, sam_level1, wd14_embedding = _extract_sam_features_onnx(sam_session, L_norm)
    else:
        sam_level0 = np.zeros((1, 256, H_in // 16, W_in // 16), dtype=np.float32)
        sam_level1 = np.zeros((1, 256, H_in // 32, W_in // 32), dtype=np.float32)
        wd14_embedding = np.zeros((1, 1024), dtype=np.float32)

    rgb_output = _colorize_onnx(session, L_bw, sam_level0, sam_level1, wd14_embedding)
    rgb_output = cv2.resize(rgb_output, (orig_W, orig_H), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(rgb_output)

# ===========================================================================
# GGUF model management
# ===========================================================================
def _is_projector_gguf(path: pathlib.Path) -> bool:
    name = path.name.lower()
    return "mmproj" in name or "mmproject" in name or "projector" in name


def _has_embedded_vision_metadata(path: pathlib.Path) -> bool:
    """Detect GGUF multimodal models whose vision adapter is embedded in the model."""
    if "gemma-4" not in path.name.lower() and "gemma4" not in path.name.lower():
        return False
    try:
        with path.open("rb") as handle:
            header = handle.read(2 * 1024 * 1024).lower()
        return b"vision" in header and (b"multimodal" in header or b"mmproj" in header or b"clip" in header)
    except OSError:
        return False


def _find_external_projector(path: pathlib.Path) -> Optional[pathlib.Path]:
    search_roots = [
        pathlib.Path.home() / ".lmstudio" / "models",
        pathlib.Path.home() / ".cache" / "lm-studio",
    ]
    model_tokens = set(re.findall(r"[a-z0-9]+", path.stem.lower())) - {
        "q4", "q8", "q6", "q5", "q3", "q2", "k", "m", "it", "gguf"
    }
    model_family = set(re.findall(r"[a-z0-9]+", path.parent.name.lower()))
    candidates: List[pathlib.Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        try:
            candidates.extend(candidate for candidate in root.rglob("*.gguf") if _is_projector_gguf(candidate))
        except OSError:
            continue
    def _size_markers(value: str) -> set[str]:
        return {marker.lower() for marker in re.findall(r"\d+(?:\.\d+)?b", value.lower())}

    model_sizes = _size_markers(path.stem)
    ranked: List[Tuple[int, pathlib.Path]] = []
    for candidate in candidates:
        candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate.stem.lower()))
        candidate_family = set(re.findall(r"[a-z0-9]+", candidate.parent.name.lower()))
        candidate_sizes = _size_markers(candidate.stem + " " + candidate.parent.name)
        if model_sizes and candidate_sizes and model_sizes.isdisjoint(candidate_sizes):
            continue
        shared_model = len(model_tokens & candidate_tokens)
        shared_family = len(model_family & candidate_family)
        same_parent_family = int(path.parent.name.lower() == candidate.parent.name.lower())
        score = shared_model * 4 + shared_family * 3 + same_parent_family * 20
        if same_parent_family or (shared_model >= 2 and shared_family >= 1):
            ranked.append((score, candidate))
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def _model_record(repo_id: str, path: pathlib.Path) -> Dict[str, Any]:
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except OSError:
        size_mb = 0.0
    projector_candidates = [
        candidate for candidate in path.parent.glob("*.gguf")
        if _is_valid_gguf(candidate) and _is_projector_gguf(candidate)
    ]
    if path.parent.resolve() == GGUF_DIR.resolve():
        repo_prefix = repo_id.replace("/", "__").lower() + "__"
        projector_candidates = [
            candidate for candidate in projector_candidates
            if candidate.name.lower().startswith(repo_prefix)
        ]
    projector = projector_candidates[0] if projector_candidates else _find_external_projector(path)
    return {
        "name": f"{repo_id.replace('/', '__')}__{path.name}",
        "repo_id": repo_id,
        "filename": path.name,
        "size_mb": round(size_mb, 1),
        "path": str(path.resolve()),
        "vision_capable": projector is not None,
        "vision_adapter": "projector" if projector else None,
        "projector_filename": projector.name if projector else None,
        "projector_path": str(projector.resolve()) if projector else None,
    }


def _is_valid_gguf(path: pathlib.Path) -> bool:
    try:
        if not path.exists():
            return False
        if path.stat().st_size < 1024:
            return False
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == b"GGUF"
    except OSError:
        return False

def _hf_hub_cache_dir() -> Optional[pathlib.Path]:
    for env_var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        val = os.environ.get(env_var)
        if val:
            p = pathlib.Path(val)
            if env_var == "HF_HOME":
                p = p / "hub"
            if p.exists():
                return p
    default = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    return default if default.exists() else None

def _hf_cache_model_path(repo_id: str, filename: str) -> Optional[pathlib.Path]:
    cache_dir = _hf_hub_cache_dir()
    if cache_dir is None:
        return None
    org, sep, name = repo_id.partition("/")
    repo_dir_name = f"models--{org}--{name}" if sep else f"models--{name}"
    repo_dir = cache_dir / repo_dir_name
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return None
    preferred_hash: Optional[str] = None
    ref_file = repo_dir / "refs" / "main"
    if ref_file.exists():
        try:
            preferred_hash = ref_file.read_text().strip()
        except OSError:
            pass
    candidates: List[pathlib.Path] = []
    if preferred_hash:
        p = snapshots / preferred_hash / filename
        if p.exists():
            candidates.append(p)
    for snap in sorted(snapshots.iterdir()):
        p = snap / filename
        if p.exists() and p not in candidates:
            candidates.append(p)
    for c in candidates:
        try:
            real = c.resolve()
            if real.exists() and _is_valid_gguf(real):
                return c
        except OSError:
            continue
    return None

def _scan_hf_cache_for_ggufs() -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    cache_dir = _hf_hub_cache_dir()
    if cache_dir is None:
        return models
    for repo_dir in cache_dir.iterdir():
        if not repo_dir.is_dir() or not repo_dir.name.startswith("models--"):
            continue
        stripped = repo_dir.name[len("models--"):]
        parts = stripped.split("--")
        repo_id = "/".join(parts) if len(parts) >= 2 else parts[0]
        snapshots = repo_dir / "snapshots"
        if not snapshots.exists():
            continue
        for snap in snapshots.iterdir():
            if not snap.is_dir():
                continue
            for f in snap.glob("*.gguf"):
                if not _is_valid_gguf(f) or _is_projector_gguf(f):
                    continue
                models.append(_model_record(repo_id, f))
    return models

def _gguf_local_path(repo_id: str, filename: str) -> pathlib.Path:
    repo_clean = repo_id.rstrip("/").replace("/", "__")
    if repo_clean.lower().endswith(".gguf"):
        repo_clean = repo_clean[:-5]
    file_stem = pathlib.Path(filename).stem
    if repo_clean.lower().endswith(file_stem.lower()):
        safe = f"{repo_clean}.gguf"
    else:
        safe = f"{repo_clean}__{filename}"
    return GGUF_DIR / safe

def download_gguf(repo_id: str, filename: Optional[str] = None) -> pathlib.Path:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")

    if not filename:
        logging.info(f"[GGUF] No filename provided for {repo_id}, scanning repo for .gguf files...")
        files = list_repo_files(repo_id)
        gguf_files = [f for f in files if f.endswith('.gguf')]
        if not gguf_files:
            raise RuntimeError(f"No .gguf files found in repo: {repo_id}")
        filename = next((f for f in gguf_files if "q4_k_m" in f.lower()), gguf_files[0])
        logging.info(f"[GGUF] Auto-selected file: {filename}")

    local_path = _gguf_local_path(repo_id, filename)

    legacy_doubled = GGUF_DIR / f"{local_path.stem}__{filename}"
    if legacy_doubled.exists() and legacy_doubled != local_path:
        logging.warning(f"[GGUF] Removing legacy doubled file to save space: {legacy_doubled}")
        try:
            legacy_doubled.unlink()
        except OSError as e:
            logging.warning(f"[GGUF] Could not remove legacy file: {e}")

    hf_cached_path = _hf_cache_model_path(repo_id, filename)
    if hf_cached_path is not None:
        resolved = hf_cached_path.resolve()
        logging.info(f"[GGUF] Using HF cache directly: {resolved}")
        return resolved

    if _is_valid_gguf(local_path):
        return local_path

    if local_path.exists():
        logging.warning(f"[GGUF] Local mirror {local_path} is missing/invalid — removing it.")
        try:
            local_path.unlink()
        except OSError as e:
            logging.warning(f"[GGUF] Could not remove stale mirror: {e}")

    logging.info(f"[GGUF] Downloading {repo_id}/{filename} via huggingface_hub...")
    try:
        cached = pathlib.Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {repo_id}/{filename}. "
            f"Check repo_id/filename (HTTP 404 / LFS pointer / network). Error: {e}"
        )

    if not _is_valid_gguf(cached):
        try:
            with open(cached, "rb") as f:
                head = f.read(64)
            raise RuntimeError(
                f"HF cache file is not a valid GGUF (bad magic). "
                f"First 64 bytes: {head!r}. "
                f"You may need `huggingface-cli download {repo_id} {filename} "
                f"--local-dir ./models/gguf --force-download`."
            )
        except OSError:
            raise RuntimeError("HF cache file is not a valid GGUF and could not be inspected.")

    resolved = cached.resolve()
    logging.info(f"[GGUF] Download complete, using HF cache path: {resolved}")
    return resolved

def list_local_gguf_models() -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    if GGUF_DIR.exists():
        for f in sorted(GGUF_DIR.glob("*.gguf")):
            if not _is_valid_gguf(f) or _is_projector_gguf(f):
                continue
            stem = f.stem
            parts = stem.split("__")
            if len(parts) >= 2:
                filename_part = parts[-1]
                repo_part = "/".join(parts[:-1])
            else:
                filename_part = stem
                repo_part = stem
            record = _model_record(repo_part, f)
            record["name"] = stem
            record["filename"] = filename_part + ".gguf"
            models.append(record)
    models.extend(_scan_hf_cache_for_ggufs())
    seen = set()
    unique = []
    for m in models:
        key = (m["repo_id"], m["filename"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    return unique

# ===========================================================================
# Hayai OCR (Japanese)
# ===========================================================================
_OCR_BOX_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ocr-box")

def get_hayai_ocr():
    global _hayai_ocr_model
    if _hayai_ocr_model is None:
        if HayaiOcr is None:
            raise RuntimeError("hayai-ocr not installed: pip install hayai-ocr")
        device = get_torch_device()
        logging.info(f"[Hayai OCR] Loading model on device: {device} ...")
        try:
            _hayai_ocr_model = HayaiOcr(device=device)
        except TypeError:
            _hayai_ocr_model = HayaiOcr()
        logging.info(f"[Hayai OCR] Model loaded (device={device}).")
    return _hayai_ocr_model

def get_yolo():
    global _global_yolo
    with _yolo_lock:
        if _global_yolo is None:
            ensure_yolo()
            device = get_torch_device()
            logging.info(f"[YOLO] Loading model on device: {device}")
            _global_yolo = YOLO(str(YOLO_MODEL_PATH))
            _global_yolo.to(device)
            logging.info(f"[YOLO] Ready on {device}.")
        return _global_yolo


def _detect_yolo_text_boxes(pil_img: Image.Image) -> List[Tuple[int, int, int, int]]:
    img_bgr = pil_to_cv2(pil_img)
    h, w = img_bgr.shape[:2]
    logging.info(f"[YOLO] Detecting text regions on {w}x{h} image...")
    with _yolo_lock:
        results = get_yolo()(img_bgr, verbose=False, conf=0.4, device=get_torch_device())
    if not results:
        return []

    img_area = h * w
    boxes: List[Tuple[int, int, int, int]] = []
    for box in results[0].boxes:
        xy = box.xyxy[0].cpu().numpy()
        x1, y1 = max(0, int(xy[0])), max(0, int(xy[1]))
        x2, y2 = min(w, int(xy[2])), min(h, int(xy[3]))
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > 0.8 * img_area or box_area < 100:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def hayai_ocr_with_yolo(pil_img: Image.Image) -> List[Dict[str, Any]]:
    boxes = _detect_yolo_text_boxes(pil_img)
    if not boxes:
        return []
    out = []
    mocr = get_hayai_ocr()
    def _ocr_one(bbox):
        x1, y1, x2, y2 = bbox
        crop = pil_img.crop((x1, y1, x2, y2))
        try:
            return bbox, mocr(crop).strip()
        except Exception as e:
            logging.error(f"Hayai OCR failed on {bbox}: {e}")
            return bbox, ""
    for bbox, text in _OCR_BOX_EXECUTOR.map(_ocr_one, boxes):
        out.append({"text": text, "bbox": bbox})
    return out

# ===========================================================================
# GLM OCR (Korean - transformers)
# ===========================================================================
def get_glm_ocr():
    global _glm_ocr_model, _glm_ocr_processor
    if _glm_ocr_model is None:
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise RuntimeError("transformers not installed: pip install transformers accelerate torch")
        if not has_cuda():
            raise RuntimeError("PyTorch can't see your CUDA GPU.")
        dtype = torch.float16
        device = "cuda"
        logging.info(f"[GLM OCR] Loading {GLM_OCR_REPO} on GPU (dtype={dtype})...")
        with _glm_ocr_lock:
            if _glm_ocr_model is None:
                _glm_ocr_model = AutoModelForImageTextToText.from_pretrained(
                    GLM_OCR_REPO, torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True,
                ).to(device)
                _glm_ocr_model.eval()
                _glm_ocr_processor = AutoProcessor.from_pretrained(GLM_OCR_REPO)
                logging.info(f"[GLM OCR] Model loaded on {device}.")
    return _glm_ocr_model, _glm_ocr_processor

def glm_ocr_korean(pil_img: Image.Image) -> List[Dict[str, Any]]:
    model, processor = get_glm_ocr()
    img_bgr = pil_to_cv2(pil_img)
    h, w = img_bgr.shape[:2]
    yolo = get_yolo()
    logging.info(f"[GLM OCR] Running YOLO text detection on {w}x{h} image...")
    results = yolo(img_bgr, verbose=False, conf=0.4, device=get_torch_device())
    if not results:
        return []
    r = results[0]
    img_area = h * w
    boxes = []
    for b in r.boxes:
        xy = b.xyxy[0].cpu().numpy()
        x1, y1 = max(0, int(xy[0])), max(0, int(xy[1]))
        x2, y2 = min(w - 1, int(xy[2])), min(h - 1, int(xy[3]))
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > 0.8 * img_area or box_area < 100:
            continue
        boxes.append((x1, y1, x2, y2))
    if not boxes:
        return []
    logging.info(f"[GLM OCR] Found {len(boxes)} valid text boxes. Running GLM OCR on each...")
    TARGET_MAX = 1024
    TARGET_MIN = 384
    def _ocr_one(bbox):
        x1, y1, x2, y2 = bbox
        crop = pil_img.crop((x1, y1, x2, y2))
        cw, ch = crop.size
        longest = max(cw, ch)
        if longest > TARGET_MAX:
            scale = TARGET_MAX / longest
            crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
        elif longest < TARGET_MIN:
            scale = TARGET_MIN / longest
            if scale > 3.0: scale = 3.0
            crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
        conversation = [{"role": "user", "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": "Extract all text in the image."},
        ]}]
        try:
            with _glm_ocr_lock, torch.inference_mode():
                inputs = processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(model.device, model.dtype)
                generate_ids = model.generate(
                    **inputs, max_new_tokens=64, do_sample=False,
                    use_cache=True, pad_token_id=processor.tokenizer.pad_token_id,
                )
                generate_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
                text = processor.decode(generate_ids_trimmed[0], skip_special_tokens=True)
            text = text.split("<|im_end|>")[0].split("</s>")[0].strip()
            logging.info(f"[GLM OCR] Box {bbox} read: '{text[:30]}'")
            return bbox, text
        except Exception as e:
            logging.error(f"GLM OCR failed on {bbox}: {e}")
            return bbox, ""
    out = []
    for bbox, text in _OCR_BOX_EXECUTOR.map(_ocr_one, boxes):
        if text:
            out.append({"text": text, "bbox": bbox})
    return out
    
# ===========================================================================
# Google Lens OCR (chrome-lens-py)
# ===========================================================================
def get_lens_api():
    global _lens_api
    if LensAPI is None:
        raise RuntimeError("chrome-lens-py not installed: pip install chrome-lens-py")
    if _lens_api is None:
        with _lens_lock:
            if _lens_api is None:
                _lens_api = LensAPI()
                logging.info("[Google Lens] LensAPI initialized.")
    return _lens_api

# ── Google Lens text tilt ──
# Lens reports a per-block rotation (rotation_z, exposed as angle_deg by
# chrome-lens-py). Reusing it lets the overlay lean with slanted dialogue and
# SFX instead of sitting bolt upright on top of it. Two guards keep it tame:
#   * angles fold into (-45, 45], so a ~90° vertical block can never flip the
#     overlay fully sideways — it just reads as a small lean,
#   * anything under TILT_MIN_DEG is treated as OCR noise and dropped, and the
#     result is clamped to TILT_MAX_DEG so text stays readable.
# Positive = clockwise on screen (image coords, y growing downward).
TILT_MIN_DEG = 3.0
TILT_MAX_DEG = 20.0
# Merged blocks whose members disagree by more than this are drawn upright —
# a mixed-tilt union box has no single honest angle.
TILT_GROUP_SPREAD_DEG = 8.0

# ── OCR block merging ──
# Two blocks join only when the real gap between them is at most this multiple
# of one text-line thickness, approximated by min(w, h) over both boxes. Scaling
# the budget by a box's own width instead would let a short fragment reach far
# across the page; the union bbox would then span the empty gap and the renderer,
# which centers text inside the union, would draw the overlay beside the glyphs
# instead of on them. Being a multiple of line thickness keeps this
# scale-invariant, so a combined strip behaves like a single page.
MERGE_GAP_RATIO = 0.8
STACKED_GROUP_X_OVERLAP = 0.75
STACKED_GROUP_Y_OVERLAP = 0.20

# Two draw candidates carrying the same string are treated as the same overlay
# when their boxes overlap by at least this IoU. Kept above 0.5 so a line that
# genuinely repeats elsewhere on the page, or two stacked bubbles saying the
# same thing, still get their own overlay.
DUPLICATE_OVERLAY_IOU = 0.55


def _normalize_tilt(angle_deg) -> float:
    """Fold a raw Lens angle into a modest, readable tilt (degrees)."""
    try:
        a = float(angle_deg)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(a):
        return 0.0
    a = ((a + 45.0) % 90.0) - 45.0
    if abs(a) < TILT_MIN_DEG:
        return 0.0
    return max(-TILT_MAX_DEG, min(TILT_MAX_DEG, a))


def _geometry_angle(geometry) -> float:
    if isinstance(geometry, dict):
        return _normalize_tilt(geometry.get("angle_deg", 0.0))
    return 0.0


def _rotated_box_points(bbox: Tuple[int, int, int, int],
                        angle_deg: float) -> List[Tuple[float, float]]:
    """Corners of bbox rotated by angle_deg (clockwise) about its own center."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    hw = (x2 - x1) / 2.0
    hh = (y2 - y1) / 2.0
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return [
        (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    ]


def _geometry_to_bbox(geometry, img_w, img_h):
    if not geometry:
        return None
    if isinstance(geometry, dict):
        try:
            cx = geometry.get("center_x")
            cy = geometry.get("center_y")
            bw = geometry.get("width")
            bh = geometry.get("height")
            if None in (cx, cy, bw, bh):
                return None
            x1 = max(0, int((cx - bw / 2) * img_w))
            y1 = max(0, int((cy - bh / 2) * img_h))
            x2 = min(img_w - 1, int((cx + bw / 2) * img_w))
            y2 = min(img_h - 1, int((cy + bh / 2) * img_h))
            if x2 - x1 < 5 or y2 - y1 < 5:
                return None
            return (x1, y1, x2, y2)
        except (TypeError, KeyError):
            return None
    if isinstance(geometry, list) and len(geometry) >= 2:
        try:
            xs = [p[0] for p in geometry]
            ys = [p[1] for p in geometry]
            x1 = max(0, int(min(xs) * img_w))
            y1 = max(0, int(min(ys) * img_h))
            x2 = min(img_w - 1, int(max(xs) * img_w))
            y2 = min(img_h - 1, int(max(ys) * img_h))
            if x2 - x1 < 5 or y2 - y1 < 5:
                return None
            return (x1, y1, x2, y2)
        except (TypeError, IndexError, ValueError):
            return None
    return None

def _bbox_iou(a, b) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter
    if union_area <= 0:
        return 0.0
    return inter / union_area


def _lens_geometry_groups(boxes: List[Tuple[int, int, int, int]]) -> List[List[int]]:
    """Return Lens-style groups for boxes without requiring OCR text."""
    if len(boxes) <= 1:
        return [list(range(len(boxes)))] if boxes else []

    parent = list(range(len(boxes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    for i, first in enumerate(boxes):
        x1_i, y1_i, x2_i, y2_i = first
        w_i = max(1, x2_i - x1_i)
        h_i = max(1, y2_i - y1_i)
        for j in range(i + 1, len(boxes)):
            x1_j, y1_j, x2_j, y2_j = boxes[j]
            w_j = max(1, x2_j - x1_j)
            h_j = max(1, y2_j - y1_j)
            gap = max(x1_i, x1_j) - min(x2_i, x2_j)
            if gap > MERGE_GAP_RATIO * min(w_i, h_i, w_j, h_j):
                continue
            vertical_overlap = min(y2_i, y2_j) - max(y1_i, y1_j)
            if vertical_overlap >= 0.3 * min(h_i, h_j):
                union(i, j)

    changed = True
    while changed:
        changed = False
        grouped: Dict[int, List[int]] = {}
        for index in range(len(boxes)):
            grouped.setdefault(find(index), []).append(index)
        group_boxes = []
        for root, members in grouped.items():
            group_boxes.append((root, (
                min(boxes[index][0] for index in members),
                min(boxes[index][1] for index in members),
                max(boxes[index][2] for index in members),
                max(boxes[index][3] for index in members),
            )))
        for left, (root_i, box_i) in enumerate(group_boxes):
            x1_i, y1_i, x2_i, y2_i = box_i
            w_i = max(1, x2_i - x1_i)
            h_i = max(1, y2_i - y1_i)
            for root_j, box_j in group_boxes[left + 1:]:
                x1_j, y1_j, x2_j, y2_j = box_j
                w_j = max(1, x2_j - x1_j)
                h_j = max(1, y2_j - y1_j)
                x_overlap = min(x2_i, x2_j) - max(x1_i, x1_j)
                y_overlap = min(y2_i, y2_j) - max(y1_i, y1_j)
                if (x_overlap >= STACKED_GROUP_X_OVERLAP * min(w_i, w_j)
                        and y_overlap >= STACKED_GROUP_Y_OVERLAP * min(h_i, h_j)):
                    union(root_i, root_j)
                    changed = True
                    break
            if changed:
                break

    result: Dict[int, List[int]] = {}
    for index in range(len(boxes)):
        result.setdefault(find(index), []).append(index)
    return list(result.values())


def _merge_close_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges OCR blocks that are side-by-side on the same line.

    Only merges horizontally — never merges boxes that are stacked
    vertically in different speech bubbles. This keeps the union bbox
    short so text overlay doesn't stretch downward across bubbles.

    Horizontal merge: the real gap between two boxes must be at most
                      MERGE_GAP_RATIO × one text-line thickness, so a short
                      fragment can't reach across an inter-bubble gap.
    Vertical merge:   STRICT — actual boxes must overlap ≥30% of the
                      smaller box's height (i.e. truly on the same line).
    """
    if len(blocks) <= 1:
        for b in blocks:
            if "bboxes" not in b:
                b["bboxes"] = [b["bbox"]]
            b.setdefault("angle", 0.0)
            b.setdefault("angles", [b["angle"]] * len(b["bboxes"]))
        return blocks

    parent = list(range(len(blocks)))

    def find(i):
        root = i
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(blocks)):
        x1_i, y1_i, x2_i, y2_i = blocks[i]["bbox"]
        w_i = max(1, x2_i - x1_i)
        h_i = max(1, y2_i - y1_i)

        for j in range(i + 1, len(blocks)):
            x1_j, y1_j, x2_j, y2_j = blocks[j]["bbox"]
            w_j = max(1, x2_j - x1_j)
            h_j = max(1, y2_j - y1_j)

            # --- Must be horizontally close (small real gap between boxes) ---
            # The gap budget is measured against the thinnest dimension of the
            # pair, which approximates one text-line thickness for both
            # horizontal lines (thin in h) and vertical CJK columns (thin in w).
            # Scaling by a box's own WIDTH instead would let a short fragment
            # reach halfway across the page: the union bbox then spans the empty
            # gap, and since the renderer centers text inside the union, the
            # overlay lands beside the original glyphs rather than on them.
            # That sideways drift is most visible on combined strips, where
            # Lens returns many small fragments.
            gap = max(x1_i, x1_j) - min(x2_i, x2_j)
            line_scale = min(w_i, h_i, w_j, h_j)
            if gap > MERGE_GAP_RATIO * line_scale:
                continue

            # --- Must NOT be vertically separated ---
            # Use ACTUAL (unexpanded) vertical overlap, not padded.
            # Only merge if the real boxes overlap vertically by at least
            # 30% of the smaller height — i.e. they're on the same text line.
            v_overlap_actual = min(y2_i, y2_j) - max(y1_i, y1_j)
            min_h = min(h_i, h_j)

            if v_overlap_actual < 0.3 * min_h:
                continue  # Different vertical levels — don't merge

            union(i, j)

    # Lens sometimes divides one vertical dialogue run into two internally
    # merged groups. Their union boxes can overlap strongly even though no
    # individual fragment cleared the strict 30% first-pass threshold. Merge
    # only groups that substantially overlap on both axes; separate bubbles
    # with whitespace between them remain untouched.
    changed = True
    while changed:
        changed = False
        roots = {}
        for idx in range(len(blocks)):
            roots.setdefault(find(idx), []).append(idx)

        group_items = []
        for root, members in roots.items():
            gx1 = min(blocks[idx]["bbox"][0] for idx in members)
            gy1 = min(blocks[idx]["bbox"][1] for idx in members)
            gx2 = max(blocks[idx]["bbox"][2] for idx in members)
            gy2 = max(blocks[idx]["bbox"][3] for idx in members)
            group_items.append((root, (gx1, gy1, gx2, gy2)))

        for left in range(len(group_items)):
            root_i, box_i = group_items[left]
            x1_i, y1_i, x2_i, y2_i = box_i
            w_i = max(1, x2_i - x1_i)
            h_i = max(1, y2_i - y1_i)
            for right in range(left + 1, len(group_items)):
                root_j, box_j = group_items[right]
                x1_j, y1_j, x2_j, y2_j = box_j
                w_j = max(1, x2_j - x1_j)
                h_j = max(1, y2_j - y1_j)
                x_overlap = min(x2_i, x2_j) - max(x1_i, x1_j)
                y_overlap = min(y2_i, y2_j) - max(y1_i, y1_j)
                if x_overlap < STACKED_GROUP_X_OVERLAP * min(w_i, w_j):
                    continue
                if y_overlap < STACKED_GROUP_Y_OVERLAP * min(h_i, h_j):
                    continue
                union(root_i, root_j)
                changed = True
                break
            if changed:
                break

    # Group blocks by their root parent
    groups = {}
    for i in range(len(blocks)):
        root = find(i)
        groups.setdefault(root, []).append(blocks[i])

    merged_blocks = []
    for group in groups.values():
        x1 = min(b["bbox"][0] for b in group)
        y1 = min(b["bbox"][1] for b in group)
        x2 = max(b["bbox"][2] for b in group)
        y2 = max(b["bbox"][3] for b in group)

        original_bboxes = [b["bbox"] for b in group]
        member_angles = [float(b.get("angle", 0.0) or 0.0) for b in group]

        # A union box has one honest angle only if its members agree.
        if member_angles and (max(member_angles) - min(member_angles)) <= TILT_GROUP_SPREAD_DEG:
            group_angle = sum(member_angles) / len(member_angles)
        else:
            group_angle = 0.0

        # Sort texts in manga reading order (right-to-left, top-to-bottom)
        group.sort(key=lambda b: (b["bbox"][0] * -1, b["bbox"][1]))
        texts = [b["text"] for b in group]
        merged_text = " ".join(texts)

        merged_blocks.append({
            "text": merged_text,
            "bbox": (x1, y1, x2, y2),
            "bboxes": original_bboxes,
            "angle": group_angle,
            "angles": member_angles,
        })

    return merged_blocks

async def google_lens_ocr(pil_img: Image.Image, ocr_lang: str = "ja") -> List[Dict[str, Any]]:
    api = get_lens_api()
    w, h = pil_img.size
    logging.info(f"[Google Lens] Running OCR on {w}x{h} image (lang={ocr_lang})...")
    lens_lang = lens_lang_code(ocr_lang)
    try:
        lens_kwargs = {
            "image_path": pil_img,
            "output_format": "blocks",
        }
        if _norm_lang(ocr_lang) != "auto":
            lens_kwargs["ocr_language"] = lens_lang
        result = await api.process_image(**lens_kwargs)
    except Exception as e:
        logging.error(f"[Google Lens] OCR failed: {e}")
        return []
    if not isinstance(result, dict):
        return []
    text_blocks = result.get("text_blocks", [])
    logging.info(f"[Google Lens DEBUG] raw text_blocks: {text_blocks!r}")
    out = []
    for block in text_blocks:
        if not isinstance(block, dict):
            continue
        text = (block.get("text") or "").strip()
        if not text:
            continue
        geometry = block.get("geometry", [])
        bbox = _geometry_to_bbox(geometry, w, h)
        if bbox is None:
            lines = block.get("lines", [])
            all_points = []
            for line in lines:
                if not isinstance(line, dict):
                    continue
                line_geom = line.get("geometry", [])
                if line_geom:
                    all_points.extend(line_geom)
            if all_points:
                bbox = _geometry_to_bbox(all_points, w, h)
        if bbox is None:
            continue
        out.append({"text": text, "bbox": bbox, "angle": _geometry_angle(geometry)})

    # Merge close blocks before returning
    merged = _merge_close_blocks(out)
    logging.info(f"[Google Lens] Found {len(out)} raw blocks -> merged to {len(merged)} blocks.")
    return merged

_HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def _norm_hex_color(val: Any) -> Optional[str]:
    """Validate a model-reported color into a '#rrggbb' string, else None.

    Accepts '#rrggbb' or 'rrggbb'. Anything else (short hex, names, garbage)
    degrades to None so the renderer falls back to local color detection.
    """
    if not isinstance(val, str):
        return None
    m = _HEX_COLOR_RE.match(val.strip())
    if not m:
        return None
    return "#" + m.group(1).lower()


def _hex_to_rgb(val: Any) -> Optional[Tuple[int, int, int]]:
    """Convert a '#rrggbb'/'rrggbb' string to an (R, G, B) tuple, else None."""
    norm = _norm_hex_color(val)
    if norm is None:
        return None
    h = norm[1:]
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# OpenAI-compatible endpoint OCR: the image is sent to a vision model that
# returns the source text regions and the lettering attributes used by the
# existing inpaint/overlay pipeline.
OPENAI_ENDPOINT_OCR_PROMPT = (
    "You are a manga OCR and lettering-analysis engine. Look at the attached"
    " manga page image and find EVERY text region (speech bubbles, captions,"
    " narration boxes, and sound effects). For each region output one JSON"
    " object. Respond with ONLY a JSON array - no prose, no markdown fences.\n"
    "Each object MUST have these keys:\n"
    '  "text": the exact original text in the region, transcribed verbatim in'
    " its original language (do NOT translate).\n"
    '  "bbox": [x1, y1, x2, y2] pixel coordinates of the region in THIS image,'
    " top-left origin, integers.\n"
    '  "angle": rotation of the text baseline in degrees, positive = clockwise,'
    " 0 for normal horizontal text.\n"
    '  "color": the main color of the glyphs as "#rrggbb".\n'
    '  "glow": true if the text has a glow, halo, white outline, or soft light'
    " behind it, else false.\n"
    '  "style": exactly one of "regular", "bold", or "italic", based on the'
    " visible source lettering.\n"
    '  "weight": stroke heaviness from 0 (normal) to 3 (very heavy/blocky); use'
    " 0 for regular and italic text that is not bold.\n"
    '  "font_px": the approximate rendered glyph height in pixels in THIS image.\n'
    "Transcribe text exactly as printed. Do not merge separate bubbles. Do not"
    " invent regions that have no text."
)


def _normalize_chat_completions_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint:
        return endpoint
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


async def openai_endpoint_ocr(pil_img: Image.Image, ocr_lang: str = "ja",
                              max_retries: int = 2) -> List[Dict[str, Any]]:
    """OCR a page through an OpenAI-compatible vision chat endpoint."""
    import aiohttp
    import random

    with _openai_ocr_config_lock:
        api_key = _openai_ocr_api_key
        model = _openai_ocr_model
        endpoint = _openai_ocr_endpoint

    if not endpoint or not model:
        logging.error("[OpenAI Endpoint OCR] Endpoint and model ID must be configured")
        return []

    data_uri, scale = _page_image_data_uri_scaled(pil_img)
    if not data_uri:
        logging.error("[OpenAI Endpoint OCR] Image encoding failed")
        return []
    inv_scale = 1.0 / scale if scale else 1.0
    img_w, img_h = pil_img.size

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": OPENAI_ENDPOINT_OCR_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Source language hint: {ocr_lang}. Extract all text regions as the requested JSON array."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_p": 0.9,
    }

    logging.info(f"[OpenAI Endpoint OCR] Sending page to {model} at {endpoint}...")
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"[OpenAI Endpoint OCR] API error {response.status}: {error_text[:300]}")
                        if response.status in (408, 429, 500, 502, 503, 504, 524) and attempt < max_retries:
                            retry_after = response.headers.get("Retry-After")
                            try:
                                wait = float(retry_after) if retry_after else float(attempt)
                            except (TypeError, ValueError):
                                wait = float(attempt)
                            await asyncio.sleep(min(8.0, max(0.5, wait)))
                            continue
                        return []

                    data = await response.json()
                    raw = None
                    try:
                        message = data["choices"][0]["message"]
                        raw = message.get("content")
                        if isinstance(raw, list):
                            raw = "".join(
                                part.get("text", "") for part in raw
                                if isinstance(part, dict) and part.get("type") in ("text", "output_text")
                            )
                    except (IndexError, KeyError, TypeError):
                        pass
                    if not raw or not isinstance(raw, str):
                        logging.warning(f"[OpenAI Endpoint OCR] Empty content on attempt {attempt}")
                        continue

                    regions = _parse_vision_ocr_json(raw)
                    if regions is None:
                        logging.warning(f"[OpenAI Endpoint OCR] Could not parse JSON on attempt {attempt}. Raw: {raw[:300]!r}")
                        continue

                    out: List[Dict[str, Any]] = []
                    for reg in regions:
                        if not isinstance(reg, dict):
                            continue
                        text = (reg.get("text") or "").strip()
                        bbox = reg.get("bbox")
                        if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                            continue
                        try:
                            x1, y1, x2, y2 = (float(v) * inv_scale for v in bbox)
                        except (TypeError, ValueError):
                            continue
                        x1, x2 = sorted((x1, x2))
                        y1, y2 = sorted((y1, y2))
                        x1 = max(0, min(img_w - 1, int(round(x1))))
                        y1 = max(0, min(img_h - 1, int(round(y1))))
                        x2 = max(0, min(img_w, int(round(x2))))
                        y2 = max(0, min(img_h, int(round(y2))))
                        if (x2 - x1) < 4 or (y2 - y1) < 4:
                            continue

                        item: Dict[str, Any] = {"text": text, "bbox": (x1, y1, x2, y2)}
                        try:
                            item["angle"] = float(reg.get("angle", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            item["angle"] = 0.0
                        color = _norm_hex_color(reg.get("color"))
                        if color:
                            item["or_color"] = color
                        item["or_glow"] = bool(reg.get("glow", False))
                        style = str(reg.get("style", "regular") or "regular").lower().strip()
                        item["or_style"] = style if style in ("regular", "bold", "italic") else "regular"
                        try:
                            item["or_bold"] = max(0, min(3, int(reg.get("weight", reg.get("bold", 0)) or 0)))
                        except (TypeError, ValueError):
                            item["or_bold"] = 0
                        try:
                            fp = float(reg.get("font_px", 0) or 0) * inv_scale
                            if fp > 0:
                                item["or_font_px"] = int(round(fp))
                        except (TypeError, ValueError):
                            pass
                        out.append(item)

                    logging.info(f"[OpenAI Endpoint OCR] Parsed {len(out)} text regions from {len(regions)} raw objects.")
                    return out
        except asyncio.TimeoutError:
            logging.warning(f"[OpenAI Endpoint OCR] Timeout on attempt {attempt}/{max_retries}.")
        except Exception as e:
            logging.error(f"[OpenAI Endpoint OCR] Request failed on attempt {attempt}/{max_retries}: {e}")
        if attempt < max_retries:
            await asyncio.sleep(min(4.0, attempt + random.uniform(0.2, 0.8)))

    logging.error(f"[OpenAI Endpoint OCR] FAILED after {max_retries} attempts.")
    return []


GOOGLE_AI_STUDIO_OCR_PROMPT = (
    "You are a comprehensive manga OCR engine. Find and transcribe EVERY visible"
    " text region in the image without filtering by purpose, size, location, or"
    " style. Include dialogue, narration, thought bubbles, titles, chapter text,"
    " sound effects, signs, labels, credits, watermarks, page numbers, handwritten"
    " text, and decorative lettering. Return only a JSON array with one object per"
    " visually distinct text region. Each object must contain the exact source text"
    " in `text` and a Gemini-native normalized bounding box in `box_2d` using"
    " `[ymin, xmin, ymax, xmax]`, where every coordinate is an integer from 0 to"
    " 1000 relative to the full image. Make each box tightly cover its complete text"
    " region with a small margin. Do not translate, omit, summarize, or merge"
    " spatially separate regions. Do not include prose or markdown fences. Return []"
    " only when the image contains no visible text."
)


def _normalize_gemini_model(model: str) -> str:
    model = model.strip().strip("/")
    return model[7:] if model.startswith("models/") else model


def _gemini_box_to_pixels(box: Any, image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (ymin, xmin, ymax, xmax)):
        return None
    ymin, ymax = sorted((max(0.0, min(1000.0, ymin)), max(0.0, min(1000.0, ymax))))
    xmin, xmax = sorted((max(0.0, min(1000.0, xmin)), max(0.0, min(1000.0, xmax))))
    image_w, image_h = image_size
    x1 = max(0, min(image_w - 1, int(round(xmin * image_w / 1000.0))))
    y1 = max(0, min(image_h - 1, int(round(ymin * image_h / 1000.0))))
    x2 = max(0, min(image_w, int(round(xmax * image_w / 1000.0))))
    y2 = max(0, min(image_h, int(round(ymax * image_h / 1000.0))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def _gemini_region_to_pixels(region: Any, image_size: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    if not isinstance(region, dict):
        return None
    text = str(
        region.get("text")
        or region.get("transcription")
        or region.get("label")
        or region.get("content")
        or ""
    ).strip()
    box = (
        region.get("box_2d")
        or region.get("box2d")
        or region.get("bounding_box")
        or region.get("boundingBox")
        or region.get("bbox")
    )
    pixel_box = _gemini_box_to_pixels(box, image_size)
    if not text or pixel_box is None:
        return None
    return {"text": text, "bbox": pixel_box, "angle": 0.0}


async def _wait_for_google_ai_ocr_slot(rpm: int) -> None:
    global _google_ai_ocr_last_request
    interval = 60.0 / max(1, rpm)
    async with _google_ai_ocr_rate_lock:
        now = time.monotonic()
        wait = interval - (now - _google_ai_ocr_last_request)
        if wait > 0:
            logging.info(f"[Google AI OCR] Waiting {wait:.1f}s for the {rpm} RPM limiter.")
            await asyncio.sleep(wait)
        _google_ai_ocr_last_request = time.monotonic()


async def google_ai_studio_ocr(pil_img: Image.Image, ocr_lang: str = "ja",
                               max_retries: int = 2) -> List[Dict[str, Any]]:
    """Detect all visible manga text regions with Google AI Studio Gemini."""
    import aiohttp

    with _google_ai_ocr_config_lock:
        api_key = _google_ai_ocr_api_key
        model = _normalize_gemini_model(_google_ai_ocr_model)
        rpm = _google_ai_ocr_rpm

    if not api_key or not model:
        logging.error("[Google AI OCR] API key and model must be configured")
        return []

    data_uri = _page_image_data_uri_original(pil_img)
    if not data_uri or "," not in data_uri:
        logging.error("[Google AI OCR] Image encoding failed")
        return []
    mime_type = data_uri[5:data_uri.index(";")]
    image_data = data_uri.split(",", 1)[1]
    img_w, img_h = pil_img.size
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model, safe='')}:generateContent"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": GOOGLE_AI_STUDIO_OCR_PROMPT}]},
        "contents": [{"role": "user", "parts": [
            {"text": f"Source language hint: {ocr_lang}. Return every visible text region."},
            {"inlineData": {"mimeType": mime_type, "data": image_data}},
        ]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "text": {"type": "STRING"},
                        "box_2d": {
                            "type": "ARRAY",
                            "items": {"type": "INTEGER"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": ["text", "box_2d"],
                },
            },
        },
    }

    for attempt in range(1, max_retries + 1):
        await _wait_for_google_ai_ocr_slot(rpm)
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"[Google AI OCR] API error {response.status}: {error_text[:300]}")
                        if response.status in (408, 429, 500, 502, 503, 504) and attempt < max_retries:
                            retry_after = response.headers.get("Retry-After")
                            try:
                                wait = float(retry_after) if retry_after else 1.0
                            except (TypeError, ValueError):
                                wait = 1.0
                            await asyncio.sleep(min(60.0, max(1.0, wait)))
                            continue
                        return []

                    data = await response.json()
                    raw = ""
                    try:
                        parts = data["candidates"][0]["content"]["parts"]
                        raw = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
                    except (IndexError, KeyError, TypeError):
                        pass
                    regions = _parse_vision_ocr_json(raw) if raw else None
                    if regions is None:
                        logging.warning(f"[Google AI OCR] Invalid JSON response: {raw[:300]!r}")
                        if attempt < max_retries:
                            continue
                        return []

                    out: List[Dict[str, Any]] = []
                    rejected = 0
                    for reg in regions:
                        item = _gemini_region_to_pixels(reg, (img_w, img_h))
                        if item is None:
                            rejected += 1
                            continue
                        out.append(item)
                    if not out and regions:
                        logging.warning(
                            f"[Google AI OCR] Rejected all {len(regions)} regions. "
                            f"Expected text + normalized box_2d; first region: {regions[0]!r}"
                        )
                    logging.info(
                        f"[Google AI OCR] Parsed {len(out)} text regions from {len(regions)} raw regions "
                        f"({rejected} rejected) with {model}."
                    )
                    return out
        except asyncio.TimeoutError:
            logging.warning(f"[Google AI OCR] Timeout on attempt {attempt}/{max_retries}.")
        except Exception as exc:
            logging.error(f"[Google AI OCR] Request failed on attempt {attempt}/{max_retries}: {exc}")
    return []


def _parse_vision_ocr_json(raw: str) -> Optional[List[Any]]:
    """Extract a JSON array of OCR region objects from a model response.

    Tolerant of markdown fences and leading/trailing prose: strips ```json
    fences, then falls back to slicing between the first '[' and last ']'.
    Returns the parsed list, or None if nothing parseable is found.
    """
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(s[start:end + 1])
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    if start < 0:
        return None

    # A local model can hit its output limit after emitting several complete
    # objects but before closing the JSON array. Recover those complete objects
    # so later pipeline stages still receive every region that was generated.
    recovered: List[Any] = []
    decoder = json.JSONDecoder()
    cursor = max(0, start + 1) if start != -1 else 0
    while cursor < len(s):
        object_start = s.find("{", cursor)
        if object_start < 0:
            break
        try:
            value, object_end = decoder.raw_decode(s, object_start)
        except json.JSONDecodeError:
            cursor = object_start + 1
            continue
        if isinstance(value, dict):
            recovered.append(value)
        cursor = object_end
    return recovered or None


LOCAL_VISION_REVIEW_PROMPT = (
    "Audit the proposed OCR regions against the image. Return ONLY a corrected JSON "
    "array using the template-required `text`, pixel `bbox`, and `angle` fields. "
    "Remove guessed, duplicated, or illegible text. Merge character-by-character "
    "fragments that are actually one visual text block. Correct every loose or "
    "oversized box so it tightly surrounds its visible text. Do not invent regions "
    "or translate text."
)


def _local_vision_inference_size(image_size: Tuple[int, int]) -> Tuple[int, int]:
    return image_size


def _page_image_data_uri_local_vision(pil_img: "Image.Image") -> str:
    try:
        img = pil_img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=78, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        logging.info(
            f"[Local Vision OCR] Encoded full-resolution {img.size[0]}x{img.size[1]} image, "
            f"{len(b64) // 1024}KB base64"
        )
        return f"data:image/jpeg;base64,{b64}"
    except Exception as exc:
        logging.warning(f"[Local Vision OCR] Image encoding failed: {exc}")
        return ""


def _selected_vision_model() -> Optional[Dict[str, Any]]:
    for model in list_local_gguf_models():
        if (model["repo_id"] == _current_qwen_repo_id
                and model["filename"] == _current_qwen_filename):
            return model if model.get("vision_capable") else None
    return None


def _vision_chat_handler(model_name: str, projector_path: str, use_gpu: Optional[bool] = None):
    try:
        from llama_cpp.llama_chat_format import MTMDChatHandler
        return MTMDChatHandler(
            clip_model_path=projector_path,
            use_gpu=llama_cpp_gpu_available() if use_gpu is None else use_gpu,
            verbose=False,
        )
    except (ImportError, TypeError) as exc:
        raise RuntimeError(
            "The installed llama-cpp-python build does not provide the generic "
            f"multimodal handler required for projector {projector_path}: {exc}"
        ) from exc


def get_local_vision_qwen():
    global _local_vision_qwen, _local_vision_model_path, _local_vision_projector_path
    selected = _selected_vision_model()
    if selected is None:
        raise RuntimeError(
            "The selected local GGUF has no compatible mmproj/projector GGUF. "
            "Install the model and its projector in the same Hugging Face repository/cache."
        )
    model_path = pathlib.Path(selected["path"])
    projector_path = pathlib.Path(selected["projector_path"])
    with _qwen_model_lock:
        if (_local_vision_qwen is not None
                and _local_vision_model_path == model_path
                and _local_vision_projector_path == projector_path):
            return _local_vision_qwen
        use_gpu = _log_llama_device("Local Vision OCR")

        def _load_local_vision(enable_gpu: bool):
            handler = _vision_chat_handler(model_path.name, str(projector_path), enable_gpu)
            if LOCAL_VISION_CHAT_TEMPLATE:
                handler._get_chat_template = lambda _llama: LOCAL_VISION_CHAT_TEMPLATE
                logging.info("[Local Vision OCR] Using jinja.txt as the multimodal chat template.")
            return Llama(
                model_path=str(model_path),
                chat_handler=handler,
                n_ctx=4096,
                n_batch=1024,
                n_ubatch=512,
                n_threads=max(4, os.cpu_count() or 4),
                n_threads_batch=max(4, os.cpu_count() or 4),
                n_gpu_layers=-1 if enable_gpu else 0,
                flash_attn=enable_gpu,
                offload_kqv=enable_gpu,
                verbose=False,
            )

        try:
            _local_vision_qwen = _load_local_vision(use_gpu)
        except Exception as exc:
            if not use_gpu:
                raise
            logging.warning(f"[Local Vision OCR] CUDA model initialization failed; retrying on CPU: {exc}")
            _local_vision_qwen = _load_local_vision(False)
            use_gpu = False
        _local_vision_model_path = model_path
        _local_vision_projector_path = projector_path
        logging.info(
            f"[Local Vision OCR] Loaded {model_path.name} with {projector_path.name} "
            f"on {'CUDA' if use_gpu else 'CPU'}"
        )
        return _local_vision_qwen


def _local_vision_box_to_pixels(
    box: Any,
    image_size: Tuple[int, int],
    *,
    normalized_yx: bool = False,
) -> Optional[Tuple[int, int, int, int]]:
    """Convert common vision-model box formats to pixel XYXY coordinates."""
    if isinstance(box, dict):
        key_sets = (
            ("x1", "y1", "x2", "y2"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("left", "top", "right", "bottom"),
        )
        values = None
        for keys in key_sets:
            if all(key in box for key in keys):
                values = [box[key] for key in keys]
                break
        if values is None:
            return None
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        values = list(box)
    else:
        return None

    try:
        coords = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in coords):
        return None

    image_w, image_h = image_size
    if normalized_yx:
        y1, x1, y2, x2 = coords
        scale = 1000.0 if max(abs(value) for value in coords) > 1.0 else 1.0
        x1, x2 = x1 * image_w / scale, x2 * image_w / scale
        y1, y2 = y1 * image_h / scale, y2 * image_h / scale
    else:
        x1, y1, x2, y2 = coords
        max_abs = max(abs(value) for value in coords)
        if max_abs <= 1.0:
            x1, x2 = x1 * image_w, x2 * image_w
            y1, y2 = y1 * image_h, y2 * image_h

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(image_w - 1, int(round(x1))))
    y1 = max(0, min(image_h - 1, int(round(y1))))
    x2 = max(0, min(image_w, int(round(x2))))
    y2 = max(0, min(image_h, int(round(y2))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def _normalize_local_vision_regions(
    regions: List[Any],
    image_size: Tuple[int, int],
    detected_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    used_region_ids = set()
    for region in regions:
        if not isinstance(region, dict):
            continue
        text = str(
            region.get("text")
            or region.get("transcription")
            or region.get("content")
            or region.get("label")
            or ""
        ).strip()
        if not text:
            continue

        bbox = None
        if detected_boxes:
            try:
                region_id = int(region.get("region_id", region.get("id", region.get("region", 0))))
            except (TypeError, ValueError):
                region_id = 0
            if 1 <= region_id <= len(detected_boxes) and region_id not in used_region_ids:
                bbox = detected_boxes[region_id - 1]
                used_region_ids.add(region_id)

        if bbox is None:
            box_key = next(
                (key for key in ("bbox", "bounding_box", "boundingBox", "box", "box_2d", "box2d")
                 if region.get(key) is not None),
                None,
            )
            if box_key is None:
                continue
            bbox = _local_vision_box_to_pixels(
                region[box_key], image_size, normalized_yx=box_key in {"box_2d", "box2d"}
            )
        if bbox is None:
            continue

        item: Dict[str, Any] = {"text": text, "bbox": bbox}
        try:
            item["angle"] = float(region.get("angle", region.get("rotation", 0.0)) or 0.0)
        except (TypeError, ValueError):
            item["angle"] = 0.0
        member_boxes = region.get("bboxes")
        if isinstance(member_boxes, (list, tuple)):
            normalized_members = [
                member_bbox
                for member in member_boxes
                if (member_bbox := _local_vision_box_to_pixels(member, image_size)) is not None
            ]
            if normalized_members:
                item["bboxes"] = normalized_members
                raw_angles = region.get("angles")
                if isinstance(raw_angles, (list, tuple)):
                    angles = []
                    for value in raw_angles[:len(normalized_members)]:
                        try:
                            angles.append(float(value or 0.0))
                        except (TypeError, ValueError):
                            angles.append(0.0)
                    angles.extend([item["angle"]] * (len(normalized_members) - len(angles)))
                    item["angles"] = angles
                else:
                    item["angles"] = [item["angle"]] * len(normalized_members)
        color = _norm_hex_color(region.get("color") or region.get("text_color"))
        if color:
            item["or_color"] = color
        item["or_glow"] = bool(region.get("glow", region.get("halo", False)))
        style = str(region.get("style", "regular") or "regular").lower().strip()
        item["or_style"] = style if style in {"regular", "bold", "italic"} else "regular"
        try:
            item["or_bold"] = max(
                0, min(3, int(region.get("weight", region.get("bold", 0)) or 0))
            )
        except (TypeError, ValueError):
            item["or_bold"] = 0
        try:
            font_px = float(region.get("font_px", region.get("font_size", 0)) or 0)
            if font_px > 0:
                item["or_font_px"] = int(round(font_px))
        except (TypeError, ValueError):
            pass
        out.append(item)
    return out


def _repair_local_vision_text(text: str) -> str:
    suspicious = sum(text.count(marker) for marker in ("π", "µ", "Φ", "Θ", "Σ", "τ", "σ"))
    if suspicious < 2:
        return text
    try:
        repaired = text.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    original_japanese = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text))
    repaired_japanese = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", repaired))
    return repaired if repaired_japanese > original_japanese else text


def _repair_local_vision_regions(regions: List[Any]) -> List[Any]:
    repaired_regions: List[Any] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        updated = dict(region)
        for key in ("text", "transcription", "content", "label"):
            value = updated.get(key)
            if isinstance(value, str):
                updated[key] = _repair_local_vision_text(value)
                break
        repaired_regions.append(updated)
    return repaired_regions


def _scale_local_vision_regions(
    regions: List[Any],
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> List[Any]:
    source_w, source_h = source_size
    target_w, target_h = target_size
    if source_size == target_size:
        return regions
    scaled: List[Any] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        box_key = next(
            (key for key in ("bbox", "bounding_box", "boundingBox", "box") if region.get(key) is not None),
            None,
        )
        if box_key is None:
            scaled.append(region)
            continue
        box = region[box_key]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            scaled.append(region)
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError):
            scaled.append(region)
            continue
        updated = dict(region)
        updated[box_key] = [
            int(round(x1 * target_w / source_w)),
            int(round(y1 * target_h / source_h)),
            int(round(x2 * target_w / source_w)),
            int(round(y2 * target_h / source_h)),
        ]
        scaled.append(updated)
    return scaled


def _local_vision_regions_need_review(regions: List[Any], image_size: Tuple[int, int]) -> bool:
    image_w, image_h = image_size
    short_texts: Dict[str, int] = {}
    boxes: List[Tuple[int, int, int, int]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        text = str(region.get("text") or region.get("transcription") or region.get("content") or "").strip()
        box_key = next(
            (key for key in ("bbox", "bounding_box", "boundingBox", "box", "box_2d", "box2d")
             if region.get(key) is not None),
            None,
        )
        if not text or box_key is None:
            continue
        bbox = _local_vision_box_to_pixels(
            region[box_key], image_size, normalized_yx=box_key in {"box_2d", "box2d"}
        )
        if bbox is None:
            continue
        boxes.append(bbox)
        if len(text.replace(" ", "")) <= 2:
            short_texts[text] = short_texts.get(text, 0) + 1
        x1, y1, x2, y2 = bbox
        if len(text.replace(" ", "")) <= 8 and (x2 - x1) * (y2 - y1) > image_w * image_h * 0.18:
            return True
    if any(count >= 3 for count in short_texts.values()):
        return True
    for index, first in enumerate(boxes):
        ax1, ay1, ax2, ay2 = first
        area = max(1, (ax2 - ax1) * (ay2 - ay1))
        for second in boxes[index + 1:]:
            bx1, by1, bx2, by2 = second
            intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
            other_area = max(1, (bx2 - bx1) * (by2 - by1))
            if intersection / min(area, other_area) >= 0.75:
                return True
    return False


def _dedupe_local_vision_regions(regions: List[Any], image_size: Tuple[int, int]) -> List[Any]:
    candidates: List[Tuple[Any, str, Tuple[int, int, int, int]]] = []
    image_w, image_h = image_size
    for region in regions:
        if not isinstance(region, dict):
            continue
        text = str(region.get("text") or region.get("transcription") or region.get("content") or "").strip()
        box_key = next(
            (key for key in ("bbox", "bounding_box", "boundingBox", "box", "box_2d", "box2d")
             if region.get(key) is not None),
            None,
        )
        if not text or box_key is None:
            continue
        bbox = _local_vision_box_to_pixels(
            region[box_key], image_size, normalized_yx=box_key in {"box_2d", "box2d"}
        )
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        if len(text.replace(" ", "")) <= 8 and (x2 - x1) * (y2 - y1) > image_w * image_h * 0.18:
            continue
        candidates.append((region, text, bbox))

    fragmented = set()
    for index, (_, text, bbox) in enumerate(candidates):
        if len(text.replace(" ", "")) > 2:
            continue
        cluster = {index}
        changed = True
        while changed:
            changed = False
            for other_index, (_, other_text, other_box) in enumerate(candidates):
                if other_index in cluster or other_text != text:
                    continue
                for member_index in cluster:
                    member_box = candidates[member_index][2]
                    mx1, my1, mx2, my2 = member_box
                    ox1, oy1, ox2, oy2 = other_box
                    horizontal_gap = max(0, max(mx1, ox1) - min(mx2, ox2))
                    vertical_gap = max(0, max(my1, oy1) - min(my2, oy2))
                    aligned = horizontal_gap <= max(mx2 - mx1, ox2 - ox1) and vertical_gap <= max(my2 - my1, oy2 - oy1)
                    if aligned:
                        cluster.add(other_index)
                        changed = True
                        break
        if len(cluster) >= 3:
            fragmented.update(cluster)

    kept: List[Any] = []
    kept_boxes: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for index, (region, text, bbox) in enumerate(candidates):
        if index in fragmented:
            continue
        x1, y1, x2, y2 = bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        duplicate = False
        for prior_text, prior_box in kept_boxes:
            if prior_text != text:
                continue
            px1, py1, px2, py2 = prior_box
            overlap = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
            if overlap / min(area, max(1, (px2 - px1) * (py2 - py1))) >= 0.75:
                duplicate = True
                break
        if not duplicate:
            kept.append(region)
            kept_boxes.append((text, bbox))
    return kept


def _complete_local_vision_ocr(
    llm: Any,
    messages: List[Dict[str, Any]],
    max_tokens: int = 512,
    expected_regions: Optional[int] = None,
) -> str:
    response_kwargs: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "stop": ["<|im_end|>", "</s>"],
        "stream": True,
    }
    if llama_cpp is not None and hasattr(llama_cpp, "LlamaGrammar"):
        properties = {
            "text": {"type": "string", "minLength": 1},
            "bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
            },
            "angle": {"type": "number"},
        }
        schema: Dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": properties,
                "required": ["text", "bbox", "angle"],
                "additionalProperties": False,
            },
        }
        if expected_regions is not None:
            schema["minItems"] = expected_regions
            schema["maxItems"] = expected_regions
        response_kwargs["grammar"] = llama_cpp.LlamaGrammar.from_json_schema(
            json.dumps(schema), verbose=False
        )
    response = llm.create_chat_completion(**response_kwargs)
    if isinstance(response, dict):
        return str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))

    output = ""
    scan_offset = 0
    array_started = False
    array_depth = 0
    in_string = False
    escaped = False
    for chunk in response:
        try:
            delta = chunk["choices"][0].get("delta", {}).get("content", "")
        except (KeyError, IndexError, TypeError):
            delta = ""
        if not delta:
            continue
        output += str(delta)
        complete_array = False
        for char in output[scan_offset:]:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                array_started = True
                array_depth += 1
            elif char == "]" and array_started:
                array_depth -= 1
                if array_depth == 0:
                    complete_array = True
                    break
        scan_offset = len(output)
        if complete_array:
            parsed = _parse_vision_ocr_json(output)
            if parsed is not None:
                logging.info(
                    f"[Local Vision OCR] Stopped generation after the first complete "
                    f"JSON array ({len(parsed)} raw regions)."
                )
                break
    return output


def _local_vision_crop_data_uri(crop: Image.Image) -> str:
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="PNG", optimize=True)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _local_vision_contact_sheet(
    pil_img: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    gutter = 24
    source_crops = [pil_img.crop(box).convert("RGB") for box in boxes]
    scales = [min(4.0, max(1.0, 128.0 / max(1, crop.width))) for crop in source_crops]
    crops = [
        crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.LANCZOS,
        ) if scale > 1.0 else crop
        for crop, scale in zip(source_crops, scales)
    ]
    columns = math.ceil(math.sqrt(len(crops)))
    rows = math.ceil(len(crops) / columns)
    column_widths = [
        max(crops[index].width for index in range(column, len(crops), columns))
        for column in range(columns)
    ]
    row_heights = [
        max(crop.height for crop in crops[row * columns:(row + 1) * columns])
        for row in range(rows)
    ]
    column_x = []
    x = 0
    for width in column_widths:
        column_x.append(x)
        x += width + gutter
    row_y = []
    y = 0
    for height in row_heights:
        row_y.append(y)
        y += height + gutter

    sheet_w = sum(column_widths) + gutter * (columns - 1)
    sheet_h = sum(row_heights) + gutter * (rows - 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    placements: List[Dict[str, Any]] = []
    for index, (crop, page_box, scale) in enumerate(zip(crops, boxes, scales)):
        row, column = divmod(index, columns)
        x = column_x[column] + (column_widths[column] - crop.width) // 2
        y = row_y[row] + (row_heights[row] - crop.height) // 2
        sheet.paste(crop, (x, y))
        placements.append({
            "sheet_bbox": (x, y, x + crop.width, y + crop.height),
            "page_bbox": page_box,
            "scale": scale,
        })
    logging.info(
        f"[Local Vision OCR] Built {columns}x{rows} crop atlas "
        f"({sheet_w}x{sheet_h}) from {len(crops)} YOLO regions."
    )
    return sheet, placements


def _local_vision_detected_regions(
    pil_img: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    llm: Any,
    ocr_lang: str,
) -> List[Dict[str, Any]]:
    sheet, placements = _local_vision_contact_sheet(pil_img, boxes)
    completion_started = time.perf_counter()
    raw = _complete_local_vision_ocr(
        llm,
        [{"role": "system", "content": ""}, {"role": "user", "content": [
            {
                "type": "text",
                "text": (
                    f"Source language hint: {ocr_lang}. This image is a compact grid atlas of "
                    "complete YOLO-detected text crops divided by blank gutters. OCR every crop "
                    "once, in reading order, and return one JSON object for each crop. For each "
                    "object, return the exact visible text and a tight second-stage bbox relative "
                    "to that crop's own top-left corner, not page coordinates. Do not omit crops "
                    "or return empty text, placeholder bboxes, reasoning, or <think> text."
                ),
            },
            {"type": "image_url", "image_url": {"url": _local_vision_crop_data_uri(sheet)}},
        ]}],
        max_tokens=min(4096, max(1024, len(boxes) * 256)),
    )
    completion_seconds = time.perf_counter() - completion_started
    parsed = _parse_vision_ocr_json(str(raw))
    logging.info(
        f"[Local Vision OCR] GGUF crop-atlas completion took {completion_seconds:.1f}s, "
        f"returned {len(str(raw))} characters and "
        f"{len(parsed) if parsed is not None else 0} parsed regions."
    )
    if parsed is None:
        preview = " ".join(str(raw).split())[:500]
        logging.warning(
            f"[Local Vision OCR] GGUF returned no parseable JSON for "
            f"{len(boxes)}-crop atlas. Raw response: {preview!r}"
        )
        return []

    regions = _repair_local_vision_regions(parsed)
    remapped: List[Dict[str, Any]] = []
    used_placements = set()
    rejected = 0
    for index, raw_region in enumerate(regions, start=1):
        if not isinstance(raw_region, dict):
            rejected += 1
            continue
        text = str(
            raw_region.get("text")
            or raw_region.get("transcription")
            or raw_region.get("content")
            or raw_region.get("label")
            or ""
        ).strip()
        raw_bbox = raw_region.get("bbox")
        if not text or not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            rejected += 1
            continue
        try:
            values = [float(value) for value in raw_bbox]
        except (TypeError, ValueError):
            rejected += 1
            continue

        placement_index: Optional[int] = None
        crop_bbox: Optional[Tuple[int, int, int, int]] = None
        coordinate_mode = "crop-local"
        ordered_index = index - 1
        if ordered_index < len(placements) and ordered_index not in used_placements:
            sx1, sy1, sx2, sy2 = placements[ordered_index]["sheet_bbox"]
            crop_w, crop_h = sx2 - sx1, sy2 - sy1
            if (min(values) >= 0 and values[0] <= crop_w and values[2] <= crop_w
                    and values[1] <= crop_h and values[3] <= crop_h):
                normalized = _normalize_local_vision_regions([raw_region], (crop_w, crop_h))
                if normalized:
                    placement_index = ordered_index
                    crop_bbox = normalized[0]["bbox"]

        if crop_bbox is None:
            atlas_boxes: List[Tuple[str, Tuple[int, int, int, int]]] = []
            atlas_pixel = _local_vision_box_to_pixels(raw_bbox, sheet.size)
            if atlas_pixel is not None:
                atlas_boxes.append(("atlas", atlas_pixel))
            if min(values) >= 0 and max(values) <= 1000 and max(values) > 1:
                normalized_atlas = (
                    round(values[0] * sheet.width / 1000.0),
                    round(values[1] * sheet.height / 1000.0),
                    round(values[2] * sheet.width / 1000.0),
                    round(values[3] * sheet.height / 1000.0),
                )
                atlas_boxes.insert(0, ("normalized-atlas", normalized_atlas))

            best_match = None
            for mode, atlas_bbox in atlas_boxes:
                ax1, ay1, ax2, ay2 = atlas_bbox
                for candidate_index, placement in enumerate(placements):
                    if candidate_index in used_placements:
                        continue
                    sx1, sy1, sx2, sy2 = placement["sheet_bbox"]
                    clipped = (
                        max(ax1, sx1), max(ay1, sy1),
                        min(ax2, sx2), min(ay2, sy2),
                    )
                    overlap_w = max(0, clipped[2] - clipped[0])
                    overlap_h = max(0, clipped[3] - clipped[1])
                    overlap = overlap_w * overlap_h
                    if overlap < 16:
                        continue
                    placement_area = max(1, (sx2 - sx1) * (sy2 - sy1))
                    score = overlap / placement_area
                    if best_match is None or score > best_match[0]:
                        best_match = (score, candidate_index, mode, clipped)
            if best_match is not None:
                _, placement_index, coordinate_mode, clipped = best_match
                sx1, sy1, _, _ = placements[placement_index]["sheet_bbox"]
                crop_bbox = (
                    clipped[0] - sx1, clipped[1] - sy1,
                    clipped[2] - sx1, clipped[3] - sy1,
                )

        if placement_index is None or crop_bbox is None:
            rejected += 1
            logging.warning(
                f"[Local Vision OCR] Rejected result {index}: text={text[:80]!r}, "
                f"bbox={raw_bbox!r}; it did not map to an unused atlas crop."
            )
            continue

        placement = placements[placement_index]
        used_placements.add(placement_index)
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        if crop_x2 - crop_x1 < 4 or crop_y2 - crop_y1 < 4:
            rejected += 1
            continue
        scale = float(placement.get("scale", 1.0) or 1.0)
        page_x1, page_y1, page_x2, page_y2 = placement["page_bbox"]
        refined = dict(raw_region)
        refined["text"] = text
        refined["bbox"] = (
            max(page_x1, min(page_x2, page_x1 + round(crop_x1 / scale))),
            max(page_y1, min(page_y2, page_y1 + round(crop_y1 / scale))),
            max(page_x1, min(page_x2, page_x1 + round(crop_x2 / scale))),
            max(page_y1, min(page_y2, page_y1 + round(crop_y2 / scale))),
        )
        try:
            refined["angle"] = float(raw_region.get("angle", 0.0) or 0.0)
        except (TypeError, ValueError):
            refined["angle"] = 0.0
        logging.info(
            f"[Local Vision OCR] Accepted result {index}/{len(regions)} for crop "
            f"{placement_index + 1}/{len(placements)} using {coordinate_mode} second bbox "
            f"{crop_bbox} -> page {refined['bbox']}."
        )
        remapped.append(refined)

    if rejected:
        logging.warning(
            f"[Local Vision OCR] Rejected {rejected}/{len(regions)} OCR results because "
            "text or the refined bbox was invalid."
        )
    if len(used_placements) < len(placements):
        logging.warning(
            f"[Local Vision OCR] GGUF represented {len(used_placements)}/{len(placements)} "
            "atlas crops with usable OCR results."
        )
    return remapped


def _local_vision_ocr_yolo_box(
    pil_img: Image.Image,
    bbox: Tuple[int, int, int, int],
    llm: Any,
    ocr_lang: str,
) -> Dict[str, Any]:
    crop = pil_img.crop(bbox).convert("RGB")
    raw = _complete_local_vision_ocr(
        llm,
        [{"role": "system", "content": ""}, {"role": "user", "content": [
            {
                "type": "text",
                "text": (
                    f"Source language hint: {ocr_lang}. OCR the complete text in this "
                    "single YOLO-detected crop. Return one JSON object in an array with "
                    "the exact visible text, a bbox covering the crop, and angle 0. Do not "
                    "split, merge, translate, explain, or return markdown."
                ),
            },
            {"type": "image_url", "image_url": {"url": _local_vision_crop_data_uri(crop)}},
        ]}],
        max_tokens=512,
        expected_regions=1,
    )
    parsed = _parse_vision_ocr_json(str(raw))
    if not parsed:
        return {"text": "", "bbox": bbox}

    repaired = _repair_local_vision_regions(parsed)
    if not repaired or not isinstance(repaired[0], dict):
        return {"text": "", "bbox": bbox}

    raw_region = repaired[0]
    text = str(
        raw_region.get("text")
        or raw_region.get("transcription")
        or raw_region.get("content")
        or raw_region.get("label")
        or ""
    ).strip()
    result: Dict[str, Any] = {"text": text, "bbox": bbox}
    try:
        result["angle"] = float(raw_region.get("angle", 0.0) or 0.0)
    except (TypeError, ValueError):
        result["angle"] = 0.0
    for key in ("or_color", "or_glow", "or_style", "or_bold", "or_font_px"):
        if key in raw_region:
            result[key] = raw_region[key]
    return result


def local_vision_ocr(pil_img: Image.Image, ocr_lang: str = "auto") -> List[Dict[str, Any]]:
    started = time.perf_counter()
    llm = get_local_vision_qwen()
    detected_boxes = _detect_yolo_text_boxes(pil_img)
    if not detected_boxes:
        logging.info("[Local Vision OCR] YOLO found no text regions.")
        return []

    logging.info(
        f"[Local Vision OCR] Running GGUF OCR independently on "
        f"{len(detected_boxes)} YOLO crop(s)."
    )
    results: List[Dict[str, Any]] = []
    with _local_vision_inference_lock:
        for index, bbox in enumerate(detected_boxes, start=1):
            try:
                item = _local_vision_ocr_yolo_box(
                    pil_img, bbox, llm, ocr_lang
                )
            except Exception as exc:
                logging.warning(
                    f"[Local Vision OCR] OCR failed for YOLO crop "
                    f"{index}/{len(detected_boxes)} at {bbox}: {exc}"
                )
                continue
            if not item["text"]:
                logging.warning(
                    f"[Local Vision OCR] YOLO crop {index}/{len(detected_boxes)} "
                    f"at {bbox} returned no text."
                )
                continue
            results.append(item)
            logging.info(
                f"[Local Vision OCR] YOLO crop {index}/{len(detected_boxes)} "
                f"read {item['text'][:40]!r} at unchanged box {bbox}."
            )

    logging.info(
        f"[Local Vision OCR] Produced {len(results)} one-to-one YOLO region(s) "
        f"for image {pil_img.width}x{pil_img.height}; "
        f"total={time.perf_counter() - started:.1f}s."
    )
    return results


async def get_ocr_results(pil_img: Image.Image, ocr_lang: str = "ja",
                          mode_override: Optional[str] = None) -> List[Dict[str, Any]]:
    if mode_override is None:
        with _ocr_mode_lock:
            mode = _ocr_mode
    else:
        mode = mode_override
    if mode == "lens":
        return await google_lens_ocr(pil_img, ocr_lang)
    elif mode == "google_ai":
        return await google_ai_studio_ocr(pil_img, ocr_lang)
    elif mode == "openai_endpoint":
        return await openai_endpoint_ocr(pil_img, ocr_lang)
    elif mode == "local_vision":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_OCR_BOX_EXECUTOR, local_vision_ocr, pil_img, ocr_lang)
    elif mode == "glm" or ocr_lang == "ko":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_OCR_BOX_EXECUTOR, glm_ocr_korean, pil_img)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_OCR_BOX_EXECUTOR, hayai_ocr_with_yolo, pil_img)

# ===========================================================================
# SFX Classifier
# ===========================================================================
# Conservative multi-signal heuristic. A region is classified as SFX only when
# at least 2 independent signals agree. This avoids false positives on big
# narration boxes while catching merged SFX+dialogue regions that Google Lens
# tends to produce.
#
# Signals:
#   1. Low character density — few chars in a large region.
#   2. Short text in a large region (≤8 non-space chars + region > 5% of image).
#   3. Near-square or wide aspect ratio at large absolute size.
#   4. Repeated kana / onomatopoeia pattern (e.g. ドドド, ゴゴゴゴ, ドゴォ).
#
# Returns (is_sfx: bool, score: int, reasons: List[str]).
# A region is SFX iff score >= 2.
def detect_sfx(
    pil_img: Image.Image,
    bbox: Tuple[int, int, int, int],
    text: str,
) -> Tuple[bool, int, List[str]]:
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    box_area = box_w * box_h
    img_w, img_h = pil_img.size
    img_area = max(1, img_w * img_h)

    cleaned = (text or "").strip()
    # Strip whitespace for character counting.
    nonspace = re.sub(r"\s+", "", cleaned)
    n_chars = len(nonspace)

    reasons: List[str] = []
    score = 0

    if not nonspace:
        return False, 0, reasons

    region_fraction = box_area / img_area

    # Signal 1: Low character density.
    # Density = chars per 1000 px² of the region. SFX typically have very few
    # characters filling a large area. Threshold: < 1 char per 4000 px².
    if n_chars > 0:
        chars_per_kpx = n_chars / (box_area / 1000.0)
        if chars_per_kpx < 0.25:  # < 1 char per 4000 px²
            score += 1
            reasons.append(f"low char density ({chars_per_kpx:.3f}/kpx)")

    # Signal 2: Short text in a large region.
    # ≤8 non-space chars AND region occupies > 5% of the image.
    if n_chars <= 8 and region_fraction > 0.05:
        score += 1
        reasons.append(f"short text ({n_chars} chars) in large region ({region_fraction*100:.1f}% of image)")

    # Signal 3: Near-square or wide aspect ratio at large absolute size.
    # SFX art is often large and roughly square or wide. Avoid flagging small
    # dialogue bubbles: require the region to be at least 8% of the image AND
    # the longer side at least 25% of the corresponding image dimension.
    aspect = box_w / box_h
    large_absolute = (region_fraction > 0.08 and
                      (box_w >= 0.25 * img_w or box_h >= 0.25 * img_h))
    square_or_wide = 0.6 <= aspect <= 4.0
    if large_absolute and square_or_wide:
        score += 1
        reasons.append(f"large {aspect:.2f}:1 region ({region_fraction*100:.1f}% of image)")

    # Signal 4: Repeated kana / onomatopoeia pattern.
    # Detects runs of 3+ identical CJK kana, possibly ending in a small vowel
    # modifier (ドドド, ゴゴゴゴ, ドゴォ). Hiragana and katakana ranges:
    #   ひらがな U+3040-U+309F, カタカナ U+30A0-U+30FF, CJK ext U+3400-U+4DBF.
    kana = re.sub(r"[^\u3040-\u30ff\u3400-\u4dbf]", "", nonspace)
    if len(kana) >= 3:
        # 3+ identical chars in a row
        if re.search(r"(.)\1{2,}", kana):
            score += 1
            reasons.append("repeated kana / onomatopoeia pattern")

    is_sfx = score >= 2
    return is_sfx, score, reasons

# ===========================================================================
# Qwen GGUF translator
# ===========================================================================
# ---------------------------------------------------------------------------
# Language support — single source of truth
# ---------------------------------------------------------------------------
# Each entry: code -> {"name": display/prompt name, "script": writing system,
#                      "featured": shown by default in the extension dropdowns,
#                      "lens": Google Lens language code}
# "script" is one of: latin, cyrillic, cjk, hangul, arabic, hebrew, thai,
# devanagari, greek, other. It drives the output sanity-check.
LANGUAGES: Dict[str, Dict[str, Any]] = {
    # ── Featured (shown first in the extension) ──
    "en": {"name": "English",    "script": "latin",    "featured": True,  "lens": "en"},
    "ja": {"name": "Japanese",   "script": "cjk",       "featured": True,  "lens": "ja"},
    "ko": {"name": "Korean",     "script": "hangul",    "featured": True,  "lens": "ko"},
    "zh": {"name": "Chinese",    "script": "cjk",       "featured": True,  "lens": "zh"},
    "id": {"name": "Indonesian", "script": "latin",     "featured": True,  "lens": "id"},
    "ru": {"name": "Russian",    "script": "cyrillic",  "featured": True,  "lens": "ru"},
    "es": {"name": "Spanish",    "script": "latin",     "featured": True,  "lens": "es"},
    # ── The long tail (revealed via "More…") ──
    "zh-tw": {"name": "Chinese (Traditional)", "script": "cjk",      "featured": False, "lens": "zh-TW"},
    "fr": {"name": "French",      "script": "latin",     "featured": False, "lens": "fr"},
    "de": {"name": "German",      "script": "latin",     "featured": False, "lens": "de"},
    "it": {"name": "Italian",     "script": "latin",     "featured": False, "lens": "it"},
    "pt": {"name": "Portuguese",  "script": "latin",     "featured": False, "lens": "pt"},
    "pt-br": {"name": "Portuguese (Brazil)", "script": "latin",   "featured": False, "lens": "pt"},
    "nl": {"name": "Dutch",       "script": "latin",     "featured": False, "lens": "nl"},
    "pl": {"name": "Polish",      "script": "latin",     "featured": False, "lens": "pl"},
    "tr": {"name": "Turkish",     "script": "latin",     "featured": False, "lens": "tr"},
    "vi": {"name": "Vietnamese",  "script": "latin",     "featured": False, "lens": "vi"},
    "th": {"name": "Thai",        "script": "thai",      "featured": False, "lens": "th"},
    "ar": {"name": "Arabic",      "script": "arabic",    "featured": False, "lens": "ar"},
    "he": {"name": "Hebrew",      "script": "hebrew",    "featured": False, "lens": "he"},
    "hi": {"name": "Hindi",       "script": "devanagari","featured": False, "lens": "hi"},
    "el": {"name": "Greek",       "script": "greek",     "featured": False, "lens": "el"},
    "uk": {"name": "Ukrainian",   "script": "cyrillic",  "featured": False, "lens": "uk"},
    "cs": {"name": "Czech",       "script": "latin",     "featured": False, "lens": "cs"},
    "sv": {"name": "Swedish",     "script": "latin",     "featured": False, "lens": "sv"},
    "fi": {"name": "Finnish",     "script": "latin",     "featured": False, "lens": "fi"},
    "no": {"name": "Norwegian",   "script": "latin",     "featured": False, "lens": "no"},
    "da": {"name": "Danish",      "script": "latin",     "featured": False, "lens": "da"},
    "hu": {"name": "Hungarian",   "script": "latin",     "featured": False, "lens": "hu"},
    "ro": {"name": "Romanian",    "script": "latin",     "featured": False, "lens": "ro"},
    "fil": {"name": "Filipino",   "script": "latin",     "featured": False, "lens": "fil"},
    "ms": {"name": "Malay",       "script": "latin",     "featured": False, "lens": "ms"},
    "fa": {"name": "Persian",     "script": "arabic",    "featured": False, "lens": "fa"},
}

# Aliases kept for backward compatibility with older clients.
_LANG_ALIASES = {"cz": "zh"}

def _norm_lang(code: Optional[str]) -> str:
    """Normalize an incoming language code to a key in LANGUAGES."""
    if not code:
        return "en"
    c = code.strip().lower()
    c = _LANG_ALIASES.get(c, c)
    return c

def get_lang_name(code: str) -> str:
    return LANGUAGES.get(_norm_lang(code), {}).get("name", "English")

def lang_script(code: str) -> str:
    return LANGUAGES.get(_norm_lang(code), {}).get("script", "latin")

def lens_lang_code(code: str) -> str:
    entry = LANGUAGES.get(_norm_lang(code))
    return entry["lens"] if entry else _norm_lang(code)

# Backward-compatible name maps (some code paths still reference these directly).
LANG_MAP = {code: meta["name"] for code, meta in LANGUAGES.items()}
LANG_MAP["cz"] = "Chinese"
SRC_LANG_MAP = dict(LANG_MAP)

def _script_hint(lang_name: str) -> str:
    """Provides explicit instructions for non-Latin scripts to prevent romanization."""
    if lang_name in ("Japanese", "Korean", "Chinese", "Chinese (Traditional)"):
        return (f"Write the translation using the native {lang_name} writing system "
                f"(e.g. kanji/kana for Japanese, hangul for Korean, hanzi for Chinese). "
                f"Do NOT romanize. Do NOT transliterate.")
    if lang_name in ("Arabic", "Persian", "Hebrew", "Thai", "Hindi", "Greek", "Russian",
                     "Ukrainian"):
        return (f"Write the translation using the native {lang_name} script. "
                f"Do NOT romanize. Do NOT transliterate.")
    return ""

SYSTEM_PROMPT = (
    "You are a professional manga translator. "
    "Translate the user's text from its original language into {lang}. "
    "Output ONLY the {lang} translation — no source text, no notes, no quotes. "
    "{script_hint}"
)

# Always appended to every translation system prompt (both backends, both
# low and high mode). Honorifics are read straight from the OCR source text and
# carried into the output by the model — there is no name-map or post-process
# reinsertion. Kept short to limit token cost (~50 input tokens).
HONORIFIC_CLAUSE = (
    " If a name in the source text has an honorific suffix or title, romanize it"
    " and attach it to the translated name with a hyphen — do NOT translate it"
    " into an English word and do NOT drop it. Japanese: -kun, -chan, -san,"
    " -sama, -senpai, -sensei, -dono, -shi. Korean: -ssi, -nim, -ya/-a, -hyung,"
    " -noona, -oppa, -unnie, -sunbae. Chinese: -ge, -jie, -shixiong, -shijie."
    " Examples: ユミさま → Yumi-sama, ヤンさま → Yang-sama, タナカさん → Tanaka-san,"
    " ジャックくん → Jack-kun. So '世界が落ちた、ユミさま' → 'The world fell, Yumi-sama'."
)

# Appended to the system prompt only when context-aware mode is on.
# Keeps the extra-token cost small (~30 input tokens per request).
# Honorific handling lives in HONORIFIC_CLAUSE (always on); this clause only
# covers name consistency and pronoun characterization.
CONTEXT_AWARE_CLAUSE = (
    " PRESERVE character names from the source text, keeping each character's"
    " name spelled the same way everywhere. Keep gendered pronouns consistent"
    " with the source characterization."
)

# Appended on top of CONTEXT_AWARE_CLAUSE when context level is "high".
# Style detection rides along inside the SAME translate call — no extra
# round-trip. The model reports how the ORIGINAL lettering looked (weight,
# slant, glow) so the overlay can match it, and _split_style_tag() strips the
# tag back off before the text reaches the renderer.
#
# Style is read off the artwork, so this clause is only ever used together with
# an attached page image (OpenRouter high mode). There is no text-only variant:
# you cannot see how thick a glyph was from a transcription.
STYLE_AWARE_VISION_CLAUSE = (
    " The manga page image is attached. Match each numbered line to the speech"
    " bubble or caption it came from, then look at how that ORIGINAL lettering"
    " was drawn and append a style tag in the exact form [B2], [I1], [R2] or"
    " [B3G]. The letter is the lettering style: B = bold/heavy/thick strokes"
    " (shouting, emphasis), I = italic/slanted/brush-swept lettering, R ="
    " regular upright lettering with normal weight. The digit is how heavy the"
    " strokes were, from 1 (light/thin) to 3 (very heavy/blocky) — judge it"
    " against the other lettering on the same page, not in the absolute. Append"
    " the letter G after the digit ONLY if the original text had a glow, halo,"
    " white outline, or soft light behind the glyphs. Example:"
    " '1. Get away from me! [B3G]' for heavy glowing lettering, '2. ...I see."
    " [R1]' for thin plain lettering. Always output exactly one tag per line, at"
    " the very end of the line. Judge style from the ARTWORK, not from the"
    " wording. Translate ONLY the lines in the numbered list — never transcribe"
    " extra text you can see in the image, and never add lines."
)

# Max edge length for the page image sent in high mode. Full-resolution manga
# scans are ~2000px tall and cost far more image tokens than the extra accuracy
# is worth; expressions and bubble outlines stay readable at this size.
STYLE_VISION_MAX_EDGE = 1024


def _page_image_data_uri(pil_img: "Image.Image") -> str:
    """Downscale + JPEG-encode a page image into a data URI for OpenRouter.

    Returns "" on any failure so the caller can silently fall back to the
    text-only path rather than losing the whole translation batch.
    """
    try:
        img = pil_img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > STYLE_VISION_MAX_EDGE:
            scale = STYLE_VISION_MAX_EDGE / float(longest)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        logging.info(f"[StyleVision] Page image encoded: {img.size[0]}x{img.size[1]}, "
                     f"{len(b64) // 1024}KB base64")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logging.warning(f"[StyleVision] Could not encode page image: {e}")
        return ""


def _page_image_data_uri_original(pil_img: "Image.Image") -> str:
    """JPEG-encode a page at its original pixel dimensions for vision OCR."""
    try:
        img = pil_img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        logging.info(
            f"[Google AI OCR] Original-resolution image encoded: "
            f"{img.size[0]}x{img.size[1]}, {len(b64) // 1024}KB base64"
        )
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logging.warning(f"[Google AI OCR] Could not encode original-resolution image: {e}")
        return ""


def _page_image_data_uri_scaled(pil_img: "Image.Image") -> Tuple[str, float]:
    """Like _page_image_data_uri but also returns the downscale factor.

    scale is (encoded edge / original edge): coordinates the model reports in
    the encoded image are multiplied by 1/scale to map back to full-res. On
    failure returns ("", 1.0).
    """
    try:
        img = pil_img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        scale = 1.0
        if longest > STYLE_VISION_MAX_EDGE:
            scale = STYLE_VISION_MAX_EDGE / float(longest)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        logging.info(f"[Vision OCR] Page image encoded: {img.size[0]}x{img.size[1]}, "
                     f"scale={scale:.4f}, {len(b64) // 1024}KB base64")
        return f"data:image/jpeg;base64,{b64}", scale
    except Exception as e:
        logging.warning(f"[Vision OCR] Could not encode page image: {e}")
        return "", 1.0


# Letter code → lettering-style bucket used for per-style font selection.
_STYLE_CODES = {"b": "bold", "i": "italic", "r": "regular"}

# Trailing style tag, e.g. "[B2]" / "[B3G]" / "( i 1 )" — tolerant of sloppy
# model output. Group 1 = style letter, 2 = stroke weight 1-3, 3 = glow flag.
_STYLE_TAG_RE = re.compile(
    r"[\[\(\{]\s*([BIRbir])\s*([1-3])?\s*([Gg])?\s*[\]\)\}]\s*$"
)

STYLE_FONT_KEYS = ("bold", "italic", "regular")


def _parse_style_fonts(raw: str) -> Dict[str, str]:
    """Parse the style_fonts FormData JSON into a {bold, italic, regular} map.

    Always returns all three keys. Anything unparseable degrades to empty
    strings, which the renderer treats as "use the configured main font".
    """
    fonts = {k: "" for k in STYLE_FONT_KEYS}
    if not raw:
        return fonts
    try:
        data = json.loads(raw)
    except Exception:
        logging.warning(f"[Style] Could not parse style_fonts payload: {raw[:120]!r}")
        return fonts
    if not isinstance(data, dict):
        return fonts
    for key in STYLE_FONT_KEYS:
        val = data.get(key)
        if isinstance(val, str):
            fonts[key] = os.path.basename(val.strip())
    return fonts


def _split_style_tag(line: str) -> Tuple[str, Optional[str], int, bool]:
    """Strip a trailing lettering-style tag off a translated line.

    Returns (clean_text, style_or_None, weight, glow). Lines without a tag come
    back untouched with style=None so non-style jobs are unaffected.
    """
    if not line:
        return line, None, 0, False
    match = _STYLE_TAG_RE.search(line)
    if not match:
        return line, None, 0, False
    style = _STYLE_CODES.get(match.group(1).lower())
    weight = int(match.group(2)) if match.group(2) else 2
    glow = bool(match.group(3))
    return line[:match.start()].strip(), style, weight, glow

def _extract_name_candidates(texts: List[str], max_n: int = 30) -> List[str]:
    """Cheap candidate extraction for the two-pass name dictionary.
    Returns short recurring CJK runs that look like proper nouns."""
    cand_re = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]{1,6}")
    counts: Dict[str, int] = {}
    for t in texts:
        for m in cand_re.findall(t or ""):
            counts[m] = counts.get(m, 0) + 1
    return sorted([c for c, n in counts.items() if n >= 2 or 2 <= len(c) <= 4])[:max_n]


def _parse_name_map_response(raw: str, candidates: List[str]) -> Dict[str, str]:
    """Parse the LLM's 'N. romanized' lines into a {source: romanized} dict."""
    name_map: Dict[str, str] = {}
    for line in (raw or "").splitlines():
        m = re.match(r"^\s*(\d+)\s*[.):\-\]]\s*(.+)$", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        romanized = m.group(2).strip()
        if 0 <= idx < len(candidates) and romanized:
            name_map[candidates[idx]] = romanized
    return name_map


def _build_name_map_llm(texts: List[str], src_lang_name: str, lang_name: str, llm) -> Dict[str, str]:
    """Two-pass step 1 (local GGUF): ask the LLM for a name dictionary.
    Returns {} if no candidates or the call fails. One extra LLM call per job."""
    if not llm or not texts:
        return {}
    candidates = _extract_name_candidates(texts)
    if not candidates:
        return {}
    list_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    user_prompt = (
        f"You are a manga name dictionary builder. For each numbered {src_lang_name} name, "
        f"output the romanized {lang_name} form with any attached honorific suffix preserved "
        f"(e.g. ジャックくん → Jack-kun). Output one line per name as 'N. romanized'. "
        f"No explanations.\n\n{list_text}"
    )
    try:
        with _llm_lock:
            out = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": f"You build name dictionaries from {src_lang_name} into {lang_name}."},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max(128, len(candidates) * 16),
                temperature=0.0,
                stop=["<|im_end|>", "</s>"],
            )
        raw = out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.warning(f"[ContextAware] name-map LLM call failed: {e}")
        return {}
    name_map = _parse_name_map_response(raw, candidates)
    logging.info(f"[ContextAware] Name map ({len(name_map)} entries): {name_map}")
    return name_map


async def _build_name_map_openrouter(texts: List[str], src_lang_name: str, lang_name: str,
                                      api_key: str, model: str) -> Dict[str, str]:
    """Two-pass step 1 (OpenRouter): one extra HTTP call per job for the name dictionary."""
    import aiohttp
    if not texts or not api_key:
        return {}
    candidates = _extract_name_candidates(texts)
    if not candidates:
        return {}
    list_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    user_prompt = (
        f"You are a manga name dictionary builder. For each numbered {src_lang_name} name, "
        f"output the romanized {lang_name} form with any attached honorific suffix preserved "
        f"(e.g. ジャックくん → Jack-kun). Output one line per name as 'N. romanized'. "
        f"No explanations.\n\n{list_text}"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Manga Translation API",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"You build name dictionaries from {src_lang_name} into {lang_name}."},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max(128, len(candidates) * 16),
        "temperature": 0.0,
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload,
            ) as resp:
                data = await resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.warning(f"[ContextAware] name-map OpenRouter call failed: {e}")
        return {}
    name_map = _parse_name_map_response(raw, candidates)
    logging.info(f"[ContextAware] Name map ({len(name_map)} entries): {name_map}")
    return name_map


def _count_script(trans: str) -> Dict[str, int]:
    """Count how many characters belong to each writing system."""
    counts = {"cjk": 0, "hangul": 0, "cyrillic": 0, "arabic": 0,
              "hebrew": 0, "thai": 0, "devanagari": 0, "greek": 0, "latin": 0}
    for c in trans:
        cp = ord(c)
        if ((0x3040 <= cp <= 0x30FF)
                or (0x3400 <= cp <= 0x9FFF)
                or (0xF900 <= cp <= 0xFAFF)
                or (0xFF00 <= cp <= 0xFFEF)):
            counts["cjk"] += 1
        elif 0xAC00 <= cp <= 0xD7AF:
            counts["hangul"] += 1
        elif (0x0400 <= cp <= 0x04FF) or (0x0500 <= cp <= 0x052F):
            counts["cyrillic"] += 1
        elif (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F) or (0xFB50 <= cp <= 0xFDFF):
            counts["arabic"] += 1
        elif 0x0590 <= cp <= 0x05FF:
            counts["hebrew"] += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            counts["thai"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0370 <= cp <= 0x03FF:
            counts["greek"] += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            counts["latin"] += 1
    return counts

def _looks_like_target(trans: str, target_lang: str) -> bool:
    """Sanity check that the output matches the expected target language script.

    Only rejects clear mismatches (e.g. a Latin-script target that came back
    mostly in CJK, meaning the model echoed the source). Scripts we cannot
    reliably distinguish are accepted rather than falsely rejected.
    """
    if not trans:
        return False

    target_script = lang_script(target_lang)
    counts = _count_script(trans)
    total_scripted = sum(counts.values())

    # CJK-family targets must contain their own script (else romanized/echoed).
    if target_script == "cjk" and (counts["cjk"] + counts["hangul"]) == 0:
        return False
    if target_script == "hangul" and counts["hangul"] == 0:
        return False

    # For non-CJK targets, reject any CJK/Hangul character. Even a short
    # untranslated fragment can render as missing-glyph boxes in a Latin font.
    if target_script not in ("cjk", "hangul"):
        cjk_like = counts["cjk"] + counts["hangul"]
        if cjk_like > 0:
            return False

    # Script-specific targets should show at least some of that script when the
    # output has a meaningful amount of scripted characters.
    strict_scripts = {"cyrillic", "arabic", "hebrew", "thai", "devanagari", "greek"}
    if target_script in strict_scripts and total_scripted >= 4:
        if counts[target_script] == 0 and counts["latin"] > total_scripted * 0.6:
            # All Latin, no target script → likely romanized/failed.
            return False

    return True

def _normalize_for_echo(s: str) -> str:
    """Normalize case, spacing, and punctuation for source-echo detection."""
    return "".join(ch.casefold() for ch in (s or "") if ch.isalnum())

def _is_echo(source: str, output: str) -> bool:
    """True when the model handed back the source instead of translating it."""
    src = _normalize_for_echo(source)
    out = _normalize_for_echo(output)
    if not src or not out:
        return False
    if out == src:
        return True
    return src in out and len(out) <= max(len(src) + 4, int(len(src) * 1.25))


def _translation_fragments(source: str, candidate: str) -> List[str]:
    raw = (candidate or "").strip()
    if not raw:
        return []

    fragments = [raw]
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        fragments.append(line)
        without_number = re.sub(r"^\s*[\[\(]?\d+[\]\)]?[\.\)\-:]\s*", "", line)
        fragments.append(without_number)
        without_label = re.sub(
            r"^\s*(?:translation|translated|output|answer)\s*:\s*",
            "",
            without_number,
            flags=re.IGNORECASE,
        )
        fragments.append(without_label)
        for separator in ("->", "=>", "→", "⇒"):
            if separator in without_label:
                fragments.append(without_label.rsplit(separator, 1)[-1].strip())

    source_clean = clean_text_for_font(source or "")
    if source_clean:
        for fragment in list(fragments):
            cleaned_fragment = clean_text_for_font(fragment)
            if source_clean in cleaned_fragment:
                fragments.append(cleaned_fragment.replace(source_clean, " ").strip(" :-=→⇒"))

    unique = []
    seen = set()
    for fragment in fragments:
        cleaned = clean_text_for_font(fragment)
        key = _normalize_for_echo(cleaned)
        if cleaned and key and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def _validated_translation(source: str, candidate: str, target_lang: str) -> str:
    """Return only drawable target-language text; fail closed on source echoes."""
    valid = [
        fragment for fragment in _translation_fragments(source, candidate)
        if _looks_like_target(fragment, target_lang) and not _is_echo(source, fragment)
    ]
    if not valid:
        return ""

    source_norm = _normalize_for_echo(source)
    valid.sort(
        key=lambda fragment: (
            source_norm in _normalize_for_echo(fragment),
            bool(re.match(r"^\s*[\[\(]?\d+[\]\)]?[\.\)\-:]", fragment)),
            bool(re.match(r"^\s*(?:translation|translated|output|answer)\s*:", fragment, re.IGNORECASE)),
            -len(fragment),
        )
    )
    return valid[0]


def _translation_source_name(ocr_lang: str, target_lang: str) -> str:
    """Describe the input language without creating same-language prompts."""
    if _norm_lang(ocr_lang) == _norm_lang(target_lang):
        return "the automatically detected source language"
    if _norm_lang(ocr_lang) in LANGUAGES:
        return get_lang_name(ocr_lang)
    return "the automatically detected source language"


def _effective_ocr_language(target_lang: str, *candidates: Optional[str]) -> str:
    selected = next((value for value in candidates if value), "auto")
    if _norm_lang(selected) == _norm_lang(target_lang):
        return "auto"
    return selected

def get_qwen():
    global _global_qwen, _current_qwen_path
    if _global_qwen is None:
        if Llama is None:
            raise RuntimeError("llama-cpp-python not installed: pip install llama-cpp-python")
        with _qwen_model_lock:
            if _global_qwen is None:
                path = _current_qwen_path
                if path is None or not _is_valid_gguf(path):
                    logging.info(f"[Qwen] Local model missing/invalid, locating via HF cache or download...")
                    path = download_gguf(_current_qwen_repo_id, _current_qwen_filename)
                    _current_qwen_path = path
                try:
                    path = path.resolve()
                except Exception:
                    pass
                if not _is_valid_gguf(path):
                    raise RuntimeError(f"Refusing to load invalid GGUF: {path}.")
                use_gpu = _log_llama_device("Qwen")
                n_gpu_layers = -1 if use_gpu else 0
                logging.info(f"[Qwen] loading {path} ({'CUDA all layers' if use_gpu else 'CPU fallback'}) ...")
                try:
                    _global_qwen = Llama(
                        model_path=str(path), n_ctx=2048,
                        n_threads=max(4, os.cpu_count() or 4),
                        n_gpu_layers=n_gpu_layers, verbose=False,
                    )
                except Exception as e:
                    logging.error(f"[Qwen] Failed to load GGUF from {path}: {e}")
                    raise RuntimeError(f"llama-cpp-python failed to load {path}. Error: {e}")
                logging.info(f"[Qwen] loaded: {_current_qwen_repo_id}/{_current_qwen_filename}")
    return _global_qwen

def _ensure_vision_projector(repo_id: str, model_filename: str) -> None:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        candidates = [name for name in list_repo_files(repo_id)
                      if name.lower().endswith(".gguf") and _is_projector_gguf(pathlib.Path(name))]
        if not candidates:
            return
        projector = candidates[0]
        hf_hub_download(repo_id=repo_id, filename=projector)
        logging.info(f"[GGUF] Vision projector available for {model_filename}: {projector}")
    except Exception as exc:
        logging.info(f"[GGUF] No downloadable vision projector for {model_filename}: {exc}")


def switch_qwen_model(repo_id: str, filename: Optional[str] = None):
    global _global_qwen, _local_vision_qwen
    global _local_vision_model_path, _local_vision_projector_path
    global _current_qwen_repo_id, _current_qwen_filename, _current_qwen_path
    path = download_gguf(repo_id, filename)
    _ensure_vision_projector(repo_id, path.name)
    with _qwen_model_lock:
        _current_qwen_repo_id = repo_id
        _current_qwen_filename = filename or path.name
        _current_qwen_path = path
        _global_qwen = None
        _local_vision_qwen = None
        _local_vision_model_path = None
        _local_vision_projector_path = None
    _save_settings()
    logging.info(f"[Qwen] Switched to {repo_id}/{filename}, preloading...")
    get_qwen()


def _retry_translate_single(text: str, lang_name: str, src_lang_name: str, llm) -> str:
    """Retry translation with a much stronger prompt that explicitly forbids source language."""
    retry_prompt = (
        f"You MUST translate this text into {lang_name}. "
        f"The source is {src_lang_name}. "
        f"DO NOT output any {src_lang_name} text. "
        f"Output ONLY {lang_name} text, nothing else. No explanations."
    )
    retry_user = f"[Source: {src_lang_name}] -> [Target: {lang_name}]\n{text}"
    msgs = [
        {"role": "system", "content": retry_prompt},
        {"role": "user", "content": retry_user},
    ]
    try:
        with _llm_lock:
            out = llm.create_chat_completion(
                messages=msgs,
                max_tokens=max(64, min(256, len(text) + 32)),
                temperature=0.1,
                top_p=0.95,
                stop=["<|im_end|>", "</s>"],
            )
        raw = out["choices"][0]["message"]["content"].strip()
        for tok in ("<|im_start|>", "<|im_end|>", "</s>"):
            raw = raw.replace(tok, "")
        # Strip common prefixes
        for prefix in ("Translation:", "Translated:", "Output:", f"{lang_name}:", f"{lang_name} translation:"):
            if raw.lower().startswith(prefix.lower()):
                raw = raw[len(prefix):].strip()
        return raw
    except Exception:
        return ""
def qwen_translate(text: str, target_lang: str = "en", ocr_lang: str = "ja") -> str:
    text = text.strip()
    if not text:
        return ""
    lang_name = get_lang_name(target_lang)
    src_lang_name = _translation_source_name(ocr_lang, target_lang)
    
    max_tok = max(64, min(256, len(text) + 32))
    logging.info(f"[LLM] Starting translation for: '{text[:40]}' -> {lang_name}")
    llm = get_qwen()
    
    sys_prompt = SYSTEM_PROMPT.format(lang=lang_name, script_hint=_script_hint(lang_name)) + HONORIFIC_CLAUSE
    user_prompt = f"[Source language: {src_lang_name}]\n{text}"
    
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    try:
        with _llm_lock:
            out = llm.create_chat_completion(
                messages=msgs, max_tokens=max_tok, temperature=0.2, top_p=0.9,
                stop=["<|im_end|>", "</s>"],
            )
        raw = out["choices"][0]["message"]["content"].strip()
        for tok in ("<|im_start|>", "<|im_end|>", "</s>"):
            if tok in raw:
                raw = raw.replace(tok, "")
                
        # Clean up common prefixes small models use despite instructions
        for prefix in ("Translation:", "Translated:", "Output:", f"{lang_name}:", f"{lang_name} translation:"):
            if raw.lower().startswith(prefix.lower()):
                raw = raw[len(prefix):].strip()
                
        # Validate translation looks like target language
        if not _looks_like_target(raw, target_lang):
            logging.warning(f"[LLM] Output appears to be wrong language ({target_lang}), retrying with stronger constraints...")
            raw = _retry_translate_single(text, lang_name, src_lang_name, llm)
            if not _looks_like_target(raw, target_lang):
                logging.error(f"[LLM] Local model failed target-language validation after retry; suppressing output")
                return ""
        validated = _validated_translation(text, raw, target_lang)
        if not validated:
            logging.error("[LLM] Local model output rejected as source echo or invalid target text")
            return ""
        logging.info(f"[LLM] Translated to: '{validated[:40]}'")
        return validated
    except Exception as e:
        logging.error(f"[LLM] Translation failed: {e}")
        return ""

def qwen_translate_batch(texts: List[str], target_lang: str = "en", ocr_lang: str = "ja",
                         context_aware: bool = False,
                         name_map: Optional[Dict[str, str]] = None,
                         llm: Any = None) -> List[str]:
    """Translate a list of texts in a SINGLE LLM call using a numbered list.

    Lettering-style detection is deliberately absent here: style is read off the
    page artwork, which needs a vision model, so it is OpenRouter-only. The
    server forces context level back to "low" for non-OpenRouter backends, so
    this path is never asked for style tags.
    """
    indexed_texts = [(i, t.strip()) for i, t in enumerate(texts) if t.strip()]
    if not indexed_texts:
        return [""] * len(texts)

    lang_name = get_lang_name(target_lang)
    src_lang_name = _translation_source_name(ocr_lang, target_lang)

    prompt_lines = [f"{idx + 1}. {text.replace(chr(10), ' ')}" for idx, (_, text) in enumerate(indexed_texts)]
    batch_text = f"[Source language: {src_lang_name}]\n" + "\n".join(prompt_lines)

    batch_system_prompt = (
        f"You are a professional manga translator. "
        f"Translate each numbered line from {src_lang_name} into {lang_name}. "
        f"Output the same numbered list, containing ONLY the {lang_name} translations. "
        f"Do not include the original text. No explanations. {_script_hint(lang_name)}"
    ).strip()
    batch_system_prompt += HONORIFIC_CLAUSE
    if context_aware:
        batch_system_prompt += CONTEXT_AWARE_CLAUSE

    total_chars = sum(len(t) for _, t in indexed_texts)
    per_line_budget = 32
    max_tok = max(256, min(4096, total_chars + (len(indexed_texts) * per_line_budget)))

    llm = llm or get_qwen()
    msgs = [
        {"role": "system", "content": batch_system_prompt},
        {"role": "user", "content": batch_text},
    ]

    try:
        with _llm_lock:
            out = llm.create_chat_completion(
                messages=msgs, max_tokens=max_tok, temperature=0.2, top_p=0.9,
                stop=["<|im_end|>", "</s>"],
            )
        raw = out["choices"][0]["message"]["content"].strip()

        results = [""] * len(texts)
        parsed_lines = [ln.strip() for ln in raw.split('\n') if ln.strip()]
        
        matched_any = False
        # Try to match numbered lines (e.g., "1. Hello")
        for line in parsed_lines:
            match = re.match(r"^(\d+)[\.\)]\s*(.*)$", line)
            if match:
                num = int(match.group(1)) - 1
                trans = match.group(2).strip()
                if 0 <= num < len(indexed_texts):
                    orig_idx = indexed_texts[num][0]
                    # Validate this translation
                    if not _looks_like_target(trans, target_lang):
                        logging.warning(f"[LLM Batch] Box {num+1} appears to be wrong language, retrying individually...")
                        # FIX: use indexed_texts[num][1] instead of undefined 'text'
                        trans = qwen_translate(indexed_texts[num][1], target_lang, ocr_lang)
                    results[orig_idx] = clean_text_for_font(trans)
                    matched_any = True

        # Fallback if model ignored numbers and just outputted translations line by line
        if not matched_any and len(parsed_lines) == len(indexed_texts):
            logging.warning("[LLM Batch] Model didn't use numbers, mapping line-by-line...")
            for i, line in enumerate(parsed_lines):
                orig_idx = indexed_texts[i][0]
                # Validate line-by-line fallback
                if not _looks_like_target(line, target_lang):
                    logging.warning(f"[LLM Batch Fallback] Line {i+1} appears to be wrong language, retrying individually...")
                    single_result = qwen_translate(indexed_texts[i][1], target_lang, ocr_lang)
                    results[orig_idx] = clean_text_for_font(single_result)
                else:
                    results[orig_idx] = clean_text_for_font(line)
            matched_any = True

        # Final fallback for any missing items OR items that came back in the wrong script
        for num, (orig_idx, text) in enumerate(indexed_texts):
            if not results[orig_idx] or not _looks_like_target(results[orig_idx], target_lang):
                logging.warning(f"[LLM Batch] Box {num+1} missed or wrong script, retrying individually...")
                results[orig_idx] = qwen_translate(text, target_lang, ocr_lang)

        return results
    except Exception as e:
        logging.error(f"[LLM Batch] Translation failed: {e}")
        return [""] * len(texts)

# ===========================================================================
# OpenRouter Translation
# ===========================================================================

def _ratelimit_wait_seconds(response) -> float:
    """How long to wait out an HTTP 429 before trying again.

    Prefer whatever the server tells us — `Retry-After` (delay-seconds or an
    HTTP-date) first, then `X-RateLimit-Reset` (epoch, seconds or milliseconds).
    When neither is present or parseable, fall back to the average lifetime of a
    free-tier per-minute window. Always clamped so a bogus header can't stall a
    job indefinitely.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, min(OPENROUTER_RATELIMIT_MAX_WAIT, float(retry_after)))
        except (TypeError, ValueError):
            try:
                from email.utils import parsedate_to_datetime
                parsed = parsedate_to_datetime(retry_after)
                if parsed is not None:
                    delta = parsed.timestamp() - time.time()
                    return max(0.0, min(OPENROUTER_RATELIMIT_MAX_WAIT, delta))
            except Exception:
                pass

    reset = response.headers.get("X-RateLimit-Reset")
    if reset:
        try:
            reset_val = float(reset)
            # OpenRouter reports this in milliseconds. Anything well past the
            # plausible epoch-seconds range is a millisecond timestamp.
            if reset_val > 1e11:
                reset_val /= 1000.0
            delta = reset_val - time.time()
            if 0 < delta <= OPENROUTER_RATELIMIT_MAX_WAIT:
                return delta
        except (TypeError, ValueError):
            pass

    return OPENROUTER_RATELIMIT_AVG_WAIT


async def openrouter_translate_batch(texts: List[str], target_lang: str = "en", ocr_lang: str = "ja",
                                      max_retries: int = 2, context_aware: bool = False,
                                      name_map: Optional[Dict[str, str]] = None,
                                      style_aware: bool = False,
                                      styles_out: Optional[List[Optional[Dict[str, Any]]]] = None,
                                      page_image: Optional["Image.Image"] = None) -> List[str]:
    import aiohttp
    import random

    if styles_out is not None and len(styles_out) < len(texts):
        styles_out.extend([None] * (len(texts) - len(styles_out)))

    with _model_type_lock:
        api_key = _openrouter_api_key
        model = _openrouter_model
    with _openrouter_free_mode_lock:
        paid_mode = _openrouter_free_mode

    if not api_key:
        logging.error("[OpenRouter] API key not configured")
        return [""] * len(texts)

    indexed_texts = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not indexed_texts:
        return [""] * len(texts)

    lang_name = get_lang_name(target_lang)
    src_lang_name = _translation_source_name(ocr_lang, target_lang)
    total_chars = sum(len(t) for _, t in indexed_texts)
    per_line_budget = 40 if style_aware else 20
    max_tok = max(256, min(4096, total_chars + (len(indexed_texts) * per_line_budget)))

    prompt_lines = [f"{idx + 1}. {text.replace(chr(10), ' ')}" for idx, (orig_i, text) in enumerate(indexed_texts)]
    batch_text = f"[Source language: {src_lang_name}]\n" + "\n".join(prompt_lines)

    # Initial system prompt
    base_system_prompt = (
        f"You are a professional manga translator. "
        f"Translate each numbered line from {src_lang_name} into {lang_name}. "
        f"CRITICAL: You must actually translate the text. Do NOT simply copy or repeat the original {src_lang_name} text. "
        f"Output ONLY the translated list, one per line, keeping the exact same numbers. "
        f"Do not include the original text. No explanations, no notes, no quotes. {_script_hint(lang_name)}"
    ).strip()
    base_system_prompt += HONORIFIC_CLAUSE
    if context_aware:
        base_system_prompt += CONTEXT_AWARE_CLAUSE
    
    # High mode vision: send the page image for style detection from the artwork.
    # Style is visual-only — if encoding fails, style detection is skipped.
    page_data_uri = ""
    if style_aware and page_image is not None:
        page_data_uri = _page_image_data_uri(page_image)
        if page_data_uri:
            base_system_prompt += STYLE_AWARE_VISION_CLAUSE
            logging.info("[OpenRouter Batch] High mode: page image attached for style detection.")
        else:
            logging.warning("[OpenRouter Batch] High mode: image encoding failed, style detection skipped.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Manga Translation API"
    }

    # Build the user message: if we have a page image, use vision format; otherwise plain text.
    if page_data_uri:
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": page_data_uri}
                },
                {
                    "type": "text",
                    "text": batch_text
                }
            ]
        }
    else:
        user_message = {"role": "user", "content": batch_text}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": base_system_prompt},
            user_message,
        ],
        "max_tokens": max_tok,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    logging.info(f"[OpenRouter Batch] Sending {len(indexed_texts)} texts in one request to {model}...")

    QUOTE_CHARS = "\"'“”‘’"

    # `attempt` counts translation-quality retries (echo/parse failures) and is
    # what max_retries bounds. A 429 is not a bad response — it's a rate limit —
    # so a 429 resend does NOT advance `attempt` or trigger the echo-escalation
    # prompt below. RATELIMIT_RESEND_CAP just stops a permanently-throttled key
    # from looping forever.
    RATELIMIT_RESEND_CAP = 50
    ratelimit_resends = 0
    # Set only when a response came back but the translation was unusable
    # (echoed source / unparseable). Drives the escalation prompt + backoff. A
    # 429, timeout, or transport error leaves it False so we resend as-is.
    retry_after_bad_translation = False
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        if retry_after_bad_translation:
            retry_after_bad_translation = False
            # An echo / wrong-script / unparseable response is a MODEL problem,
            # not a rate limit — waiting changes nothing. Re-prompt immediately
            # (the escalated instruction below is what actually fixes it) so a
            # few bad attempts can't add tens of seconds of dead time per batch.
            logging.info(f"[OpenRouter Batch] Retry {attempt}/{max_retries} (re-prompting immediately)...")

            # Escalate the prompt on retries to force the LLM to stop echoing
            escalated_prompt = (
                f"YOUR PREVIOUS RESPONSE WAS INVALID BECAUSE YOU ECHOED THE SOURCE TEXT. "
                f"You MUST translate each numbered line from {src_lang_name} into {lang_name}. "
                f"CRITICAL: Do NOT repeat the original {src_lang_name} text. You MUST output {lang_name} text only. "
                f"Output ONLY the translated list, one per line, keeping the exact same numbers. "
                f"No explanations, no notes, no quotes. {_script_hint(lang_name)}"
            ).strip()
            escalated_prompt += HONORIFIC_CLAUSE
            if context_aware:
                escalated_prompt += CONTEXT_AWARE_CLAUSE
            if style_aware:
                # The image is still attached on retries, so keep the vision
                # wording when available.
                if page_data_uri:
                    escalated_prompt += STYLE_AWARE_VISION_CLAUSE
            payload["messages"][0]["content"] = escalated_prompt

        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 429:
                        # A 429 is a rate limit, not a bad translation, so it
                        # must not consume the echo/parse retry budget. Roll
                        # `attempt` back and bound the resends separately.
                        ratelimit_resends += 1
                        if ratelimit_resends > RATELIMIT_RESEND_CAP:
                            logging.error(f"[OpenRouter Batch] Still rate limited after {RATELIMIT_RESEND_CAP} resends, giving up.")
                            return [""] * len(texts)
                        attempt -= 1
                        if paid_mode:
                            # Paid account: no per-minute quota to wait out, so
                            # re-send the whole batch immediately. Never fan out
                            # to per-box requests just because of a 429.
                            logging.warning("[OpenRouter Batch] Rate limited (429) — paid mode ON, retrying batch immediately.")
                            continue
                        wait = _ratelimit_wait_seconds(response)
                        logging.warning(f"[OpenRouter Batch] Rate limited (429). Waiting {wait:.1f}s...")
                        await asyncio.sleep(wait)
                        continue

                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"[OpenRouter Batch] API error {response.status} on attempt {attempt}: {error_text[:200]}")
                        # Transient upstream/provider errors (500, 502, 503, 504,
                        # 524, 408) are not bad translations and not client
                        # mistakes — OpenRouter's provider routing throws these
                        # even on paid keys with no rate limit. Resend the batch
                        # with a short backoff WITHOUT consuming the translation
                        # retry budget, so a couple of hiccups can't collapse the
                        # whole page to slow per-box requests. Bounded by the same
                        # resend cap as 429s.
                        if response.status in (408, 500, 502, 503, 504, 524):
                            ratelimit_resends += 1
                            if ratelimit_resends > RATELIMIT_RESEND_CAP:
                                logging.error(f"[OpenRouter Batch] Persistent {response.status} after {RATELIMIT_RESEND_CAP} resends, giving up.")
                                return [""] * len(texts)
                            attempt -= 1
                            backoff = min(8.0, 1.0 + ratelimit_resends * 0.5) + random.uniform(0.2, 0.8)
                            logging.warning(f"[OpenRouter Batch] Transient {response.status}, resending batch in {backoff:.1f}s (resend {ratelimit_resends}).")
                            await asyncio.sleep(backoff)
                            continue
                        # Genuine client error (400/401/403/404 etc.) — retrying
                        # won't help. Consume the attempt and let it fail through.
                        continue

                    data = await response.json()

                    raw = None
                    try:
                        if (data and isinstance(data.get("choices"), list)
                            and len(data["choices"]) > 0
                            and isinstance(data["choices"][0].get("message"), dict)):
                            raw = data["choices"][0]["message"].get("content")
                    except (IndexError, KeyError, TypeError) as e:
                        logging.warning(f"[OpenRouter Batch] Unexpected structure on attempt {attempt}: {e}")

                    if not raw or not isinstance(raw, str):
                        logging.warning(f"[OpenRouter Batch] Empty/None content on attempt {attempt}")
                        retry_after_bad_translation = True
                        continue

                    # Strip markdown code fences if present
                    raw = raw.strip()
                    if raw.startswith("```"):
                        fence_lines = raw.split('\n')
                        if fence_lines[0].startswith("```"):
                            fence_lines = fence_lines[1:]
                        if fence_lines and fence_lines[-1].startswith("```"):
                            fence_lines = fence_lines[:-1]
                        raw = '\n'.join(fence_lines).strip()

                    results = [""] * len(texts)
                    if styles_out is not None:
                        for _mi in range(len(styles_out)):
                            styles_out[_mi] = None
                    parsed_lines = [ln.strip() for ln in raw.split('\n') if ln.strip()]

                    # Try numbered format: "1. text", "1) text", "1: text", "[1] text"
                    matched_count = 0
                    for line in parsed_lines:
                        match = re.match(
                            r"^\s*[\[\(]?(\d+)[\]\)]?[\.\)\-:]\s*(.*)$", line
                        )
                        if match:
                            num = int(match.group(1)) - 1
                            trans = match.group(2).strip()
                            line_style, line_weight, line_glow = None, 0, False
                            if style_aware:
                                trans, line_style, line_weight, line_glow = _split_style_tag(trans)
                            # Strip wrapping quotes (standard and smart)
                            if len(trans) >= 2 and trans[0] in QUOTE_CHARS and trans[-1] in QUOTE_CHARS:
                                trans = trans[1:-1].strip()

                            if 0 <= num < len(indexed_texts):
                                orig_idx = indexed_texts[num][0]
                                source = indexed_texts[num][1]
                                results[orig_idx] = _validated_translation(source, trans, target_lang)
                                if styles_out is not None and line_style and results[orig_idx]:
                                    styles_out[orig_idx] = {"style": line_style, "weight": line_weight, "glow": line_glow}
                                matched_count += 1

                    # Fallback: line-by-line if no numbers but count matches
                    if matched_count == 0 and len(parsed_lines) == len(indexed_texts):
                        logging.warning("[OpenRouter Batch] No numbered lines found, trying line-by-line mapping...")
                        for i, line in enumerate(parsed_lines):
                            trans = line.strip()
                            line_style, line_weight, line_glow = None, 0, False
                            if style_aware:
                                trans, line_style, line_weight, line_glow = _split_style_tag(trans)
                            if len(trans) >= 2 and trans[0] in QUOTE_CHARS and trans[-1] in QUOTE_CHARS:
                                trans = trans[1:-1].strip()
                            orig_idx = indexed_texts[i][0]
                            source = indexed_texts[i][1]
                            results[orig_idx] = _validated_translation(source, trans, target_lang)
                            if styles_out is not None and line_style and results[orig_idx]:
                                styles_out[orig_idx] = {"style": line_style, "weight": line_weight, "glow": line_glow}
                            matched_count += 1

                    # Validate script + reject echoes, then clear failures.
                    # Map original index -> source text so we can echo-check the
                    # output against the exact line it was meant to translate.
                    src_by_orig = {orig_i: src for (orig_i, src) in indexed_texts}
                    valid_count = 0
                    for i, r in enumerate(results):
                        if r and _looks_like_target(r, target_lang) and not _is_echo(src_by_orig.get(i, ""), r):
                            valid_count += 1
                        else:
                            results[i] = ""
                            if styles_out is not None:
                                styles_out[i] = None

                    logging.info(f"[OpenRouter Batch] Parsed {valid_count}/{len(indexed_texts)} valid translations "
                                 f"(matched {matched_count} lines).")
                    
                    if valid_count > 0:
                        return results
                    else:
                        logging.warning(f"[OpenRouter Batch] Failed to parse any valid translations. "
                                        f"Raw response (first 500 chars): {raw[:500]!r}")
                        retry_after_bad_translation = True
                        continue

        except asyncio.TimeoutError:
            # A network timeout is transient, not a bad translation. Resend the
            # batch without consuming the translation-retry budget, bounded by
            # the same resend cap, so one slow response can't force sequential.
            ratelimit_resends += 1
            if ratelimit_resends > RATELIMIT_RESEND_CAP:
                logging.error(f"[OpenRouter Batch] Persistent timeouts after {RATELIMIT_RESEND_CAP} resends, giving up.")
                return [""] * len(texts)
            attempt -= 1
            logging.warning(f"[OpenRouter Batch] Timeout, resending batch (resend {ratelimit_resends}).")
            continue
        except Exception as e:
            ratelimit_resends += 1
            if ratelimit_resends > RATELIMIT_RESEND_CAP:
                logging.error(f"[OpenRouter Batch] Persistent transport errors after {RATELIMIT_RESEND_CAP} resends, giving up.")
                return [""] * len(texts)
            attempt -= 1
            backoff = min(8.0, 1.0 + ratelimit_resends * 0.5) + random.uniform(0.2, 0.8)
            logging.error(f"[OpenRouter Batch] Transport error on resend {ratelimit_resends} (retrying in {backoff:.1f}s): {e}")
            await asyncio.sleep(backoff)
            continue

    logging.error(f"[OpenRouter Batch] FAILED after {max_retries} retries. Falling back to sequential.")
    return [""] * len(texts)


async def openrouter_translate(text: str, target_lang: str = "en", ocr_lang: str = "ja", max_retries: int = 5) -> str:
    import aiohttp
    import random

    with _model_type_lock:
        api_key = _openrouter_api_key
        model = _openrouter_model
    with _openrouter_free_mode_lock:
        free_mode = _openrouter_free_mode

    if not api_key:
        logging.error("[OpenRouter] API key not configured")
        return ""

    if not text.strip():
        return ""

    lang_name = get_lang_name(target_lang)
    src_lang_name = _translation_source_name(ocr_lang, target_lang)
    max_tok = max(16, min(96, len(text) + 16))

    logging.info(f"[OpenRouter] Translating '{text[:40]}' -> {lang_name} using {model}")

    sys_prompt = SYSTEM_PROMPT.format(lang=lang_name, script_hint=_script_hint(lang_name)) + HONORIFIC_CLAUSE
    user_prompt = f"[Source language: {src_lang_name}]\n{text}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Manga Translation API"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tok,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    # Only back off before a retry caused by a transient FAILURE (timeout,
    # transport error, non-200). An echo / wrong-script rejection is a model
    # problem that waiting won't fix, so those retries re-prompt immediately.
    backoff_next_retry = False
    for attempt in range(1, max_retries + 1):
        if attempt > 1 and backoff_next_retry:
            backoff_next_retry = False
            wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
            logging.info(f"[OpenRouter] Retry {attempt}/{max_retries} after {wait_time:.1f}s wait...")
            await asyncio.sleep(wait_time)

        try:
            timeout = aiohttp.ClientTimeout(total=90)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 429:
                        if free_mode:
                            logging.warning("[OpenRouter] Rate limited (429) — free mode ON, skipping retries.")
                            return ""
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait = float(retry_after)
                        else:
                            wait = 10.0
                        logging.warning(f"[OpenRouter] Rate limited (429). Waiting {wait:.1f}s...")
                        await asyncio.sleep(wait)
                        continue

                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"[OpenRouter] API error {response.status} on attempt {attempt}/{max_retries}: {error_text[:200]}")
                        backoff_next_retry = True
                        continue

                    data = await response.json()

                    raw = None
                    try:
                        if (data
                            and isinstance(data.get("choices"), list)
                            and len(data["choices"]) > 0
                            and isinstance(data["choices"][0].get("message"), dict)):
                            raw = data["choices"][0]["message"].get("content")
                    except (IndexError, KeyError, TypeError) as e:
                        logging.warning(f"[OpenRouter] Unexpected response structure on attempt {attempt}: {e}")

                    if not raw or not isinstance(raw, str):
                        logging.warning(f"[OpenRouter] Empty/None content on attempt {attempt}/{max_retries} for '{text[:30]}'")
                        backoff_next_retry = True
                        continue

                    result = _validated_translation(text, raw, target_lang)
                    # Reject a wrong-script or echoed response so raw source
                    # never reaches the overlay. Retry with the stronger prompt.
                    if not result:
                        logging.warning(f"[OpenRouter] Rejected echo/wrong-script output on attempt {attempt} for '{text[:30]}'")
                        payload["messages"][0]["content"] = (
                            f"You MUST translate into {lang_name}. The previous answer was rejected because "
                            f"it repeated the {src_lang_name} source instead of translating it. "
                            f"Output ONLY the {lang_name} translation, nothing else. {_script_hint(lang_name)}"
                        )
                        continue
                    logging.info(f"[OpenRouter] Translated to: '{result[:40]}'")
                    return result

        except asyncio.TimeoutError:
            logging.warning(f"[OpenRouter] Timeout on attempt {attempt}/{max_retries}")
            backoff_next_retry = True
            continue
        except Exception as e:
            logging.error(f"[OpenRouter] Error on attempt {attempt}/{max_retries}: {e}")
            backoff_next_retry = True
            continue

    logging.error(f"[OpenRouter] FAILED after {max_retries} retries for: '{text[:40]}'")
    return ""

def translate_with_current_backend(text: str, target_lang: str = "en", ocr_lang: str = "ja") -> str:
    with _model_type_lock:
        model_type = _current_model_type
    if model_type == "openrouter":
        try:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(openrouter_translate(text, target_lang, ocr_lang))
                return result
            finally:
                loop.close()
        except Exception as e:
            logging.error(f"[OpenRouter] Failed to run async translation: {e}")
            return ""
    else:
        return qwen_translate(text, target_lang, ocr_lang)

async def translate_with_current_backend_async(text: str, target_lang: str = "en", ocr_lang: str = "ja") -> str:
    with _model_type_lock:
        model_type = _current_model_type
    if model_type == "openrouter":
        return await openrouter_translate(text, target_lang, ocr_lang)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_llm_executor, qwen_translate, text, target_lang, ocr_lang)

# ===========================================================================
# Inpainting (SimpleLama + cv2 fallback) — with Low/High mode support
# ===========================================================================

def load_lama_low():
    """Load the default SimpleLama model (low quality, big-lama.pt packaged)."""
    global _simple_lama_model
    if _simple_lama_model is None and SimpleLama is not None:
        device = get_torch_device()
        logging.info(f"[SimpleLama] Loading (low/default) on device: {device}")
        _simple_lama_model = SimpleLama()
        if device == "cuda":
            try:
                inner = getattr(_simple_lama_model, "model", None)
                if inner is not None and hasattr(inner, "to"):
                    inner.to("cuda")
                    logging.info("[SimpleLama] Model moved to CUDA.")
                else:
                    logging.warning("[SimpleLama] Could not access .model to move to CUDA.")
            except Exception as e:
                logging.warning(f"[SimpleLama] Failed moving to CUDA: {e}")
    return _simple_lama_model


class HighQualityLama:
    """Custom wrapper for anime-manga-big-lama.pt.
    
    Optimized for speed and color accuracy:
    - Processes connected components (text clusters) individually instead of the whole image.
    - Runs single-pass inference for crops smaller than 512x512.
    - Uses LAB color space for precise color correction to prevent washed-out results.
    - Uses targeted feathered blending to avoid light halos.
    """

    def __init__(self):
        self.device = get_torch_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        ensure_lama_large()
        logging.info(f"[Lama High] Loading {LAMA_LARGE_PATH} on device: {self.device} (dtype={self.dtype})")
        
        try:
            self.model = torch.jit.load(str(LAMA_LARGE_PATH), map_location=self.device)
        except Exception as e:
            logging.warning(f"[Lama High] Failed to load on {self.device} ({e}), falling back to CPU.")
            self.device = "cpu"
            self.dtype = torch.float32
            self.model = torch.jit.load(str(LAMA_LARGE_PATH), map_location="cpu")
            
        self.model.eval()
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.patch_size = 512
        self.stride = 384  # 25% overlap is plenty for seamless blending
        self.batch_size = 4 if self.device == "cuda" else 1

    def _infer_batch(self, batch_imgs: np.ndarray, batch_masks: np.ndarray) -> np.ndarray:
        """Run the LaMa model on a batch of patches."""
        img_t = (
            torch.from_numpy(batch_imgs).float().permute(0, 3, 1, 2)
            .to(self.device).to(self.dtype) / 255.0
        )
        mask_t = (
            torch.from_numpy(batch_masks).float().unsqueeze(1)
            .to(self.device).to(self.dtype) / 255.0
        )
        mask_t = (mask_t > 0.5).float()

        with torch.no_grad():
            out = self.model(img_t, mask_t).clamp(0, 1)

        return (out.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.float32)

    def _infer_crop(self, crop_img: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        """Run LaMa on a single crop. Pads to 512x512 if small enough, otherwise patches."""
        H, W = crop_img.shape[:2]
        ps = self.patch_size
        
        # Fast path: crop fits inside a single patch
        if H <= ps and W <= ps:
            pad_h = ps - H
            pad_w = ps - W
            padded_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            padded_mask = np.pad(crop_mask, ((0, pad_h), (0, pad_w)), mode="reflect")
            
            out = self._infer_batch(
                np.expand_dims(padded_img, 0),
                np.expand_dims(padded_mask, 0)
            )[0]
            return out[:H, :W].clip(0, 255).astype(np.uint8)

        # Slow path: crop requires patching
        gauss = cv2.getGaussianKernel(ps, ps // 3)
        gauss_2d = (gauss @ gauss.T).astype(np.float32)
        gauss_2d /= gauss_2d.max()

        inpainted_acc = np.zeros((H, W, 3), dtype=np.float32)
        inpainted_weight = np.zeros((H, W), dtype=np.float32)

        if H <= ps:
            ys = [0]
        else:
            ys = list(range(0, H - ps + 1, self.stride))
            if ys[-1] != H - ps: ys.append(H - ps)
            
        if W <= ps:
            xs = [0]
        else:
            xs = list(range(0, W - ps + 1, self.stride))
            if xs[-1] != W - ps: xs.append(W - ps)

        patches = []
        coords = []
        for y in ys:
            for x in xs:
                y1, y2 = y, min(H, y + ps)
                x1, x2 = x, min(W, x + ps)
                p_img = crop_img[y1:y2, x1:x2]
                p_mask = crop_mask[y1:y2, x1:x2]

                if p_mask.sum() == 0:
                    continue

                patch_h, patch_w = p_img.shape[:2]
                pad_h = ps - patch_h
                pad_w = ps - patch_w
                if pad_h or pad_w:
                    p_img = np.pad(
                        p_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
                    )
                    p_mask = np.pad(
                        p_mask, ((0, pad_h), (0, pad_w)), mode="reflect"
                    )

                patches.append((p_img, p_mask))
                coords.append((y1, y2, x1, x2))

        for i in range(0, len(patches), self.batch_size):
            batch_imgs = np.stack([p[0] for p in patches[i:i+self.batch_size]])
            batch_masks = np.stack([p[1] for p in patches[i:i+self.batch_size]])
            
            outs = self._infer_batch(batch_imgs, batch_masks)
            
            for j, out_patch in enumerate(outs):
                y1, y2, x1, x2 = coords[i+j]
                ph, pw = y2 - y1, x2 - x1
                out_patch = out_patch[:ph, :pw]
                g = gauss_2d[:ph, :pw]

                inpainted_acc[y1:y2, x1:x2] += out_patch * g[:, :, None]
                inpainted_weight[y1:y2, x1:x2] += g

        inpainted_weight_safe = inpainted_weight.copy()
        inpainted_weight_safe[inpainted_weight_safe == 0] = 1.0
        result = (inpainted_acc / inpainted_weight_safe[:, :, None]).clip(0, 255).astype(np.uint8)
        return result

    def _border_ring(self, mask: np.ndarray):
        """Return a boolean mask of the ring of ORIGINAL pixels just outside the
        text mask — the local background that the fill should match."""
        border = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=2)
        return (border > 0) & (mask == 0)

    def _flat_fill(self, mask: np.ndarray, original: np.ndarray, border: np.ndarray):
        """If the surrounding background is essentially a single flat colour
        (the common case for manga speech bubbles / solid panels), fill the
        masked region with that exact colour.

        Returns the filled RGB image, or None if the background is textured and
        needs real inpainting. Filling directly is both faster (no model call)
        and eliminates the washed-out lightening LaMa introduces on flat areas.
        """
        if border.sum() < 20 or mask.sum() == 0:
            return None
        ref = original[border]  # (N, 3) RGB
        # Per-channel spread of the surrounding background.
        std = ref.std(axis=0)
        if std.max() > 6.0:
            return None  # textured background — let LaMa handle it
        fill_color = np.median(ref, axis=0)
        out = original.copy()
        out[mask > 0] = fill_color.astype(np.uint8)
        return out

    def _color_correct(self, inpainted: np.ndarray, mask: np.ndarray,
                       original: np.ndarray, border: np.ndarray) -> np.ndarray:
        """Pin the inpainted region's colour to the surrounding background.

        Uses a robust MEDIAN offset per LAB channel instead of mean/std
        scaling. Std-scaling was what let the fill drift brighter than the
        background; a pure median offset aligns the central tone exactly and
        cannot systematically lighten the region.
        """
        if border.sum() < 20 or mask.sum() == 0:
            return inpainted

        inp_lab = cv2.cvtColor(inpainted, cv2.COLOR_RGB2LAB).astype(np.float32)
        orig_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
        mask_bool = mask > 0

        for c in range(3):
            ref = orig_lab[border, c]
            inp = inp_lab[mask_bool, c]
            if len(ref) < 10 or len(inp) < 10:
                continue
            ref_med = np.median(ref)
            inp_med = np.median(inp)
            ref_std = ref.std()
            inp_std = max(inp.std(), 1.0)

            # Central-tone offset (never brightens on average).
            shifted = inp_lab[:, :, c] - inp_med + ref_med
            # Gently pull texture variance toward the background, but only ever
            # REDUCE spread (scale <= 1) so we can't amplify into a bright halo.
            scale = min(1.0, ref_std / inp_std)
            shifted = (shifted - ref_med) * scale + ref_med
            inp_lab[:, :, c] = np.where(mask_bool, shifted, inp_lab[:, :, c])

        out = np.clip(inp_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)

    def __call__(self, pil_img: Image.Image, pil_mask: Image.Image) -> Image.Image:
        w, h = pil_img.size
        img = np.array(pil_img.convert("RGB"))
        mask = np.array(pil_mask.convert("L"))
        
        # Ensure mask is strictly 0 or 255
        mask = (mask > 127).astype(np.uint8) * 255

        if mask.sum() == 0:
            return pil_img

        # 1. Merge close text regions to group adjacent boxes into single crops
        # This prevents hard seams between boxes that are close together
        merge_kernel = np.ones((51, 51), np.uint8)
        merged_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, merge_kernel)
        
        # 2. Find connected components (clusters of text)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged_mask, connectivity=8)
        
        out_img = img.copy()

        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]
            if cw < 2 or ch < 2:
                continue
                
            # Add generous padding to the crop for background context
            pad = 64
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad)
            y2 = min(h, y + ch + pad)
            
            crop_img = img[y1:y2, x1:x2].copy()
            crop_mask = mask[y1:y2, x1:x2].copy()

            border = self._border_ring(crop_mask)

            # 3a. Fast path: if the local background is a flat colour (most
            #     manga bubbles/panels), fill it directly. No model call → much
            #     faster, and an exact colour match → no washed-out lightening.
            flat = self._flat_fill(crop_mask, crop_img, border)
            if flat is not None:
                corrected_crop = flat
            else:
                # 3b. Textured background: run LaMa, then pin its colour to the
                #     surrounding background with a median offset (no brightening).
                inpainted_crop = self._infer_crop(crop_img, crop_mask)
                corrected_crop = self._color_correct(inpainted_crop, crop_mask, crop_img, border)

            # 5. Blend the crop back into the original image
            # Feather the mask edges slightly for a seamless transition
            blend_mask = cv2.GaussianBlur(crop_mask, (7, 7), 2.0)
            blend_mask = np.maximum(blend_mask, crop_mask.astype(np.float32))
            blend_mask = (blend_mask / 255.0).clip(0, 1)
            
            # Alpha blend only the masked region
            out_img[y1:y2, x1:x2] = (
                corrected_crop.astype(np.float32) * blend_mask[:, :, None] +
                out_img[y1:y2, x1:x2].astype(np.float32) * (1.0 - blend_mask[:, :, None])
            ).clip(0, 255).astype(np.uint8)

        return Image.fromarray(out_img)


def load_lama_high():
    """Load the high-quality LaMa model (anime-manga-big-lama.pt)."""
    global _simple_lama_high_model
    if _simple_lama_high_model is None:
        logging.info(f"[Lama High] Initializing high-quality model wrapper...")
        _simple_lama_high_model = HighQualityLama()
        logging.info(f"[Lama High] Successfully loaded high-quality model.")
    return _simple_lama_high_model


def load_lama():
    """Load the appropriate LaMa model based on the current inpaint mode.
    Returns None for 'none' mode — no model is loaded."""
    with _inpaint_mode_lock:
        mode = _inpaint_mode

    if mode == "none":
        logging.info("[Inpaint] Mode is 'none' — no LaMa model loaded.")
        return None
    if mode == "high":
        return load_lama_high()
    return load_lama_low()


def lama_inpaint(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Thread-safe SimpleLama inpainting."""
    with _inpaint_lock:
        sl = load_lama()
        if sl is None:
            raise RuntimeError("SimpleLama unavailable")
        pil_img  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask).convert("L")
        out_pil  = sl(pil_img, pil_mask)
        return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

def cv2_inpaint_fallback(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Thread-safe cv2 inpainting."""
    with _inpaint_lock:
        return cv2.inpaint(img_bgr, mask, INPAINT_RADIUS_CV2, cv2.INPAINT_TELEA)

async def inpaint_image_async(img_bgr: np.ndarray, mask: np.ndarray, use_lama: bool = True) -> np.ndarray:
    """Run inpainting in a thread pool so it doesn't block the event loop.
    In 'none' mode, returns the original image unchanged."""
    with _inpaint_mode_lock:
        mode = _inpaint_mode

    if mode == "none":
        logging.info("[Inpaint] Mode is 'none' — skipping inpainting entirely.")
        return img_bgr

    should_use_lama = use_lama and (mode == "high" or SimpleLama is not None)

    if should_use_lama:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(_inpaint_executor, lama_inpaint, img_bgr, mask)
            logging.info(f"[Inpaint] LaMa inpainting complete (mode={mode}).")
            return result
        except Exception as e:
            logging.warning(f"[Inpaint] LaMa failed ({e}), falling back to cv2.inpaint")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_inpaint_executor, cv2_inpaint_fallback, img_bgr, mask)
    logging.info("[Inpaint] cv2.inpaint fallback complete.")
    return result

def build_inpaint_mask(img_shape: Tuple[int, int, int],
                       bboxes: List[Tuple[int, int, int, int]],
                       padding: int = 2,
                       dilate_kernel: int = 3,
                       angles: Optional[List[float]] = None) -> np.ndarray:
    """Build a strict binary mask tailored tightly to the text bounding boxes.

    When `angles` is supplied (Google Lens tilt), a tilted box is filled as a
    rotated quad so the mask follows the slanted text instead of covering the
    larger axis-aligned area around it.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(w, x2 + padding)
        py2 = min(h, y2 + padding)

        angle = 0.0
        if angles and i < len(angles):
            angle = float(angles[i] or 0.0)

        if angle:
            pts = _rotated_box_points((px1, py1, px2, py2), angle)
            cv2.fillPoly(mask, [np.int32([pts])], 255)
        else:
            mask[py1:py2, px1:px2] = 255

    if dilate_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
    return mask

# ===========================================================================
# Text color detection (per box + global batch voting for consistency)
# ===========================================================================
def detect_text_and_bg_colors(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
                              ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Detect text (ink) and background (outline) colors within a bbox."""
    x1, y1, x2, y2 = bbox
    h, w = img_bgr.shape[:2]

    # --- Expand bbox slightly to capture surrounding background context ---
    pad_x = max(3, (x2 - x1) // 6)
    pad_y = max(3, (y2 - y1) // 6)
    ex_x1 = max(0, x1 - pad_x)
    ex_y1 = max(0, y1 - pad_y)
    ex_x2 = min(w, x2 + pad_x)
    ex_y2 = min(h, y2 + pad_y)

    region = img_bgr[ex_y1:ex_y2, ex_x1:ex_x2]
    if region.size == 0:
        return (255, 255, 255), (0, 0, 0)

    rh, rw = region.shape[:2]

    # --- Resize for consistent processing speed ---
    max_dim = 180
    if rh > max_dim or rw > max_dim:
        scale = max_dim / max(rh, rw)
        new_w = max(8, int(rw * scale))
        new_h = max(8, int(rh * scale))
        region = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # --- Convert to LAB for perceptually uniform color distance ---
    region_lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    pixels_lab = np.ascontiguousarray(region_lab.reshape(-1, 3).astype(np.float32))
    pixels_bgr = region.reshape(-1, 3).astype(np.float32)

    n_pixels = int(pixels_bgr.shape[0])
    if n_pixels < 8:
        return (255, 255, 255), (0, 0, 0)

    # --- K-means clustering in LAB space (k=3: bg, text, transition) ---
    K = 3
    try:
        _, labels, centers_lab = cv2.kmeans(
            pixels_lab, K, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0),
            10, cv2.KMEANS_PP_CENTERS
        )
    except cv2.error:
        # Degenerate region — fall back to simple luminance split
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        mean_lum = float(gray.mean())
        if mean_lum > 127:
            return (0, 0, 0), (255, 255, 255)
        return (255, 255, 255), (0, 0, 0)

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=K)
    sorted_idx = np.argsort(-counts)

    # --- Identify background: the largest cluster ---
    bg_idx = int(sorted_idx[0])
    bg_lab = centers_lab[bg_idx]
    bg_mask = (labels == bg_idx)
    bg_bgr = np.median(pixels_bgr[bg_mask], axis=0)

    # --- Identify text: highest perceptual distance from bg, enough pixels ---
    min_text_count = max(5, int(n_pixels * 0.04))
    best_text_idx = None
    best_text_dist = -1.0
    for i in range(K):
        if i == bg_idx:
            continue
        if counts[i] < min_text_count:
            continue
        d = float(np.linalg.norm(centers_lab[i] - bg_lab))
        if d > best_text_dist:
            best_text_dist = d
            best_text_idx = i

    if best_text_idx is not None:
        text_mask = (labels == best_text_idx)
        text_bgr = np.median(pixels_bgr[text_mask], axis=0)
    else:
        # --- Fallback: Otsu threshold on luminance ---
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if rh >= 2 and rw >= 2:
            border = np.concatenate([
                gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
            ])
        else:
            border = gray.flatten()
        border_mean = float(border.mean()) if border.size else float(gray.mean())
        if border_mean > 127:
            text_sel = (otsu.flatten() == 0)
        else:
            text_sel = (otsu.flatten() != 0)
        if int(text_sel.sum()) > 0:
            text_bgr = np.median(pixels_bgr[text_sel], axis=0)
        else:
            bg_lum = float(bg_bgr.mean())
            text_bgr = (np.array([0, 0, 0], dtype=np.float32)
                        if bg_lum > 127
                        else np.array([255, 255, 255], dtype=np.float32))

    # --- Gentle snap: ONLY when extremely close to extremes ---
    def gentle_snap(c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=np.float32)
        if np.all(c <= 20):
            return np.array([0, 0, 0], dtype=np.float32)
        if np.all(c >= 235):
            return np.array([255, 255, 255], dtype=np.float32)
        return c

    text_bgr = gentle_snap(text_bgr)
    bg_bgr = gentle_snap(bg_bgr)

    # --- Final contrast enforcement (last resort) ---
    contrast = float(np.linalg.norm(text_bgr - bg_bgr))
    if contrast < 60:
        bg_lum = float(bg_bgr.mean())
        if bg_lum > 127:
            text_bgr = np.array([0, 0, 0], dtype=np.float32)
        else:
            text_bgr = np.array([255, 255, 255], dtype=np.float32)

    # BGR -> RGB
    text_rgb = (int(text_bgr[2]), int(text_bgr[1]), int(text_bgr[0]))
    outline_rgb = (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0]))
    return text_rgb, outline_rgb


def measure_source_glyph_height(img_bgr: np.ndarray,
                                 bbox: Tuple[int, int, int, int]) -> Optional[float]:
    """Measure the height (px) of the ORIGINAL lettering inside bbox.

    Used by high mode to size the overlay to match how big the source text was
    actually drawn, instead of sizing purely off the bubble dimensions. Works by
    binarising the region (Otsu, auto-polarity from the border), keeping
    connected components that look like glyphs, and returning their median
    height. Returns None when it can't find a reliable glyph run so the caller
    falls back to bubble-proportional sizing.
    """
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = img_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
        if (x2 - x1) < 6 or (y2 - y1) < 6:
            return None

        region = img_bgr[y1:y2, x1:x2]
        if region.size == 0:
            return None
        box_h = y2 - y1

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Auto-polarity: glyphs are the pixels that DIFFER from the border,
        # which is almost always background (bubble fill / art behind text).
        rh, rw = gray.shape[:2]
        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        border_mean = float(border.mean()) if border.size else float(gray.mean())
        glyph_mask = (otsu == 0) if border_mean > 127 else (otsu == 255)
        glyph_u8 = glyph_mask.astype(np.uint8)

        num, _labels, stats, _cent = cv2.connectedComponentsWithStats(glyph_u8, connectivity=8)
        if num <= 1:
            return None

        heights = []
        for i in range(1, num):  # skip background label 0
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            # Discard noise (tiny specks) and full-region blobs (a component as
            # tall as the whole box is a frame/inpaint artifact, not a glyph).
            if ch < max(4, box_h * 0.12):
                continue
            if ch > box_h * 0.98:
                continue
            if area < 6:
                continue
            if cw > 0 and ch / cw > 12:  # thin vertical rule, not a glyph
                continue
            heights.append(float(ch))

        if len(heights) < 1:
            return None
        return float(np.median(heights))
    except Exception as e:
        logging.warning(f"[GlyphSize] measure failed for {bbox}: {e}")
        return None


def detect_text_colors_batch(img_bgr: np.ndarray,
                             bboxes: List[Tuple[int, int, int, int]]
                             ) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Detect text/bg colors for a list of bboxes WITH global consistency."""
    if not bboxes:
        return []

    # --- Pass 1: independent detection ---
    results: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    for bbox in bboxes:
        try:
            results.append(detect_text_and_bg_colors(img_bgr, bbox))
        except Exception as e:
            logging.warning(f"[Color] detect_text_and_bg_colors failed for {bbox}: {e}")
            results.append(((255, 255, 255), (0, 0, 0)))

    # --- Pass 2: global polarity voting ---
    light_votes = 0
    dark_votes = 0
    for text_rgb, bg_rgb in results:
        text_lum = (text_rgb[0] + text_rgb[1] + text_rgb[2]) / 3.0
        bg_lum = (bg_rgb[0] + bg_rgb[1] + bg_rgb[2]) / 3.0
        if text_lum > bg_lum:
            light_votes += 1
        else:
            dark_votes += 1

    total_votes = light_votes + dark_votes
    if total_votes == 0:
        return results

    force_light = light_votes >= 2 * dark_votes and light_votes >= 2
    force_dark = dark_votes >= 2 * light_votes and dark_votes >= 2

    if not force_light and not force_dark:
        return results

    logging.info(f"[Color] Global vote: light={light_votes} dark={dark_votes} "
                 f"-> force_light={force_light} force_dark={force_dark}")

    final_results = []
    for (text_rgb, bg_rgb) in results:
        if force_light:
            text_lum = (text_rgb[0] + text_rgb[1] + text_rgb[2]) / 3.0
            bg_lum = (bg_rgb[0] + bg_rgb[1] + bg_rgb[2]) / 3.0
            if text_lum <= bg_lum:
                outline = bg_rgb if bg_lum < 90 else (0, 0, 0)
                final_results.append(((255, 255, 255), outline))
            else:
                final_results.append((text_rgb, bg_rgb))
        else:  # force_dark
            text_lum = (text_rgb[0] + text_rgb[1] + text_rgb[2]) / 3.0
            bg_lum = (bg_rgb[0] + bg_rgb[1] + bg_rgb[2]) / 3.0
            if text_lum >= bg_lum:
                outline = bg_rgb if bg_lum > 165 else (255, 255, 255)
                final_results.append(((0, 0, 0), outline))
            else:
                final_results.append((text_rgb, bg_rgb))

    return final_results

# ===========================================================================
# Text wrapping & auto-fit
# ===========================================================================
@functools.lru_cache(maxsize=256)
def _get_font_cached(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()

def clear_font_cache() -> None:
    _get_font_cached.cache_clear()

def get_font(font_path, size: int) -> ImageFont.FreeTypeFont:
    return _get_font_cached(str(font_path), size)

def get_current_font(size: int) -> ImageFont.FreeTypeFont:
    with _font_config_lock:
        font_path = _current_font_path
    return get_font(font_path, size)

def _add_cutoff_marker(draw, line, font, max_width):
    marker = "-"
    if line.endswith(marker) and draw.textlength(line, font=font) <= max_width:
        return line
    candidate = line.rstrip()
    while candidate and draw.textlength(candidate + marker, font=font) > max_width:
        candidate = candidate[:-1].rstrip()
    if candidate:
        return candidate + marker
    return marker if draw.textlength(marker, font=font) <= max_width else ""


def _break_long_word(draw, word, font, max_width):
    """Split an oversized word into width-safe, hyphen-terminated pieces."""
    pieces = []
    remaining = word
    while remaining:
        if draw.textlength(remaining, font=font) <= max_width:
            pieces.append(remaining)
            break

        chunk = ""
        for ch in remaining:
            candidate = chunk + ch
            if draw.textlength(candidate + "-", font=font) > max_width:
                break
            chunk = candidate

        if not chunk:
            pieces.append(_add_cutoff_marker(draw, "", font, max_width))
            break

        pieces.append(chunk + "-")
        remaining = remaining[len(chunk):]

    return pieces or [word]

def wrap_text(draw, text, font, max_width, allow_break=False, is_vertical=False):
    """Wrap text by words, optionally hyphenating oversized words."""
    if is_vertical:
        return [text] if text else [""]

    words = text.split()
    if not words:
        return [""]

    lines = []
    cur = ""

    for word in words:
        test = (cur + " " + word) if cur else word
        if draw.textlength(test, font=font) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
                cur = ""
            # The word itself is wider than max_width.
            if allow_break and draw.textlength(word, font=font) > max_width:
                pieces = _break_long_word(draw, word, font, max_width)
                # All but the last piece are complete lines.
                lines.extend(pieces[:-1])
                cur = pieces[-1]
            else:
                cur = word

    if cur:
        lines.append(cur)

    return lines

def _measure_block(draw, lines, font):
    try:
        ascent, descent = font.getmetrics()
        line_h = int(ascent + descent)
    except Exception:
        line_h = int(font.size * 1.2)
    line_h = max(line_h, int(font.size * 1.1))
    heights = [line_h] * len(lines)
    total_h = line_h * len(lines)
    max_w = 0.0
    for ln in lines:
        w = draw.textlength(ln, font=font)
        if w > max_w: max_w = w
    return heights, total_h, max_w

def fit_font_and_wrap(draw, text, box_w, box_h,
                      font_path=None,
                      max_size=96, min_size=17, is_vertical=False,
                      inner_padding_ratio=0.0):
    """Fit text at the largest size that fits the selected portion of a region.

    With the default zero padding, the complete OCR bbox is available. The
    returned font size never drops below min_size.

    Returns:
        (font_size, lines, heights, inner_w, inner_h)
        inner_w, inner_h = dimensions of the inner box used for fitting,
        so the caller can position text correctly within the outer bbox.
    """
    if font_path is None:
        with _font_config_lock:
            font_path = str(_current_font_path)

    # ── Create inner box ──
    pad_x = max(0, int(box_w * inner_padding_ratio))
    pad_y = max(0, int(box_h * inner_padding_ratio))
    inner_w = max(8, box_w - 2 * pad_x)
    inner_h = max(8, box_h - 2 * pad_y)

    if not text.strip():
        return min_size, [""], [0], inner_w, inner_h

    if not hasattr(fit_font_and_wrap, '_cache'):
        fit_font_and_wrap._cache = {}
    cache = fit_font_and_wrap._cache

    if is_vertical:
        lo, hi = min_size, max_size
        best_size, best_cols, best_col_widths = None, None, None
        clean_v_text = text.replace(" ", "").replace("\n", "")
        while lo <= hi:
            mid = (lo + hi) // 2
            key = (font_path, mid)
            if key not in cache:
                try: cache[key] = ImageFont.truetype(font_path, mid)
                except Exception: cache[key] = ImageFont.load_default()
            font = cache[key]
            cols = []
            cur_col = ""
            cur_h = 0
            bb = draw.textbbox((0, 0), "字", font=font)
            char_h = (bb[3] - bb[1]) * 1.2
            if char_h == 0:
                char_h = mid
            for ch in clean_v_text:
                if cur_h + char_h > inner_h and cur_col:
                    cols.append(cur_col)
                    cur_col = ch
                    cur_h = char_h
                else:
                    cur_col += ch
                    cur_h += char_h
            if cur_col:
                cols.append(cur_col)
            if not cols:
                cols = [clean_v_text]
            max_char_w = max(draw.textlength(ch, font=font) for ch in clean_v_text) if clean_v_text else mid
            col_w = max(max_char_w, mid * 0.8)
            total_w = len(cols) * col_w
            if total_w <= inner_w:
                best_size = mid
                best_cols = cols
                best_col_widths = [col_w] * len(cols)
                lo = mid + 1
            else:
                hi = mid - 1
        if best_cols is None:
            key = (font_path, min_size)
            if key not in cache:
                try: cache[key] = ImageFont.truetype(font_path, min_size)
                except Exception: cache[key] = ImageFont.load_default()
            font = cache[key]
            bb = draw.textbbox((0, 0), "字", font=font)
            char_h = (bb[3] - bb[1]) * 1.2
            if char_h == 0:
                char_h = min_size
            cols = []
            cur_col = ""
            cur_h = 0
            for ch in clean_v_text:
                if cur_h + char_h > inner_h and cur_col:
                    cols.append(cur_col)
                    cur_col = ch
                    cur_h = char_h
                else:
                    cur_col += ch
                    cur_h += char_h
            if cur_col:
                cols.append(cur_col)
            best_cols = cols if cols else [text]
            max_char_w = max(draw.textlength(ch, font=font) for ch in clean_v_text) if clean_v_text else min_size
            best_col_widths = [max_char_w] * len(best_cols)
            best_size = min_size
        return best_size, best_cols, best_col_widths, inner_w, inner_h

    # ── Horizontal text: binary search for largest font that fits inner box ──
    lo, hi = min_size, max_size
    best_size = None
    best_lines = None
    best_heights = None

    while lo <= hi:
        mid = (lo + hi) // 2
        key = (font_path, mid)
        if key not in cache:
            try:
                cache[key] = ImageFont.truetype(font_path, mid)
            except Exception:
                cache[key] = ImageFont.load_default()
        font = cache[key]

        lines = wrap_text(draw, text, font, inner_w, allow_break=False, is_vertical=False)
        heights, total_h, max_w = _measure_block(draw, lines, font)

        if max_w <= inner_w and total_h <= inner_h:
            best_size, best_lines, best_heights = mid, lines, heights
            lo = mid + 1
        else:
            hi = mid - 1

    if best_lines is None:
        # Preserve the readable floor while containing dense translations within
        # the region. Oversized words are hyphenated, and text beyond the
        # available line count is marked as cut off on the final visible line.
        key = (font_path, min_size)
        if key not in cache:
            try:
                cache[key] = ImageFont.truetype(font_path, min_size)
            except Exception:
                cache[key] = ImageFont.load_default()
        font = cache[key]
        wrapped_lines = wrap_text(
            draw, text, font, inner_w, allow_break=True, is_vertical=False
        )
        wrapped_heights, _, _ = _measure_block(draw, wrapped_lines, font)
        line_h = wrapped_heights[0] if wrapped_heights else max(1, min_size)
        visible_line_count = max(1, int(inner_h // line_h))
        was_cut_off = len(wrapped_lines) > visible_line_count
        best_lines = wrapped_lines[:visible_line_count]
        if was_cut_off and best_lines:
            best_lines[-1] = _add_cutoff_marker(
                draw, best_lines[-1], font, inner_w
            )
        best_heights, _, _ = _measure_block(draw, best_lines, font)
        best_size = min_size

    return best_size, best_lines, best_heights, inner_w, inner_h

# ===========================================================================
# Text drawing with configurable stroke
# ===========================================================================
def draw_text_with_config(draw: ImageDraw.ImageDraw,
                          position: Tuple[float, float],
                          text: str,
                          font: ImageFont.FreeTypeFont,
                          fill: Tuple[int, int, int],
                          stroke_fill: Optional[Tuple[int, int, int]] = None,
                          anchor: Optional[str] = None,
                          embolden: int = 0,
                          glow_radius: float = 0.0,
                          target_image: Optional[Image.Image] = None):
    """
    Draw text with optional stroke, synthetic bold, and glow.
    """
    with _font_config_lock:
        base_stroke = _current_stroke_width

    total_stroke = base_stroke + embolden
    
    # Glow pass: draw text onto a temporary layer, blur it, paste behind
    if glow_radius > 0 and target_image is not None:
        try:
            bbox = draw.textbbox(position, text, font=font, anchor=anchor)
            pad = int(glow_radius * 3)
            x1, y1, x2, y2 = bbox
            layer_w = max(1, int(x2 - x1) + pad * 2)
            layer_h = max(1, int(y2 - y1) + pad * 2)
            glow_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_layer)
            glow_draw.text((pad, pad), text, font=font, fill=stroke_fill or fill, anchor=anchor)
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=glow_radius))
            target_image.paste(glow_layer, (int(x1) - pad, int(y1) - pad), glow_layer)
        except Exception as e:
            logging.warning(f"[Render] Glow rendering failed (drawing without glow): {e}")

    # Crisp text with optional stroke
    try:
        if total_stroke > 0 and stroke_fill is not None:
            draw.text(position, text, font=font, fill=fill,
                      stroke_width=int(total_stroke), stroke_fill=stroke_fill, anchor=anchor)
        else:
            draw.text(position, text, font=font, fill=fill, anchor=anchor)
    except Exception as e:
        logging.warning(f"[Render] Text drawing failed: {e}")

# ===========================================================================
# Font Management Endpoints (Set, Get, GetFonts)
# ===========================================================================
def list_available_fonts() -> List[Dict[str, Any]]:
    fonts = []
    if FONT_DIR.exists():
        for ext in ('*.ttf', '*.otf', '*.ttc'):
            for f in sorted(FONT_DIR.glob(ext)):
                try:
                    size_kb = f.stat().st_size / 1024
                except OSError:
                    size_kb = 0
                fonts.append({
                    "name": f.stem,
                    "filename": f.name,
                    "path": str(f),
                    "size_kb": round(size_kb, 1)
                })
    return fonts

class SetFontRequest(BaseModel):
    font_path: Optional[str] = None
    font_url: Optional[str] = None
    font_name: Optional[str] = None
    stroke_width: int = 0

@app.post("/SetFont")
async def set_font(req: SetFontRequest):
    global _current_font_path, _current_stroke_width
    with _font_config_lock:
        provided_params = sum(1 for p in [req.font_path, req.font_url, req.font_name] if p)
        if provided_params > 1:
            raise HTTPException(400, "Provide either font_path, font_url, or font_name, not multiple")
        
        if req.font_url:
            filename = pathlib.Path(req.font_url).name
            if not filename.lower().endswith(('.ttf', '.otf', '.ttc')):
                filename += '.ttf'
            new_path = FONT_DIR / filename
            try:
                logging.info(f"[Font] Downloading from {req.font_url} -> {new_path}")
                urllib.request.urlretrieve(req.font_url, new_path)
                _current_font_path = new_path
                clear_font_cache()
                logging.info(f"[Font] Downloaded and set: {new_path}")
            except Exception as e:
                raise HTTPException(500, f"Failed to download font: {e}")
        elif req.font_path:
            p = pathlib.Path(req.font_path).resolve()
            if not p.exists():
                raise HTTPException(400, f"Font file not found: {req.font_path}")
            if not p.suffix.lower() in ('.ttf', '.otf', '.ttc'):
                raise HTTPException(400, f"Unsupported font format: {p.suffix}")
            _current_font_path = p
            clear_font_cache()
            logging.info(f"[Font] Set to: {_current_font_path}")
        elif req.font_name:
            req_font_name = req.font_name.strip().lower()
            available_fonts = list_available_fonts()
            
            matched_font = None
            for f in available_fonts:
                if f["filename"].lower() == req_font_name or f["name"].lower() == req_font_name:
                    matched_font = f
                    break
            
            if not matched_font:
                for f in available_fonts:
                    if f["filename"].lower().startswith(req_font_name):
                        matched_font = f
                        break

            if not matched_font:
                raise HTTPException(404, f"Font '{req.font_name}' not found in fonts folder. Available: {[f['name'] for f in available_fonts]}")
            
            _current_font_path = pathlib.Path(matched_font["path"])
            clear_font_cache()
            logging.info(f"[Font] Set to by name: {_current_font_path}")
            
        _current_stroke_width = max(0, min(20, req.stroke_width))
        logging.info(f"[Font] Stroke width set to: {_current_stroke_width}")
    _save_settings()
    return {"status": "ok", "font_path": str(_current_font_path), "stroke_width": _current_stroke_width}

@app.get("/GetFont")
async def get_font_config():
    with _font_config_lock:
        return {"font_path": str(_current_font_path), "stroke_width": _current_stroke_width}

@app.get("/GetFonts")
async def get_fonts():
    fonts = list_available_fonts()
    return {"fonts": fonts, "count": len(fonts)}

@app.get("/v1/font")
async def get_font_file():
    """Serve the currently active font file bytes so clients (e.g. the browser
    extension) can render accurate font previews without needing the file
    installed locally."""
    with _font_config_lock:
        path = pathlib.Path(_current_font_path)

    if not path.exists():
        raise HTTPException(404, "Font file not found on server")

    suffix = path.suffix.lower()
    media_type = {
        ".ttf": "font/ttf",
        ".otf": "font/otf",
        ".ttc": "font/collection",
    }.get(suffix, "application/octet-stream")

    return FileResponse(str(path), media_type=media_type, filename=path.name)

@app.get("/v1/font/{filename}")
async def get_font_file_by_name(filename: str):
    """Serve any font file (by filename) from the fonts folder so clients can
    render an accurate preview of *each* available font in its own typeface,
    not just the currently-active one."""
    # Guard against path traversal — only allow bare filenames inside FONT_DIR.
    safe_name = pathlib.Path(filename).name
    path = (FONT_DIR / safe_name).resolve()
    try:
        path.relative_to(FONT_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid font filename")
    if not path.exists() or path.suffix.lower() not in ('.ttf', '.otf', '.ttc'):
        raise HTTPException(404, f"Font '{safe_name}' not found")

    media_type = {
        ".ttf": "font/ttf",
        ".otf": "font/otf",
        ".ttc": "font/collection",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(str(path), media_type=media_type, filename=path.name)


# ===========================================================================
# Languages endpoint — full list + which are featured (shown by default)
# ===========================================================================
@app.get("/languages")
async def get_languages():
    """Expose the full supported language list plus which codes are 'featured'
    (shown by default in the extension, the rest behind a 'More…' toggle)."""
    langs = [
        {"code": code, "name": meta["name"],
         "script": meta["script"], "featured": meta["featured"]}
        for code, meta in LANGUAGES.items()
    ]
    featured = [l["code"] for l in langs if l["featured"]]
    return {"languages": langs, "featured": featured, "count": len(langs)}


# ===========================================================================
# SetInpaintMode Endpoint (Low/High inpainting model switching)
# ===========================================================================
class SetInpaintModeRequest(BaseModel):
    mode: str  # "low" or "high"

@app.post("/SetInpaintMode")
async def set_inpaint_mode(req: SetInpaintModeRequest):
    global _inpaint_mode
    mode = req.mode.lower().strip()
    if mode not in ("low", "high", "none"):
        raise HTTPException(400, "mode must be 'low', 'high', or 'none'")

    if mode == "high":
        try:
            ensure_lama_large()
        except Exception as e:
            raise HTTPException(500, f"Failed to download high-quality inpainting model: {e}")

    # If switching away from "none", clear cached lama models so they reload
    # on next use. If switching TO "none", no model load happens at all.
    with _inpaint_mode_lock:
        _inpaint_mode = mode

    _save_settings()
    logging.info(f"[Inpaint] Mode set to: {mode}")

    return {
        "status": "ok",
        "inpaint_mode": _inpaint_mode,
        "high_model_path": str(LAMA_LARGE_PATH),
        "high_model_downloaded": LAMA_LARGE_PATH.exists() if LAMA_LARGE_PATH.exists() else False,
        "high_model_size_mb": round(LAMA_LARGE_PATH.stat().st_size / (1024 * 1024), 1) if LAMA_LARGE_PATH.exists() else 0,
    }

    mode = req.mode.lower().strip()
    if mode not in ("low", "high"):
        raise HTTPException(400, "mode must be 'low' or 'high'")

    if mode == "high":
        try:
            ensure_lama_large()
        except Exception as e:
            raise HTTPException(500, f"Failed to download high-quality inpainting model: {e}")

    with _inpaint_mode_lock:
        _inpaint_mode = mode

    _save_settings()
    logging.info(f"[Inpaint] Mode set to: {mode}")

    return {
        "status": "ok",
        "inpaint_mode": _inpaint_mode,
        "high_model_path": str(LAMA_LARGE_PATH),
        "high_model_downloaded": LAMA_LARGE_PATH.exists() if LAMA_LARGE_PATH.exists() else False,
        "high_model_size_mb": round(LAMA_LARGE_PATH.stat().st_size / (1024 * 1024), 1) if LAMA_LARGE_PATH.exists() else 0,
    }

@app.get("/GetInpaintMode")
async def get_inpaint_mode():
    with _inpaint_mode_lock:
        mode = _inpaint_mode
    return {
        "inpaint_mode": mode,
        "high_model_path": str(LAMA_LARGE_PATH),
        "high_model_downloaded": LAMA_LARGE_PATH.exists(),
        "high_model_size_mb": round(LAMA_LARGE_PATH.stat().st_size / (1024 * 1024), 1) if LAMA_LARGE_PATH.exists() else 0,
        "low_model": "big-lama.pt (SimpleLama default)",
        "high_model": "anime-manga-big-lama.pt (df1412/anime-big-lama)",
        "none_model": "No inpainting — fills text regions with detected background color",
    }
    with _inpaint_mode_lock:
        mode = _inpaint_mode
    return {
        "inpaint_mode": mode,
        "high_model_path": str(LAMA_LARGE_PATH),
        "high_model_downloaded": LAMA_LARGE_PATH.exists(),
        "high_model_size_mb": round(LAMA_LARGE_PATH.stat().st_size / (1024 * 1024), 1) if LAMA_LARGE_PATH.exists() else 0,
        "low_model": "big-lama.pt (SimpleLama default)",
        "high_model": "anime-manga-big-lama.pt (df1412/anime-big-lama)",
    }

# ===========================================================================
# SetOcrMode Endpoint (OCR backend switching)
# ===========================================================================
class SetOcrModeRequest(BaseModel):
    mode: str

@app.post("/SetOcrMode")
async def set_ocr_mode(req: SetOcrModeRequest):
    global _ocr_mode
    mode = req.mode.lower().strip()
    if mode not in ("hayai", "glm", "lens", "openai_endpoint", "google_ai", "local_vision"):
        raise HTTPException(400, "mode must be 'hayai', 'glm', 'lens', 'openai_endpoint', 'google_ai', or 'local_vision'")

    if mode == "local_vision" and _selected_vision_model() is None:
        raise HTTPException(400, "Local GGUF OCR requires a selected GGUF model with a compatible mmproj/projector")

    if mode == "lens" and LensAPI is None:
        raise HTTPException(500, "chrome-lens-py not installed. Run: pip install chrome-lens-py")

    if mode == "lens":
        try:
            get_lens_api()
        except Exception as e:
            raise HTTPException(500, f"Failed to initialize Google Lens API: {e}")

    if mode == "openai_endpoint":
        with _openai_ocr_config_lock:
            endpoint = _openai_ocr_endpoint
            model = _openai_ocr_model
        if not endpoint or not model:
            raise HTTPException(400, "OpenAI Endpoint OCR needs an endpoint URL and model ID")

    if mode == "google_ai":
        with _google_ai_ocr_config_lock:
            if not _google_ai_ocr_api_key or not _google_ai_ocr_model:
                raise HTTPException(400, "Google AI Studio OCR needs an API key and model")

    with _ocr_mode_lock:
        _ocr_mode = mode
    _save_settings()

    logging.info(f"[OCR] Mode set to: {mode}")

    return {
        "status": "ok",
        "ocr_mode": _ocr_mode,
        "lens_available": LensAPI is not None,
    }

@app.get("/GetOcrMode")
async def get_ocr_mode():
    with _ocr_mode_lock:
        mode = _ocr_mode
    return {
        "ocr_mode": mode,
        "available_modes": ["hayai", "glm", "lens", "openai_endpoint", "google_ai", "local_vision"],
        "lens_available": LensAPI is not None,
        "local_vision_available": _selected_vision_model() is not None,
        "descriptions": {
            "hayai": "Hayai OCR (Japanese, local model + YOLO)",
            "glm": "GLM-OCR (Korean, transformers + YOLO)",
            "lens": "Google Lens OCR (all languages, cloud API)",
            "openai_endpoint": "OpenAI-compatible vision OCR with configurable endpoint/model and lettering analysis",
            "google_ai": "Google AI Studio Gemini OCR for every visible manga text region",
            "local_vision": "Selected local vision GGUF with its compatible mmproj/projector",
        }
    }


class SetOpenAiOcrConfigRequest(BaseModel):
    endpoint: str
    model: str
    api_key: Optional[str] = None


@app.post("/SetOpenAiOcrConfig")
async def set_openai_ocr_config(req: SetOpenAiOcrConfigRequest):
    global _openai_ocr_endpoint, _openai_ocr_model, _openai_ocr_api_key
    endpoint = _normalize_chat_completions_endpoint(req.endpoint)
    model = req.model.strip()
    if not endpoint.startswith(("http://", "https://")):
        raise HTTPException(400, "endpoint must be an http:// or https:// URL")
    if not model:
        raise HTTPException(400, "model is required")
    with _openai_ocr_config_lock:
        _openai_ocr_endpoint = endpoint
        _openai_ocr_model = model
        if req.api_key is not None:
            _openai_ocr_api_key = req.api_key.strip() or None
    _save_settings()
    return {
        "status": "ok",
        "endpoint": endpoint,
        "model": model,
        "api_key_set": _openai_ocr_api_key is not None,
    }


@app.get("/GetOpenAiOcrConfig")
async def get_openai_ocr_config():
    with _openai_ocr_config_lock:
        return {
            "endpoint": _openai_ocr_endpoint,
            "model": _openai_ocr_model,
            "api_key_set": _openai_ocr_api_key is not None,
        }


class SetGoogleAiOcrConfigRequest(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None
    rpm: Optional[int] = None


@app.post("/SetGoogleAiOcrConfig")
async def set_google_ai_ocr_config(req: SetGoogleAiOcrConfigRequest):
    global _google_ai_ocr_api_key, _google_ai_ocr_model, _google_ai_ocr_rpm
    with _google_ai_ocr_config_lock:
        if req.api_key is not None:
            _google_ai_ocr_api_key = req.api_key.strip() or None
        if req.model is not None:
            model = _normalize_gemini_model(req.model)
            if not model:
                raise HTTPException(400, "google AI model cannot be empty")
            _google_ai_ocr_model = model
        if req.rpm is not None:
            if not 1 <= req.rpm <= 15:
                raise HTTPException(400, "Google AI OCR RPM must be between 1 and 15")
            _google_ai_ocr_rpm = int(req.rpm)
        result = {
            "status": "ok",
            "model": _google_ai_ocr_model,
            "rpm": _google_ai_ocr_rpm,
            "api_key_set": _google_ai_ocr_api_key is not None,
        }
    _save_settings()
    return result


@app.get("/GetGoogleAiOcrConfig")
async def get_google_ai_ocr_config():
    with _google_ai_ocr_config_lock:
        return {
            "model": _google_ai_ocr_model,
            "rpm": _google_ai_ocr_rpm,
            "api_key_set": _google_ai_ocr_api_key is not None,
            "default_model": "gemini-2.5-flash-lite",
            "free_tier_rpm_default": 5,
        }

# ===========================================================================
# SetModelType Endpoint
# ===========================================================================
class SetModelTypeRequest(BaseModel):
    model_type: str
    api_key: Optional[str] = None
    model: Optional[str] = None

@app.post("/SetModelType")
async def set_model_type(req: SetModelTypeRequest):
    global _current_model_type, _openrouter_api_key, _openrouter_model
    model_type = req.model_type.lower().strip()
    if model_type not in ("local", "openrouter"):
        raise HTTPException(400, "model_type must be 'local' or 'openrouter'")
    with _model_type_lock:
        _current_model_type = model_type
        if model_type == "openrouter":
            if req.api_key:
                _openrouter_api_key = req.api_key
            if not _openrouter_api_key:
                raise HTTPException(400, "OpenRouter API key is required. Provide api_key parameter.")
            if req.model:
                _openrouter_model = req.model
            logging.info(f"[ModelType] Set to openrouter, model={_openrouter_model}")
        else:
            logging.info(f"[ModelType] Set to local (GGUF)")
    _save_settings()
    return {
        "status": "ok",
        "model_type": _current_model_type,
        "local_model": f"{_current_qwen_repo_id}/{_current_qwen_filename}" if _current_model_type == "local" else None,
        "openrouter_model": _openrouter_model if _current_model_type == "openrouter" else None,
        "openrouter_configured": _openrouter_api_key is not None
    }

@app.get("/GetModelType")
async def get_model_type():
    with _model_type_lock:
        return {
            "model_type": _current_model_type,
            "local_model": f"{_current_qwen_repo_id}/{_current_qwen_filename}" if _current_model_type == "local" else None,
            "openrouter_model": _openrouter_model if _current_model_type == "openrouter" else None,
            "openrouter_configured": _openrouter_api_key is not None
        }

# ===========================================================================
# Cloud Mode Endpoint — one switch that offloads everything to the cloud so the
# machine running the backend uses minimal resources: Google Lens OCR +
# OpenRouter translation + no local inpainting model (fill with bg colour).
# Reuses any OpenRouter model/key already configured so nothing must be re-sent.
# ===========================================================================
class SetCloudModeRequest(BaseModel):
    enabled: bool
    model: Optional[str] = None       # optional OpenRouter model override
    api_key: Optional[str] = None     # optional OpenRouter key override

@app.post("/SetCloudMode")
async def set_cloud_mode(req: SetCloudModeRequest):
    global _cloud_mode, _ocr_mode, _inpaint_mode
    global _current_model_type, _openrouter_api_key, _openrouter_model

    if not req.enabled:
        # Turning cloud mode OFF just clears the flag; the individual modes keep
        # whatever they are currently set to (the client restores its own state).
        with _cloud_mode_lock:
            _cloud_mode = False
        _save_settings()
        logging.info("[CloudMode] Disabled.")
        return {"status": "ok", "cloud_mode": False}

    # ── Enabling: force lens + openrouter + none ──
    if LensAPI is None:
        raise HTTPException(500, "chrome-lens-py not installed. Run: pip install chrome-lens-py")
    try:
        get_lens_api()
    except Exception as e:
        raise HTTPException(500, f"Failed to initialize Google Lens API: {e}")

    with _model_type_lock:
        if req.model:
            _openrouter_model = req.model.strip()
        if req.api_key:
            _openrouter_api_key = req.api_key
        if not _openrouter_api_key:
            raise HTTPException(
                400,
                "Cloud mode needs an OpenRouter API key. Set one once via the model "
                "box (or pass api_key here) — it is then reused automatically."
            )
        _current_model_type = "openrouter"
        active_model = _openrouter_model

    with _ocr_mode_lock:
        _ocr_mode = "lens"
    with _inpaint_mode_lock:
        _inpaint_mode = "none"
    with _cloud_mode_lock:
        _cloud_mode = True

    _save_settings()
    logging.info(f"[CloudMode] Enabled — lens OCR + OpenRouter ({active_model}) + no local inpainting.")
    return {
        "status": "ok",
        "cloud_mode": True,
        "ocr_mode": "lens",
        "inpaint_mode": "none",
        "model_type": "openrouter",
        "openrouter_model": active_model,
        "openrouter_configured": _openrouter_api_key is not None,
    }

@app.get("/GetCloudMode")
async def get_cloud_mode():
    with _cloud_mode_lock:
        enabled = _cloud_mode
    return {
        "cloud_mode": enabled,
        "lens_available": LensAPI is not None,
        "openrouter_configured": _openrouter_api_key is not None,
    }

# ===========================================================================
# SetOpenRouterModel Endpoint
# ===========================================================================
class SetOpenRouterModelRequest(BaseModel):
    model: str
    api_key: Optional[str] = None

@app.post("/SetOpenRouterModel")
async def set_openrouter_model(req: SetOpenRouterModelRequest):
    global _openrouter_model, _openrouter_api_key
    if not req.model or not req.model.strip():
        raise HTTPException(400, "model is required")
    with _model_type_lock:
        _openrouter_model = req.model.strip()
        if req.api_key:
            _openrouter_api_key = req.api_key
        logging.info(f"[OpenRouter] Model changed to: {_openrouter_model}")
    _save_settings()
    return {
        "status": "ok",
        "openrouter_model": _openrouter_model,
        "api_key_set": _openrouter_api_key is not None,
        "note": "This only takes effect when model_type is 'openrouter'. Use /SetModelType to switch."
    }

# ===========================================================================
# OpenRouter Free-Mode Endpoints
# ===========================================================================
class SetOpenRouterFreeModeRequest(BaseModel):
    enabled: bool

@app.post("/SetOpenRouterFreeMode")
async def set_openrouter_free_mode(req: SetOpenRouterFreeModeRequest):
    global _openrouter_free_mode
    with _openrouter_free_mode_lock:
        _openrouter_free_mode = bool(req.enabled)
        logging.info(f"[OpenRouter] Free mode (skip 429 retries) set to: {_openrouter_free_mode}")
    _save_settings()
    return {"status": "ok", "free_mode": _openrouter_free_mode}

@app.get("/GetOpenRouterFreeMode")
async def get_openrouter_free_mode():
    with _openrouter_free_mode_lock:
        return {"enabled": _openrouter_free_mode}

# ===========================================================================
# Context-Aware Mode Endpoints
# ===========================================================================
class SetContextAwareRequest(BaseModel):
    enabled: bool

@app.post("/SetContextAware")
async def set_context_aware(req: SetContextAwareRequest):
    global _context_aware_mode
    with _context_aware_lock:
        _context_aware_mode = bool(req.enabled)
        logging.info(f"[ContextAware] Mode set to: {_context_aware_mode}")
    _save_settings()
    return {"status": "ok", "context_aware": _context_aware_mode}

@app.get("/GetContextAware")
async def get_context_aware():
    with _context_aware_lock:
        return {"enabled": _context_aware_mode}

# ===========================================================================
# SetAllSettings / GetAllSettings — unified settings push
# ===========================================================================
# The extension calls this on Translate to (re)apply the full settings payload
# in one shot. This handles the cloud-mode-restart problem: if the backend was
# restarted (resetting all in-memory globals to defaults) while cloud mode was
# on in the extension, the next Translate click re-pushes the correct state
# before the translation request goes out.
#
# All fields are optional — only the non-null ones are applied. Each field uses
# the SAME internal logic (locks, validation, side effects like model download)
# as the individual /Set* endpoints so behavior is identical.
class SetAllSettingsRequest(BaseModel):
    cloud_mode: Optional[bool] = None
    ocr_mode: Optional[str] = None
    inpaint_mode: Optional[str] = None
    model_type: Optional[str] = None
    openrouter_model: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openai_ocr_endpoint: Optional[str] = None
    openai_ocr_model: Optional[str] = None
    openai_ocr_api_key: Optional[str] = None
    google_ai_ocr_api_key: Optional[str] = None
    google_ai_ocr_model: Optional[str] = None
    google_ai_ocr_rpm: Optional[int] = None
    font_filename: Optional[str] = None
    stroke_width: Optional[int] = None
    free_openrouter: Optional[bool] = None
    context_aware: Optional[bool] = None
    # Note: skip_sfx is per-job (passed to /v1/translate), not a global setting,
    # so it's not included here.

@app.post("/SetAllSettings")
async def set_all_settings(req: SetAllSettingsRequest):
    global _cloud_mode, _ocr_mode, _inpaint_mode
    global _current_model_type, _openrouter_api_key, _openrouter_model
    global _openai_ocr_endpoint, _openai_ocr_model, _openai_ocr_api_key
    global _google_ai_ocr_api_key, _google_ai_ocr_model, _google_ai_ocr_rpm
    global _current_font_path, _current_stroke_width, _openrouter_free_mode
    applied = {}

    # ── Cloud mode (when forcing on, applies the cloud-forced state) ──
    if req.cloud_mode is True:
        if LensAPI is None:
            raise HTTPException(500, "chrome-lens-py not installed. Run: pip install chrome-lens-py")
        try:
            get_lens_api()
        except Exception as e:
            raise HTTPException(500, f"Failed to initialize Google Lens API: {e}")
        with _model_type_lock:
            if req.openrouter_model:
                _openrouter_model = req.openrouter_model.strip()
            if req.openrouter_api_key:
                _openrouter_api_key = req.openrouter_api_key
            if not _openrouter_api_key:
                raise HTTPException(
                    400,
                    "Cloud mode needs an OpenRouter API key. Set one once via the model "
                    "box (or pass api_key here) — it is then reused automatically."
                )
            _current_model_type = "openrouter"
        with _ocr_mode_lock:
            _ocr_mode = "lens"
        with _inpaint_mode_lock:
            _inpaint_mode = "none"
        with _cloud_mode_lock:
            _cloud_mode = True
        applied["cloud_mode"] = True
        applied["ocr_mode"] = "lens"
        applied["inpaint_mode"] = "none"
        applied["model_type"] = "openrouter"
        applied["openrouter_model"] = _openrouter_model
    elif req.cloud_mode is False:
        with _cloud_mode_lock:
            _cloud_mode = False
        applied["cloud_mode"] = False

    # ── OpenAI-compatible OCR configuration ──
    if (req.openai_ocr_endpoint is not None or req.openai_ocr_model is not None
            or req.openai_ocr_api_key is not None):
        with _openai_ocr_config_lock:
            if req.openai_ocr_endpoint is not None:
                endpoint = _normalize_chat_completions_endpoint(req.openai_ocr_endpoint)
                if not endpoint.startswith(("http://", "https://")):
                    raise HTTPException(400, "openai_ocr_endpoint must be an http:// or https:// URL")
                _openai_ocr_endpoint = endpoint
            if req.openai_ocr_model is not None:
                model = req.openai_ocr_model.strip()
                if not model:
                    raise HTTPException(400, "openai_ocr_model cannot be empty")
                _openai_ocr_model = model
            if req.openai_ocr_api_key is not None:
                _openai_ocr_api_key = req.openai_ocr_api_key.strip() or None
            applied["openai_ocr_endpoint"] = _openai_ocr_endpoint
            applied["openai_ocr_model"] = _openai_ocr_model
            applied["openai_ocr_api_key_set"] = _openai_ocr_api_key is not None

    # ── Google AI Studio OCR configuration ──
    if (req.google_ai_ocr_api_key is not None or req.google_ai_ocr_model is not None
            or req.google_ai_ocr_rpm is not None):
        with _google_ai_ocr_config_lock:
            if req.google_ai_ocr_api_key is not None:
                _google_ai_ocr_api_key = req.google_ai_ocr_api_key.strip() or None
            if req.google_ai_ocr_model is not None:
                model = _normalize_gemini_model(req.google_ai_ocr_model)
                if not model:
                    raise HTTPException(400, "google_ai_ocr_model cannot be empty")
                _google_ai_ocr_model = model
            if req.google_ai_ocr_rpm is not None:
                if not 1 <= req.google_ai_ocr_rpm <= 15:
                    raise HTTPException(400, "google_ai_ocr_rpm must be between 1 and 15")
                _google_ai_ocr_rpm = int(req.google_ai_ocr_rpm)
            applied["google_ai_ocr_model"] = _google_ai_ocr_model
            applied["google_ai_ocr_rpm"] = _google_ai_ocr_rpm
            applied["google_ai_ocr_api_key_set"] = _google_ai_ocr_api_key is not None

    # ── OCR mode (skip if cloud mode just forced it) ──
    if req.ocr_mode is not None and req.cloud_mode is not True:
        mode = req.ocr_mode.lower().strip()
        if mode not in ("hayai", "glm", "lens", "openai_endpoint", "google_ai", "local_vision"):
            raise HTTPException(400, "ocr_mode must be 'hayai', 'glm', 'lens', 'openai_endpoint', 'google_ai', or 'local_vision'")
        if mode == "local_vision" and _selected_vision_model() is None:
            raise HTTPException(400, "Local GGUF OCR requires a selected GGUF model with a compatible mmproj/projector")
        if mode == "lens" and LensAPI is None:
            raise HTTPException(500, "chrome-lens-py not installed for lens OCR")
        if mode == "lens":
            try:
                get_lens_api()
            except Exception as e:
                raise HTTPException(500, f"Failed to initialize Google Lens API: {e}")
        if mode == "openai_endpoint":
            with _openai_ocr_config_lock:
                if not _openai_ocr_endpoint or not _openai_ocr_model:
                    raise HTTPException(400, "OpenAI Endpoint OCR needs an endpoint URL and model ID")
        if mode == "google_ai":
            with _google_ai_ocr_config_lock:
                if not _google_ai_ocr_api_key or not _google_ai_ocr_model:
                    raise HTTPException(400, "Google AI Studio OCR needs an API key and model")
        with _ocr_mode_lock:
            _ocr_mode = mode
        applied["ocr_mode"] = mode

    # ── Inpaint mode (skip if cloud mode just forced it) ──
    if req.inpaint_mode is not None and req.cloud_mode is not True:
        mode = req.inpaint_mode.lower().strip()
        if mode not in ("low", "high", "none"):
            raise HTTPException(400, "inpaint_mode must be 'low', 'high', or 'none'")
        if mode == "high":
            try:
                ensure_lama_large()
            except Exception as e:
                raise HTTPException(500, f"Failed to download high-quality inpainting model: {e}")
        with _inpaint_mode_lock:
            _inpaint_mode = mode
        applied["inpaint_mode"] = mode

    # ── Model type + openrouter config (skip if cloud mode just forced it) ──
    if req.model_type is not None and req.cloud_mode is not True:
        mt = req.model_type.lower().strip()
        if mt not in ("local", "openrouter"):
            raise HTTPException(400, "model_type must be 'local' or 'openrouter'")
        with _model_type_lock:
            _current_model_type = mt
            if mt == "openrouter":
                if req.openrouter_api_key is not None:
                    _openrouter_api_key = req.openrouter_api_key
                if not _openrouter_api_key:
                    raise HTTPException(400, "OpenRouter API key is required for openrouter model_type.")
                if req.openrouter_model is not None:
                    _openrouter_model = req.openrouter_model.strip()
        applied["model_type"] = mt
    else:
        # Even in cloud mode or when not switching model_type, allow the
        # openrouter model/key to be updated if provided.
        if req.openrouter_model is not None or req.openrouter_api_key is not None:
            with _model_type_lock:
                if req.openrouter_model is not None:
                    _openrouter_model = req.openrouter_model.strip()
                if req.openrouter_api_key is not None:
                    _openrouter_api_key = req.openrouter_api_key
            applied["openrouter_model"] = _openrouter_model

    # ── Font (filename) ──
    if req.font_filename is not None:
        req_font_name = req.font_filename.strip().lower()
        available_fonts = list_available_fonts()
        matched_font = None
        for f in available_fonts:
            if f["filename"].lower() == req_font_name or f["name"].lower() == req_font_name:
                matched_font = f
                break
        if not matched_font:
            # Not fatal — keep the current font, just warn.
            logging.warning(f"[SetAllSettings] Font '{req.font_filename}' not found; keeping current font.")
        else:
            with _font_config_lock:
                _current_font_path = pathlib.Path(matched_font["path"])
                clear_font_cache()
            applied["font_filename"] = matched_font["filename"]

    # ── Stroke width ──
    if req.stroke_width is not None:
        sw = max(0, min(20, int(req.stroke_width)))
        with _font_config_lock:
            _current_stroke_width = sw
        applied["stroke_width"] = sw

    # ── OpenRouter free mode ──
    if req.free_openrouter is not None:
        with _openrouter_free_mode_lock:
            _openrouter_free_mode = bool(req.free_openrouter)
        applied["free_openrouter"] = _openrouter_free_mode

    # ── Context-aware mode ──
    if req.context_aware is not None:
        with _context_aware_lock:
            _context_aware_mode = bool(req.context_aware)
        applied["context_aware"] = _context_aware_mode

    logging.info(f"[SetAllSettings] Applied: {applied}")
    _save_settings()
    return {"status": "ok", "applied": applied}

@app.get("/GetAllSettings")
async def get_all_settings():
    with _ocr_mode_lock:
        ocr_mode = _ocr_mode
    with _inpaint_mode_lock:
        inpaint_mode = _inpaint_mode
    with _cloud_mode_lock:
        cloud_mode = _cloud_mode
    with _model_type_lock:
        model_type = _current_model_type
        openrouter_model = _openrouter_model
        openrouter_configured = _openrouter_api_key is not None
    with _font_config_lock:
        font_path = str(_current_font_path)
        stroke_width = _current_stroke_width
    with _openrouter_free_mode_lock:
        free_mode = _openrouter_free_mode
    with _openai_ocr_config_lock:
        openai_ocr_endpoint = _openai_ocr_endpoint
        openai_ocr_model = _openai_ocr_model
        openai_ocr_api_key_set = _openai_ocr_api_key is not None
    with _google_ai_ocr_config_lock:
        google_ai_ocr_model = _google_ai_ocr_model
        google_ai_ocr_rpm = _google_ai_ocr_rpm
        google_ai_ocr_api_key_set = _google_ai_ocr_api_key is not None
    with _context_aware_lock:
        context_aware = _context_aware_mode
    return {
        "cloud_mode": cloud_mode,
        "ocr_mode": ocr_mode,
        "openai_ocr_endpoint": openai_ocr_endpoint,
        "openai_ocr_model": openai_ocr_model,
        "openai_ocr_api_key_set": openai_ocr_api_key_set,
        "google_ai_ocr_model": google_ai_ocr_model,
        "google_ai_ocr_rpm": google_ai_ocr_rpm,
        "google_ai_ocr_api_key_set": google_ai_ocr_api_key_set,
        "inpaint_mode": inpaint_mode,
        "model_type": model_type,
        "openrouter_model": openrouter_model,
        "openrouter_configured": openrouter_configured,
        "font_path": font_path,
        "stroke_width": stroke_width,
        "free_openrouter": free_mode,
        "context_aware": context_aware,
        "lens_available": LensAPI is not None,
    }

# ===========================================================================
# Health / Meta endpoints
# ===========================================================================
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/version")
async def version():
    return {"version": BUILD_ID}

@app.get("/meta")
async def meta():
    with _inpaint_mode_lock:
        inpaint_mode = _inpaint_mode
    with _ocr_mode_lock:
        ocr_mode = _ocr_mode
    with _openrouter_free_mode_lock:
        free_mode = _openrouter_free_mode
    with _context_aware_lock:
        context_aware = _context_aware_mode
    return {
        "version": BUILD_ID,
        "cuda": has_cuda(),
        "device": get_torch_device(),
        "ocr_mode": ocr_mode,
        "ocr_model": _current_ocr_model,
        "lens_available": LensAPI is not None,
        "font_path": str(_current_font_path),
        "stroke_width": _current_stroke_width,
        "model_type": _current_model_type,
        "openrouter_model": _openrouter_model if _current_model_type == "openrouter" else None,
        "local_model": f"{_current_qwen_repo_id}/{_current_qwen_filename}" if _current_model_type == "local" else None,
        "inpaint_lama_available": SimpleLama is not None,
        "inpaint_mode": inpaint_mode,
        "inpaint_high_model_downloaded": LAMA_LARGE_PATH.exists(),
        "inpaint_high_model_path": str(LAMA_LARGE_PATH),
        "openrouter_free_mode": free_mode,
        "context_aware": context_aware,
    }

@app.post("/warmup")
async def warmup():
    errors = []
    with _ocr_mode_lock:
        ocr_mode = _ocr_mode
    with _inpaint_mode_lock:
        inpaint_mode = _inpaint_mode

    try:
        get_yolo()
    except Exception as e:
        errors.append(f"YOLO: {e}")
    try:
        if ocr_mode != "lens":
            get_hayai_ocr()
    except Exception as e:
        errors.append(f"Hayai OCR: {e}")
    try:
        if ocr_mode == "lens":
            get_lens_api()
            logging.info("[Warmup] Google Lens API initialized.")
    except Exception as e:
        errors.append(f"Google Lens: {e}")
    try:
        if _current_model_type == "local":
            get_qwen()
    except Exception as e:
        errors.append(f"Qwen: {e}")
    try:
        if inpaint_mode == "none":
            logging.info("[Warmup] Inpaint mode is 'none' — skipping LaMa load entirely.")
        elif SimpleLama is not None:
            if inpaint_mode == "high":
                load_lama_high()
                logging.info("[Warmup] HighQualityLama loaded for inpainting.")
            else:
                load_lama_low()
                logging.info("[Warmup] SimpleLama loaded for inpainting.")
        else:
            logging.info("[Warmup] SimpleLama not installed; cv2.inpaint will be used as fallback.")
    except Exception as e:
        errors.append(f"SimpleLama: {e}")
    return {"status": "warmed" if not errors else "partial", "errors": errors}
    errors = []
    with _ocr_mode_lock:
        ocr_mode = _ocr_mode

    try:
        get_yolo()
    except Exception as e:
        errors.append(f"YOLO: {e}")
    try:
        if ocr_mode != "lens":
            get_hayai_ocr()
    except Exception as e:
        errors.append(f"Hayai OCR: {e}")
    try:
        if ocr_mode == "lens":
            get_lens_api()
            logging.info("[Warmup] Google Lens API initialized.")
    except Exception as e:
        errors.append(f"Google Lens: {e}")
    try:
        if _current_model_type == "local":
            get_qwen()
    except Exception as e:
        errors.append(f"Qwen: {e}")
    try:
        if SimpleLama is not None:
            with _inpaint_mode_lock:
                mode = _inpaint_mode
            if mode == "high":
                load_lama_high()
                logging.info("[Warmup] HighQualityLama loaded for inpainting.")
            else:
                load_lama_low()
                logging.info("[Warmup] SimpleLama loaded for inpainting.")
        else:
            logging.info("[Warmup] SimpleLama not installed; cv2.inpaint will be used as fallback.")
    except Exception as e:
        errors.append(f"SimpleLama: {e}")
    return {"status": "warmed" if not errors else "partial", "errors": errors}

# ===========================================================================
# Console / Logs endpoint
# ===========================================================================
@app.get("/console")
async def console():
    html = """<!DOCTYPE html>
<html><head><title>Console Logs</title>
<style>
body { background: #1a1a2e; color: #e0e0e0; font-family: 'Consolas', 'Monaco', monospace; padding: 20px; margin: 0; }
.log-line { padding: 2px 8px; border-bottom: 1px solid #2a2a4a; font-size: 13px; }
.log-line:hover { background: #2a2a4a; }
.level-INFO { color: #a0d0ff; }
.level-WARNING { color: #ffd060; }
.level-ERROR { color: #ff6060; }
.level-DEBUG { color: #808080; }
h1 { color: #60a0ff; margin-bottom: 10px; }
.controls { margin-bottom: 15px; }
button { background: #2a4a8a; color: white; border: 1px solid #4080c0; padding: 8px 16px;
         cursor: pointer; border-radius: 4px; margin-right: 8px; }
button:hover { background: #3a5a9a; }
#logs { max-height: calc(100vh - 120px); overflow-y: auto; }
</style></head><body>
<h1>Backend Console</h1>
<div class="controls">
<button onclick="fetchLogs()">Refresh</button>
<button onclick="autoRefresh=!autoRefresh;this.textContent=autoRefresh?'Stop Auto':'Auto Refresh'">Auto Refresh</button>
<button onclick="location.href='/fonts'">Fonts</button>
<span id="count"></span>
</div>
<div id="logs"></div>
<script>
let autoRefresh = false;
async function fetchLogs() {
  const r = await fetch('/console/json');
  const logs = await r.json();
  const el = document.getElementById('logs');
  document.getElementById('count').textContent = logs.length + ' entries';
  el.innerHTML = logs.map(l => {
    const cls = 'level-' + (l.match(/\\b(INFO|WARNING|ERROR|DEBUG)\\b/) || ['','INFO'])[1];
    return '<div class="log-line ' + cls + '">' + l.replace(/</g,'&lt;') + '</div>';
  }).join('');
  el.scrollTop = el.scrollHeight;
}
fetchLogs();
setInterval(() => { if(autoRefresh) fetchLogs(); }, 2000);
</script></body></html>"""
    return HTMLResponse(content=html)

@app.get("/console/json")
async def console_json():
    return JSONResponse(content=log_handler.get_logs())

# ===========================================================================
# Fonts preview page — renders every available font in its OWN typeface
# ===========================================================================
@app.get("/fonts")
async def fonts_page():
    """A small web page that lists every font in the fonts folder and renders a
    live sample of each in its own typeface (via @font-face + /v1/font/{name}).
    Clicking a card sets it as the active font through /SetFont."""
    html = """<!DOCTYPE html>
<html><head><title>Fonts</title><meta charset="utf-8">
<style>
body { background: #14141f; color: #e0e0e0; font-family: Arial, sans-serif; padding: 20px; margin: 0; }
h1 { color: #60a0ff; margin: 0 0 4px 0; }
.sub { color: #888; font-size: 13px; margin-bottom: 16px; }
.controls { margin-bottom: 16px; }
input[type=text] { padding: 8px; border-radius: 4px; border: 1px solid #444; background: #2a2a3c; color: #e0e0e0; width: 260px; }
button { background: #2a4a8a; color: #fff; border: 1px solid #4080c0; padding: 8px 14px; cursor: pointer; border-radius: 4px; margin-left: 8px; }
button:hover { background: #3a5a9a; }
#grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
.card { background: #1e1e2e; border: 2px solid #333; border-radius: 8px; padding: 14px; cursor: pointer; transition: border-color .15s; }
.card:hover { border-color: #4080c0; }
.card.active { border-color: #28a745; }
.name { font-size: 12px; color: #aaa; margin-bottom: 6px; word-break: break-all; }
.sample { font-size: 30px; line-height: 1.25; color: #fff; min-height: 44px; }
.sample.small { font-size: 15px; }
.meta { font-size: 11px; color: #666; margin-top: 8px; }
.badge { color: #28a745; font-size: 11px; font-weight: bold; }
#status { margin-top: 14px; font-size: 13px; color: #28a745; min-height: 18px; }
</style></head><body>
<h1>Fonts</h1>
<div class="sub">Each card previews a font in its own typeface. Click one to make it the active font.</div>
<div class="controls">
  <input type="text" id="sampleText" value="The quick brown fox — 123 あア 한글" placeholder="Preview text">
  <button onclick="render()">Update preview</button>
  <button onclick="load()">Refresh list</button>
</div>
<div id="grid"></div>
<div id="status"></div>
<script>
let FONTS = [];
let ACTIVE = '';
let STROKE = 0;
function filenameFromPath(p){ return (p||'').split(/[\\\\/]/).pop(); }
async function load(){
  const [fr, ar] = await Promise.all([fetch('/GetFonts'), fetch('/GetFont')]);
  const fd = await fr.json(); const ad = await ar.json();
  FONTS = fd.fonts || [];
  ACTIVE = filenameFromPath(ad.font_path);
  STROKE = (typeof ad.stroke_width === 'number') ? ad.stroke_width : 0;
  // Inject an @font-face for each font so previews use the real typeface.
  let css = '';
  FONTS.forEach(f => {
    const fam = 'PF_' + f.filename.replace(/[^a-zA-Z0-9]/g,'_');
    css += '@font-face{font-family:"'+fam+'";src:url("/v1/font/'+encodeURIComponent(f.filename)+'");}\\n';
  });
  let styleEl = document.getElementById('pfFaces');
  if (!styleEl){ styleEl = document.createElement('style'); styleEl.id='pfFaces'; document.head.appendChild(styleEl); }
  styleEl.textContent = css;
  render();
}
function render(){
  const sample = document.getElementById('sampleText').value || 'AaBbCc';
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  if (!FONTS.length){ grid.innerHTML = '<div class="sub">No fonts found in the fonts folder.</div>'; return; }
  FONTS.forEach(f => {
    const fam = 'PF_' + f.filename.replace(/[^a-zA-Z0-9]/g,'_');
    const card = document.createElement('div');
    card.className = 'card' + (f.filename === ACTIVE ? ' active' : '');
    const isActive = f.filename === ACTIVE;
    card.innerHTML =
      '<div class="name">'+f.name+(isActive?' <span class="badge">● active</span>':'')+'</div>'+
      '<div class="sample" style="font-family:\\''+fam+'\\',sans-serif">'+
        sample.replace(/</g,'&lt;')+'</div>'+
      '<div class="meta">'+f.filename+' · '+f.size_kb+' KB</div>';
    card.onclick = () => setActive(f.filename);
    grid.appendChild(card);
  });
}
async function setActive(filename){
  const status = document.getElementById('status');
  status.textContent = 'Switching to '+filename+'…';
  try {
    const res = await fetch('/SetFont', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({font_name: filename, stroke_width: STROKE})});
    const data = await res.json();
    if (!res.ok){ status.innerHTML = '<span style="color:#ff6060">Error: '+(data.detail||res.status)+'</span>'; return; }
    ACTIVE = filenameFromPath(data.font_path) || filename;
    status.textContent = 'Active font: '+ACTIVE;
    render();
  } catch(e){ status.innerHTML = '<span style="color:#ff6060">Error: '+e+'</span>'; }
}
load();
</script></body></html>"""
    return HTMLResponse(content=html)

# ===========================================================================
# Model management endpoints
# ===========================================================================
@app.post("/setmodel")
async def setmodel(req: SetModelTypeRequest):
    return await set_model_type(req)

@app.get("/getmodel")
async def getmodel():
    with _model_type_lock:
        result = {"model_type": _current_model_type}
        if _current_model_type == "local":
            result["local"] = {
                "repo_id": _current_qwen_repo_id,
                "filename": _current_qwen_filename,
                "path": str(_current_qwen_path) if _current_qwen_path else None,
            }
        else:
            result["openrouter"] = {
                "model": _openrouter_model,
                "api_key_set": _openrouter_api_key is not None,
            }
        return result

class ChangeModelRequest(BaseModel):
    repo_id: str
    filename: Optional[str] = None


@app.post("/v1/changemodel")
async def change_model(req: ChangeModelRequest):
    global _current_model_type, _inpaint_mode
    repo_id = req.repo_id.strip()
    filename = req.filename.strip() if req.filename else None
    if not repo_id:
        raise HTTPException(400, "repo_id is required")
    try:
        switch_qwen_model(repo_id, filename)
        with _model_type_lock:
            _current_model_type = "local"
        with _inpaint_mode_lock:
            _inpaint_mode = "low"
        _save_settings()
        selected = _selected_vision_model()
        return {
            "status": "ok",
            "repo_id": _current_qwen_repo_id,
            "filename": _current_qwen_filename,
            "model_type": "local",
            "vision_capable": selected is not None,
            "projector_filename": selected.get("projector_filename") if selected else None,
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/v1/listmodels")
async def list_models():
    models = list_local_gguf_models()
    return {"models": models, "count": len(models)}

# ===========================================================================
# OCR resolve endpoint
# ===========================================================================
@app.post("/v1/ai/resolve")
async def ai_resolve(image: UploadFile = File(...), lang: str = Form("ja")):
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    results = await get_ocr_results(pil_img, lang)
    return {"results": results, "count": len(results)}

# ===========================================================================
# Default prompt endpoint
# ===========================================================================
@app.get("/v1/ai/prompt/default")
async def get_default_prompt():
    return {"prompt": SYSTEM_PROMPT}

# ===========================================================================
# Colorize endpoint
# ===========================================================================
@app.post("/v1/colorize")
async def colorize_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        result = colorize_pil(pil_img)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"Colorization failed: {e}")

# ===========================================================================
# Translation Job endpoints
# ===========================================================================
class JobHealthRequest(BaseModel):
    job_id: str
    active: bool = True


def _job_health_expired(job: Dict[str, Any], now: Optional[float] = None) -> bool:
    if not job.get("health_required", False):
        return False
    checked_at = time.time() if now is None else now
    return checked_at - float(job.get("last_health", job.get("created", checked_at))) > JOB_HEALTH_TIMEOUT_SECONDS


def _cancel_job(job: Dict[str, Any], reason: str) -> None:
    status = job.get("status")
    if status in {"failed", "cancelled"} or (status == "completed" and not job.get("rendering", False)):
        return
    job["status"] = "cancelled"
    job["error"] = reason


async def _ensure_job_active(job_id: str) -> Dict[str, Any]:
    async with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            raise asyncio.CancelledError(f"Job {job_id} no longer exists")
        if _job_health_expired(job):
            _cancel_job(job, "Translation stopped because the originating page stopped sending health reports.")
        if job.get("status") == "cancelled":
            raise asyncio.CancelledError(job.get("error") or "Translation cancelled")
        return job


@app.post("/v1/health")
async def update_job_health(request: JobHealthRequest):
    async with _job_lock:
        job = _jobs.get(request.job_id)
        if not job:
            raise HTTPException(404, f"Job {request.job_id} not found")
        if request.active:
            if job.get("status") == "cancelled":
                return {"job_id": request.job_id, "status": "cancelled"}
            job["health_required"] = True
            job["last_health"] = time.time()
        else:
            job["health_required"] = True
            _cancel_job(job, "Translation stopped because the originating page was closed or navigated away.")
        return {"job_id": request.job_id, "status": job["status"]}


@app.post("/v1/translate")
async def create_translate_job(
    image: UploadFile = File(...),
    target_lang: str = Form(DEFAULT_LANG),
    ocr_lang: str = Form("ja"),
    source_lang: Optional[str] = Form(None),
    src_lang: Optional[str] = Form(None),
    sl: Optional[str] = Form(None),
    inpaint: bool = Form(True),
    skip_sfx: bool = Form(False),
    context_aware: bool = Form(False),
    context_level: str = Form("low"),
    style_aware: bool = Form(False),
    style_fonts: str = Form(""),
):
    # ── Source-language priority: accept the source language from any
    # common field name the extension might use. If none are provided,
    # default to "auto" instead of "ja" so OpenRouter and the OCR engine
    # auto-detect the language rather than forcing Japanese.
    effective_ocr_lang = _effective_ocr_language(
        target_lang, source_lang, src_lang, sl, ocr_lang
    )
    if _norm_lang(effective_ocr_lang) == "auto" and any(
        _norm_lang(value) == _norm_lang(target_lang)
        for value in (source_lang, src_lang, sl, ocr_lang)
        if value
    ):
        logging.warning(
            f"[Translate] Source and target are both {target_lang!r}; "
            "using automatic OCR language detection."
        )
    logging.info(
        f"[Translate] target_lang={target_lang!r} "
        f"ocr_lang(raw)={ocr_lang!r} effective={effective_ocr_lang!r}"
    )
    ocr_lang = effective_ocr_lang

    job_id = str(uuid.uuid4())[:8]
    contents = await image.read()
    request_fingerprint = hashlib.sha256(contents).hexdigest()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    level = "high" if str(context_level).lower() == "high" else "low"
    # High mode rides on a vision request, which only the OpenRouter backend
    # can make. On the local GGUF backend it silently degrades to low.
    with _model_type_lock:
        backend_is_openrouter = _current_model_type == "openrouter"
    if level == "high" and not backend_is_openrouter:
        logging.info(f"[Job {job_id}] Context level 'high' requires the OpenRouter "
                     f"backend; falling back to 'low'.")
        level = "low"
    style_font_map = _parse_style_fonts(style_fonts)
    style_on = bool(style_aware) and level == "high"
    with _ocr_mode_lock:
        job_ocr_mode = _ocr_mode
    with _model_type_lock:
        job_model_type = _current_model_type

    async with _job_lock:
        for existing_id, existing in _jobs.items():
            if (
                existing.get("request_fingerprint") == request_fingerprint
                and existing.get("target_lang") == target_lang
                and existing.get("ocr_lang") == ocr_lang
                and existing.get("ocr_mode") == job_ocr_mode
                and existing.get("status") in {"pending", "processing"}
            ):
                existing["last_health"] = time.time()
                logging.info(f"[Job {job_id}] Coalesced duplicate request into active job {existing_id}.")
                return {
                    "job_id": existing_id,
                    "status": existing["status"],
                    "inpaint": existing.get("inpaint", inpaint),
                    "skip_sfx": existing.get("skip_sfx", skip_sfx),
                    "context_aware": existing.get("context_aware", context_aware),
                    "context_level": existing.get("context_level", level),
                    "style_aware": existing.get("style_aware", style_on),
                    "coalesced": True,
                }
        _jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "image": pil_img,
            "request_fingerprint": request_fingerprint,
            "target_lang": target_lang,
            "ocr_lang": ocr_lang,
            "ocr_mode": job_ocr_mode,
            "model_type": job_model_type,
            "inpaint": inpaint,
            "skip_sfx": skip_sfx,
            "context_aware": context_aware,
            "context_level": level,
            "style_aware": style_on,
            "style_fonts": style_font_map,
            "result": None,
            "error": None,
            "created": time.time(),
            "health_required": True,
            "last_health": time.time(),
        }

    asyncio.create_task(_process_job(job_id))
    return {"job_id": job_id, "status": "pending", "inpaint": inpaint, "skip_sfx": skip_sfx,
            "context_aware": context_aware, "context_level": level, "style_aware": style_on}


async def _process_job(job_id: str):
    """Background task: OCR -> Translate (Batch for OpenRouter & Local).

    Each translation entry now includes a "bboxes" list containing all original
    bounding boxes for merged groups, so the image renderer can draw in-place
    without repositioning to the center of a merged region.
    """
    async with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["status"] = "processing"

    try:
        job = await _ensure_job_active(job_id)
        pil_img = job["image"]
        target_lang = job["target_lang"]
        ocr_lang = job["ocr_lang"]

        ocr_results = await get_ocr_results(pil_img, ocr_lang, job.get("ocr_mode"))
        job = await _ensure_job_active(job_id)
        logging.info(
            f"[Job {job_id}] OCR mode={job.get('ocr_mode')} produced "
            f"{len(ocr_results)} renderable region(s)."
        )

        if not ocr_results:
            logging.warning(
                f"[Job {job_id}] OCR produced no regions; returning the original image unchanged."
            )
            async with _job_lock:
                job["status"] = "completed"
                job["result"] = {"boxes": [], "translations": []}
            return

        texts_to_translate = [item["text"] for item in ocr_results]
        translations = []

        # Classify each OCR region as SFX once, up front. The result is attached
        # to every translation entry so the renderer can skip SFX overlay when
        # the job's skip_sfx flag is set. Classification is cheap (pure
        # arithmetic + regex on the bbox/text) so it's fine to always compute
        # it, even when skip_sfx is False.
        sfx_flags: List[Dict[str, Any]] = []
        skip_sfx = job.get("skip_sfx", False)
        for item in ocr_results:
            is_sfx, score, reasons = detect_sfx(pil_img, item["bbox"], item["text"])
            sfx_flags.append({"is_sfx": is_sfx, "score": score, "reasons": reasons})
            if is_sfx:
                logging.info(f"[Job {job_id}] SFX detected (score={score}): "
                             f"'{item['text'][:40]}' bbox={item['bbox']} reasons={reasons}")

        # ── Context-aware two-pass: build a name dictionary ONCE per job ──
        # Adds ~1 extra LLM/OpenRouter call + ~200-600 tokens when enabled.
        # Per-job context_aware (from FormData) takes precedence; fall back to
        # the global toggle so /SetContextAware still works as a default.
        with _context_aware_lock:
            global_ca = _context_aware_mode
        context_aware = job.get("context_aware", False) or global_ca
        name_map: Dict[str, str] = {}
        if context_aware and texts_to_translate:
            src_lang_name = _translation_source_name(ocr_lang, target_lang)
            lang_name = get_lang_name(target_lang)
            with _model_type_lock:
                model_type_for_names = _current_model_type
                or_model_for_names = _openrouter_model
                or_key_for_names = _openrouter_api_key
            try:
                if model_type_for_names == "openrouter" and or_key_for_names:
                    name_map = await _build_name_map_openrouter(
                        texts_to_translate, src_lang_name, lang_name,
                        or_key_for_names, or_model_for_names,
                    )
                else:
                    llm_for_names = get_qwen()
                    if llm_for_names:
                        loop = asyncio.get_event_loop()
                        name_map = await loop.run_in_executor(
                            _llm_executor,
                            _build_name_map_llm,
                            texts_to_translate, src_lang_name, lang_name, llm_for_names,
                        )
            except Exception as e:
                logging.warning(f"[Job {job_id}] Context-aware name-map build failed: {e}")

        # ── Style-aware: the batch translator writes a per-line lettering-style
        # tag into this parallel list. It stays index-aligned with
        # texts_to_translate; entries the model didn't tag (or that failed
        # validation) stay None.
        style_aware = bool(job.get("style_aware", False))
        style_fonts = job.get("style_fonts", {}) or {}
        styles: List[Optional[Dict[str, Any]]] = [None] * len(texts_to_translate)

        def _style_fields(idx: int) -> Dict[str, Any]:
            entry = styles[idx] if (style_aware and idx < len(styles)) else None
            if not entry:
                return {"style": None, "weight": 0, "glow": False}
            return {
                "style": entry.get("style"),
                "weight": entry.get("weight", 0),
                "glow": bool(entry.get("glow", False)),
            }

        def _tilt_fields(idx: int) -> Dict[str, Any]:
            """Carry the OCR tilt through to the renderer. Non-Lens paths have none."""
            box = ocr_results[idx] if idx < len(ocr_results) else {}
            angle = float(box.get("angle", 0.0) or 0.0)
            angles = box.get("angles")
            if not angles:
                angles = [angle] * len(box.get("bboxes", [box.get("bbox")]))
            return {"angle": angle, "angles": angles}

        def _ocr_attr_fields(idx: int) -> Dict[str, Any]:
            """Carry vision-OCR overlay attributes through to the renderer."""
            box = ocr_results[idx] if idx < len(ocr_results) else {}
            out: Dict[str, Any] = {}
            for k in ("or_color", "or_glow", "or_style", "or_bold", "or_font_px"):
                if k in box:
                    out[k] = box[k]
            return out

        model_type = job.get("model_type", "local")

        if model_type == "openrouter":
            logging.info(f"[Job {job_id}] Using OpenRouter BATCH strategy for {len(texts_to_translate)} boxes.")
            batch_results = await openrouter_translate_batch(
                texts_to_translate, target_lang, ocr_lang,
                context_aware=context_aware, name_map=name_map,
                style_aware=style_aware, styles_out=styles,
                page_image=pil_img if style_aware else None,
            )

            needs_sequential_fallback = not any(batch_results)

            if needs_sequential_fallback:
                logging.warning(f"[Job {job_id}] Batch failed entirely, falling back to sequential requests.")
                for idx, text in enumerate(texts_to_translate):
                    ocr_bbox = ocr_results[idx]["bbox"]
                    ocr_bboxes = ocr_results[idx].get("bboxes", [ocr_bbox])
                    if not text.strip():
                        translations.append({
                            "text": text, "translation": "",
                            "bbox": ocr_bbox, "bboxes": ocr_bboxes,
                            "is_sfx": sfx_flags[idx]["is_sfx"],
                            **_style_fields(idx),
                            **_tilt_fields(idx),
                            **_ocr_attr_fields(idx),
                        })
                        continue
                    translated = await openrouter_translate(text, target_lang, ocr_lang)
                    await asyncio.sleep(1.0)
                    translations.append({
                        "text": text,
                        "translation": translated,
                        "bbox": ocr_bbox,
                        "bboxes": ocr_bboxes,
                        "is_sfx": sfx_flags[idx]["is_sfx"],
                        **_style_fields(idx),
                        **_tilt_fields(idx),
                        **_ocr_attr_fields(idx),
                    })
            else:
                for idx, text in enumerate(texts_to_translate):
                    ocr_bbox = ocr_results[idx]["bbox"]
                    ocr_bboxes = ocr_results[idx].get("bboxes", [ocr_bbox])
                    translated = batch_results[idx]
                    if not translated and text.strip():
                        logging.warning(f"[Job {job_id}] Box {idx+1} missed in batch, retrying individually...")
                        translated = await openrouter_translate(text, target_lang, ocr_lang)
                        await asyncio.sleep(1.0)

                    translations.append({
                        "text": text,
                        "translation": translated,
                        "bbox": ocr_bbox,
                        "bboxes": ocr_bboxes,
                        "is_sfx": sfx_flags[idx]["is_sfx"],
                        **_style_fields(idx),
                        **_tilt_fields(idx),
                        **_ocr_attr_fields(idx),
                    })
        else:
            # --- BATCH STRATEGY FOR LOCAL GGUF ---
            logging.info(f"[Job {job_id}] Using Local GGUF BATCH strategy for {len(texts_to_translate)} boxes.")
            await asyncio.get_running_loop().run_in_executor(_llm_executor, get_qwen)
            logging.info(f"[Job {job_id}] Local GGUF ready: {_current_qwen_repo_id}/{_current_qwen_filename}")
            batch_results = await asyncio.get_running_loop().run_in_executor(
                _llm_executor, qwen_translate_batch,
                texts_to_translate, target_lang, ocr_lang, context_aware, name_map,
                None,
            )

            for idx, text in enumerate(texts_to_translate):
                ocr_bbox = ocr_results[idx]["bbox"]
                ocr_bboxes = ocr_results[idx].get("bboxes", [ocr_bbox])
                translations.append({
                    "text": text,
                    "translation": batch_results[idx] if batch_results[idx] else "",
                    "bbox": ocr_bbox,
                    "bboxes": ocr_bboxes,
                    "is_sfx": sfx_flags[idx]["is_sfx"],
                    **_style_fields(idx),
                    **_tilt_fields(idx),
                    **_ocr_attr_fields(idx),
                })

        # Fail closed at the job boundary. Provider parsing success does not
        # guarantee that a candidate was translated rather than echoed.
        await _ensure_job_active(job_id)
        rejected_count = 0
        for entry in translations:
            source = entry.get("text", "")
            candidate = entry.get("translation", "")
            validated = _validated_translation(source, candidate, target_lang)
            if candidate and not validated:
                rejected_count += 1
                logging.warning(
                    f"[Job {job_id}] Rejected untranslated/source-like output: "
                    f"source={source[:40]!r}, output={candidate[:40]!r}"
                )
            entry["translation"] = validated
            entry["render_text"] = validated
        if rejected_count:
            logging.warning(
                f"[Job {job_id}] Rejected {rejected_count}/{len(translations)} "
                f"parsed outputs before rendering."
            )
        renderable_count = sum(
            1 for entry in translations
            if _render_translation_text(entry, target_lang)
        )
        logging.info(
            f"[Job {job_id}] Translation pipeline produced {renderable_count}/"
            f"{len(translations)} renderable overlay(s)."
        )
        if translations and not renderable_count:
            logging.error(
                f"[Job {job_id}] All translations were rejected or empty; "
                f"the rendered image will be unchanged."
            )

        # Honorifics are handled inline by the always-on HONORIFIC_CLAUSE in the
        # translation prompt (romanized straight from the OCR source text). No
        # mechanical post-process reinsertion.

        await _ensure_job_active(job_id)
        async with _job_lock:
            job["status"] = "completed"
            job["result"] = {
                "boxes": ocr_results,
                "translations": translations,
                "sfx_flags": sfx_flags,
                "skip_sfx": skip_sfx,
                "name_map": name_map if context_aware else {},
                "context_aware": context_aware,
                "style_aware": style_aware,
                "style_fonts": style_fonts,
            }

    except asyncio.CancelledError as exc:
        logging.info(f"[Job {job_id}] Cancelled: {exc}")
        async with _job_lock:
            current = _jobs.get(job_id)
            if current:
                _cancel_job(current, str(exc) or "Translation cancelled")
    except Exception as e:
        logging.error(f"[Job {job_id}] Failed: {e}\n{traceback.format_exc()}")
        async with _job_lock:
            job["status"] = "failed"
            job["error"] = str(e)

@app.get("/v1/translate/{job_id}")
async def get_translate_job(job_id: str):
    async with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        if _job_health_expired(job):
            _cancel_job(job, "Translation stopped because the originating page stopped sending health reports.")
        result = {
            "id": job["id"],
            "status": job["status"],
            "target_lang": job["target_lang"],
            "ocr_lang": job["ocr_lang"],
            "inpaint": job.get("inpaint", True),
        }
        if job["status"] == "completed":
            result["result"] = job["result"]
        elif job["status"] in {"failed", "cancelled"}:
            result["error"] = job["error"]
        return result


def _render_translation_text(item: Dict[str, Any], target_lang: str) -> str:
    render_text = clean_text_for_font(item.get("render_text", ""))
    if not render_text:
        return ""
    if not _looks_like_target(render_text, target_lang):
        return ""
    return render_text


async def _rendering_job(job_id: str):
    async with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        if _job_health_expired(job):
            _cancel_job(job, "Translation stopped because the originating page stopped sending health reports.")
        if job.get("status") == "cancelled":
            raise HTTPException(409, job.get("error") or "Translation cancelled")
        if job.get("status") != "completed":
            raise HTTPException(400, f"Job {job_id} is not completed (status: {job.get('status')})")
        job["rendering"] = True
    try:
        yield job
    finally:
        async with _job_lock:
            current = _jobs.get(job_id)
            if current:
                current["rendering"] = False


async def _render_translated_image(job_id: str, job: Dict[str, Any]):
    """Generate the final translated image.

    Inpaint mode behaviour:
      - 'low'/'high': Standard LaMa inpainting erases original text.
      - 'none':       No inpainting model is loaded. Text regions are
                      filled with the background color detected by the
                      text-color algorithm, then translations are drawn
                      on top.
    """
    async with _job_lock:
        pil_img = job["image"]
        translations = job["result"].get("translations", [])
        do_inpaint = job.get("inpaint", True)
        ocr_mode = job.get("ocr_mode", _ocr_mode)
        skip_sfx = job.get("skip_sfx", False)
        style_aware = bool(job["result"].get("style_aware", False))
        style_fonts = job["result"].get("style_fonts", {}) or {}

    await _ensure_job_active(job_id)

    if not translations:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    img_bgr = pil_to_cv2(pil_img)
    h, w = img_bgr.shape[:2]

    boxes_to_inpaint = []
    inpaint_angles = []
    items_to_draw = []

    # ── Collect inpaint boxes (all original sub-boxes) and draw items (union bbox) ──
    for t_idx, item in enumerate(translations):
        source_text = item.get("text", "")
        translated_text = _render_translation_text(item, job["target_lang"])
        if not translated_text:
            logging.warning(
                f"[Render] Box {t_idx + 1}/{len(translations)} has no valid translation text "
                f"(source='{source_text[:40]}') — nothing to overlay."
            )
            continue
        bbox = item.get("bbox")
        if not bbox:
            logging.warning(
                f"[Render] Box {t_idx + 1}/{len(translations)} has no bbox — skipping overlay."
            )
            continue

        # ── SFX skip: when the job requests skip_sfx and this region was
        # classified as SFX, leave the original art completely untouched —
        # no inpainting, no overlay. This is the fix for Google Lens merging
        # SFX + dialogue into one giant region that then gets wiped by the
        # 'none'-mode bg-color fill.
        if skip_sfx and item.get("is_sfx"):
            logging.info(f"[Render] Skipping SFX overlay for '{translated_text[:40]}' at {bbox}")
            continue

        bboxes = item.get("bboxes", [bbox])
        item_angle = float(item.get("angle", 0.0) or 0.0)
        sub_angles = item.get("angles") or []
        for bx_idx, bx in enumerate(bboxes):
            bx1, by1, bx2, by2 = bx
            if (bx2 - bx1) < 10 or (by2 - by1) < 10:
                continue
            boxes_to_inpaint.append(bx)
            sub_angle = sub_angles[bx_idx] if bx_idx < len(sub_angles) else item_angle
            inpaint_angles.append(float(sub_angle or 0.0))

        x1, y1, x2, y2 = bbox
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            logging.warning(
                f"[Render] Box {t_idx + 1}/{len(translations)} too small to overlay "
                f"({x2 - x1}x{y2 - y1}px at {bbox}) — translation '{translated_text[:40]}' dropped."
            )
            continue

        # ── Duplicate-overlay guard ──
        # OCR can hand us several near-identical regions carrying the same
        # string (YOLO detections without NMS, or Lens returning overlapping
        # paragraphs). Nothing downstream deduplicates, so the renderer would
        # stack N copies of the same overlay on the same spot. Drop a candidate
        # only when the text matches AND the box substantially coincides with
        # one already collected — a legitimately repeated line elsewhere on the
        # page has a low IoU and still gets drawn.
        # This applies to items_to_draw only: the boxes_to_inpaint fan-out above
        # is intentional, since inpainting must erase every original glyph run.
        norm_translation = translated_text.strip()
        is_duplicate = False
        for prev in items_to_draw:
            if prev["translated_text"].strip() != norm_translation:
                continue
            if _bbox_iou(prev["bbox"], bbox) >= DUPLICATE_OVERLAY_IOU:
                is_duplicate = True
                break
        if is_duplicate:
            logging.info(f"[Render] Skipping duplicate overlay for '{norm_translation[:40]}' at {bbox}")
            continue

        items_to_draw.append({
            "translated_text": translated_text,
            "bbox": bbox,
            "style": item.get("style"),
            "weight": item.get("weight", 0),
            "glow": bool(item.get("glow", False)),
            "angle": item_angle,
            "or_color": item.get("or_color"),
            "or_glow": item.get("or_glow"),
            "or_style": item.get("or_style"),
            "or_bold": item.get("or_bold"),
            "or_font_px": item.get("or_font_px"),
        })

    if len(items_to_draw) != len(translations):
        logging.warning(
            f"[Render] Overlaying {len(items_to_draw)}/{len(translations)} boxes — "
            f"{len(translations) - len(items_to_draw)} dropped (see [Render] lines above). "
            f"skip_sfx={skip_sfx}"
        )
    else:
        logging.info(f"[Render] Overlaying all {len(items_to_draw)} boxes.")

    # ── Detect text/background colors FIRST (from original image) ──
    # This must happen before any inpainting/filling so the colors are
    # read from the original text. In 'none' mode we need the bg_color
    # to fill regions.
    orig_bgr = pil_to_cv2(pil_img)
    all_bboxes_for_color = [item["bbox"] for item in items_to_draw]
    all_box_colors = detect_text_colors_batch(orig_bgr, all_bboxes_for_color)
    color_by_idx = {i: all_box_colors[i] for i in range(len(items_to_draw))}
    # Vision-OCR text-color override: replace only the detected text color
    # with the model-provided hex; keep the auto-detected bg for the outline.
    for _ci, _it in enumerate(items_to_draw):
        _or_rgb = _hex_to_rgb(_it.get("or_color"))
        if _or_rgb is not None:
            _det_text, _det_bg = color_by_idx[_ci]
            color_by_idx[_ci] = (_or_rgb, _det_bg)

    # ── High mode: measure the ORIGINAL lettering height per region ──
    # style_aware ("high" context) means we already have the page image, so we
    # measure how tall the source text was actually drawn and size the overlay
    # to match instead of guessing from the bubble. Only computed in high mode
    # to avoid the extra per-box work on the standard path.
    glyph_h_by_idx: Dict[int, Optional[float]] = {}
    if style_aware or ocr_mode == "local_vision":
        for i, item in enumerate(items_to_draw):
            glyph_h_by_idx[i] = measure_source_glyph_height(orig_bgr, item["bbox"])

    # ── Get inpaint mode ──
    with _inpaint_mode_lock:
        inpaint_mode = _inpaint_mode

    # ── Erase original text ──
    if do_inpaint and boxes_to_inpaint:
        if inpaint_mode == "none":
            # ── None mode: fill text regions with detected background color ──
            # No inpainting model is used. Each item's union bbox is filled
            # with the background color detected by the text-color algorithm.
            logging.info(f"[Inpaint] Mode='none' — filling {len(items_to_draw)} text regions "
                         f"with detected background color (no inpainting model loaded).")
            out_pil_temp = cv2_to_pil(img_bgr)
            draw_fill = ImageDraw.Draw(out_pil_temp)
            for item_idx, item in enumerate(items_to_draw):
                _, bg_color = color_by_idx[item_idx]
                bx1, by1, bx2, by2 = item["bbox"]
                fill_angle = float(item.get("angle", 0.0) or 0.0)
                if fill_angle:
                    draw_fill.polygon(
                        _rotated_box_points((bx1, by1, bx2, by2), fill_angle),
                        fill=bg_color,
                    )
                else:
                    # Fill the union bbox with the detected background color
                    draw_fill.rectangle([bx1, by1, bx2, by2], fill=bg_color)
            img_bgr = pil_to_cv2(out_pil_temp)
            logging.info(f"[Inpaint] None-mode background fill complete for {len(items_to_draw)} regions.")
        else:
            # ── Low/High mode: standard inpainting ──
            logging.info(f"[Inpaint] Building mask for {len(boxes_to_inpaint)} text regions "
                         f"(from {len(translations)} translation groups)...")
            mask_padding = 2
            mask_dilate_kernel = 3
            if ocr_mode == "local_vision":
                mask_padding = LOCAL_VISION_INPAINT_PADDING
                mask_dilate_kernel = LOCAL_VISION_INPAINT_DILATE_KERNEL
            mask = build_inpaint_mask(
                img_bgr.shape,
                boxes_to_inpaint,
                padding=mask_padding,
                dilate_kernel=mask_dilate_kernel,
                angles=inpaint_angles,
            )
            use_lama = inpaint_mode == "high" or SimpleLama is not None
            img_bgr = await inpaint_image_async(img_bgr, mask, use_lama=use_lama)
            await _ensure_job_active(job_id)
            logging.info(f"[Inpaint] Inpainting complete for {len(boxes_to_inpaint)} regions.")

    await _ensure_job_active(job_id)
    out_pil = cv2_to_pil(img_bgr)
    draw = ImageDraw.Draw(out_pil)

    with _font_config_lock:
        fp = str(_current_font_path)

    # ── Per-style font resolution ──
    # High-mode style awareness lets the user pick a font per lettering bucket.
    # An unset bucket, an unknown style, or a filename that is no longer on disk
    # all fall back to the globally configured font.
    _style_font_cache: Dict[str, str] = {}

    def _font_for_style(style: Optional[str]) -> str:
        if not style:
            return fp
        if style in _style_font_cache:
            return _style_font_cache[style]
        resolved = fp
        name = (style_fonts.get(style) or "").strip()
        if name:
            candidate = FONT_DIR / os.path.basename(name)
            if candidate.is_file():
                resolved = str(candidate)
            else:
                logging.warning(f"[Style] Font for '{style}' not found on disk: {name!r} — using main font.")
        _style_font_cache[style] = resolved
        return resolved

    # Preserve the complete translation by allowing compact text in dense OCR
    # boxes. The fitter still chooses the largest size that fits first.
    ABS_MIN_SIZE = 8

    is_lens = (ocr_mode == "lens")

    for item_idx, item in enumerate(items_to_draw):
        if item_idx % 4 == 0:
            await asyncio.sleep(0)
            await _ensure_job_active(job_id)
        translated_text = item["translated_text"]
        bbox = item["bbox"]
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        text_color, bg_color = color_by_idx[item_idx]

        # Use the complete OCR bbox for wrapping and sizing.
        INNER_PADDING_RATIO = 0.0

        # ── Per-bubble dynamic size range ──
        # A translated line can be much larger than the source lettering. Search
        # up to twice the largest region dimension; fit_font_and_wrap verifies
        # the complete wrapped block against both region dimensions.
        dyn_max = max(ABS_MIN_SIZE + 1, max(box_w, box_h) * 2)
        dyn_min = ABS_MIN_SIZE
        if ocr_mode == "local_vision":
            measured_glyph_h = glyph_h_by_idx.get(item_idx)
            if measured_glyph_h:
                dyn_max = max(dyn_min, min(dyn_max, int(round(measured_glyph_h * 1.35))))

        # OCR/source font measurements remain available as metadata, but they do
        # not limit the translated overlay. The geometry fit is authoritative.

        # ── Fit text to the complete bbox ──
        effective_style = item.get("or_style") or item.get("style")
        item_fp = _font_for_style(effective_style)

        font_size, lines, heights, inner_w, inner_h = fit_font_and_wrap(
            draw, translated_text, box_w, box_h, font_path=item_fp,
            max_size=dyn_max, min_size=dyn_min,
            inner_padding_ratio=INNER_PADDING_RATIO,
        )
        font = get_font(item_fp, font_size)

        # Never split words: text is always wrapped by whole words only. If the
        # block is taller than the bubble it overflows (the whole translation
        # stays intact and readable) rather than being cut off or broken into
        # fragments across many lines.

        # ── Compute inner box position (centered within outer bbox) ──
        inner_x = x1 + (box_w - inner_w) // 2
        inner_y = y1 + (box_h - inner_h) // 2

        # ── Compute vertical placement (center text block within inner box) ──
        if heights:
            total_text_h = sum(heights)
        else:
            total_text_h = font_size * len(lines)

        start_y = inner_y + (inner_h - total_text_h) // 2

        # ── Style effects from the original lettering ──
        # Bold and glow are only applied when the model reported them off the
        # source artwork. Both scale with the rendered font size so a small
        # bubble doesn't get a stroke wider than its own glyphs.
        item_style = item.get("style")
        item_weight = int(item.get("weight", 0) or 0)
        item_glow = bool(item.get("glow", False))

        embolden = 0
        if style_aware and item_style == "bold" and item_weight > 0:
            # weight 1..3 → roughly 1.5%..4.5% of the font size, min 1px.
            embolden = max(1, int(round(font_size * 0.015 * item_weight)))
        # Vision-OCR boldness override (1..3), independent of style_aware.
        or_bold = item.get("or_bold")
        if isinstance(or_bold, (int, float)) and or_bold > 0:
            embolden = max(1, int(round(font_size * 0.015 * min(3, or_bold))))

        glow_radius = 0.0
        or_glow = item.get("or_glow")
        if (style_aware and item_glow) or bool(or_glow):
            glow_radius = max(2.0, font_size * 0.12)

        def _draw_lines(target_draw, block_x, block_y, block_w, target_image=None):
            """Draw the wrapped lines centered horizontally in block_w."""
            cur_y = block_y
            for i, line in enumerate(lines):
                line_h = heights[i] if i < len(heights) else font_size
                if line:
                    line_w = target_draw.textlength(line, font=font)
                    draw_text_with_config(
                        target_draw,
                        (block_x + (block_w - line_w) / 2, cur_y),
                        line,
                        font=font,
                        fill=text_color,
                        stroke_fill=bg_color,
                        embolden=embolden,
                        glow_radius=glow_radius,
                        target_image=target_image,
                    )
                cur_y += line_h

        # ── Draw the block ──
        # Upright text goes straight onto the page. Tilted text (Google Lens
        # angle) is drawn on a transparent layer, rotated, then composited so
        # the glyphs follow the slant of the original line.
        item_angle = float(item.get("angle", 0.0) or 0.0)

        if not item_angle:
            _draw_lines(draw, inner_x, start_y, inner_w, out_pil)
        else:
            # Margin absorbs glyph overshoot (ascenders/descenders) plus the
            # glow halo, so nothing is clipped before the rotation widens the
            # layer.
            margin = max(8, int(font_size) + int(glow_radius * 3))
            layer = Image.new(
                "RGBA",
                (max(1, int(inner_w) + margin * 2),
                 max(1, int(total_text_h) + margin * 2)),
                (0, 0, 0, 0),
            )
            _draw_lines(ImageDraw.Draw(layer), margin, margin, inner_w, layer)
            # _rotated_box_points treats positive as clockwise in image coords;
            # PIL rotates counter-clockwise, hence the negated angle.
            layer = layer.rotate(-item_angle, expand=True, resample=Image.BICUBIC)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            out_pil.paste(
                layer,
                (int(round(cx - layer.width / 2.0)),
                 int(round(cy - layer.height / 2.0))),
                layer,
            )

    await _ensure_job_active(job_id)
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    await _ensure_job_active(job_id)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/v1/translate/{job_id}/image")
async def get_translated_image(job_id: str, job: Optional[Dict[str, Any]] = Depends(_rendering_job)):
    if not isinstance(job, dict):
        dependency = _rendering_job(job_id)
        direct_job = await anext(dependency)
        try:
            return await _render_translated_image(job_id, direct_job)
        finally:
            await dependency.aclose()
    return await _render_translated_image(job_id, job)

# ===========================================================================
# SFX Detection Endpoint
# ===========================================================================
@app.post("/v1/detect_sfx")
async def detect_sfx_endpoint(
    image: UploadFile = File(...),
    text: str = Form(...),
    bbox: str = Form(...),
):
    """Classify a single text region as SFX or dialogue.

    Form fields:
      image: the source manga page image.
      text:  the OCR text for the region.
      bbox:  "x1,y1,x2,y2" (comma-separated pixel coords).

    Returns {is_sfx, score, reasons}.
    """
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        parts = [int(p.strip()) for p in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must have 4 ints")
        bbox_tuple = (parts[0], parts[1], parts[2], parts[3])
    except Exception as e:
        raise HTTPException(400, f"Invalid bbox '{bbox}': {e}")

    is_sfx, score, reasons = detect_sfx(pil_img, bbox_tuple, text)
    return {"is_sfx": is_sfx, "score": score, "reasons": reasons}

# ===========================================================================
# Main entry point
# ===========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
