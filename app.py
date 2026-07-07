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
import os
import pathlib
import time
import traceback
import urllib.request
import uuid
import logging
import threading
import functools
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

# --- FastAPI ---------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query, Request, Form
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
    return -1 if has_cuda() else 0

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
    from llama_cpp import Llama
except Exception:
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

_jobs: Dict[str, Dict[str, Any]] = {}
_job_lock = asyncio.Lock()
_job_queue: Optional[asyncio.Queue] = None
_worker_task = None

_llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
_llm_lock = threading.Lock()

# Inpainting executor (runs in thread pool since it can be slow)
_inpaint_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inpaint")
_inpaint_lock = threading.Lock()

# --- Inpainting Mode Globals ---
_inpaint_mode = "low"  # "low" or "high"
_inpaint_mode_lock = threading.Lock()

# --- OCR Mode Globals ---
_ocr_mode = "hayai"  # "hayai", "glm", or "lens"
_ocr_mode_lock = threading.Lock()

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
                if not _is_valid_gguf(f):
                    continue
                try:
                    size_mb = f.stat().st_size / (1024 * 1024)
                except OSError:
                    continue
                models.append({
                    "name": f"{repo_id.replace('/', '__')}__{f.name}",
                    "repo_id": repo_id,
                    "filename": f.name,
                    "size_mb": round(size_mb, 1),
                    "path": str(f.resolve()),
                })
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
            if not _is_valid_gguf(f):
                continue
            try:
                size_mb = f.stat().st_size / (1024 * 1024)
            except OSError:
                continue
            stem = f.stem
            parts = stem.split("__")
            if len(parts) >= 2:
                filename_part = parts[-1]
                repo_part = "/".join(parts[:-1])
            else:
                filename_part = stem
                repo_part = stem
            models.append({
                "name": stem,
                "repo_id": repo_part,
                "filename": filename_part + ".gguf",
                "size_mb": round(size_mb, 1),
                "path": str(f),
            })
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
    if _global_yolo is None:
        ensure_yolo()
        device = get_torch_device()
        logging.info(f"[YOLO] Loading model on device: {device}")
        _global_yolo = YOLO(str(YOLO_MODEL_PATH))
        _global_yolo.to(device)
        logging.info(f"[YOLO] Ready on {device}.")
    return _global_yolo

def hayai_ocr_with_yolo(pil_img: Image.Image) -> List[Dict[str, Any]]:
    img_bgr = pil_to_cv2(pil_img)
    h, w = img_bgr.shape[:2]
    yolo = get_yolo()
    logging.info(f"[OCR] Running YOLO text detection on {w}x{h} image...")
    use_half = has_cuda()
    results = yolo(img_bgr, verbose=False, conf=0.4, device=get_torch_device(), half=use_half)
    if not results:
        return []
    r = results[0]
    out = []
    img_area = h * w
    mocr = get_hayai_ocr()
    boxes = []
    for b in r.boxes:
        xy = b.xyxy[0].cpu().numpy()
        x1, y1 = max(0, int(xy[0])), max(0, int(xy[1]))
        x2, y2 = min(w - 1, int(xy[2])), min(h - 1, int(xy[3]))
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > 0.8 * img_area or box_area < 100:
            continue
        boxes.append((x1, y1, x2, y2))
        
    logging.info(f"[Hayai OCR] Found {len(boxes)} valid text boxes. Running OCR sequentially...")
    if not boxes:
        return []
        
    def _ocr_one(bbox):
        x1, y1, x2, y2 = bbox
        crop = pil_img.crop((x1, y1, x2, y2))
        try:
            return bbox, mocr(crop).strip()
        except Exception as e:
            logging.error(f"Hayai OCR failed on {bbox}: {e}")
            return bbox, ""
            
    # FIX: Removed _OCR_BOX_EXECUTOR.map to prevent thread pool deadlock
    for bbox in boxes:
        bbox, text = _ocr_one(bbox)
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
    use_half = has_cuda()
    results = yolo(img_bgr, verbose=False, conf=0.4, device=get_torch_device(), half=use_half)
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
    logging.info(f"[GLM OCR] Found {len(boxes)} valid text boxes. Running GLM OCR sequentially...")
    
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
    # FIX: Removed _OCR_BOX_EXECUTOR.map to prevent thread pool deadlock
    for bbox in boxes:
        bbox, text = _ocr_one(bbox)
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

def _merge_close_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges OCR blocks that are physically close to each other.
    Expands bounding boxes horizontally by 50% of their width to catch side-by-side text.
    Expands bounding boxes vertically by a smaller margin (20% of height) to prevent 
    merging distinct speech bubbles stacked on top of each other.
    """
    if len(blocks) <= 1:
        return blocks
        
    parent = list(range(len(blocks)))
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            
    for i in range(len(blocks)):
        x1_i, y1_i, x2_i, y2_i = blocks[i]["bbox"]
        w_i = x2_i - x1_i
        h_i = y2_i - y1_i
        
        # Expand horizontally by 50% of width
        # Expand vertically by 20% of height (or at least 10px) - reduces vertical over-merging
        v_pad_i = max(10, h_i * 0.2)
        exp_x1_i = x1_i - w_i * 0.5
        exp_y1_i = y1_i - v_pad_i
        exp_x2_i = x2_i + w_i * 0.5
        exp_y2_i = y2_i + v_pad_i
        
        for j in range(i + 1, len(blocks)):
            x1_j, y1_j, x2_j, y2_j = blocks[j]["bbox"]
            w_j = x2_j - x1_j
            h_j = y2_j - y1_j
            
            v_pad_j = max(10, h_j * 0.2)
            exp_x1_j = x1_j - w_j * 0.5
            exp_y1_j = y1_j - v_pad_j
            exp_x2_j = x2_j + w_j * 0.5
            exp_y2_j = y2_j + v_pad_j
            
            # Check intersection of expanded boxes
            ix1 = max(exp_x1_i, exp_x1_j)
            iy1 = max(exp_y1_i, exp_y1_j)
            ix2 = min(exp_x2_i, exp_x2_j)
            iy2 = min(exp_y2_i, exp_y2_j)
            
            if ix1 < ix2 and iy1 < iy2:
                union(i, j)
                
    # Group blocks by their root parent
    groups = {}
    for i in range(len(blocks)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(blocks[i])
        
    merged_blocks = []
    for group in groups.values():
        # Combined bounding box
        x1 = min(b["bbox"][0] for b in group)
        y1 = min(b["bbox"][1] for b in group)
        x2 = max(b["bbox"][2] for b in group)
        y2 = max(b["bbox"][3] for b in group)
        
        # Sort texts in manga reading order (right-to-left, top-to-bottom)
        group.sort(key=lambda b: (b["bbox"][0] * -1, b["bbox"][1]))
        texts = [b["text"] for b in group]
        merged_text = " ".join(texts) # Join with space for LLM context
        
        merged_blocks.append({
            "text": merged_text,
            "bbox": (x1, y1, x2, y2)
        })
        
    return merged_blocks

async def google_lens_ocr(pil_img: Image.Image, ocr_lang: str = "ja") -> List[Dict[str, Any]]:
    api = get_lens_api()
    w, h = pil_img.size
    logging.info(f"[Google Lens] Running OCR on {w}x{h} image (lang={ocr_lang})...")
    lens_lang_map = {
        "ja": "ja", "ko": "ko", "en": "en", "zh": "zh",
        "ru": "ru", "es": "es", "id": "id", "cz": "zh",
    }
    lens_lang = lens_lang_map.get(ocr_lang, ocr_lang)
    try:
        result = await api.process_image(
            image_path=pil_img, ocr_language=lens_lang, output_format='blocks'
        )
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
        out.append({"text": text, "bbox": bbox})
    
    # Merge close blocks before returning
    merged = _merge_close_blocks(out)
    logging.info(f"[Google Lens] Found {len(out)} raw blocks -> merged to {len(merged)} blocks.")
    return merged

async def get_ocr_results(pil_img: Image.Image, ocr_lang: str = "ja") -> List[Dict[str, Any]]:
    with _ocr_mode_lock:
        mode = _ocr_mode
    if mode == "lens":
        return await google_lens_ocr(pil_img, ocr_lang)
    elif mode == "glm" or ocr_lang == "ko":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_OCR_BOX_EXECUTOR, glm_ocr_korean, pil_img)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_OCR_BOX_EXECUTOR, hayai_ocr_with_yolo, pil_img)

# ===========================================================================
# Qwen GGUF translator
# ===========================================================================
LANG_MAP = {
    "en": "English", "ja": "Japanese", "ko": "Korean",
    "id": "Indonesian", "ru": "Russian", "es": "Spanish", "cz": "Chinese"
}

SYSTEM_PROMPT = (
    "You are a manga translation engine. "
    "Translate the user's text into {lang}. "
    "Output ONLY the {lang} translation, with no explanations, no notes, and no quotes."
)

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
                use_gpu = has_cuda()
                n_gpu_layers = -1 if use_gpu else 0
                logging.info(f"[Qwen] loading {path} (GPU layers: {n_gpu_layers}) ...")
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

def switch_qwen_model(repo_id: str, filename: Optional[str] = None):
    global _global_qwen, _current_qwen_repo_id, _current_qwen_filename, _current_qwen_path
    path = download_gguf(repo_id, filename)
    with _qwen_model_lock:
        _current_qwen_repo_id = repo_id
        _current_qwen_filename = filename or path.name
        _current_qwen_path = path
        _global_qwen = None
    logging.info(f"[Qwen] Switched to {repo_id}/{filename}, preloading...")
    get_qwen()

def qwen_translate(text: str, target_lang: str = "en") -> str:
    text = text.strip()
    if not text:
        return ""
    lang_name = LANG_MAP.get(target_lang, "English")
    # FIX: Increased max_tokens to prevent cutoffs on small models that add fluff text
    max_tok = max(64, min(256, len(text) + 32))
    logging.info(f"[LLM] Starting translation for: '{text[:40]}' -> {lang_name}")
    llm = get_qwen()
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT.format(lang=lang_name)},
        {"role": "user",   "content": text},
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
                
        # FIX: Clean up common prefixes small models use despite instructions
        for prefix in ("Translation:", "Translated:", "Output:", f"{lang_name}:", f"{lang_name} translation:"):
            if raw.lower().startswith(prefix.lower()):
                raw = raw[len(prefix):].strip()
                
        logging.info(f"[LLM] Translated to: '{raw[:40]}'")
        return clean_text_for_font(raw)
    except Exception as e:
        logging.error(f"[LLM] Translation failed: {e}")
        return ""

def qwen_translate_batch(texts: List[str], target_lang: str = "en") -> List[str]:
    """Translate a list of texts in a SINGLE LLM call using a numbered list."""
    indexed_texts = [(i, t.strip()) for i, t in enumerate(texts) if t.strip()]
    if not indexed_texts:
        return [""] * len(texts)

    lang_name = LANG_MAP.get(target_lang, "English")
    prompt_lines = [f"{idx + 1}. {text.replace(chr(10), ' ')}" for idx, (_, text) in enumerate(indexed_texts)]
    batch_text = "\n".join(prompt_lines)

    # FIX: Simplified prompt to be more robust for 0.8B models
    batch_system_prompt = (
        f"Translate the following numbered texts into {lang_name}. "
        "Output only the translations, keeping the exact same numbers. No extra text."
    )

    total_chars = sum(len(t) for _, t in indexed_texts)
    # FIX: Increased token limit generously
    max_tok = max(256, min(4096, total_chars + (len(indexed_texts) * 32)))

    llm = get_qwen()
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
                    results[orig_idx] = clean_text_for_font(trans)
                    matched_any = True
                    
        # FIX: Fallback if model ignored numbers and just outputted translations line by line
        if not matched_any and len(parsed_lines) == len(indexed_texts):
            logging.warning("[LLM Batch] Model didn't use numbers, mapping line-by-line...")
            for i, line in enumerate(parsed_lines):
                orig_idx = indexed_texts[i][0]
                results[orig_idx] = clean_text_for_font(line)
            matched_any = True

        # Final fallback for any missing items
        for num, (orig_idx, text) in enumerate(indexed_texts):
            if not results[orig_idx]:
                logging.warning(f"[LLM Batch] Box {num+1} missed in batch, retrying individually...")
                results[orig_idx] = qwen_translate(text, target_lang)

        return results
    except Exception as e:
        logging.error(f"[LLM Batch] Translation failed: {e}")
        return [""] * len(texts)

# ===========================================================================
# OpenRouter Translation
# ===========================================================================

async def openrouter_translate_batch(texts: List[str], target_lang: str = "en", max_retries: int = 5) -> List[str]:
    import aiohttp
    import random

    with _model_type_lock:
        api_key = _openrouter_api_key
        model = _openrouter_model

    if not api_key:
        logging.error("[OpenRouter] API key not configured")
        return [""] * len(texts)

    indexed_texts = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not indexed_texts:
        return [""] * len(texts)

    lang_name = LANG_MAP.get(target_lang, "English")
    total_chars = sum(len(t) for _, t in indexed_texts)
    max_tok = max(256, min(4096, total_chars + (len(indexed_texts) * 20)))

    # Sanitize input: replace newlines with spaces so the LLM list doesn't break
    prompt_lines = [f"{idx + 1}. {text.replace(chr(10), ' ')}" for idx, (orig_i, text) in enumerate(indexed_texts)]
    batch_text = "\n".join(prompt_lines)

    batch_system_prompt = (
        f"You are a manga translation engine. Translate the user's numbered list of texts into {lang_name}. "
        "Output ONLY the translated list, one per line, keeping the exact same numbers. "
        "Do not include the original text. "
        "No explanations, no notes, no quotes."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Manga Translation API"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": batch_system_prompt},
            {"role": "user", "content": batch_text},
        ],
        "max_tokens": max_tok,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    logging.info(f"[OpenRouter Batch] Sending {len(indexed_texts)} texts in one request to {model}...")

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
            logging.info(f"[OpenRouter Batch] Retry {attempt}/{max_retries} after {wait_time:.1f}s wait...")
            await asyncio.sleep(wait_time)

        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        wait = float(retry_after) if retry_after else 10.0
                        logging.warning(f"[OpenRouter Batch] Rate limited (429). Waiting {wait:.1f}s...")
                        await asyncio.sleep(wait)
                        continue

                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"[OpenRouter Batch] API error {response.status} on attempt {attempt}: {error_text[:200]}")
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
                        continue

                    results = [""] * len(texts)
                    parsed_lines = raw.split('\n')
                    for line in parsed_lines:
                        match = re.match(r"^\s*(\d+)\.\s*(.*)$", line)
                        if match:
                            num = int(match.group(1)) - 1
                            trans = match.group(2).strip()
                            if 0 <= num < len(indexed_texts):
                                orig_idx = indexed_texts[num][0]
                                results[orig_idx] = clean_text_for_font(trans)

                    success_count = sum(1 for r in results if r)
                    logging.info(f"[OpenRouter Batch] Parsed {success_count}/{len(indexed_texts)} translations.")
                    
                    if success_count > 0:
                        return results
                    else:
                        logging.warning(f"[OpenRouter Batch] Failed to parse any numbered lines from response.")
                        continue

        except asyncio.TimeoutError:
            logging.warning(f"[OpenRouter Batch] Timeout on attempt {attempt}/{max_retries}")
            continue
        except Exception as e:
            logging.error(f"[OpenRouter Batch] Error on attempt {attempt}/{max_retries}: {e}")
            continue

    logging.error(f"[OpenRouter Batch] FAILED after {max_retries} retries. Falling back to sequential.")
    return [""] * len(texts)

async def openrouter_translate(text: str, target_lang: str = "en", max_retries: int = 5) -> str:
    import aiohttp
    import random

    with _model_type_lock:
        api_key = _openrouter_api_key
        model = _openrouter_model

    if not api_key:
        logging.error("[OpenRouter] API key not configured")
        return ""

    if not text.strip():
        return ""

    lang_name = LANG_MAP.get(target_lang, "English")
    max_tok = max(16, min(96, len(text) + 16))

    logging.info(f"[OpenRouter] Translating '{text[:40]}' -> {lang_name} using {model}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Manga Translation API"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(lang=lang_name)},
            {"role": "user", "content": text},
        ],
        "max_tokens": max_tok,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
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
                        continue

                    result = clean_text_for_font(raw)
                    logging.info(f"[OpenRouter] Translated to: '{result[:40]}'")
                    return result

        except asyncio.TimeoutError:
            logging.warning(f"[OpenRouter] Timeout on attempt {attempt}/{max_retries}")
            continue
        except Exception as e:
            logging.error(f"[OpenRouter] Error on attempt {attempt}/{max_retries}: {e}")
            continue

    logging.error(f"[OpenRouter] FAILED after {max_retries} retries for: '{text[:40]}'")
    return ""

def translate_with_current_backend(text: str, target_lang: str = "en") -> str:
    with _model_type_lock:
        model_type = _current_model_type
    if model_type == "openrouter":
        try:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(openrouter_translate(text, target_lang))
                return result
            finally:
                loop.close()
        except Exception as e:
            logging.error(f"[OpenRouter] Failed to run async translation: {e}")
            return ""
    else:
        return qwen_translate(text, target_lang)

async def translate_with_current_backend_async(text: str, target_lang: str = "en") -> str:
    with _model_type_lock:
        model_type = _current_model_type
    if model_type == "openrouter":
        return await openrouter_translate(text, target_lang)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_llm_executor, qwen_translate, text, target_lang)

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
    Uses patch-based inference with:
    - 50% overlap (stride=256) for seamless blending
    - Single pass inference (with color correction)
    - Proper Gaussian-weighted accumulation and normalization
    - Feathered mask boundary for smooth transitions
    - Post-processing color correction to match surroundings
    - TF32 + cuDNN benchmark on CUDA for speed
    - FP16 precision on CUDA for doubled throughput
    - Batched patch inference for massive speedups
    """

    def __init__(self):
        self.device = get_torch_device()
        # Use FP16 on CUDA for massive speedup, FP32 on CPU
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Enable TF32 on CUDA for tensor-core acceleration
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        ensure_lama_large()
        logging.info(f"[Lama High] Loading {LAMA_LARGE_PATH} on device: {self.device} (dtype={self.dtype})")
        
        # TorchScript load with CPU fallback
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
        self.stride = 256  # 50% overlap is enough for seamless blending
        self.batch_size = 4 if self.device == "cuda" else 1 # Batch patches on GPU

        # Wider Gaussian (sigma = patch_size/3) for softer blending
        gauss = cv2.getGaussianKernel(self.patch_size, self.patch_size // 3)
        self.gauss_2d = (gauss @ gauss.T).astype(np.float32)
        self.gauss_2d /= self.gauss_2d.max()

    def _infer_patches_batch(self, batch_imgs: np.ndarray, batch_masks: np.ndarray) -> np.ndarray:
        """Run the LaMa model on a batch of 512×512 patches. Returns float32 RGB array."""
        img_t = (
            torch.from_numpy(batch_imgs)
            .float()
            .permute(0, 3, 1, 2)
            .to(self.device)
            .to(self.dtype)
            / 255.0
        )
        mask_t = (
            torch.from_numpy(batch_masks)
            .float()
            .unsqueeze(1)
            .to(self.device)
            .to(self.dtype)
            / 255.0
        )
        mask_t = (mask_t > 0.5).float()

        with torch.no_grad():
            out = self.model(img_t, mask_t).clamp(0, 1)

        return (out.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.float32)

    def _gen_coords(self, length: int) -> List[int]:
        ps = self.patch_size
        if length <= ps:
            return [0]
        coords = list(range(0, length - ps + 1, self.stride))
        if coords[-1] != length - ps:
            coords.append(length - ps)
        return coords

    def _run_pass(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Single inpainting pass over the image. Returns the blended result."""
        H, W = img.shape[:2]
        ps = self.patch_size

        inpainted_acc = np.zeros((H, W, 3), dtype=np.float32)
        inpainted_weight = np.zeros((H, W), dtype=np.float32)

        ys = self._gen_coords(H)
        xs = self._gen_coords(W)

        valid_patches = []
        valid_coords = []

        # Gather all valid patches first
        for y in ys:
            for x in xs:
                y1, y2 = y, y + ps
                x1, x2 = x, x + ps

                patch_img = img[y1:y2, x1:x2]
                patch_mask = mask[y1:y2, x1:x2]
                ph, pw = patch_img.shape[:2]

                # Pad to full patch size if at image edge
                if ph < ps or pw < ps:
                    patch_img = np.pad(
                        patch_img,
                        ((0, ps - ph), (0, ps - pw), (0, 0)),
                        mode="reflect",
                    )
                    patch_mask = np.pad(
                        patch_mask,
                        ((0, ps - ph), (0, ps - pw)),
                        mode="reflect",
                    )

                # Skip patches with nothing to inpaint
                if patch_mask.sum() == 0:
                    continue

                valid_patches.append((patch_img, patch_mask))
                valid_coords.append((y1, y2, x1, x2, ph, pw))

        # Process patches in batches
        for i in range(0, len(valid_patches), self.batch_size):
            batch_imgs = np.stack([p[0] for p in valid_patches[i:i+self.batch_size]])
            batch_masks = np.stack([p[1] for p in valid_patches[i:i+self.batch_size]])
            
            outs = self._infer_patches_batch(batch_imgs, batch_masks)
            
            for j, out_patch in enumerate(outs):
                y1, y2, x1, x2, ph, pw = valid_coords[i+j]
                out_patch = out_patch[:ph, :pw]
                g = self.gauss_2d[:ph, :pw]

                inpainted_acc[y1:y2, x1:x2] += out_patch * g[:, :, None]
                inpainted_weight[y1:y2, x1:x2] += g

        # Normalize accumulated inpainted results
        inpainted_weight_safe = inpainted_weight.copy()
        inpainted_weight_safe[inpainted_weight_safe == 0] = 1.0
        inpainted_result = (inpainted_acc / inpainted_weight_safe[:, :, None]).clip(
            0, 255
        )

        # Feather the mask for a smooth transition between original and inpainted
        feathered = cv2.GaussianBlur(
            (mask > 0).astype(np.float32), (7, 7), 2.0
        )
        feathered = np.maximum(feathered, (mask > 0).astype(np.float32))

        final = (
            inpainted_result * feathered[:, :, None]
            + img.astype(np.float32) * (1.0 - feathered[:, :, None])
        )
        return final.clip(0, 255).astype(np.uint8)

    def _color_correct(
        self, inpainted: np.ndarray, mask: np.ndarray, original: np.ndarray
    ) -> np.ndarray:
        """Match inpainted region's per-channel mean/std to surrounding context."""
        border = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=2)
        border = (border > 0) & (mask == 0)

        if border.sum() < 20 or mask.sum() == 0:
            return inpainted

        out = inpainted.astype(np.float32).copy()
        mask_bool = mask > 0

        for c in range(3):
            ref_pixels = original[border, c].astype(np.float32)
            if len(ref_pixels) < 10:
                continue
            ref_mean = ref_pixels.mean()
            ref_std = max(ref_pixels.std(), 1.0)

            inp_pixels = out[mask_bool, c]
            if len(inp_pixels) < 10:
                continue
            inp_mean = inp_pixels.mean()
            inp_std = max(inp_pixels.std(), 1.0)

            corrected = (out[:, :, c] - inp_mean) / inp_std * ref_std + ref_mean
            out[:, :, c] = np.where(mask_bool, corrected, out[:, :, c])

        return out.clip(0, 255).astype(np.uint8)

    def __call__(self, pil_img: Image.Image, pil_mask: Image.Image) -> Image.Image:
        w, h = pil_img.size
        img = np.array(pil_img.convert("RGB"))
        mask = np.array(pil_mask.convert("L"))

        # Pad to multiple of 8 (LaMa dimension requirement)
        pad_w = (8 - w % 8) % 8
        pad_h = (8 - h % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")

        # ── Pass 1: Initial inpainting ──
        logging.info("[Lama High] Pass 1/1 — batched inpainting...")
        result_1 = self._run_pass(img, mask)

        # ── Color correction: match inpainted stats to surrounding original ──
        logging.info("[Lama High] Color correction...")
        corrected = self._color_correct(result_1, mask, img)

        # Restore original dimensions
        corrected = corrected[:h, :w, :]
        return Image.fromarray(corrected)

def load_lama_high():
    """Load the high-quality LaMa model (anime-manga-big-lama.pt)."""
    global _simple_lama_high_model
    if _simple_lama_high_model is None:
        logging.info(f"[Lama High] Initializing high-quality model wrapper...")
        _simple_lama_high_model = HighQualityLama()
        logging.info(f"[Lama High] Successfully loaded high-quality model.")
    return _simple_lama_high_model


def load_lama():
    """Load the appropriate LaMa model based on the current inpaint mode."""
    with _inpaint_mode_lock:
        mode = _inpaint_mode

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
    """Run inpainting in a thread pool so it doesn't block the event loop."""
    with _inpaint_mode_lock:
        mode = _inpaint_mode

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
                       dilate_kernel: int = 3) -> np.ndarray:
    """Build a strict binary mask tailored tightly to the text bounding boxes."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for x1, y1, x2, y2 in bboxes:
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(w, x2 + padding)
        py2 = min(h, y2 + padding)
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
    """Detect text (ink) and background (outline) colors within a bbox.

    Strategy:
      1. Slightly expand the bbox to capture surrounding background pixels
         for a more representative background sample.
      2. Cluster pixels in LAB color space using k-means (k=3) to identify
         dominant color groups (background, text, anti-aliasing/transition).
      3. Background = the largest cluster.
      4. Text = the cluster with the greatest perceptual distance from the
         background that has at least 4% of the pixels.
      5. Use per-cluster MEDIAN color (robust to outliers) instead of mean.
      6. Only "snap" to pure black/white when colors are extremely close to
         the extremes (<=20 or >=235), avoiding premature binarization that
         causes inconsistent black/white flipping between similar regions.
      7. If final text/bg contrast is still too low, force the text to the
         opposite extreme of the background luminance — but only as a last
         resort.

    Returns (text_rgb, bg_rgb) as 0-255 RGB tuples.
    """
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
        # Border sampling decides which side is background
        if rh >= 2 and rw >= 2:
            border = np.concatenate([
                gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
            ])
        else:
            border = gray.flatten()
        border_mean = float(border.mean()) if border.size else float(gray.mean())
        if border_mean > 127:
            text_sel = (otsu.flatten() == 0)   # light bg -> dark text
        else:
            text_sel = (otsu.flatten() != 0)   # dark bg  -> light text
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


def detect_text_colors_batch(img_bgr: np.ndarray,
                             bboxes: List[Tuple[int, int, int, int]]
                             ) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Detect text/bg colors for a list of bboxes WITH global consistency.

    Two passes:
      1. Run detect_text_and_bg_colors on each box independently.
      2. Tally light-text vs dark-text votes across all boxes. If there is a
         strong majority (>= 2x the other), force the minority boxes to flip
         to the majority polarity. This is what fixes "same region, half
         white / half black text" — visually similar boxes now agree.

    Returns a list of (text_rgb, bg_rgb) tuples aligned with `bboxes`.
    """
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

    # Require a 2:1 majority before forcing flips — protects mixed scenes
    # (e.g. light text in a dark bubble AND dark text on a light sign).
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
                # Flip to light text. Use the actual detected bg color as outline
                # if it's dark enough; otherwise fall back to pure black outline.
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

def wrap_text(draw, text, font, max_width, allow_break=False, is_vertical=False):
    if is_vertical:
        return [text] if text else [""]
    words = text.split()
    if not words:
        return [""]
    lines = []
    cur = ""
    for word in words:
        word_width = draw.textlength(word, font=font)
        if word_width > max_width:
            if not allow_break:
                return None
            if cur:
                lines.append(cur)
                cur = ""
            while word:
                split_idx = len(word)
                while split_idx > 1 and draw.textlength(word[:split_idx], font=font) > max_width:
                    split_idx -= 1
                if split_idx == 0:
                    split_idx = 1
                part = word[:split_idx]
                if draw.textlength(part + "-", font=font) <= max_width and split_idx < len(word):
                    part += "-"
                lines.append(part)
                word = word[split_idx:]
            continue
        test = (cur + " " + word) if cur else word
        if draw.textlength(test, font=font) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
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
                      max_size=96, min_size=8, is_vertical=False):
    if font_path is None:
        with _font_config_lock:
            font_path = str(_current_font_path)
    if not text.strip():
        return min_size, [""], [0]
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
            bb = draw.textbbox((0,0), "字", font=font)
            char_h = (bb[3] - bb[1]) * 1.2
            if char_h == 0: char_h = mid
            for ch in clean_v_text:
                if cur_h + char_h > box_h and cur_col:
                    cols.append(cur_col)
                    cur_col = ch
                    cur_h = char_h
                else:
                    cur_col += ch
                    cur_h += char_h
            if cur_col: cols.append(cur_col)
            if not cols: cols = [clean_v_text]
            max_char_w = max(draw.textlength(ch, font=font) for ch in clean_v_text) if clean_v_text else mid
            col_w = max(max_char_w, mid * 0.8)
            total_w = len(cols) * col_w
            if total_w <= box_w - 4:
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
            bb = draw.textbbox((0,0), "字", font=font)
            char_h = (bb[3] - bb[1]) * 1.2
            if char_h == 0: char_h = min_size
            cols = []
            cur_col = ""
            cur_h = 0
            for ch in clean_v_text:
                if cur_h + char_h > box_h and cur_col:
                    cols.append(cur_col)
                    cur_col = ch
                    cur_h = char_h
                else:
                    cur_col += ch
                    cur_h += char_h
            if cur_col: cols.append(cur_col)
            best_cols = cols if cols else [text]
            max_char_w = max(draw.textlength(ch, font=font) for ch in clean_v_text) if clean_v_text else min_size
            best_col_widths = [max_char_w] * len(best_cols)
            best_size = min_size
        return best_size, best_cols, best_col_widths

    lo, hi = min_size, max_size
    best_size = None
    best_lines = None
    best_heights = None
    while lo <= hi:
        mid = (lo + hi) // 2
        key = (font_path, mid)
        if key not in cache:
            try: cache[key] = ImageFont.truetype(font_path, mid)
            except Exception: cache[key] = ImageFont.load_default()
        font = cache[key]
        lines = wrap_text(draw, text, font, box_w - 4, allow_break=False, is_vertical=False)
        if lines is None:
            hi = mid - 1
            continue
        heights, total_h, max_w = _measure_block(draw, lines, font)
        if max_w <= box_w - 4 and total_h <= box_h - 4:
            best_size, best_lines, best_heights = mid, lines, heights
            lo = mid + 1
        else:
            hi = mid - 1
    if best_lines is None:
        key = (font_path, min_size)
        if key not in cache:
            try: cache[key] = ImageFont.truetype(font_path, min_size)
            except Exception: cache[key] = ImageFont.load_default()
        font = cache[key]
        fallback_lines = wrap_text(draw, text, font, box_w - 4, allow_break=True, is_vertical=False)
        best_lines = fallback_lines if fallback_lines else [text]
        heights, _, _ = _measure_block(draw, best_lines, font)
        best_size = min_size
        best_heights = heights
    return best_size, best_lines, best_heights

# ===========================================================================
# Text drawing with configurable stroke
# ===========================================================================
def draw_text_with_config(draw: ImageDraw.ImageDraw,
                          position: Tuple[float, float],
                          text: str,
                          font: ImageFont.FreeTypeFont,
                          fill: Tuple[int, int, int],
                          stroke_fill: Optional[Tuple[int, int, int]] = None,
                          anchor: Optional[str] = None):
    with _font_config_lock:
        stroke_width = _current_stroke_width
    if stroke_width > 0 and stroke_fill is not None:
        draw.text(position, text, font=font, fill=fill,
                  stroke_width=stroke_width, stroke_fill=stroke_fill, anchor=anchor)
    else:
        draw.text(position, text, font=font, fill=fill, anchor=anchor)

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


# ===========================================================================
# SetInpaintMode Endpoint (Low/High inpainting model switching)
# ===========================================================================
class SetInpaintModeRequest(BaseModel):
    mode: str  # "low" or "high"

@app.post("/SetInpaintMode")
async def set_inpaint_mode(req: SetInpaintModeRequest):
    global _inpaint_mode
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
    }

# ===========================================================================
# SetOcrMode Endpoint (OCR backend switching: hayai / glm / lens)
# ===========================================================================
class SetOcrModeRequest(BaseModel):
    mode: str  # "hayai", "glm", or "lens"

@app.post("/SetOcrMode")
async def set_ocr_mode(req: SetOcrModeRequest):
    global _ocr_mode
    mode = req.mode.lower().strip()
    if mode not in ("hayai", "glm", "lens"):
        raise HTTPException(400, "mode must be 'hayai', 'glm', or 'lens'")

    if mode == "lens" and LensAPI is None:
        raise HTTPException(500, "chrome-lens-py not installed. Run: pip install chrome-lens-py")

    if mode == "lens":
        try:
            get_lens_api()
        except Exception as e:
            raise HTTPException(500, f"Failed to initialize Google Lens API: {e}")

    with _ocr_mode_lock:
        _ocr_mode = mode

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
        "available_modes": ["hayai", "glm", "lens"],
        "lens_available": LensAPI is not None,
        "descriptions": {
            "hayai": "Hayai OCR (Japanese, local model + YOLO)",
            "glm": "GLM-OCR (Korean, transformers + YOLO)",
            "lens": "Google Lens OCR (all languages, cloud API)",
        }
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
    return {
        "status": "ok",
        "openrouter_model": _openrouter_model,
        "api_key_set": _openrouter_api_key is not None,
        "note": "This only takes effect when model_type is 'openrouter'. Use /SetModelType to switch."
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
    }

@app.post("/warmup")
async def warmup():
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

@app.post("/v1/changemodel")
async def change_model(repo_id: str = Form(...), filename: Optional[str] = Form(None)):
    try:
        switch_qwen_model(repo_id, filename)
        return {"status": "ok", "repo_id": repo_id, "filename": filename}
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
@app.post("/v1/translate")
async def create_translate_job(
    image: UploadFile = File(...),
    target_lang: str = Form(DEFAULT_LANG),
    ocr_lang: str = Form("ja"),
    inpaint: bool = Form(True),
):
    """Create a new translation job.
    
    - inpaint: If true, erase original text via inpainting before overlaying translations.
               If false, overlay translations directly on top of the original text.
    """
    job_id = str(uuid.uuid4())[:8]
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    async with _job_lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "image": pil_img,
            "target_lang": target_lang,
            "ocr_lang": ocr_lang,
            "inpaint": inpaint,
            "result": None,
            "error": None,
            "created": time.time(),
        }

    asyncio.create_task(_process_job(job_id))
    return {"job_id": job_id, "status": "pending", "inpaint": inpaint}


async def _process_job(job_id: str):
    """Background task: OCR -> Translate (Batch for OpenRouter & Local)."""
    async with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["status"] = "processing"
        # Capture the OCR mode that produced these boxes (lens boxes are tighter)
        with _ocr_mode_lock:
            job["ocr_mode"] = _ocr_mode

    try:
        pil_img = job["image"]
        target_lang = job["target_lang"]
        ocr_lang = job["ocr_lang"]

        ocr_results = await get_ocr_results(pil_img, ocr_lang)

        if not ocr_results:
            async with _job_lock:
                job["status"] = "completed"
                job["result"] = {"boxes": [], "translations": []}
            return

        texts_to_translate = [item["text"] for item in ocr_results]
        translations = []

        with _model_type_lock:
            model_type = _current_model_type

        if model_type == "openrouter":
            logging.info(f"[Job {job_id}] Using OpenRouter BATCH strategy for {len(texts_to_translate)} boxes.")
            batch_results = await openrouter_translate_batch(texts_to_translate, target_lang)

            needs_sequential_fallback = not any(batch_results)

            if needs_sequential_fallback:
                logging.warning(f"[Job {job_id}] Batch failed entirely, falling back to sequential requests.")
                for idx, text in enumerate(texts_to_translate):
                    if not text.strip():
                        translations.append({"text": text, "translation": "", "bbox": ocr_results[idx]["bbox"]})
                        continue
                    translated = await openrouter_translate(text, target_lang)
                    await asyncio.sleep(1.0)
                    translations.append({
                        "text": text,
                        "translation": translated,
                        "bbox": ocr_results[idx]["bbox"],
                    })
            else:
                for idx, text in enumerate(texts_to_translate):
                    translated = batch_results[idx]
                    if not translated and text.strip():
                        logging.warning(f"[Job {job_id}] Box {idx+1} missed in batch, retrying individually...")
                        translated = await openrouter_translate(text, target_lang)
                        await asyncio.sleep(1.0)

                    translations.append({
                        "text": text,
                        "translation": translated,
                        "bbox": ocr_results[idx]["bbox"],
                    })
        else:
            # --- BATCH STRATEGY FOR LOCAL GGUF ---
            logging.info(f"[Job {job_id}] Using Local BATCH strategy for {len(texts_to_translate)} boxes.")
            loop = asyncio.get_event_loop()
            batch_results = await loop.run_in_executor(_llm_executor, qwen_translate_batch, texts_to_translate, target_lang)

            for idx, text in enumerate(texts_to_translate):
                translations.append({
                    "text": text,
                    "translation": batch_results[idx] if batch_results[idx] else "",
                    "bbox": ocr_results[idx]["bbox"],
                })

        async with _job_lock:
            job["status"] = "completed"
            job["result"] = {
                "boxes": ocr_results,
                "translations": translations,
            }

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
        result = {
            "id": job["id"],
            "status": job["status"],
            "target_lang": job["target_lang"],
            "ocr_lang": job["ocr_lang"],
            "inpaint": job.get("inpaint", True),
        }
        if job["status"] == "completed":
            result["result"] = job["result"]
        elif job["status"] == "failed":
            result["error"] = job["error"]
        return result

@app.post("/v1/translate/{job_id}/image")
async def get_translated_image(job_id: str):
    """Generate the final translated image."""
    async with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        if job["status"] != "completed":
            raise HTTPException(400, f"Job {job_id} is not completed (status: {job['status']})")

        pil_img = job["image"]
        translations = job["result"].get("translations", [])
        do_inpaint = job.get("inpaint", True)
        ocr_mode = job.get("ocr_mode", _ocr_mode)  # captured at OCR time

    if not translations:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    img_bgr = pil_to_cv2(pil_img)
    h, w = img_bgr.shape[:2]

    boxes_to_inpaint = []
    items_to_draw = []

    for item in translations:
        text = item.get("translation", "")
        if not text or not text.strip():
            continue
        bbox = item.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 10 or box_h < 10:
            continue

        boxes_to_inpaint.append(bbox)
        items_to_draw.append(item)

    if do_inpaint and boxes_to_inpaint:
        logging.info(f"[Inpaint] Building mask for {len(boxes_to_inpaint)} text regions...")
        mask = build_inpaint_mask(
            img_bgr.shape,
            boxes_to_inpaint,
            padding=2,
            dilate_kernel=3,
        )

        with _inpaint_mode_lock:
            inpaint_mode = _inpaint_mode
        use_lama = inpaint_mode == "high" or SimpleLama is not None
        img_bgr = await inpaint_image_async(img_bgr, mask, use_lama=use_lama)
        logging.info(f"[Inpaint] Inpainting complete for {len(boxes_to_inpaint)} regions.")

    orig_bgr = pil_to_cv2(pil_img)

    out_pil = cv2_to_pil(img_bgr)
    draw = ImageDraw.Draw(out_pil)

    with _font_config_lock:
        fp = str(_current_font_path)

    # --- Detect colors for ALL boxes at once with global polarity voting ---
    # This ensures visually similar regions (e.g. multiple boxes inside one
    # speech bubble) get the same text color instead of flipping black/white.
    all_bboxes_for_color = [item["bbox"] for item in items_to_draw]
    all_box_colors = detect_text_colors_batch(orig_bgr, all_bboxes_for_color)
    color_by_idx = {i: all_box_colors[i] for i in range(len(items_to_draw))}

    # ----- Lens-specific text sizing rules -----
    # Lens bboxes are tight around text, so we let the rendered text
    # slightly overflow by 1px on every side, and we force a readable
    # minimum size for very small boxes (which would otherwise render
    # as blurry tiny text).
    LENS_OVERFLOW_PX          = 1   # text may exceed box by 1px each side
    LENS_SMALL_BOX_THRESHOLD  = 24  # box dim below this => "really small"
    LENS_SMALL_READABLE_SIZE  = 14  # forced readable font size for tiny boxes

    is_lens = (ocr_mode == "lens")

    for item_idx, item in enumerate(items_to_draw):
        text = item["translation"]
        bbox = item["bbox"]
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        text_color, bg_color = color_by_idx[item_idx]

        # -------- Pick font size + lines --------
        if is_lens:
            really_small = (box_w < LENS_SMALL_BOX_THRESHOLD or
                            box_h < LENS_SMALL_BOX_THRESHOLD)

            if really_small:
                # Force a small-but-readable size and let it overflow the box.
                font_size = LENS_SMALL_READABLE_SIZE
                font = get_font(fp, font_size)
                # Allow wrapping if the text is long, but also allow overflow.
                lines = wrap_text(draw, text, font,
                                  max_width=max(box_w, font_size * 3),
                                  allow_break=True, is_vertical=False)
                if not lines:
                    lines = [text]
                heights, _, _ = _measure_block(draw, lines, font)
            else:
                # Normal Lens box: size text so it fills box + 1px overflow each side.
                fit_w = box_w + LENS_OVERFLOW_PX * 2
                fit_h = box_h + LENS_OVERFLOW_PX * 2
                dynamic_max_size = max(96, min(int(fit_h * 0.98),
                                               int(fit_w * 0.95), 320))
                dynamic_min_size = max(10, min(int(box_h * 0.25), 18))
                font_size, lines, heights = fit_font_and_wrap(
                    draw, text, fit_w, fit_h, font_path=fp,
                    max_size=dynamic_max_size, min_size=dynamic_min_size
                )
                font = get_font(fp, font_size)
        else:
            # ----- Non-Lens OCR: original behavior -----
            dynamic_max_size = max(96, min(int(box_h * 0.95), int(box_w * 0.85), 300))
            dynamic_min_size = max(8, min(int(box_h * 0.15), 16))
            font_size, lines, heights = fit_font_and_wrap(
                draw, text, box_w, box_h, font_path=fp,
                max_size=dynamic_max_size, min_size=dynamic_min_size
            )
            font = get_font(fp, font_size)

        # -------- Compute vertical placement --------
        if heights:
            total_text_h = sum(heights)
        else:
            total_text_h = font_size * len(lines)

        if is_lens and not (box_w < LENS_SMALL_BOX_THRESHOLD or
                            box_h < LENS_SMALL_BOX_THRESHOLD):
            # Vertically center within (box + 1px overflow) region
            outer_h = box_h + LENS_OVERFLOW_PX * 2
            start_y = (y1 - LENS_OVERFLOW_PX) + (outer_h - total_text_h) // 2
        else:
            start_y = y1 + (box_h - total_text_h) // 2

        # -------- Draw each line --------
        current_y = start_y
        for i, line in enumerate(lines):
            if not line:
                current_y += heights[i] if i < len(heights) else font_size
                continue

            line_w = draw.textlength(line, font=font)

            if is_lens and not (box_w < LENS_SMALL_BOX_THRESHOLD or
                                box_h < LENS_SMALL_BOX_THRESHOLD):
                # Horizontally center within (box + 1px overflow) region
                outer_w = box_w + LENS_OVERFLOW_PX * 2
                line_x = (x1 - LENS_OVERFLOW_PX) + (outer_w - line_w) / 2
            else:
                line_x = x1 + (box_w - line_w) / 2

            draw_text_with_config(
                draw,
                (line_x, current_y),
                line,
                font=font,
                fill=text_color,
                stroke_fill=bg_color,
            )

            current_y += heights[i] if i < len(heights) else font_size

    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ===========================================================================
# Main entry point
# ===========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)