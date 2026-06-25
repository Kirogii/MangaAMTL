#!/usr/bin/env python3
"""
Manga translation service (FastAPI, OpenAI-style /v1/ endpoints).

- OCR: Hayai OCR (Japanese, with YOLO text box detection) /
       PaddleOCR korean_PP-OCRv5_mobile_rec (Korean, end-to-end).
- Translation: Qwen 0.8B GGUF  (switchable via /v1/changemodel).
- Inpainting: SimpleLama (preferred) with cv2.inpaint fallback.
- Colorizer: Manga Light Colorizer v6 (ONNX) — optional per-request or global.
- Text render: auto-fit binary-search font sizing + per-box ink-color sampling.
- API: /health /version /meta /warmup /setmodel /getmodel
       /v1/translate /v1/translate/{id} /v1/translate/{id}/image
       /v1/changemodel /v1/listmodels /v1/colorize
       /v1/ai/resolve /v1/ai/prompt/default
- UI: Embedded HTML testing interface at /
- Logs: /console endpoint to view all backend logs and errors
"""
from __future__ import annotations

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
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

# --- FastAPI ---------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query, Request, Form
from fastapi.responses import JSONResponse, Response, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Optional deps ---------------------------------------------------------
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from simple_lama_inpainting import SimpleLama
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
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

# --- Sanitization ----
import re

_ALLOWED_RANGES = (
    (0x0020, 0x007E),   # Basic Latin (English)
    (0x00A0, 0x00FF),   # Latin-1 Supplement (Spanish)
    (0x0100, 0x017F),   # Latin Extended-A
    (0x0180, 0x024F),   # Latin Extended-B
    (0x0400, 0x04FF),   # Cyrillic (Russian)
    (0x0500, 0x052F),   # Cyrillic Supplement
    (0x2000, 0x206F),   # General Punctuation
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0x3040, 0x309F),   # Japanese Hiragana
    (0x30A0, 0x30FF),   # Japanese Katakana
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs (Kanji/Hanja)
    (0xAC00, 0xD7AF),   # Korean Hangul Syllables
    (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
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

INPAINT_RADIUS_CV2 = 3

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

_simple_lama_model = None
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

# ===========================================================================
# Image utils (Optimized: no-copy conversion)
# ===========================================================================
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.asarray(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

# ===========================================================================
# Colorizer (ONNX) — Manga Light Colorizer v6
# ===========================================================================
def ensure_colorizer_models():
    import shutil
    
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

def _hf_cache_model_path(repo_id: str, filename: str) -> Optional[pathlib.Path]:
    """Find a specific file in the HF Hub cache.

    Cache layout: <cache>/models--{org}--{repo}/snapshots/{hash}/{filename}
    Snapshot files are symlinks to blobs/.
    """
    cache_dir = _hf_hub_cache_dir()
    if cache_dir is None:
        return None
    org, sep, name = repo_id.partition("/")
    repo_dir_name = f"models--{org}--{name}" if sep else f"models--{name}"
    repo_dir = cache_dir / repo_dir_name
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return None

    # Prefer the commit hash from refs/main, but fall back to any snapshot
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
                return c  # return symlink path; caller can .resolve() if needed
        except OSError:
            continue
    return None


def _scan_hf_cache_for_ggufs() -> List[Dict[str, Any]]:
    """Scan the HF Hub cache directory for all valid .gguf files."""
    models: List[Dict[str, Any]] = []
    cache_dir = _hf_hub_cache_dir()
    if cache_dir is None:
        return models
    for repo_dir in cache_dir.iterdir():
        if not repo_dir.is_dir() or not repo_dir.name.startswith("models--"):
            continue
        # Reconstruct repo_id: models--{org}--{name}  =>  org/name
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

def _is_valid_gguf(path: pathlib.Path) -> bool:
    """Return True only if the file exists, is reasonably sized, and starts with
    the GGUF magic header ('GGUF' = 0x46554747 little-endian)."""
    try:
        if not path.exists():
            return False
        if path.stat().st_size < 1024:  # GGUFs are always much larger
            return False
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == b"GGUF"
    except OSError:
        return False

def _hf_hub_cache_dir() -> Optional[pathlib.Path]:
    """Return the HuggingFace Hub local cache root, if it exists."""
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
    """Find a specific file in the HF Hub cache."""
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
    """Scan the HF Hub cache directory for all valid .gguf files."""
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
    
    # FIX: If the repo name already ends with the file stem, avoid doubling the name.
    # E.g., "Manojb__Qwen_Qwen3.5-0.8B-Q4_K_M" + "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf" 
    # becomes "Manojb__Qwen_Qwen3.5-0.8B-Q4_K_M.gguf" instead of duplicating it.
    if repo_clean.lower().endswith(file_stem.lower()):
        safe = f"{repo_clean}.gguf"
    else:
        safe = f"{repo_clean}__{filename}"
        
    return GGUF_DIR / safe

import shutil

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

    # Clean up legacy doubled filenames
    legacy_doubled = GGUF_DIR / f"{local_path.stem}__{filename}"
    if legacy_doubled.exists() and legacy_doubled != local_path:
        logging.warning(f"[GGUF] Removing legacy doubled file to save space: {legacy_doubled}")
        try:
            legacy_doubled.unlink()
        except OSError as e:
            logging.warning(f"[GGUF] Could not remove legacy file: {e}")

    # 1) Check HuggingFace Hub Cache FIRST — use it directly, skip mirroring
    #    (Windows often can't hardlink across drives, so we just use the resolved path)
    hf_cached_path = _hf_cache_model_path(repo_id, filename)
    if hf_cached_path is not None:
        resolved = hf_cached_path.resolve()
        logging.info(f"[GGUF] Using HF cache directly: {resolved}")
        return resolved

    # 2) If our local mirror is already a valid GGUF, use it.
    if _is_valid_gguf(local_path):
        return local_path

    # Otherwise delete the stale/corrupt mirror if present.
    if local_path.exists():
        logging.warning(f"[GGUF] Local mirror {local_path} is missing/invalid — removing it.")
        try:
            local_path.unlink()
        except OSError as e:
            logging.warning(f"[GGUF] Could not remove stale mirror: {e}")

    # 3) Pull from HF Hub via API
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

    # 4) Return the HF cache path directly (skip mirroring to avoid cross-drive issues)
    resolved = cached.resolve()
    logging.info(f"[GGUF] Download complete, using HF cache path: {resolved}")
    return resolved

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

    # 4) Mirror to local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size != cached.stat().st_size:
        try: local_path.unlink()
        except OSError: pass
    if not local_path.exists():
        try:
            os.link(cached, local_path)
            logging.info(f"[GGUF] Hardlinked HF cache -> {local_path}")
        except OSError:
            try:
                shutil.copy2(cached, local_path)
                logging.info(f"[GGUF] Copied HF cache -> {local_path}")
            except OSError as e:
                logging.warning(f"[GGUF] Mirroring failed ({e}); using HF cache path: {cached}")
                return cached

    if not _is_valid_gguf(local_path):
        logging.warning(f"[GGUF] Mirror {local_path} failed validation; using HF cache: {cached}")
        return cached

    return local_path

def list_local_gguf_models() -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    
    # 1. Scan hardcoded local models directory
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
            
    # 2. Scan HuggingFace Hub cache directory
    models.extend(_scan_hf_cache_for_ggufs())

    # Deduplicate by (repo_id, filename) in case it exists in both places
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
# Hayai OCR (Japanese) — Optimized with Parallel Box OCR
# ===========================================================================
_OCR_BOX_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ocr-box")

def get_hayai_ocr():
    global _hayai_ocr_model
    if _hayai_ocr_model is None:
        if HayaiOcr is None:
            raise RuntimeError("hayai-ocr not installed: pip install hayai-ocr")
        logging.info("[Hayai OCR] Loading model (may take a few minutes on first run)...")
        _hayai_ocr_model = HayaiOcr()
        logging.info("[Hayai OCR] Model loaded.")
    return _hayai_ocr_model

def get_yolo():
    global _global_yolo
    if _global_yolo is None:
        ensure_yolo()
        _global_yolo = YOLO(str(YOLO_MODEL_PATH))
    return _global_yolo

def hayai_ocr_with_yolo(pil_img: Image.Image) -> List[Dict[str, Any]]:
    img_bgr = pil_to_cv2(pil_img)
    h, w = img_bgr.shape[:2]
    yolo = get_yolo()

    results = yolo(img_bgr, verbose=False, conf=0.4)
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

    for bbox, text in _OCR_BOX_EXECUTOR.map(_ocr_one, boxes):
        out.append({"text": text, "bbox": bbox})
    return out

# ===========================================================================
# PaddleOCR (Korean)
# ===========================================================================
def get_paddle_ocr():
    global _paddle_ocr_model
    if _paddle_ocr_model is None:
        if PaddleOCR is None:
            raise RuntimeError("paddleocr not installed: pip install paddleocr paddlepaddle")
        logging.info("[PaddleOCR] Loading Korean model (korean_PP-OCRv5_mobile_rec)...")
        for attempt_kwargs in [
            dict(lang='korean', use_textline_orientation=False,
                 use_doc_orientation_classify=False, use_doc_unwarping=False,
                 text_rec_model_name='korean_PP-OCRv5_mobile_rec'),
            dict(lang='korean', use_textline_orientation=False,
                 use_doc_orientation_classify=False, use_doc_unwarping=False),
            dict(lang='korean', use_angle_cls=False, show_log=False,
                 rec_model_name='korean_PP-OCRv5_mobile_rec'),
            dict(lang='korean'),
        ]:
            try:
                _paddle_ocr_model = PaddleOCR(**attempt_kwargs)
                logging.info(f"[PaddleOCR] Model loaded with kwargs: {attempt_kwargs}")
                return _paddle_ocr_model
            except (ValueError, TypeError) as e:
                logging.warning(f"[PaddleOCR] Failed with {attempt_kwargs}: {e}")
        raise RuntimeError("Failed to initialize PaddleOCR with any known API variant")
    return _paddle_ocr_model

def paddle_ocr_korean(pil_img: Image.Image) -> List[Dict[str, Any]]:
    img_bgr = pil_to_cv2(pil_img)
    paddle = get_paddle_ocr()
    try:
        result = paddle.ocr(img_bgr, cls=False)
    except (ValueError, TypeError):
        result = paddle.ocr(img_bgr)
    return _parse_paddle_result(result)

def _parse_paddle_result(result) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not result:
        return out
    first = result[0] if isinstance(result, list) else result
    if first is None:
        return out
    if isinstance(first, list):
        for line in first:
            try:
                box_pts, (text, conf) = line
                if not text or not text.strip():
                    continue
                xs = [p[0] for p in box_pts]
                ys = [p[1] for p in box_pts]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                out.append({"text": text.strip(), "bbox": (x1, y1, x2, y2)})
            except (ValueError, TypeError):
                continue
    elif hasattr(first, 'rec_texts') and hasattr(first, 'rec_polys'):
        texts = first.rec_texts
        polys = first.rec_polys
        for text, poly in zip(texts, polys):
            if not text or not text.strip():
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            out.append({"text": text.strip(), "bbox": (x1, y1, x2, y2)})
    return out

# ===========================================================================
# Qwen GGUF translator (Optimized max_tokens & stops)
# ===========================================================================
LANG_MAP = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "id": "Indonesian",
    "ru": "Russian",
    "es": "Spanish",
    "cz": "Chinese"
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

                # Resolve symlinks — on Windows without Developer Mode these are real files
                try:
                    path = path.resolve()
                except Exception:
                    pass

                if not _is_valid_gguf(path):
                    raise RuntimeError(
                        f"Refusing to load invalid GGUF: {path}. "
                        f"Delete it and restart, or call /v1/changemodel with a valid repo."
                    )

                logging.info(f"[Qwen] loading {path} ...")
                try:
                    _global_qwen = Llama(
                        model_path=str(path),
                        n_ctx=2048,
                        n_threads=max(4, os.cpu_count() or 4),
                        n_gpu_layers=-1,
                        verbose=False,
                    )
                except Exception as e:
                    logging.error(f"[Qwen] Failed to load GGUF from {path}: {e}")
                    raise RuntimeError(
                        f"llama-cpp-python failed to load {path}. "
                        f"This is likely a version mismatch, not a corrupt file. "
                        f"Run: pip uninstall llama-cpp-python -y && "
                        f"pip install llama-cpp-python --extra-index-url "
                        f"https://abetlen.github.io/llama-cpp-python/whl/cpu"
                    )

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
    max_tok = max(16, min(96, len(text) + 16))
    
    llm = get_qwen()
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT.format(lang=lang_name)},
        {"role": "user",   "content": text},
    ]
    with _llm_lock:
        out = llm.create_chat_completion(
            messages=msgs,
            max_tokens=max_tok,
            temperature=0.2,
            top_p=0.9,
            stop=["<|im_end|>", "</s>"],
        )
    try:
        raw = out["choices"][0]["message"]["content"].strip()
        for tok in ("<|im_start|>", "<|im_end|>", "</s>"):
            if tok in raw:
                raw = raw.replace(tok, "")
        return clean_text_for_font(raw)
    except Exception:
        return ""

# ===========================================================================
# Inpainting
# ===========================================================================
def load_lama():
    global _simple_lama_model
    if _simple_lama_model is None and SimpleLama is not None:
        _simple_lama_model = SimpleLama()
    return _simple_lama_model

def lama_inpaint(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sl = load_lama()
    if sl is None:
        raise RuntimeError("SimpleLama unavailable")
    pil_img  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask).convert("L")
    out_pil  = sl(pil_img, pil_mask)
    return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

def cv2_inpaint_fallback(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(img_bgr, mask, INPAINT_RADIUS_CV2, cv2.INPAINT_TELEA)

# ===========================================================================
# Text color detection (Optimized: Median + Threshold vs K-means)
# ===========================================================================
def detect_text_and_bg_colors(img_bgr: np.ndarray, bbox: Tuple[int,int,int,int]
                              ) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    x1, y1, x2, y2 = bbox
    region = img_bgr[max(0, y1):y2, max(0, x1):x2]
    if region.size == 0:
        return (0, 0, 0), (255, 255, 255)

    h_r, w_r = region.shape[:2]
    if h_r > 80 or w_r > 80:
        scale = 80.0 / max(h_r, w_r)
        region = cv2.resize(region, (max(1, int(w_r * scale)), max(1, int(h_r * scale))),
                            interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    bg_val = int(np.median(gray))

    diff = np.abs(gray.astype(np.int16) - bg_val)
    ink_mask = diff > 60
    flat = region.reshape(-1, 3).astype(np.float32)
    ink_flat = ink_mask.reshape(-1)

    if ink_flat.sum() >= 4:
        ink_bgr = flat[ink_flat].mean(axis=0)
        bg_bgr  = flat[~ink_flat].mean(axis=0)
    else:
        bg_bgr = flat.mean(axis=0)
        ink_bgr = np.array([0, 0, 0] if bg_val > 127 else [255, 255, 255], dtype=np.float32)

    def snap(c):
        c = np.asarray(c, dtype=np.float32)
        if np.all(c < 40):  return np.array([0, 0, 0], dtype=np.float32)
        if np.all(c > 215): return np.array([255, 255, 255], dtype=np.float32)
        return c

    bg_bgr, ink_bgr = snap(bg_bgr), snap(ink_bgr)
    bg_lum  = 0.299 * bg_bgr[2]  + 0.587 * bg_bgr[1]  + 0.114 * bg_bgr[0]
    ink_lum = 0.299 * ink_bgr[2] + 0.587 * ink_bgr[1] + 0.114 * ink_bgr[0]
    if abs(bg_lum - ink_lum) < 60:
        ink_bgr = np.array([0, 0, 0] if bg_lum > 127 else [255, 255, 255], dtype=np.float32)

    return (int(ink_bgr[2]), int(ink_bgr[1]), int(ink_bgr[0])), \
           (int(bg_bgr[2]),  int(bg_bgr[1]),  int(bg_bgr[0]))

# ===========================================================================
# Text wrapping & auto-fit (Optimized: Precomputed word widths)
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
    # Use fixed font metrics for line height to prevent glyph overlap.
    # PIL's textbbox varies per line depending on ascenders/descenders, 
    # which causes clipping when lines are stacked.
    try:
        ascent, descent = font.getmetrics()
        line_h = int(ascent + descent)
    except Exception:
        line_h = int(font.size * 1.2)
    
    # Add a tiny bit of breathing room just in case the font has tight internal leading
    line_h = max(line_h, int(font.size * 1.1))
    
    heights = [line_h] * len(lines)
    total_h = line_h * len(lines)
    max_w = 0.0
    for ln in lines:
        w = draw.textlength(ln, font=font)
        if w > max_w: max_w = w
    return heights, total_h, max_w

def fit_font_and_wrap(draw, text, box_w, box_h,
                      font_path=str(FONT_PATH), max_size=96, min_size=8, is_vertical=False):
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
            logging.warning(f"Vertical text could not fit cleanly even at min_size={min_size} in box ({box_w}x{box_h}).")

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
        logging.warning(f"Text could not fit cleanly even at min_size={min_size} in box ({box_w}x{box_h}).")
    return best_size, best_lines, best_heights

def draw_text_outline(draw, pos, text, font, fill, outline, outline_width):
    x, y = pos
    draw.text((x, y), text, font=font, fill=fill,
              stroke_width=outline_width, stroke_fill=outline)

# ===========================================================================
# Core pipeline (Highly Concurrent Inpainting, Translation, Render Prep)
# ===========================================================================
_BOX_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="box-prep")
_tls = threading.local()

def _tls_draw():
    d = getattr(_tls, "draw", None)
    if d is None:
        # Using a larger dummy canvas prevents any potential internal PIL clipping metrics
        d = ImageDraw.Draw(Image.new("RGBA", (2048, 2048)))
        _tls.draw = d
    return d

def _prep_one_box(c, trans, inpainted_bgr, target_lang):
    x1, y1, x2, y2 = c["bbox"]
    trans = clean_text_for_font(trans)
    if not trans.strip():
        return None
    
    text_rgb, outline_rgb = detect_text_and_bg_colors(inpainted_bgr, (x1, y1, x2, y2))
    box_w, box_h = x2 - x1, y2 - y1
    draw = _tls_draw()
    
    font_size, lines, heights = fit_font_and_wrap(
        draw, trans, box_w, box_h, str(FONT_PATH),
        is_vertical=(target_lang in ("ja", "ko", "cz"))
    )
    return (c["bbox"], trans, text_rgb, outline_rgb, box_w, box_h,
            font_size, lines, heights)

def _colorize_sam_precompute(pil_img):
    session, sam_session = get_colorizer_sessions()
    gray = np.array(pil_img.convert("L"))
    L_bw = cv2.resize(gray, (COLORIZER_DEFAULT_INFER_SIZE, COLORIZER_DEFAULT_INFER_SIZE), 
                      interpolation=cv2.INTER_AREA)
    L_norm = (L_bw.astype(np.float32) / 127.5) - 1.0
    if sam_session is not None:
        return _extract_sam_features_onnx(sam_session, L_norm)
    H_in, W_in = L_bw.shape
    return (np.zeros((1, 256, H_in//16, W_in//16), dtype=np.float32),
            np.zeros((1, 256, H_in//32, W_in//32), dtype=np.float32),
            np.zeros((1, 1024), dtype=np.float32))

def _colorize_generate_only(inpainted_pil, sam_features):
    session, _ = get_colorizer_sessions()
    gray = np.array(inpainted_pil.convert("L"))
    orig_H, orig_W = gray.shape
    L_bw = cv2.resize(gray, (COLORIZER_DEFAULT_INFER_SIZE, COLORIZER_DEFAULT_INFER_SIZE), 
                      interpolation=cv2.INTER_AREA)
    
    sam_level0, sam_level1, wd14_embedding = sam_features
    rgb_output = _colorize_onnx(session, L_bw, sam_level0, sam_level1, wd14_embedding)
    rgb_output = cv2.resize(rgb_output, (orig_W, orig_H), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(rgb_output)

async def detect_translate_inpaint(pil_img: Image.Image,
                                   use_lama: bool = True,
                                   colorize: bool = False,
                                   target_lang: str = "en") -> Tuple[Image.Image, List[Dict]]:
    img_bgr = pil_to_cv2(pil_img)
    h, w = img_bgr.shape[:2]
    loop = asyncio.get_running_loop()

    is_vertical = target_lang in ("ja", "ko", "cz")

    with _ocr_model_lock:
        current_model = _current_ocr_model

    if current_model == "ko":
        logging.info("Using PaddleOCR for Korean...")
        blocks = await loop.run_in_executor(None, paddle_ocr_korean, pil_img)
    else:
        logging.info("Using Hayai OCR + YOLO for Japanese...")
        blocks = await loop.run_in_executor(None, hayai_ocr_with_yolo, pil_img)

    cleaned = []
    for b in blocks:
        x1, y1, x2, y2 = b["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        cleaned.append({"text": (b.get("text") or "").strip(), "bbox": (x1, y1, x2, y2)})
    cleaned.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    if not cleaned:
        if colorize and ort is not None:
            try:
                pil_img = colorize_pil(pil_img)
            except Exception as e:
                logging.error(f"Colorization failed: {e}")
        return pil_img, []

    sam_future = None
    if colorize:
        if ort is None:
            raise RuntimeError("onnxruntime not installed but colorization requested.")
        sam_future = loop.run_in_executor(None, _colorize_sam_precompute, pil_img)

    def _do_inpaint():
        mask = np.zeros((h, w), dtype=np.uint8)
        for c in cleaned:
            x1, y1, x2, y2 = c["bbox"]
            pad = max(2, int(min(x2 - x1, y2 - y1) * 0.06))
            y0, y1p = max(0, y1 - pad), min(h, y2 + pad)
            x0, x1p = max(0, x1 - pad), min(w, x2 + pad)
            mask[y0:y1p, x0:x1p] = 255
        mask_area = int(mask.sum() // 255)
        try:
            if use_lama and SimpleLama:
                return lama_inpaint(img_bgr, mask)
            if mask_area > 0.25 * h * w:
                logging.warning("Mask too large for cv2.inpaint; solid-filling boxes.")
                filled = img_bgr.copy()
                for c in cleaned:
                    x1, y1, x2, y2 = c["bbox"]
                    _, bg_rgb = detect_text_and_bg_colors(img_bgr, (x1, y1, x2, y2))
                    cv2.rectangle(filled, (x1, y1), (x2, y2),
                                  (bg_rgb[2], bg_rgb[1], bg_rgb[0]), -1)
                return filled
            return cv2_inpaint_fallback(img_bgr, mask)
        except Exception as e:
            logging.error(f"Inpainting failed, falling back to cv2: {e}")
            return cv2_inpaint_fallback(img_bgr, mask)

    async def _do_translation():
        texts = [c["text"] for c in cleaned]
        tasks = []
        for t in texts:
            if not t.strip():
                tasks.append(asyncio.sleep(0, result=""))
            else:
                tasks.append(loop.run_in_executor(_llm_executor, qwen_translate, t, target_lang))
        return await asyncio.gather(*tasks)

    inpaint_task = loop.run_in_executor(None, _do_inpaint)
    translation_task = _do_translation()

    futures = [inpaint_task, translation_task]
    if sam_future:
        futures.append(sam_future)
    
    try:
        results = await asyncio.wait_for(asyncio.gather(*futures), timeout=120.0)
        inpainted, translations = results[0], results[1]
        sam_features = results[2] if sam_future else None
    except asyncio.TimeoutError:
        logging.error("Pipeline timed out after 120s")
        raise RuntimeError("Translation pipeline timed out")

    if colorize and sam_features is not None:
        try:
            inpainted_pil = cv2_to_pil(inpainted)
            colorized_pil = await loop.run_in_executor(
                None, _colorize_generate_only, inpainted_pil, sam_features
            )
            inpainted = pil_to_cv2(colorized_pil)
            logging.info("Colorization applied to inpainted image.")
        except Exception as e:
            logging.error(f"Colorization failed, using non-colorized: {e}")
            raise RuntimeError(f"Colorization ONNX failed: {e}")

    base_pil = cv2_to_pil(inpainted).convert("RGBA")
    out = base_pil.copy()
    draw = ImageDraw.Draw(out)

    if not hasattr(fit_font_and_wrap, '_cache'):
        fit_font_and_wrap._cache = {}
    cache = fit_font_and_wrap._cache

    prep_args = [(c, trans, inpainted, target_lang) for c, trans in zip(cleaned, translations)]
    prepped = list(_BOX_EXECUTOR.map(lambda args: _prep_one_box(*args), prep_args))

    boxes_info: List[Dict] = []
    
    for p, c in zip(prepped, cleaned):
        if p is None:
            boxes_info.append({
                "bbox": c["bbox"], "orig": c["text"], "trans": "",
                "font_size": 0, "text_color": None, "outline_color": None
            })
            continue
            
        (bbox, trans, text_rgb, outline_rgb, 
         box_w, box_h, font_size, lines, heights) = p
        x1, y1, x2, y2 = bbox
        
        key = (str(FONT_PATH), font_size)
        if key not in cache:
            try: cache[key] = ImageFont.truetype(str(FONT_PATH), font_size)
            except Exception: cache[key] = ImageFont.load_default()
        font = cache[key]
        
        outline_w = max(1, int(font_size * 0.08))

        if is_vertical:
            col_widths = heights
            total_w = sum(col_widths)
            cur_x_right = x2 - max(0, (box_w - total_w) // 2)

            for col, col_w in zip(lines, col_widths):
                col_total_h = 0
                char_heights = []
                for ch in col:
                    bb_ch = draw.textbbox((0,0), ch, font=font)
                    ch_h = (bb_ch[3] - bb_ch[1]) * 1.2
                    if ch_h == 0: ch_h = font_size
                    char_heights.append(ch_h)
                    col_total_h += ch_h

                cur_y = y1 + max(0, (box_h - col_total_h) // 2)
                cur_x_left = cur_x_right - col_w

                for ch, ch_h in zip(col, char_heights):
                    bb_ch = draw.textbbox((0,0), ch, font=font)
                    ch_w = bb_ch[2] - bb_ch[0]
                    pos_x = cur_x_left + max(0, (col_w - ch_w) / 2)
                    draw_text_outline(draw, (pos_x, cur_y), ch, font,
                                      fill=text_rgb, outline=outline_rgb, outline_width=outline_w)
                    cur_y += ch_h
                cur_x_right = cur_x_left
        else:
            total_h = sum(heights)
            cur_y = y1 + max(0, (box_h - total_h) // 2)
            for ln, hln in zip(lines, heights):
                tw = draw.textlength(ln, font=font)
                pos_x = x1 + max(0, int((box_w - tw) // 2))
                draw_text_outline(draw, (pos_x, cur_y), ln, font,
                                  fill=text_rgb, outline=outline_rgb, outline_width=outline_w)
                cur_y += hln

        boxes_info.append({
            "bbox": bbox, "orig": c["text"], "trans": trans,
            "font_size": font_size, "text_color": text_rgb, "outline_color": outline_rgb,
        })
        
    return out, boxes_info

# ===========================================================================
# Job queue / worker (Optimized PNG Compression)
# ===========================================================================
def _cleanup_old_jobs():
    now = time.time()
    to_remove = [
        jid for jid, j in _jobs.items()
        if j["status"] in ("done", "error")
        and now - j.get("completed_at", j.get("created_at", now)) > 600
    ]
    for jid in to_remove:
        _jobs.pop(jid, None)

async def job_worker():
    global _job_queue
    while True:
        _cleanup_old_jobs()
        job_id = await _job_queue.get()
        job = _jobs.get(job_id)
        if not job:
            continue
        job["status"] = "running"
        job["started_at"] = time.time()
        try:
            pil = job["_pil"]
            use_lama = job["_use_lama"]
            colorize = job.get("_colorize", False)
            target_lang = job.get("_target_lang", "en")
            
            result_img, boxes = await detect_translate_inpaint(
                pil, use_lama=use_lama, colorize=colorize, target_lang=target_lang
            )
            buf = io.BytesIO()
            try:
                result_img.convert("RGB").save(buf, format="PNG", optimize=False, compress_level=1)
            except Exception:
                buf = io.BytesIO()
                result_img.convert("RGB").save(buf, format="PNG")
            job["image_bytes"] = buf.getvalue()
            job["boxes"] = boxes
            job["status"] = "done"
            job["completed_at"] = time.time()
            logging.info(f"Job {job_id} completed successfully.")
        except Exception as e:
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            job["traceback"] = traceback.format_exc()
            logging.error(f"Job {job_id} failed: {e}\n{traceback.format_exc()}")
        finally:
            job.pop("_pil", None)

@app.on_event("startup")
async def _start_worker():
    global _job_queue, _worker_task
    _job_queue = asyncio.Queue()
    _worker_task = asyncio.create_task(job_worker())
    
    logging.info("[Startup] Preloading models in background...")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_llm_executor, get_qwen)
    loop.run_in_executor(None, get_yolo)
    if _current_ocr_model == "ko":
        loop.run_in_executor(None, get_paddle_ocr)
    else:
        loop.run_in_executor(None, get_hayai_ocr)

# ===========================================================================
# API models
# ===========================================================================
class TranslateRequest(BaseModel):
    image_b64: Optional[str] = None
    use_lama: bool = True
    lang: str = DEFAULT_LANG
    source: Optional[str] = None
    colorize: bool = False

class AIResolveRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    model_list: Optional[List[str]] = None

class ChangeModelRequest(BaseModel):
    repo_id: str
    filename: Optional[str] = None

# ===========================================================================
# Embedded HTML Testing UI
# ===========================================================================
@app.get("/", response_class=HTMLResponse)
async def root_ui():
    return r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Manga Translator Testing UI</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }
            h2 { color: #333; }
            .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .upload-box { margin-bottom: 20px; display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
            .images { display: flex; gap: 20px; flex-wrap: wrap; }
            .image-block { flex: 1; min-width: 300px; }
            img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; display: none; }
            #status { margin: 10px 0; font-weight: bold; color: #0066cc; }
            pre { background: #eee; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; }
            button { padding: 10px 20px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0052a3; }
            input[type="file"] { margin-right: 10px; }
            .switcher { margin-bottom: 15px; padding: 10px; background: #eef; border-radius: 5px; }
            .switcher label { margin-right: 10px; font-weight: bold; }
            .switcher input { margin-right: 5px; }
            .model-row { display: flex; gap: 8px; align-items: center; margin-top: 8px; flex-wrap: wrap; }
            select, input[type="text"] { padding: 6px; border-radius: 4px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Manga Translator API - Testing UI</h2>

            <div class="switcher">
                <label>OCR Model:</label>
                <input type="radio" id="ocrJa" name="ocrModel" value="ja" checked onchange="switchModel()">
                <label for="ocrJa">Japanese (Hayai+YOLO)</label>
                <input type="radio" id="ocrKo" name="ocrModel" value="ko" onchange="switchModel()" style="margin-left: 20px;">
                <label for="ocrKo">Korean (PaddleOCR)</label>
            </div>

            <div class="switcher">
                <label>Colorize:</label>
                <input type="checkbox" id="colorizeChk"> <label for="colorizeChk">Enable manga colorization (ONNX v6)</label>
                <span id="colorizeStatus" style="margin-left:10px; color:#666;"></span>
            </div>

            <div class="switcher">
                <label>Translation GGUF Model:</label>
                <div class="model-row">
                    <select id="ggufSelect" style="min-width:350px;"></select>
                    <button onclick="loadModelList()">Refresh List</button>
                    <button onclick="changeModel()">Switch Model</button>
                    <input type="text" id="customRepo" placeholder="repo_id (e.g. hugging-quants/Llama-3.2-1B-Instruct-GGUF)" style="min-width:300px;">
                    <input type="text" id="customFile" placeholder="filename (leave blank to auto-find)" style="min-width:220px;">
                    <button onclick="changeModelCustom()">Switch (custom)</button>
                </div>
                <div id="modelStatus" style="margin-top:6px; color:#666;"></div>
            </div>

            <div class="upload-box">
                <input type="file" id="fileInput" accept="image/*">
                <button onclick="translateImage()">Translate</button>
            </div>
            <div id="status"></div>
            <div class="images">
                <div class="image-block">
                    <h3>Original</h3>
                    <img id="origImg" alt="Original Image">
                </div>
                <div class="image-block">
                    <h3>Translated</h3>
                    <img id="transImg" alt="Translated Image">
                </div>
            </div>
            <h3>Debug Data (Boxes & Text)</h3>
            <pre id="boxesInfo">No data yet.</pre>
        </div>

        <script>
            async function switchModel() {
                const selected = document.querySelector('input[name="ocrModel"]:checked').value;
                try {
                    const res = await fetch(`/setmodel?model=${selected}`, { method: "POST" });
                    const data = await res.json();
                    console.log("Switched OCR model:", data);
                } catch (e) {
                    console.error("Failed to switch OCR model:", e);
                }
            }

            async function loadModelList() {
                const sel = document.getElementById('ggufSelect');
                const status = document.getElementById('modelStatus');
                status.innerText = "Fetching models...";
                try {
                    const res = await fetch('/v1/listmodels');
                    const data = await res.json();
                    sel.innerHTML = '';
                    if (data.models && data.models.length > 0) {
                        data.models.forEach(m => {
                            const opt = document.createElement('option');
                            opt.value = m.name;
                            opt.textContent = `${m.repo_id}/${m.filename} (${m.size_mb} MB)`;
                            opt.dataset.repo = m.repo_id;
                            opt.dataset.file = m.filename;
                            sel.appendChild(opt);
                        });
                        status.innerText = "Models loaded.";
                    } else {
                        status.innerText = "No local models found.";
                    }
                } catch (e) {
                    status.innerText = "Error loading models.";
                    console.error(e);
                }
            }

            async function changeModel() {
                const sel = document.getElementById('ggufSelect');
                const status = document.getElementById('modelStatus');
                if (!sel.value) {
                    alert("Select a model first or refresh list.");
                    return;
                }
                const opt = sel.options[sel.selectedIndex];
                const repo = opt.dataset.repo;
                const file = opt.dataset.file;
                status.innerText = `Switching to ${repo}/${file}...`;
                try {
                    const res = await fetch('/v1/changemodel', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({repo_id: repo, filename: file})
                    });
                    const data = await res.json();
                    if (res.ok) {
                        status.innerText = `Active model: ${data.repo_id}/${data.filename}`;
                    } else {
                        status.innerText = `Error: ${data.detail || 'Failed'}`;
                    }
                } catch (e) {
                    status.innerText = "Request failed.";
                }
            }

            async function changeModelCustom() {
                const repo = document.getElementById('customRepo').value.trim();
                const file = document.getElementById('customFile').value.trim();
                const status = document.getElementById('modelStatus');
                if (!repo) {
                    alert("Provide at least the repo_id.");
                    return;
                }
                status.innerText = `Downloading/Switching to ${repo}/${file || 'auto'}...`;
                try {
                    const res = await fetch('/v1/changemodel', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({repo_id: repo, filename: file || null})
                    });
                    const data = await res.json();
                    if (res.ok) {
                        status.innerText = `Active model: ${data.repo_id}/${data.filename}`;
                        loadModelList(); // refresh dropdown
                    } else {
                        status.innerText = `Error: ${data.detail || 'Failed'}`;
                    }
                } catch (e) {
                    status.innerText = "Request failed.";
                }
            }

            async function translateImage() {
                const fileInput = document.getElementById('fileInput');
                const statusDiv = document.getElementById('status');
                const origImg = document.getElementById('origImg');
                const transImg = document.getElementById('transImg');
                const boxesInfo = document.getElementById('boxesInfo');
                const colorizeChk = document.getElementById('colorizeChk').checked;

                if (!fileInput.files[0]) {
                    alert("Please select an image file.");
                    return;
                }

                statusDiv.innerText = "Uploading...";
                origImg.src = URL.createObjectURL(fileInput.files[0]);
                origImg.style.display = "block";
                transImg.style.display = "none";
                boxesInfo.textContent = "Processing...";

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                formData.append("colorize", colorizeChk);

                try {
                    const upRes = await fetch("/v1/translate/upload", { method: "POST", body: formData });
                    if (!upRes.ok) {
                        const errBody = await upRes.text();
                        statusDiv.innerText = `Upload failed (${upRes.status})`;
                        boxesInfo.textContent = errBody;
                        return;
                    }
                    const upData = await upRes.json();
                    const jobId = upData.id;
                    if (!jobId) {
                        statusDiv.innerText = "Upload returned no job ID";
                        boxesInfo.textContent = JSON.stringify(upData, null, 2);
                        return;
                    }
                    
                    statusDiv.innerText = `Job ${jobId} queued. Polling...`;
                    
                    let done = false;
                    let pollCount = 0;
                    const MAX_POLLS = 30;
                    
                    while (!done && pollCount < MAX_POLLS) {
                        pollCount++;
                        const pollRes = await fetch(`/v1/translate/${jobId}?wait=20`);
                        const pollData = await pollRes.json();
                        
                        if (pollData.status === "done") {
                            done = true;
                            transImg.src = pollData.image_url + "?t=" + Date.now();
                            transImg.style.display = "block";
                            statusDiv.innerText = "Done!";
                            boxesInfo.textContent = JSON.stringify(pollData.boxes, null, 2);
                        } else if (pollData.status === "error") {
                            done = true;
                            statusDiv.innerText = "Error: " + pollData.error;
                            boxesInfo.textContent = "Error details:\n" + (pollData.error || "Unknown error");
                        } else {
                            statusDiv.innerText = `Status: ${pollData.status}. Polling again...`;
                        }
                    }
                    
                    if (!done) {
                        statusDiv.innerText = "Timed out waiting for result.";
                        boxesInfo.textContent = "The request timed out after " + MAX_POLLS + " polls.";
                    }
                } catch (e) {
                    statusDiv.innerText = "Request failed: " + e.message;
                    boxesInfo.textContent = e.stack;
                }
            }

            // Initial load
            window.onload = () => {
                loadModelList();
            };
        </script>
    </body>
    </html>
    """

# ===========================================================================
# Console / Logs
# ===========================================================================
@app.get("/console", response_class=PlainTextResponse)
async def console_logs():
    """Return all captured backend logs for debugging and error reporting."""
    return "\n".join(log_handler.get_logs())

# ===========================================================================
# Model Switching Endpoints
# ===========================================================================
@app.post("/setmodel")
async def set_model(model: str = Query(..., pattern="^(ja|ko)$")):
    """Change the current OCR model. Accepts 'ja' or 'ko'."""
    global _current_ocr_model
    with _ocr_model_lock:
        _current_ocr_model = model
    logging.info(f"OCR model switched to: {model}")
    return {"status": "ok", "current_model": _current_ocr_model}

@app.get("/getmodel")
async def get_model():
    """Get the current OCR model being used."""
    return {"current_model": _current_ocr_model}

# ===========================================================================
# Root-level endpoints
# ===========================================================================
@app.post("/reloadfont")
async def reload_font():
    """Clear the font cache so a swapped TTF is picked up."""
    clear_font_cache()
    return {"status": "ok", "font": str(FONT_PATH), "exists": FONT_PATH.exists()}

@app.get("/health")
async def health():
    return {"status": "ok", "ts": time.time()}

@app.get("/version")
async def version():
    return {"version": "1.0.0", "build": BUILD_ID}

@app.get("/meta")
async def meta():
    return {
        "languages": [
            {"code": "en", "label": "English"},
            {"code": "ja", "label": "Japanese (source)"},
            {"code": "ko", "label": "Korean (source)"},
        ],
        "sources": ["hayai-yolo", "paddleocr"],
        "server_ai_key": False,
        "translation_model": _current_qwen_repo_id,
        "ocr_backend": "switchable",
        "current_ocr_model": _current_ocr_model,
        "colorize_enabled": _colorize_enabled,
    }

@app.get("/warmup")
async def warmup(lang: Optional[str] = None):
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_llm_executor, get_qwen)
    return {"warmed": True, "lang": lang or DEFAULT_LANG}

# ===========================================================================
# /v1/* endpoints (OpenAI-style)
# ===========================================================================
@app.post("/v1/translate")
async def v1_translate(
    req: TranslateRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    if not req.image_b64:
        raise HTTPException(400, "image_b64 required")
    try:
        raw = base64.b64decode(req.image_b64)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"bad image: {e}")

    if idempotency_key and idempotency_key in _jobs:
        existing = _jobs[idempotency_key]
        if existing["status"] in ("queued", "running", "done", "error"):
            return {"id": idempotency_key, "status": existing["status"],
                    "hint": "idempotent reuse"}

    job_id = idempotency_key or uuid.uuid4().hex
    _jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "_pil": pil,
        "_use_lama": req.use_lama,
        "_colorize": req.colorize or _colorize_enabled,
        "_target_lang": req.lang,
    }
    if _job_queue is None:
        raise HTTPException(503, "Server is still starting up, please try again in a moment.")
    await _job_queue.put(job_id)
    return {"id": job_id, "status": "queued", "hint": "poll /v1/translate/{id}?wait=N"}

@app.post("/v1/translate/upload")
async def v1_translate_upload(
    file: UploadFile = File(...),
    use_lama: str = Form("true"),
    lang: str = Form("en"),
    colorize: str = Form("false"),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    use_lama_bool = use_lama.lower() in ("true", "1", "on", "yes")
    colorize_bool = colorize.lower() in ("true", "1", "on", "yes")
    
    if lang not in LANG_MAP:
        lang = "en"
    
    raw = await file.read()
    logging.info(f"[Upload] Received lang='{lang}', colorize={colorize_bool}")

    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"bad image: {e}")
    
    if (colorize_bool or _colorize_enabled) and ort is None:
        raise HTTPException(500, "Colorization requested but 'onnxruntime' is not installed.")

    job_id = idempotency_key or uuid.uuid4().hex
    _jobs[job_id] = {
        "id": job_id, "status": "queued", "created_at": time.time(),
        "_pil": pil, "_use_lama": use_lama_bool,
        "_colorize": colorize_bool or _colorize_enabled,
        "_target_lang": lang,
    }
    if _job_queue is None:
        raise HTTPException(503, "Server is still starting up, please try again in a moment.")
    await _job_queue.put(job_id)
    return {"id": job_id, "status": "queued"}

@app.get("/v1/translate/{job_id}")
async def v1_translate_status(job_id: str, wait: float = Query(0, ge=0, le=25)):
    if job_id not in _jobs:
        raise HTTPException(404, "job not found")
    job = _jobs[job_id]
    deadline = time.time() + wait
    while wait > 0 and job["status"] in ("queued", "running") and time.time() < deadline:
        await asyncio.sleep(0.1)
        job = _jobs[job_id]
    resp = {"id": job_id, "status": job["status"]}
    if job["status"] == "done":
        b64 = base64.b64encode(job["image_bytes"]).decode("ascii")
        resp.update({
            "image_b64": b64,
            "image_url": f"/v1/translate/{job_id}/image",
            "boxes": [
                {
                    "bbox": b["bbox"],
                    "orig": b["orig"],
                    "trans": b["trans"],
                    "font_size": b["font_size"],
                    "text_color": b["text_color"],
                    "outline_color": b["outline_color"],
                } for b in job["boxes"]
            ],
            "completed_at": job.get("completed_at"),
        })
    elif job["status"] == "error":
        resp["error"] = job.get("error")
    return resp

@app.get("/v1/translate/{job_id}/image")
async def v1_translate_image(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "job not found")
    job = _jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(409, f"job status={job['status']}")
    return Response(
        content=job["image_bytes"], 
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=600, immutable"}
    )

# ===========================================================================
# GGUF Model Management Endpoints
# ===========================================================================
@app.post("/v1/changemodel")
async def v1_change_model(req: ChangeModelRequest):
    """Switch the active Qwen GGUF translation model."""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, switch_qwen_model, req.repo_id, req.filename)
        return {"status": "ok", "repo_id": req.repo_id, "filename": _current_qwen_filename}
    except Exception as e:
        logging.error(f"Failed to switch model: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, detail=str(e))

@app.get("/v1/listmodels")
async def v1_list_models():
    """List all locally downloaded GGUF models available for translation."""
    return {"models": list_local_gguf_models()}

# ===========================================================================
# Colorize Global Toggle Endpoint
# ===========================================================================
@app.post("/v1/colorize")
async def v1_toggle_colorize(enable: bool = Query(...)):
    """Globally enable or disable the manga colorizer for all future translations."""
    global _colorize_enabled
    _colorize_enabled = enable
    logging.info(f"Global colorization toggled to: {enable}")
    return {"status": "ok", "colorize_enabled": _colorize_enabled}

# ===========================================================================
# AI Resolve / Prompt Endpoints
# ===========================================================================
@app.post("/v1/ai/resolve")
async def v1_ai_resolve(req: AIResolveRequest):
    return {
        "provider": req.provider or "qwen-gguf",
        "model":    req.model or _current_qwen_repo_id,
        "model_list": req.model_list or [_current_qwen_repo_id],
        "resolved": True,
    }

@app.get("/v1/ai/prompt/default")
async def v1_ai_prompt_default(lang: str = DEFAULT_LANG):
    return {
        "lang": lang,
        "system": SYSTEM_PROMPT,
        "user_template": "{text}",
    }

# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
