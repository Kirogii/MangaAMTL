#!/usr/bin/env python3
"""
- API: /health /version /meta /warmup /setmodel /getmodel
       /v1/translate /v1/translate/{id} /v1/translate/{id}/image
       /v1/changemodel /v1/listmodels /v1/colorize
       /v1/ai/resolve /v1/ai/prompt/default
       /SetFont /GetFont /SetModelType /GetModelType /SetOpenRouterModel
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
from fastapi.responses import JSONResponse, Response, HTMLResponse, PlainTextResponse
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

INPAINT_RADIUS_CV2 = 7  # Increased from 5
INPAINT_TELEA_RADIUS = 10
INPAINT_NS_RADIUS = 7
INPAINT_DILATE_PASSES = 2
INPAINT_FEATHER_PX = 3
INPAINT_USE_MULTI_PASS = True
INPAINT_COLOR_MATCH = True

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

# Inpainting executor (runs in thread pool since it can be slow)
_inpaint_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inpaint")
_inpaint_lock = threading.Lock()

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
    results = yolo(img_bgr, verbose=False, conf=0.4, device=get_torch_device())
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
    max_tok = max(16, min(96, len(text) + 16))
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
        logging.info(f"[LLM] Translated to: '{raw[:40]}'")
        return clean_text_for_font(raw)
    except Exception as e:
        logging.error(f"[LLM] Translation failed: {e}")
        return ""

# ===========================================================================
# OpenRouter Translation
# ===========================================================================

async def openrouter_translate_batch(texts: List[str], target_lang: str = "en", max_retries: int = 5) -> List[str]:
    """Translate a list of texts in a SINGLE OpenRouter API call using a numbered list format."""
    import aiohttp
    import random

    with _model_type_lock:
        api_key = _openrouter_api_key
        model = _openrouter_model

    if not api_key:
        logging.error("[OpenRouter] API key not configured")
        return [""] * len(texts)

    # Filter out empty texts but keep their original indices
    indexed_texts = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not indexed_texts:
        return [""] * len(texts)

    lang_name = LANG_MAP.get(target_lang, "English")
    
    # Scale max_tokens based on total input length to prevent truncation
    total_chars = sum(len(t) for _, t in indexed_texts)
    max_tok = max(256, min(4096, total_chars + (len(indexed_texts) * 20)))

    # Build the numbered list prompt
    prompt_lines = [f"{idx + 1}. {text}" for idx, (orig_i, text) in enumerate(indexed_texts)]
    batch_text = "\n".join(prompt_lines)

    batch_system_prompt = (
        f"You are a manga translation engine. Translate the user's numbered list of texts into {lang_name}. "
        "Output ONLY the translated list, one per line, keeping the exact same numbers. "
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
            timeout = aiohttp.ClientTimeout(total=120) # Longer timeout for batches
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

                    # Parse the numbered list back out
                    results = [""] * len(texts)
                    parsed_lines = raw.split('\n')
                    for line in parsed_lines:
                        match = re.match(r"^\s*(\d+)\.\s*(.*)$", line)
                        if match:
                            num = int(match.group(1)) - 1 # Convert to 0-based index
                            trans = match.group(2).strip()
                            # Map back to the original text index
                            if 0 <= num < len(indexed_texts):
                                orig_idx = indexed_texts[num][0]
                                results[orig_idx] = clean_text_for_font(trans)

                    # Verify we got most of them
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
    return [""] * len(texts) # Signal to fallback to sequential

async def openrouter_translate(text: str, target_lang: str = "en", max_retries: int = 5) -> str:
    """Translate text using OpenRouter API with retries and rate-limit handling."""
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
        # Exponential backoff: 2s, 4s, 8s, 16s, 32s
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
                    # Handle rate limiting (429) specially
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

                    # Safely extract content
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
# Inpainting (SimpleLama + cv2 fallback)
# ===========================================================================
def load_lama():
    global _simple_lama_model
    if _simple_lama_model is None and SimpleLama is not None:
        device = get_torch_device()
        logging.info(f"[SimpleLama] Loading on device: {device}")
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

def _lama_inpaint_sync(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Thread-safe SimpleLama inpainting."""
    with _inpaint_lock:
        sl = load_lama()
        if sl is None:
            raise RuntimeError("SimpleLama unavailable")
        pil_img  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask).convert("L")
        out_pil  = sl(pil_img, pil_mask)
        return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

def _cv2_inpaint_sync(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Thread-safe cv2 inpainting."""
    with _inpaint_lock:
        return cv2.inpaint(img_bgr, mask, INPAINT_RADIUS_CV2, cv2.INPAINT_TELEA)

async def inpaint_image_async(img_bgr: np.ndarray, mask: np.ndarray, use_lama: bool = True) -> np.ndarray:
    """Run inpainting in a thread pool so it doesn't block the event loop.
    
    Tries SimpleLama first (if use_lama=True and available), falls back to cv2.inpaint.
    """
    if use_lama and SimpleLama is not None:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(_inpaint_executor, _lama_inpaint_sync, img_bgr, mask)
            logging.info("[Inpaint] SimpleLama inpainting complete.")
            return result
        except Exception as e:
            logging.warning(f"[Inpaint] SimpleLama failed ({e}), falling back to cv2.inpaint")
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_inpaint_executor, _cv2_inpaint_sync, img_bgr, mask)
    logging.info("[Inpaint] cv2.inpaint fallback complete.")
    return result

def build_inpaint_mask(img_shape: Tuple[int, int, int],
                       bboxes: List[Tuple[int, int, int, int]],
                       padding: int = 6,
                       dilate_kernel: int = 7,
                       feather_pixels: int = 3,
                       adaptive_dilate: bool = True) -> np.ndarray:
    """Build a high-quality binary mask from bounding boxes with padding, 
    adaptive dilation, and feathered edges for smoother inpainting.
    
    Args:
        img_shape: (H, W, C) of the image
        bboxes: list of (x1, y1, x2, y2) tuples
        padding: pixels to expand each box
        dilate_kernel: kernel size for morphological dilation
        feather_pixels: pixels to feather the mask edges (gradient)
        adaptive_dilate: if True, scale dilation based on box size
    
    Returns:
        uint8 mask, 255 = inpaint region, 0 = keep
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for x1, y1, x2, y2 in bboxes:
        box_w = x2 - x1
        box_h = y2 - y1
        box_size = max(box_w, box_h)
        
        # Adaptive padding based on text box size
        adaptive_pad = padding + max(0, (box_size - 50) // 20)
        
        # Apply padding, clamped to image bounds
        px1 = max(0, x1 - adaptive_pad)
        py1 = max(0, y1 - adaptive_pad)
        px2 = min(w, x2 + adaptive_pad)
        py2 = min(h, y2 + adaptive_pad)
        mask[py1:py2, px1:px2] = 255
    
    # Multi-pass dilation with different kernels for better coverage
    if dilate_kernel > 0:
        # First pass: small kernel to catch edge details
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        
        # Second pass: larger kernel for broader coverage
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, kernel_large, iterations=1)
        
        # Optional third pass for larger text areas
        if adaptive_dilate:
            kernel_xl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel + 2, dilate_kernel + 2))
            mask = cv2.dilate(mask, kernel_xl, iterations=1)
    
    # Feather edges using Gaussian blur for smoother transitions
    if feather_pixels > 0:
        # Convert to float for smooth gradient
        mask_float = mask.astype(np.float32) / 255.0
        # Apply Gaussian blur to create gradient at edges
        blurred = cv2.GaussianBlur(mask_float, (feather_pixels * 2 + 1, feather_pixels * 2 + 1), 0)
        # Threshold back to binary but with anti-aliased edges
        mask = (blurred * 255).astype(np.uint8)
        # Re-threshold to keep it mostly binary but with smoother edges
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Close small holes in the mask
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    return mask

# ===========================================================================
# Text color detection
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
# SetFont Endpoint
# ===========================================================================
class SetFontRequest(BaseModel):
    font_path: Optional[str] = None
    font_url: Optional[str] = None
    stroke_width: int = 0

@app.post("/SetFont")
async def set_font(req: SetFontRequest):
    global _current_font_path, _current_stroke_width
    with _font_config_lock:
        if req.font_url and req.font_path:
            raise HTTPException(400, "Provide either font_path or font_url, not both")
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
        _current_stroke_width = max(0, min(20, req.stroke_width))
        logging.info(f"[Font] Stroke width set to: {_current_stroke_width}")
    return {"status": "ok", "font_path": str(_current_font_path), "stroke_width": _current_stroke_width}

@app.get("/GetFont")
async def get_font_config():
    with _font_config_lock:
        return {"font_path": str(_current_font_path), "stroke_width": _current_stroke_width}

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
    return {
        "version": BUILD_ID,
        "cuda": has_cuda(),
        "device": get_torch_device(),
        "ocr_model": _current_ocr_model,
        "font_path": str(_current_font_path),
        "stroke_width": _current_stroke_width,
        "model_type": _current_model_type,
        "openrouter_model": _openrouter_model if _current_model_type == "openrouter" else None,
        "local_model": f"{_current_qwen_repo_id}/{_current_qwen_filename}" if _current_model_type == "local" else None,
        "inpaint_lama_available": SimpleLama is not None,
    }

@app.post("/warmup")
async def warmup():
    errors = []
    try:
        get_yolo()
    except Exception as e:
        errors.append(f"YOLO: {e}")
    try:
        get_hayai_ocr()
    except Exception as e:
        errors.append(f"Hayai OCR: {e}")
    try:
        if _current_model_type == "local":
            get_qwen()
    except Exception as e:
        errors.append(f"Qwen: {e}")
    try:
        if SimpleLama is not None:
            load_lama()
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
    if lang == "ko":
        results = glm_ocr_korean(pil_img)
    else:
        results = hayai_ocr_with_yolo(pil_img)
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
    """Background task: OCR -> Translate (Batch for OpenRouter, Sequential for Local)."""
    async with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["status"] = "processing"

    try:
        pil_img = job["image"]
        target_lang = job["target_lang"]
        ocr_lang = job["ocr_lang"]

        # Step 1: OCR
        if ocr_lang == "ko":
            ocr_results = glm_ocr_korean(pil_img)
        else:
            ocr_results = hayai_ocr_with_yolo(pil_img)

        if not ocr_results:
            async with _job_lock:
                job["status"] = "completed"
                job["result"] = {"boxes": [], "translations": []}
            return

        texts_to_translate = [item["text"] for item in ocr_results]
        translations = []

        with _model_type_lock:
            model_type = _current_model_type

        # Step 2: Translate
        if model_type == "openrouter":
            # --- BATCH STRATEGY FOR OPENROUTER ---
            logging.info(f"[Job {job_id}] Using OpenRouter BATCH strategy for {len(texts_to_translate)} boxes.")
            batch_results = await openrouter_translate_batch(texts_to_translate, target_lang)
            
            # Check if batch completely failed or missed items, fallback to sequential for missing ones
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
                # Batch succeeded (even if partially), fill in translations
                for idx, text in enumerate(texts_to_translate):
                    translated = batch_results[idx]
                    
                    # If a specific line failed to parse in the batch, retry it individually
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
            # --- SEQUENTIAL STRATEGY FOR LOCAL GGUF ---
            logging.info(f"[Job {job_id}] Using Local SEQUENTIAL strategy for {len(texts_to_translate)} boxes.")
            loop = asyncio.get_event_loop()
            for idx, text in enumerate(texts_to_translate):
                if not text.strip():
                    translations.append({"text": text, "translation": "", "bbox": ocr_results[idx]["bbox"]})
                    continue
                
                logging.info(f"[Job {job_id}] Translating box {idx + 1}/{len(ocr_results)}: '{text[:40]}'")
                translated = await loop.run_in_executor(_llm_executor, qwen_translate, text, target_lang)
                
                translations.append({
                    "text": text,
                    "translation": translated,
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
    """Generate the final translated image.
    
    Pipeline:
    1. Get OCR boxes + translations from completed job
    2. Build inpaint mask from all text bounding boxes
    3. Inpaint to erase original text (SimpleLama if available, else cv2.inpaint)
    4. Detect text/bg colors from ORIGINAL image (before inpainting)
    5. Overlay translated text with proper sizing and centering
    """
    async with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        if job["status"] != "completed":
            raise HTTPException(400, f"Job {job_id} is not completed (status: {job['status']})")

        pil_img = job["image"]
        translations = job["result"].get("translations", [])
        do_inpaint = job.get("inpaint", True)

    if not translations:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    # Convert original image to cv2 for color detection and inpainting
    img_bgr = pil_to_cv2(pil_img)
    h, w = img_bgr.shape[:2]

    # Collect all bounding boxes that have translations
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

    # --- Step 1: Inpainting (erase original text) ---
    if do_inpaint and boxes_to_inpaint:
        logging.info(f"[Inpaint] Building mask for {len(boxes_to_inpaint)} text regions...")
        mask = build_inpaint_mask(
            img_bgr.shape,
            boxes_to_inpaint,
            padding=4,
            dilate_kernel=5,
        )

        # Use lama if available, otherwise cv2 fallback
        use_lama = SimpleLama is not None
        img_bgr = await inpaint_image_async(img_bgr, mask, use_lama=use_lama)
        logging.info(f"[Inpaint] Inpainting complete for {len(boxes_to_inpaint)} regions.")

    # --- Step 2: Detect colors from ORIGINAL image (before inpainting) ---
    # We use the original for color detection so we get accurate text/bg colors
    orig_bgr = pil_to_cv2(pil_img)

    # --- Step 3: Overlay translated text ---
    out_pil = cv2_to_pil(img_bgr)
    draw = ImageDraw.Draw(out_pil)

    with _font_config_lock:
        fp = str(_current_font_path)

    for item in items_to_draw:
        text = item["translation"]
        bbox = item["bbox"]
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        # Detect colors from the ORIGINAL image at this bbox location
        text_color, bg_color = detect_text_and_bg_colors(orig_bgr, bbox)

        # Fit text to box
        font_size, lines, heights = fit_font_and_wrap(draw, text, box_w, box_h, font_path=fp)
        font = get_font(fp, font_size)

        # Calculate vertical centering
        if heights:
            total_text_h = sum(heights)
        else:
            total_text_h = font_size * len(lines)
        start_y = y1 + (box_h - total_text_h) // 2

        # Draw each line
        current_y = start_y
        for i, line in enumerate(lines):
            if not line:
                current_y += heights[i] if i < len(heights) else font_size
                continue

            line_w = draw.textlength(line, font=font)
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

    # Return the final image
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

# ===========================================================================
# Main entry point
# ===========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
