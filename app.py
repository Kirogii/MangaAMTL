#!/usr/bin/env python3
"""
Manga translation service (FastAPI, OpenAI-style /v1/ endpoints).

- OCR: Hayai OCR (Japanese, with YOLO text box detection) /
       PaddleOCR korean_PP-OCRv5_mobile_rec (Korean, end-to-end).
- Translation: Qwen 0.8B GGUF  (Manojb/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf).
- Inpainting: SimpleLama (preferred) with cv2.inpaint fallback.
- Text render: auto-fit binary-search font sizing + per-box ink-color sampling.
- API: /health /version /meta /warmup /setmodel /getmodel
       /v1/translate /v1/translate/{id} /v1/translate/{id}/image
       /v1/ai/resolve /v1/ai/prompt/default
- UI: Embedded HTML testing interface at / with Korean/Japanese OCR model switches
- Logs: /console endpoint to view all backend logs and errors
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import pathlib
import time
import traceback
import urllib.request
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

# --- FastAPI ---------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, HTMLResponse, PlainTextResponse

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

# --- Sanitization ----
import re

# Unicode ranges your English comic font (AnimeAce / Arial) actually supports.
# Anything outside → drop, so PIL never draws a □.
_ALLOWED_RANGES = (
    (0x0020, 0x007E),   # Basic Latin (ASCII printable)
    (0x00A0, 0x00FF),   # Latin-1 supplement (à, é, ñ, etc.)
    (0x0100, 0x017F),   # Latin Extended-A
    (0x0180, 0x024F),   # Latin Extended-B
    (0x2000, 0x206F),   # General Punctuation (quotes, dashes, ellipsis)
)

# Translate common "smart" Unicode punctuation → ASCII equivalents so we
# keep meaning instead of deleting it.
_PUNCT_MAP = {
    0x2018: "'", 0x2019: "'",   # ‘ ’
    0x201C: '"', 0x201D: '"',   # “ ”
    0x2013: '-', 0x2014: '-',   # – —
    0x2026: '...',              # …
    0x00A0: ' ',                # non-breaking space
    0x2022: '*',                # •
    0x2122: '(TM)', 0x00A9: '(c)', 0x00AE: '(R)',
}

def clean_text_for_font(text: str) -> str:
    """Drop any character the rendering font can't display.

    - Maps smart punctuation → ASCII
    - Keeps Latin / Latin-1 / Latin Extended / general punctuation
    - Drops CJK, Hangul, emoji, music notes, decorative glyphs, etc.
    - Collapses runs of whitespace
    Returns '' if nothing renderable remains (caller should then skip the box).
    """
    if not text:
        return ""

    out = []
    for ch in text:
        cp = ord(ch)
        # Skip control chars (keep tab/newline for safety)
        if cp < 0x20 and ch not in '\t\n':
            continue
        # Smart-punctuation → ASCII
        if cp in _PUNCT_MAP:
            out.append(_PUNCT_MAP[cp])
            continue
        # Keep if inside any allowed range
        if any(lo <= cp <= hi for lo, hi in _ALLOWED_RANGES):
            out.append(ch)
        # else: silently drop (CJK, Hangul, symbols, emoji, etc.)

    result = ''.join(out)
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r'\n+', ' ', result)
    return result.strip()




# --- Config ----------------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
YOLO_MODEL_PATH = MODEL_DIR / "yolo_manga_textbox.pt"
YOLO_HF_RAW = "https://huggingface.co/Kirogii/Yolo-Manga_Textbox-Region_Detect/resolve/main/model.pt"

Qwen_REPO_ID = "Manojb/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
Qwen_MODEL_FILENAME = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"

INPAINT_RADIUS_CV2 = 3

# Font path prioritization
FONT_DIR = ROOT_DIR / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)
FONT_PATH = FONT_DIR / "animeace2_reg.ttf"

if not FONT_PATH.exists():
    logging.warning(f"Custom font {FONT_PATH} not found. Falling back to arial.ttf or PIL default.")
    FONT_PATH = pathlib.Path("arial.ttf")

DEFAULT_LANG       = "en"
BUILD_ID           = "manga-v1-2025.01"

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

# Attach to root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(log_handler)

# Capture uvicorn logs
logging.getLogger("uvicorn").addHandler(log_handler)
logging.getLogger("uvicorn.access").addHandler(log_handler)

# --- Globals ---------------------------------------------------------------
app = FastAPI(title="Manga Translation API", version="1.0.0")

_simple_lama_model = None
_global_yolo       = None
_global_qwen       = None
_hayai_ocr_model   = None
_paddle_ocr_model  = None

# Current OCR model: "ja" (Hayai + YOLO) or "ko" (PaddleOCR)
_current_ocr_model = "ja"
_ocr_model_lock = threading.Lock()

# Job queue
_jobs: Dict[str, Dict[str, Any]] = {}
_job_lock = asyncio.Lock()
_job_queue: Optional[asyncio.Queue] = None  # Initialized on startup
_worker_task = None  # Keep a strong reference to prevent GC

# LLM Concurrency Control
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
# Image utils
# ===========================================================================
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

# ===========================================================================
# Hayai OCR (Japanese) — YOLO detection + Hayai recognition
# ===========================================================================
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
    """
    Japanese OCR pipeline:
      1. YOLO detects manga text-box regions.
      2. Hayai OCR reads each cropped region (supports multi-line, furigana, SFX).
    Returns list of {"text": str, "bbox": (x1,y1,x2,y2)}.
    """
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

    for b in r.boxes:
        xy = b.xyxy[0].cpu().numpy()
        x1, y1 = max(0, int(xy[0])), max(0, int(xy[1]))
        x2, y2 = min(w - 1, int(xy[2])), min(h - 1, int(xy[3]))

        box_area = (x2 - x1) * (y2 - y1)
        # Reject boxes that cover more than 80% of the image or are too small
        if box_area > 0.8 * img_area or box_area < 100:
            continue

        crop = pil_img.crop((x1, y1, x2, y2))
        try:
            text = mocr(crop).strip()
        except Exception as e:
            logging.error(f"Hayai OCR failed on box ({x1},{y1},{x2},{y2}): {e}")
            text = ""

        out.append({"text": text, "bbox": (x1, y1, x2, y2)})
    return out

# ===========================================================================
# PaddleOCR (Korean) — korean_PP-OCRv5_mobile_rec
# ===========================================================================
def get_paddle_ocr():
    global _paddle_ocr_model
    if _paddle_ocr_model is None:
        if PaddleOCR is None:
            raise RuntimeError("paddleocr not installed: pip install paddleocr paddlepaddle")
        logging.info("[PaddleOCR] Loading Korean model (korean_PP-OCRv5_mobile_rec)...")
        
        # PaddleOCR 3.x enables doc orientation/unwarping by default (causing PP-LCNet downloads).
        # We disable them and explicitly request the Korean v5 mobile recognition model.
        for attempt_kwargs in [
            # PaddleOCR 3.x (newest API) - Explicit model names
            dict(
                lang='korean', 
                use_textline_orientation=False, 
                use_doc_orientation_classify=False, 
                use_doc_unwarping=False,
                text_rec_model_name='korean_PP-OCRv5_mobile_rec'
            ),
            # PaddleOCR 3.x (fallback without explicit rec name)
            dict(
                lang='korean', 
                use_textline_orientation=False, 
                use_doc_orientation_classify=False, 
                use_doc_unwarping=False
            ),
            # PaddleOCR 2.x (legacy API)
            dict(
                lang='korean', 
                use_angle_cls=False, 
                show_log=False, 
                rec_model_name='korean_PP-OCRv5_mobile_rec'
            ),
            # Minimal fallback
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
    # PaddleOCR 3.x removed the 'cls' parameter from .ocr()
    try:
        result = paddle.ocr(img_bgr, cls=False)
    except (ValueError, TypeError):
        result = paddle.ocr(img_bgr)
    return _parse_paddle_result(result)

def _parse_paddle_result(result) -> List[Dict[str, Any]]:
    """Parse PaddleOCR result into our standard {text, bbox} format.
    Handles both PaddleOCR 2.x and 3.x result structures.
    """
    out: List[Dict[str, Any]] = []
    if not result:
        return out

    first = result[0] if isinstance(result, list) else result
    if first is None:
        return out

    # --- Case 1: PaddleOCR 2.x — list of [box_points, (text, conf)] ---
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

    # --- Case 2: PaddleOCR 3.x — OCRResult with rec_texts / rec_polys ---
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
# Qwen GGUF translator
# ===========================================================================
SYSTEM_PROMPT = (
    "You are a professional manga translator. Translate the user's Japanese or Korean text into natural, fluent English. "
    "If the text is already in English, or is a single character, or is meaningless, just return it exactly as is. "
    "Output ONLY the translation, no notes, no romanization, no quotes."
)

def get_qwen():
    global _global_qwen
    if _global_qwen is None:
        if Llama is None:
            raise RuntimeError("llama-cpp-python not installed: pip install llama-cpp-python")
        logging.info(f"[Qwen] loading {Qwen_REPO_ID} ...")
        _global_qwen = Llama.from_pretrained(
            repo_id=Qwen_REPO_ID,
            filename=Qwen_MODEL_FILENAME,
            n_ctx=2048,
            n_threads=max(4, os.cpu_count() or 4),
            n_gpu_layers=-1,       # Auto-offload to GPU if available
            verbose=False,
        )
    return _global_qwen

def qwen_translate(text: str) -> str:
    if not text.strip():
        return ""
    llm = get_qwen()
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]

    # Serialize LLM access to prevent concurrent CUDA deadlocks
    with _llm_lock:
        out = llm.create_chat_completion(
            messages=msgs,
            max_tokens=128,
            temperature=0.2,
            top_p=0.9,
            stop=["<|im_end|>", "</s>"],
        )
    try:
        raw = out["choices"][0]["message"]["content"].strip()
        # Strip any chat-template tokens that slipped through as literal text
        for tok in ("<|im_start|>", "<|im_end|>", "</s>", "<|endoftext|>",
                    "<|system|>", "<|user|>", "<|assistant|>"):
            raw = raw.replace(tok, "")
        # Drop any character the render font can't draw
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
# Text color detection (per box)
# ===========================================================================
def detect_text_and_bg_colors(img_bgr: np.ndarray, bbox: Tuple[int,int,int,int]
                              ) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    x1,y1,x2,y2 = bbox
    region = img_bgr[max(0,y1):y2, max(0,x1):x2]
    if region.size == 0:
        return (0,0,0), (255,255,255)

    # Downsample for speed
    if region.shape[0] > 120 or region.shape[1] > 120:
        region = cv2.resize(region, (120, 120), interpolation=cv2.INTER_AREA)

    pixels = region.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 8:
        return (0,0,0), (255,255,255)

    # --- Background = most common quantized color (mode, robust to thin text) ---
    quant = (pixels / 32).astype(np.int32)
    keys = quant[:,0] * 64 + quant[:,1] * 8 + quant[:,2]
    uniq, counts = np.unique(keys, return_counts=True)
    bg_key = int(uniq[int(np.argmax(counts))])
    bg_bgr = np.array([bg_key // 64, (bg_key // 8) % 8, bg_key % 8], dtype=np.float32) * 32 + 16

    # --- Text pixels = those far from background ---
    dists = np.linalg.norm(pixels - bg_bgr, axis=1)
    thresh = max(60.0, float(np.percentile(dists, 75)))
    text_mask = dists > thresh

    if int(text_mask.sum()) < 5:
        # No clear text — pick ink by luminance contrast
        bg_lum = float(bg_bgr.mean())
        ink_bgr = np.array([0,0,0], dtype=np.float32) if bg_lum > 127 else np.array([255,255,255], dtype=np.float32)
    else:
        text_pixels = pixels[text_mask]
        text_dists = np.linalg.norm(text_pixels - bg_bgr, axis=1)
        # Take the most extreme 30% — these are the true ink, not anti-aliasing
        ext_t = float(np.percentile(text_dists, 70))
        ext_mask = text_dists >= ext_t
        if int(ext_mask.sum()) >= 3:
            ink_bgr = np.median(text_pixels[ext_mask], axis=0)
        else:
            ink_bgr = np.median(text_pixels, axis=0)

    # --- Snap near-pure colors to pure black/white ---
    def snap(c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=np.float32)
        if np.all(c < 40):  return np.array([0,0,0], dtype=np.float32)
        if np.all(c > 215): return np.array([255,255,255], dtype=np.float32)
        return c

    ink_bgr = snap(ink_bgr)
    bg_bgr  = snap(bg_bgr)

    # --- Guarantee contrast between ink and outline ---
    if float(np.linalg.norm(ink_bgr - bg_bgr)) < 80:
        bg_lum = float(bg_bgr.mean())
        ink_bgr = np.array([0,0,0], dtype=np.float32) if bg_lum > 127 else np.array([255,255,255], dtype=np.float32)

    text_rgb    = (int(ink_bgr[2]), int(ink_bgr[1]), int(ink_bgr[0]))
    outline_rgb = (int(bg_bgr[2]),  int(bg_bgr[1]),  int(bg_bgr[0]))
    return text_rgb, outline_rgb

# ===========================================================================
# Text wrapping & auto-fit (binary search)
# ===========================================================================
import functools  # ← also add to your top-level imports if not present

@functools.lru_cache(maxsize=256)
def _get_font_cached(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Return a cached FreeTypeFont for (font_path, size).

    PIL's ImageFont.truetype re-parses the TTF file on every call. During
    binary-search auto-fit (sizes 8..96) that's ~7 disk reads per text box,
    per image. This cache keeps one ImageFont object per (path,size) pair in
    memory so the disk hit happens at most once per size ever used.

    lru_cache is thread-safe under the GIL for lookups; a rare double-load
    on first miss is harmless (only the winner is stored).
    """
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        # Don't poison the cache with the fallback default — a later
        # successful load at the same key would be skipped. Return a fresh
        # default each time the truetype call fails.
        return ImageFont.load_default()


def clear_font_cache() -> None:
    """Drop all cached font objects (call after hot-swapping the TTF file)."""
    _get_font_cached.cache_clear()


def get_font(font_path, size: int) -> ImageFont.FreeTypeFont:
    """Public accessor — accepts str or pathlib.Path, normalizes to str key."""
    return _get_font_cached(str(font_path), size)

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
              max_width: int, allow_break: bool = False) -> Optional[List[str]]:
    """Wrap text to fit within max_width.
    
    - Normal mode (allow_break=False): never breaks a word mid-character.
      Returns None if a single word is wider than max_width (signals the
      binary search to try a smaller font size).
    - Fallback mode (allow_break=True): breaks long words by character
      as an absolute last resort.
    
    Guarantees: every returned line contains at least one word (or one
    character fragment in fallback mode).
    """
    words = text.split()
    if not words:
        return [""]
    
    lines = []
    cur = ""
    
    for word in words:
        word_width = draw.textlength(word, font=font)
        
        # --- Single word doesn't fit on its own line ---
        if word_width > max_width:
            if not allow_break:
                # Signal: font is too big, binary search should go smaller
                return None
            # Last-resort character breaking
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
        
        # --- Try to append word to current line ---
        test = (cur + " " + word) if cur else word
        if draw.textlength(test, font=font) <= max_width:
            cur = test
        else:
            # Current line is full; push it and start a new line with this word
            if cur:
                lines.append(cur)
            cur = word  # word alone fits (checked above), so this line has ≥1 word
    
    if cur:
        lines.append(cur)
    return lines

def _measure_block(draw, lines, font):
    heights, total_h, max_w = [], 0, 0
    for ln in lines:
        bb = draw.textbbox((0,0), ln, font=font)
        h = bb[3] - bb[1]
        heights.append(h); total_h += h
        w = draw.textlength(ln, font=font)
        if w > max_w: max_w = w
    return heights, total_h, max_w

def fit_font_and_wrap(draw: ImageDraw.ImageDraw, text: str,
                      box_w: int, box_h: int,
                      font_path: str = str(FONT_PATH),
                      max_size: int = 96, min_size: int = 8
                      ) -> Tuple[int, List[str], List[int]]:
    """Binary-search the largest font size where text fits in (box_w, box_h).
    
    Each line is guaranteed to contain at least one complete word (no mid-word
    breaks). For tall narrow boxes, this naturally produces one word per line
    going downward, at the largest font size that fits horizontally.
    """
    if not text.strip():
        return min_size, [""], [0]
    
    lo, hi = min_size, max_size
    best_size: Optional[int] = None
    best_lines: Optional[List[str]] = None
    best_heights: Optional[List[int]] = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        font = get_font(font_path, mid)
        
        # Try wrapping without breaking any word
        lines = wrap_text(draw, text, font, box_w - 4, allow_break=False)
        
        if lines is None:
            # A single word doesn't fit at this size → need smaller font
            hi = mid - 1
            continue
        
        heights, total_h, max_w = _measure_block(draw, lines, font)
        
        if max_w <= box_w - 4 and total_h <= box_h - 4:
            # Fits! Try larger
            best_size, best_lines, best_heights = mid, lines, heights
            lo = mid + 1
        else:
            # Doesn't fit (too tall) → try smaller
            hi = mid - 1
    
    # --- Absolute fallback: no font size produced clean word-wrapping ---
    # Use min_size and allow character breaking as last resort
    if best_lines is None:
        font = get_font(font_path, min_size)

        fallback_lines = wrap_text(draw, text, font, box_w - 4, allow_break=True)
        best_lines = fallback_lines if fallback_lines else [text]
        heights, _, _ = _measure_block(draw, best_lines, font)
        best_size = min_size
        best_heights = heights
        logging.warning(
            f"Text could not fit cleanly even at min_size={min_size} "
            f"in box ({box_w}x{box_h}). Using character-break fallback."
        )
    
    return best_size, best_lines, best_heights
def draw_text_outline(draw, pos, text, font, fill, outline, outline_width):
    x, y = pos
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0: continue
            draw.text((x+dx, y+dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)

# ===========================================================================
# Core pipeline (Concurrent Inpainting & Translation)
# ===========================================================================
async def detect_translate_inpaint(pil_img: Image.Image,
                                   use_lama: bool = True) -> Tuple[Image.Image, List[Dict]]:
    img_bgr = pil_to_cv2(pil_img)
    h, w = img_bgr.shape[:2]
    loop = asyncio.get_running_loop()

    # --- OCR ---
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
        x1,y1,x2,y2 = b["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        cleaned.append({"text": (b.get("text") or "").strip(), "bbox": (x1,y1,x2,y2)})
    cleaned.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    if not cleaned:
        return pil_img, []

    # --- Concurrent Translation and Inpainting ---
    def _do_inpaint():
        mask = np.zeros((h, w), dtype=np.uint8)
        for c in cleaned:
            x1,y1,x2,y2 = c["bbox"]
            pad = max(2, int(min(x2-x1, y2-y1) * 0.06))
            mask[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)] = 255
            
        mask_area = np.count_nonzero(mask)
        total_area = h * w
        
        try:
            if use_lama and SimpleLama:
                return lama_inpaint(img_bgr, mask)
            elif mask_area > 0.25 * total_area:
                # cv2.inpaint will create a wavy green mess if the mask is too large.
                # Fallback to filling the boxes with the detected background color.
                logging.warning("Mask too large for cv2.inpaint. Filling boxes with solid colors instead.")
                filled = img_bgr.copy()
                for c in cleaned:
                    x1,y1,x2,y2 = c["bbox"]
                    _, bg_rgb = detect_text_and_bg_colors(img_bgr, (x1,y1,x2,y2))
                    bg_bgr = (bg_rgb[2], bg_rgb[1], bg_rgb[0]) # RGB to BGR
                    cv2.rectangle(filled, (x1, y1), (x2, y2), bg_bgr, -1)
                return filled
            else:
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
                # Use the dedicated single-thread LLM executor
                tasks.append(loop.run_in_executor(_llm_executor, qwen_translate, t))
        return await asyncio.gather(*tasks)

    # Run both concurrently, but with a timeout safety net
    inpaint_task = loop.run_in_executor(None, _do_inpaint)
    translation_task = _do_translation()
    
    try:
        inpainted, translations = await asyncio.wait_for(
            asyncio.gather(inpaint_task, translation_task),
            timeout=120.0  # 2 minutes max
        )
    except asyncio.TimeoutError:
        logging.error("Pipeline timed out after 120s")
        raise RuntimeError("Translation pipeline timed out")

    # --- Render ---
    base_pil = cv2_to_pil(inpainted).convert("RGBA")
    out = base_pil.copy()
    draw = ImageDraw.Draw(out)

    boxes_info: List[Dict] = []
    for c, trans in zip(cleaned, translations):
        x1,y1,x2,y2 = c["bbox"]
        # Belt-and-suspenders: clean again in case the LLM output bypassed
        # qwen_translate (e.g. idempotent cache, future OpenAI backend, etc.)
        trans = clean_text_for_font(trans)
        if not trans.strip():
            boxes_info.append({"bbox": c["bbox"], "orig": c["text"], "trans": "",
                               "font_size": 0, "text_color": None, "outline_color": None})
            continue

        text_rgb, outline_rgb = detect_text_and_bg_colors(img_bgr, (x1,y1,x2,y2))
        box_w, box_h = x2 - x1, y2 - y1
        font_size, lines, heights = fit_font_and_wrap(draw, trans, box_w, box_h, str(FONT_PATH))
        font = get_font(FONT_PATH, font_size)

        total_h = sum(heights)
        cur_y = y1 + max(0, (box_h - total_h) // 2)
        outline_w = max(1, int(font_size * 0.08))
        for ln, hln in zip(lines, heights):
            tw = draw.textlength(ln, font=font)
            pos_x = x1 + max(0, int((box_w - tw) // 2))
            draw_text_outline(draw, (pos_x, cur_y), ln, font,
                              fill=text_rgb, outline=outline_rgb, outline_width=outline_w)
            cur_y += hln

        boxes_info.append({
            "bbox": c["bbox"],
            "orig": c["text"],
            "trans": trans,
            "font_size": font_size,
            "text_color": text_rgb,
            "outline_color": outline_rgb,
        })

    return out, boxes_info

# ===========================================================================
# Job queue / worker
# ===========================================================================
def _cleanup_old_jobs():
    """Remove completed/errored jobs older than 10 minutes."""
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
            result_img, boxes = await detect_translate_inpaint(pil, use_lama=use_lama)
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
    _worker_task = asyncio.create_task(job_worker())  # Keep reference to prevent GC

# ===========================================================================
# API models
# ===========================================================================
class TranslateRequest(BaseModel):
    image_b64: Optional[str] = None
    use_lama: bool = True
    lang: str = DEFAULT_LANG
    source: Optional[str] = None

class AIResolveRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    model_list: Optional[List[str]] = None

# ===========================================================================
# Embedded HTML Testing UI (Raw string prevents \n JS bugs)
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
                    console.log("Switched model:", data);
                } catch (e) {
                    console.error("Failed to switch model:", e);
                }
            }

            async function translateImage() {
                const fileInput = document.getElementById('fileInput');
                const statusDiv = document.getElementById('status');
                const origImg = document.getElementById('origImg');
                const transImg = document.getElementById('transImg');
                const boxesInfo = document.getElementById('boxesInfo');

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
                    const MAX_POLLS = 30; // ~10 min at wait=20
                    
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
# Endpoints (root-level, matching the spec you pasted)
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
        "translation_model": Qwen_REPO_ID,
        "ocr_backend": "switchable",
        "current_ocr_model": _current_ocr_model,
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
        "lang": req.lang,
    }
    if _job_queue is None:
        raise HTTPException(503, "Server is still starting up, please try again in a moment.")
    await _job_queue.put(job_id)
    return {"id": job_id, "status": "queued", "hint": "poll /v1/translate/{id}?wait=N"}

@app.post("/v1/translate/upload")
async def v1_translate_upload(
    file: UploadFile = File(...),
    use_lama: bool = True,
    lang: str = DEFAULT_LANG,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """Multipart convenience endpoint for extensions that prefer file uploads."""
    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"bad image: {e}")
    job_id = idempotency_key or uuid.uuid4().hex
    _jobs[job_id] = {
        "id": job_id, "status": "queued", "created_at": time.time(),
        "_pil": pil, "_use_lama": use_lama, "lang": lang,
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
        await asyncio.sleep(0.25)
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
    return Response(content=job["image_bytes"], media_type="image/png")

@app.post("/v1/ai/resolve")
async def v1_ai_resolve(req: AIResolveRequest):
    return {
        "provider": req.provider or "qwen-gguf",
        "model":    req.model or Qwen_REPO_ID,
        "model_list": req.model_list or [Qwen_REPO_ID],
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
