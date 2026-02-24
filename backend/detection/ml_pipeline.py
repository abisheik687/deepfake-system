"""
DeepShield AI — ML Detection Pipeline
Replaces heuristics with real pretrained neural networks.

Model priority order (fastest to most accurate):
  1. EfficientNet-B4   (~2s CPU / ~80ms GPU)  — primary
  2. XceptionNet       (~3s CPU / ~100ms GPU)  — secondary
  3. ViT-B/16          (~4s CPU / ~120ms GPU)  — transformer option
  4. Heuristic fallback (existing pipeline.py) — CPU-only fallback

On first run: downloads pretrained ImageNet weights via timm (~90MB each).
Fine-tuned deepfake-specific weights loaded from models/<name>.pth if present.

Usage:
    from backend.detection.ml_pipeline import ml_analyze_frame
    result = ml_analyze_frame(base64_str)
    # result → {"verdict": "FAKE", "confidence": 0.87, "faces": [...], "gradcam_heatmap": "data:image/png;base64,..."}
"""

import os
import io
import cv2
import base64
import numpy as np
from pathlib import Path
from loguru import logger

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from backend.detection.model_zoo import get_model, AVAILABLE_MODELS
from backend.detection.face_extractor import FaceExtractor
from backend.detection.gradcam import GradCAM

# ─── Config ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[ML Pipeline] Device: {DEVICE}")

FAKE_THRESHOLD = 0.50      # Probability above this → FAKE
FACE_SIZE      = 224        # All models use 224×224 inputs
MIN_FACE_PX    = 80         # Ignore tiny detections

# ImageNet normalisation (standard for all timm / HuggingFace models)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

# Singletons — lazy-loaded on first call
_face_extractor = None
_active_model   = None
_gradcam        = None
_active_name    = None


def _get_face_extractor() -> FaceExtractor:
    global _face_extractor
    if _face_extractor is None:
        _face_extractor = FaceExtractor()
    return _face_extractor


def _load_model(model_name: str = "efficientnet_b4"):
    """Load model and Grad-CAM wrapper — cached after first load."""
    global _active_model, _gradcam, _active_name
    if _active_name == model_name and _active_model is not None:
        return _active_model, _gradcam

    logger.info(f"[ML Pipeline] Loading model: {model_name}")
    model = get_model(model_name, pretrained=True)
    model.eval().to(DEVICE)

    # GradCAM target layer heuristics (last conv layer)
    target_layer = _find_last_conv(model)
    cam = GradCAM(model, target_layer) if target_layer else None

    _active_model = model
    _gradcam      = cam
    _active_name  = model_name
    logger.success(f"[ML Pipeline] Model '{model_name}' ready on {DEVICE}")
    return model, cam


def _find_last_conv(model) -> torch.nn.Module | None:
    """Walk the model graph to find the last Conv2d layer for Grad-CAM."""
    last = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module
    return last


def _face_to_tensor(face_bgr: np.ndarray) -> torch.Tensor:
    """BGR numpy → normalised RGB tensor on DEVICE."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return _preprocess(pil).unsqueeze(0).to(DEVICE)


def _frequency_score(face_bgr: np.ndarray) -> float:
    """
    DCT/FFT-based forgery signal.
    GAN-generated faces have abnormal high-frequency energy distribution.
    Returns 0.0 (likely real) to 1.0 (likely fake).
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (128, 128))

    # DCT analysis
    dct = cv2.dct(gray)
    dct_energy = np.abs(dct)
    low_freq   = dct_energy[:16, :16].mean()
    high_freq  = dct_energy[48:, 48:].mean()
    ratio = high_freq / (low_freq + 1e-6)

    # FFT analysis
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    h, w = magnitude.shape
    center_energy = magnitude[h//2-16:h//2+16, w//2-16:w//2+16].mean()
    outer_energy   = magnitude.mean()
    fft_ratio = center_energy / (outer_energy + 1e-6)

    # Normalise: high DCT ratio + low FFT centralisation → suspicious
    dct_score = float(np.clip(ratio * 8, 0.0, 1.0))
    fft_score = float(np.clip(1.0 - fft_ratio / 20.0, 0.0, 1.0))
    return round(0.5 * dct_score + 0.5 * fft_score, 4)


def _heatmap_to_base64(heatmap: np.ndarray, original_bgr: np.ndarray) -> str:
    """
    Overlay Grad-CAM heatmap on the original face image.
    Returns a base64-encoded PNG data-URI.
    """
    h, w = original_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_color   = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(original_bgr, 0.55, heatmap_color, 0.45, 0)

    success, buf = cv2.imencode(".png", overlay)
    if not success:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# ─── Public API ───────────────────────────────────────────────────────────────

def ml_analyze_frame(
    base64_image: str,
    model_name: str = "efficientnet_b4",
    return_heatmap: bool = True,
) -> dict:
    """
    Analyse a single frame/image for deepfake content using a real neural network.

    Parameters
    ----------
    base64_image  : base64 string (with or without data-URI prefix)
    model_name    : key from AVAILABLE_MODELS (default: efficientnet_b4)
    return_heatmap: include Grad-CAM heatmap PNG in response?

    Returns
    -------
    {
      "verdict"        : "FAKE" | "REAL" | "NO_FACE" | "ERROR"
      "confidence"     : float 0–1
      "faces"          : list[{x,y,w,h,xn,yn,wn,hn}]
      "model"          : str
      "gradcam_heatmap": "data:image/png;base64,..."  (if return_heatmap True)
      "frequency_score": float 0–1
      "scores"         : {per-face DNN scores}
      "message"        : str
    }
    """
    try:
        # ── 1. Decode base64 image ─────────────────────────────────────────
        if "," in base64_image:
            base64_image = base64_image.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_image)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return _error("Could not decode image")

        h_f, w_f = frame.shape[:2]

        # ── 2. Face detection ──────────────────────────────────────────────
        extractor   = _get_face_extractor()
        face_rects  = extractor.detect(frame)  # list of (x,y,w,h)

        if not face_rects:
            return {
                "verdict": "NO_FACE",
                "confidence": 0.0,
                "faces": [],
                "model": model_name,
                "gradcam_heatmap": "",
                "frequency_score": 0.0,
                "scores": {},
                "message": "No face detected — cannot analyse deepfakes without a face."
            }

        # ── 3. Load model (cached after first call) ────────────────────────
        try:
            model, cam = _load_model(model_name)
        except Exception as e:
            logger.warning(f"ML model failed ({e}), falling back to heuristic")
            from backend.detection.pipeline import analyze_frame
            return analyze_frame(base64_image)

        # ── 4. Per-face inference ──────────────────────────────────────────
        face_results   = []
        heatmap_b64    = ""
        best_heatmap   = None
        best_face_crop = None

        model.eval()
        with torch.no_grad() if cam is None else torch.enable_grad():
            for (x, y, w, h) in face_rects:
                if w < MIN_FACE_PX or h < MIN_FACE_PX:
                    continue

                x1 = max(0, x - int(w * 0.1))
                y1 = max(0, y - int(h * 0.1))
                x2 = min(w_f, x + w + int(w * 0.1))
                y2 = min(h_f, y + h + int(h * 0.1))
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                tensor = _face_to_tensor(face_crop)

                # Forward pass
                if cam is not None:
                    logits, heatmap_np = cam.generate(tensor, class_idx=1)  # class 1 = FAKE
                else:
                    logits   = model(tensor)
                    heatmap_np = None

                prob = float(F.softmax(logits, dim=-1)[0][1].item()) if logits.shape[-1] == 2 \
                       else float(torch.sigmoid(logits)[0][0].item())

                freq_score = _frequency_score(face_crop)

                # Combine DNN probability with frequency signal (80/20 blend)
                combined = round(0.80 * prob + 0.20 * freq_score, 4)

                face_results.append({
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "xn": round(x / w_f, 4), "yn": round(y / h_f, 4),
                    "wn": round(w / w_f, 4), "hn": round(h / h_f, 4),
                    "dnn_score":   round(prob, 4),
                    "freq_score":  round(freq_score, 4),
                    "combined":    combined,
                })

                if heatmap_np is not None and best_heatmap is None:
                    best_heatmap   = heatmap_np
                    best_face_crop = face_crop

        if not face_results:
            return _error("Faces detected but too small to analyse")

        # ── 5. Aggregate results ───────────────────────────────────────────
        max_score = max(f["combined"] for f in face_results)
        mean_score = float(np.mean([f["combined"] for f in face_results]))
        # Take the more conservative (higher) score
        final_score = round(0.7 * max_score + 0.3 * mean_score, 4)
        is_fake     = final_score >= FAKE_THRESHOLD
        verdict     = "FAKE" if is_fake else "REAL"
        confidence  = final_score if is_fake else round(1.0 - final_score, 4)

        # ── 6. Grad-CAM overlay ────────────────────────────────────────────
        if return_heatmap and best_heatmap is not None and best_face_crop is not None:
            heatmap_b64 = _heatmap_to_base64(best_heatmap, best_face_crop)

        pct = int(confidence * 100)
        prefix = "⚠️ FAKE DETECTED" if is_fake else "✅ REAL"

        return {
            "verdict":         verdict,
            "confidence":      confidence,
            "faces":           face_results,
            "model":           model_name,
            "gradcam_heatmap": heatmap_b64,
            "frequency_score": float(np.mean([f["freq_score"] for f in face_results])),
            "scores":          {"max": max_score, "mean": mean_score, "final": final_score},
            "message": f"{prefix} ({pct}% confidence) — model: {model_name}",
        }

    except Exception as exc:
        logger.error(f"ml_analyze_frame error: {exc}", exc_info=True)
        return _error(str(exc))


def _error(msg: str) -> dict:
    return {
        "verdict": "ERROR",
        "confidence": 0.0,
        "faces": [],
        "model": "unknown",
        "gradcam_heatmap": "",
        "frequency_score": 0.0,
        "scores": {},
        "message": f"Analysis error: {msg}",
    }


def get_active_model_name() -> str:
    return _active_name or "none"


def switch_model(name: str) -> dict:
    """Switch the active inference model. Returns status dict."""
    if name not in AVAILABLE_MODELS:
        return {"status": "error", "message": f"Unknown model: {name}. Available: {list(AVAILABLE_MODELS.keys())}"}
    try:
        _load_model(name)
        return {"status": "ok", "model": name, "device": str(DEVICE)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
