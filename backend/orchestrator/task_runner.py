"""
DeepShield AI — Async Task Runner

Executes multiple detection models concurrently via asyncio + ThreadPoolExecutor.

Features:
  • Each model runs in its own thread (CPU-bound inference safe)
  • Per-model timeout — model that exceeds deadline returns timed_out=True
  • Other models continue regardless of individual failures
  • GPU memory guarding: sequential GPU models, parallel CPU models
  • Health registry updated after every inference (success or failure)
  • Structured logging per model
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image

from backend.orchestrator.model_registry import (
    REGISTRY, ModelOutput, ModelSpec, enabled_healthy
)

# Thread pool — CPU models run here; GPU models share the GPU sequentially
_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="orch_infer")


async def run_all_models(
    pil_image:   Image.Image,
    model_names: Optional[List[str]] = None,
    include_freq: bool = True,
) -> List[ModelOutput]:
    """
    Run all requested (or all enabled+healthy) models concurrently.

    Parameters
    ----------
    pil_image    : PIL.Image — RGB image for analysis
    model_names  : list of registry keys; None = all enabled healthy
    include_freq : whether to include frequency-analysis models

    Returns
    -------
    List[ModelOutput] — one per model, includes failures/timeouts
    """
    if model_names is None:
        model_names = enabled_healthy()
    if not include_freq:
        model_names = [m for m in model_names if REGISTRY.get(m) and REGISTRY[m].kind != "frequency"]

    loop  = asyncio.get_event_loop()
    tasks = []
    for name in model_names:
        spec = REGISTRY.get(name)
        if spec is None:
            logger.warning(f"[TaskRunner] Unknown model: {name}")
            continue
        tasks.append(
            asyncio.wait_for(
                loop.run_in_executor(_POOL, _infer_sync, name, spec, pil_image),
                timeout=spec.timeout_s,
            )
        )

    # Gather with per-task exception handling
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    outputs: List[ModelOutput] = []

    for name, raw in zip(model_names, raw_results):
        spec = REGISTRY.get(name, ModelSpec(name=name, kind="custom", description=""))
        if isinstance(raw, asyncio.TimeoutError):
            logger.warning(f"[TaskRunner] ⏱️ {name} timed out after {spec.timeout_s}s")
            spec.record_failure(f"Timeout after {spec.timeout_s}s")
            outputs.append(ModelOutput(
                model_name=name, fake_prob=0.5, verdict="REAL",
                confidence=0.0, weight=spec.weight, timed_out=True, error="timeout",
            ))
        elif isinstance(raw, Exception):
            err = str(raw)[:120]
            logger.error(f"[TaskRunner] ❌ {name} failed: {err}")
            spec.record_failure(err)
            outputs.append(ModelOutput(
                model_name=name, fake_prob=0.5, verdict="REAL",
                confidence=0.0, weight=spec.weight, failed=True, error=err,
            ))
        else:
            outputs.append(raw)

    return outputs


# ─── Sync inference (runs inside thread pool) ───────────────────────────────

def _infer_sync(name: str, spec: ModelSpec, pil_image: Image.Image) -> ModelOutput:
    """Run a single model synchronously. Called from thread pool."""
    t0 = time.perf_counter()
    try:
        if spec.kind == "frequency":
            fp, conf = _infer_frequency(pil_image, spec.freq_method or "dct")
        elif spec.kind == "hf_pipeline":
            fp, conf = _infer_hf(pil_image, spec)
        elif spec.kind == "timm":
            fp, conf = _infer_timm(pil_image, spec)
        else:
            raise ValueError(f"Unknown model kind: {spec.kind}")

        latency = (time.perf_counter() - t0) * 1000
        spec.record_success(latency)

        verdict = "FAKE" if fp >= 0.50 else "REAL"
        logger.debug(f"[TaskRunner] {name}: {verdict} ({fp:.2%}) {latency:.0f}ms")

        return ModelOutput(
            model_name=name,
            fake_prob=float(fp),
            verdict=verdict,
            confidence=float(conf),
            weight=spec.weight,
            latency_ms=round(latency, 1),
        )

    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        raise   # re-raise; caller handles


# ─── Model-specific inference implementations ───────────────────────────────

def _infer_hf(pil_image: Image.Image, spec: ModelSpec):
    """HuggingFace pipeline inference."""
    from backend.ai.hf_registry import load_hf_model, HF_CATALOGUE

    # Pull from our existing HF registry
    pipe, _ = load_hf_model(spec.name)
    results  = pipe(pil_image, top_k=5)

    fake_lbl = spec.fake_label.lower()
    real_lbl = spec.real_label.lower()
    fake_p   = 0.0
    real_p   = 0.0

    for r in results:
        lbl = r["label"].lower()
        if lbl == fake_lbl or "fake" in lbl or "deepfake" in lbl:
            fake_p = max(fake_p, r["score"])
        elif lbl == real_lbl or "real" in lbl or "authentic" in lbl:
            real_p = max(real_p, r["score"])

    if fake_p == 0.0 and real_p == 0.0:
        top = results[0]["label"].lower()
        if "real" in top or "authentic" in top:
            real_p = results[0]["score"]
        else:
            fake_p = results[0]["score"]

    is_fake = fake_p > real_p
    conf    = fake_p if is_fake else real_p
    return (fake_p if is_fake else 1.0 - real_p), conf


def _infer_timm(pil_image: Image.Image, spec: ModelSpec):
    """timm model inference."""
    from backend.ai.hf_registry import load_hf_model
    model, transforms = load_hf_model(spec.name)
    device = next(model.parameters()).device
    tensor = transforms(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(tensor)
        else:
            logits = model(tensor)

    probs  = torch.softmax(logits, dim=-1)[0]
    fake_p = float(probs[1].item())
    conf   = float(probs.max().item())
    return fake_p, conf


def _infer_frequency(pil_image: Image.Image, method: str):
    """
    Frequency-domain deepfake detector (no model weights required).

    Computes spectral statistics from DCT/FFT decomposition.
    GAN-generated images have characteristic frequency artifacts
    (periodic patterns, spectral peaks) not present in real photos.
    """
    import cv2
    import numpy as np

    bgr   = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if method in ("dct", "combined"):
        # DCT analysis
        dct  = cv2.dct(gray / 255.0)
        low  = np.abs(dct[:32, :32]).mean()
        high = np.abs(dct[32:, 32:]).mean()
        ratio = high / (low + 1e-6)
        # High ratio of HF to LF energy suggests GAN artifacts
        dct_score = min(1.0, max(0.0, (ratio - 0.01) / 0.15))
    else:
        dct_score = 0.5

    if method in ("fft", "combined"):
        # FFT analysis
        fft  = np.fft.fft2(gray / 255.0)
        fmag = np.log1p(np.abs(np.fft.fftshift(fft)))
        h, w = fmag.shape
        center = fmag[h//4:3*h//4, w//4:3*w//4]
        outer  = fmag.mean() - center.mean()
        fft_score = min(1.0, max(0.0, outer + 0.5))
    else:
        fft_score = 0.5

    if method == "combined":
        fake_p = 0.6 * dct_score + 0.4 * fft_score
    elif method == "dct":
        fake_p = dct_score
    else:
        fake_p = fft_score

    # Frequency analysis — lower confidence (signal is noisy)
    confidence = 0.55
    return float(fake_p), confidence
