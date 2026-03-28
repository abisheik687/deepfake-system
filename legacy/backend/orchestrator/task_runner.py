"""
task_runner.py — Hardened concurrent model inference runner
Includes: asyncio.Semaphore for concurrency control,
          CUDA OOM guard with CPU fallback,
          memory logging, and graceful error recovery.
"""
from __future__ import annotations

import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

import torch
from PIL import Image

from backend.config import settings
from backend.orchestrator.model_registry import (
    REGISTRY, ModelOutput, ModelSpec, enabled_healthy
)
from backend.metrics import MODEL_LATENCY

logger = logging.getLogger(__name__)

# Thread pool — size = MAX_CONCURRENT_MODELS + 2 for headroom
_POOL = ThreadPoolExecutor(
    max_workers=settings.MAX_CONCURRENT_MODELS + 2,
    thread_name_prefix="orch_infer",
)

# Semaphore: hard cap on simultaneous model inferences (3 to avoid OOM)
_INFER_SEMAPHORE = asyncio.Semaphore(min(3, settings.MAX_CONCURRENT_MODELS) if settings.MAX_CONCURRENT_MODELS else 3)

MODEL_TIMEOUT: float = 60.0  # seconds per model


def _log_memory(stage: str) -> None:
    """Log CUDA and system memory at a given stage (best-effort)."""
    try:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(
                f"[{stage}] CUDA memory: alloc {alloc:.2f}GB, res {reserved:.2f}GB"
            )
    except Exception:
        pass


def _infer_sync_safe(
    name: str,
    spec: Any,
    pil_image: Image.Image,
    force_cpu: bool = False,
) -> ModelOutput:
    """
    Run a single model inference synchronously.
    If force_cpu=True, moves inputs/model to CPU before inference.
    """
    t0 = time.perf_counter()
    try:
        device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        _log_memory(f"before:{name}")

        if spec.kind == "frequency":
            fp, conf = _infer_frequency(pil_image, spec.freq_method or "dct")
        elif spec.kind == "hf_pipeline":
            # Pass device preference explicitly if supported
            fp, conf = _infer_hf(pil_image, spec, device=device)
        elif spec.kind == "timm":
            fp, conf = _infer_timm(pil_image, spec, device=device)
        else:
            raise ValueError(f"Unknown model kind: {spec.kind}")

        latency = (time.perf_counter() - t0) * 1000
        spec.record_success(latency)
        MODEL_LATENCY.labels(model_name=name).observe(latency / 1000.0)

        verdict = "FAKE" if fp >= 0.50 else "REAL"
        
        _log_memory(f"after:{name}")
        return ModelOutput(
            model_name=name,
            fake_prob=float(fp),
            verdict=verdict,
            confidence=float(conf),
            weight=spec.weight,
            latency_ms=round(latency, 1),
        )

    except (torch.cuda.OutOfMemoryError, RuntimeError) as oom:
        if torch.cuda.is_available() and "memory" in str(oom).lower() and not force_cpu:
            logger.warning(f"[{name}] CUDA OOM detected — clearing cache and retrying on CPU.")
            torch.cuda.empty_cache()
            try:
                return _infer_sync_safe(name, spec, pil_image, force_cpu=True)
            except Exception as cpu_err:
                logger.error(f"[{name}] CPU fallback also failed: {cpu_err}")
                return ModelOutput(
                    model_name=name, fake_prob=0.5, verdict="REAL",
                    confidence=0.0, weight=spec.weight, failed=True, error=str(cpu_err)
                )
        raise # otherwise raise mapped error

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        spec.record_failure(str(exc))
        MODEL_LATENCY.labels(model_name=name).observe(latency / 1000.0)
        logger.error(f"[{name}] Unexpected inference error: {exc}", exc_info=True)
        return ModelOutput(
            model_name=name, fake_prob=0.5, verdict="REAL",
            confidence=0.0, weight=spec.weight, failed=True, error=str(exc),
            latency_ms=round(latency, 1)
        )

    finally:
        # Clear cache between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def _run_model_with_semaphore(
    loop: asyncio.AbstractEventLoop,
    name: str,
    spec: Any,
    pil_image: Image.Image,
) -> ModelOutput:
    """Acquire semaphore, run model, release semaphore."""
    async with _INFER_SEMAPHORE:
        logger.debug(f"[{name}] Semaphore acquired — starting inference")
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    _POOL,
                    _infer_sync_safe,
                    name,
                    spec,
                    pil_image,
                    False,  # force_cpu
                ),
                timeout=MODEL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            spec.record_failure(f"Timeout after {MODEL_TIMEOUT}s")
            return ModelOutput(
                model_name=name, fake_prob=0.5, verdict="REAL",
                confidence=0.0, weight=spec.weight, timed_out=True, error="timeout",
            )


async def run_all_models(
    pil_image: Image.Image,
    model_names: Optional[List[str]] = None,
    include_freq: bool = True,
) -> List[ModelOutput]:
    """Run requested models concurrently through the semaphore pool."""
    if model_names is None:
        model_names = enabled_healthy()
    if not include_freq:
        model_names = [m for m in model_names if REGISTRY.get(m) and REGISTRY[m].kind != "frequency"]

    loop = asyncio.get_event_loop()
    tasks = []
    
    for name in model_names:
        spec = REGISTRY.get(name)
        if not spec:
            continue
        tasks.append(_run_model_with_semaphore(loop, name, spec, pil_image))

    logger.info(
        f"Running {len(tasks)} models with semaphore={settings.MAX_CONCURRENT_MODELS}"
    )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    outputs: List[ModelOutput] = []
    for i, r in enumerate(results):
        name = model_names[i]
        spec = REGISTRY.get(name, ModelSpec(name=name, kind="custom", description=""))
        if isinstance(r, Exception):
            logger.error(f"[{name}] gather exception: {r}")
            outputs.append(ModelOutput(
                model_name=name, fake_prob=0.5, verdict="REAL",
                confidence=0.0, weight=spec.weight, failed=True, error=str(r),
            ))
        else:
            outputs.append(r)

    return outputs


def _infer_hf(pil_image: Image.Image, spec: ModelSpec, device: str = "cpu"):
    from backend.ai.hf_registry import load_hf_model
    pipe, _ = load_hf_model(spec.name)
    pipe.model = pipe.model.to(device) # force device
    
    results  = pipe(pil_image, top_k=5)

    fake_lbl = spec.fake_label.lower()
    real_lbl = spec.real_label.lower()
    fake_p, real_p = 0.0, 0.0

    for r in results:
        lbl = r["label"].lower()
        if lbl == fake_lbl or "fake" in lbl or "deepfake" in lbl:
            fake_p = max(fake_p, r["score"])
        elif lbl == real_lbl or "real" in lbl or "authentic" in lbl:
            real_p = max(real_p, r["score"])

    is_fake = fake_p > real_p
    conf    = fake_p if is_fake else real_p
    return (fake_p if is_fake else 1.0 - real_p), conf


def _infer_timm(pil_image: Image.Image, spec: ModelSpec, device: str = "cpu"):
    from backend.ai.hf_registry import load_hf_model
    model, transforms = load_hf_model(spec.name)
    model = model.to(device)
    tensor = transforms(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(tensor)
        else:
            logits = model(tensor)

    probs  = torch.softmax(logits, dim=-1)[0]
    fake_p = float(probs[1].item())
    conf   = float(probs.max().item())
    return fake_p, conf


def _infer_frequency(pil_image: Image.Image, method: str):
    import cv2
    import numpy as np

    bgr   = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if method in ("dct", "combined"):
        dct  = cv2.dct(gray / 255.0)
        low  = np.abs(dct[:32, :32]).mean()
        high = np.abs(dct[32:, 32:]).mean()
        ratio = high / (low + 1e-6)
        dct_score = min(1.0, max(0.0, (ratio - 0.01) / 0.15))
    else:
        dct_score = 0.5

    if method in ("fft", "combined"):
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

    return float(fake_p), 0.55
