"""
DeepShield AI — Central Model Orchestrator

The entry point for all unified deepfake analysis.
Coordinates: cache lookup → task runner → ensemble aggregation → structured output.

Design principles:
  • Single async function `analyze()` = public API used by unified_api.py
  • All models run concurrently via task_runner.py
  • Results normalized via temperature_scaler.py
  • Fused via ensemble_aggregator.py
  • Cached via cache_service.py
  • Health tracked via model_registry.py
"""

from __future__ import annotations

import base64
import io
import time
from typing import List, Optional

import numpy as np
from loguru import logger
from PIL import Image

from backend.orchestrator import (
    cache_service,
    task_runner,
)
from backend.orchestrator.ensemble_aggregator import aggregate, EnsembleResult
from backend.orchestrator.model_registry     import REGISTRY, enabled_healthy


# ─── Default model tiers ──────────────────────────────────────────────────────
# "balanced"  = all enabled models
# "fast"      = frequency only (instant, no GPU)
# "hf_only"   = HF deepfake-specific models only
# "full"      = all models including frequency

TIER_PRESETS: dict[str, List[str]] = {
    "fast":    ["frequency_dct", "frequency_fft"],
    "hf_only": ["vit_deepfake_primary", "vit_deepfake_secondary"],
    "balanced":["vit_deepfake_primary", "efficientnet_b4", "frequency_dct"],
    "full":    list(REGISTRY.keys()),
}
DEFAULT_TIER = "balanced"


# ─── Main analysis function ───────────────────────────────────────────────────

async def analyze(
    image_input:    "str | bytes | Image.Image",   # base64 str, raw bytes, or PIL
    models:         Optional[List[str]] = None,    # None = DEFAULT_TIER
    tier:           str                 = DEFAULT_TIER,
    use_cache:      bool                = True,
    detect_faces:   bool                = True,
    return_heatmap: bool                = False,
    source:         str                 = "",      # "web" | "extension" | "webcam" | "desktop"
) -> dict:
    """
    Full orchestrated deepfake analysis.

    Steps:
      1. Decode input (base64 / bytes / PIL)
      2. Face extraction (optional)
      3. Cache lookup
      4. Run all selected models concurrently (task_runner)
      5. Ensemble aggregation (temperature scaling + weighted soft voting)
      6. Build structured response
      7. Cache result
      8. Return

    Returns deterministic JSON-serialisable dict.
    """
    t0 = time.perf_counter()

    # ── 1. Decode image ───────────────────────────────────────────────────────
    pil_image = _to_pil(image_input)
    img_bytes = _pil_to_bytes(pil_image)

    # ── 2. Resolve model list ─────────────────────────────────────────────────
    if models:
        requested = [m for m in models if m in REGISTRY]
    else:
        requested = TIER_PRESETS.get(tier, TIER_PRESETS[DEFAULT_TIER])

    # Filter to enabled+healthy only
    active_models = [m for m in requested if m in REGISTRY and REGISTRY[m].enabled and REGISTRY[m].healthy]
    if not active_models:
        # All unhealthy? Try everything anyway (will record failures)
        active_models = requested
        logger.warning("[Orchestrator] All selected models unhealthy — attempting anyway")

    # ── 3. Cache lookup ───────────────────────────────────────────────────────
    cache_key = cache_service.make_key(img_bytes, active_models)
    if use_cache:
        hit = cache_service.get(cache_key)
        if hit is not None:
            hit["cached"] = True
            hit["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return hit

    # ── 4. Face detection ─────────────────────────────────────────────────────
    analysis_image = pil_image
    faces_found    = 0
    if detect_faces:
        try:
            from backend.ai.face_pipeline import get_face_pipeline
            pipe  = get_face_pipeline()
            crops = pipe.extract_faces(pil_image)
            if crops:
                analysis_image = crops[0]   # analyse primary face
                faces_found    = len(crops)
        except Exception as e:
            logger.debug(f"[Orchestrator] Face extraction failed: {e}")

    # ── 5. Run all models concurrently ────────────────────────────────────────
    model_outputs = await task_runner.run_all_models(analysis_image, active_models)

    # ── 6. Ensemble aggregation ───────────────────────────────────────────────
    result: EnsembleResult = aggregate(model_outputs)

    # ── 7. Grad-CAM heatmap (optional, only for timm models) ─────────────────
    heatmap_b64 = ""
    if return_heatmap:
        try:
            from backend.ai.inference_service import _generate_heatmap
            timm_models = [m for m in active_models
                           if REGISTRY.get(m) and REGISTRY[m].kind == "timm"]
            heatmap_b64 = _generate_heatmap(pil_image, timm_models or active_models)
        except Exception as e:
            logger.debug(f"[Orchestrator] Heatmap skipped: {e}")

    # ── 8. Build structured response ──────────────────────────────────────────
    latency = round((time.perf_counter() - t0) * 1000, 1)

    response = {
        # Top-level verdict
        "verdict":           result.verdict,
        "risk_score":        result.risk_score,
        "confidence":        result.confidence,
        "fake_prob":         result.fake_prob,

        # Ensemble quality metrics
        "agreement_score":   result.agreement_score,
        "variance":          result.variance,
        "reliability_index": result.reliability_index,

        # Model counts
        "models_used":       result.models_used,
        "models_timed_out":  result.models_timed_out,
        "models_failed":     result.models_failed,
        "models_requested":  len(active_models),

        # Per-model breakdown
        "per_model":         result.per_model,

        # Explainability
        "heatmap_b64":       heatmap_b64,

        # Metadata
        "faces_found":       faces_found,
        "source":            source,
        "tier":              tier,
        "models":            active_models,
        "latency_ms":        latency,
        "cached":            False,
        "message":           result.message,
    }

    # ── 9. Cache result ───────────────────────────────────────────────────────
    if use_cache and result.models_used > 0:
        # Don't cache results with zero successful models
        cache_service.set(cache_key, response)

    logger.info(f"[Orchestrator] {result.verdict} risk={result.risk_score} "
                f"agreement={result.agreement_score:.2f} lat={latency}ms "
                f"src={source} models={result.models_used}/{len(active_models)}")

    return response


async def analyze_live_frame(
    base64_frame: str,
    models:       Optional[List[str]] = None,
    source:       str                 = "webcam",
) -> dict:
    """
    Optimized path for live webcam frames.
    - Skips face detection (too slow for real-time)
    - Skips heatmap
    - Uses "fast" tier by default (frequency models only for <100ms)
    - Falls back to balanced if time < 200ms
    """
    if models is None:
        models = TIER_PRESETS["balanced"]

    return await analyze(
        image_input    = base64_frame,
        models         = models,
        use_cache      = False,   # live frames shouldn't be cached
        detect_faces   = False,
        return_heatmap = False,
        source         = source,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_pil(inp: "str | bytes | Image.Image") -> Image.Image:
    if isinstance(inp, Image.Image):
        return inp.convert("RGB")
    if isinstance(inp, str):
        if "," in inp:
            inp = inp.split(",", 1)[1]
        data = base64.b64decode(inp)
        return Image.open(io.BytesIO(data)).convert("RGB")
    if isinstance(inp, bytes):
        return Image.open(io.BytesIO(inp)).convert("RGB")
    raise TypeError(f"Cannot convert {type(inp)} to PIL.Image")


def _pil_to_bytes(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
