"""
DeepShield AI — Unified API Endpoints (Orchestration Layer)

New isolated endpoints — do NOT modify any existing routes.

Endpoints:
  POST /analyze-unified         → full orchestrated image analysis
  POST /live-unified            → optimized webcam frame analysis
  POST /extension-scan          → browser extension primary endpoint
  POST /analyze-unified-video   → temporal video orchestration
  GET  /orchestrator-health     → full system health report
  GET  /orchestrator-status     → model registry status

All return standardized JSON with:
  verdict, risk_score, confidence, agreement_score, variance,
  reliability_index, per_model breakdown, latency_ms
"""

import io
import base64
import time
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from backend.orchestrator          import orchestrator, cache_service
from backend.orchestrator.model_registry import REGISTRY, list_all, enabled_healthy, TIER_PRESETS

router = APIRouter()


# ─── Request schemas ──────────────────────────────────────────────────────────

class UnifiedImageRequest(BaseModel):
    data:           str                      # base64 data-URI or raw
    models:         Optional[List[str]] = None
    tier:           Optional[str]       = "balanced"
    detect_faces:   Optional[bool]      = True
    return_heatmap: Optional[bool]      = False
    use_cache:      Optional[bool]      = True
    source:         Optional[str]       = "web"


class LiveFrameRequest(BaseModel):
    frame:      str                      # base64 frame
    models:     Optional[List[str]] = None
    tier:       Optional[str]       = "balanced"
    frame_idx:  Optional[int]       = 0
    source:     Optional[str]       = "webcam"


class ExtensionScanRequest(BaseModel):
    data:       Optional[str]       = None   # base64 image
    url:        Optional[str]       = None   # OR image URL
    models:     Optional[List[str]] = None
    tier:       Optional[str]       = "balanced"
    source:     str                 = "extension"
    page_url:   Optional[str]       = ""


# ─── 1. Unified Image Analysis ────────────────────────────────────────────────

@router.post("/analyze-unified")
async def analyze_unified(req: UnifiedImageRequest):
    """
    Orchestrated multi-model deepfake analysis.

    Runs all selected/tier models concurrently, applies temperature-scaled
    weighted ensemble, returns Deepfake Risk Score 0–100 + 3-tier verdict.
    """
    if not req.data:
        raise HTTPException(400, "Image data (base64) is required")

    _validate_tier(req.tier)

    try:
        result = await orchestrator.analyze(
            image_input    = req.data,
            models         = req.models,
            tier           = req.tier or "balanced",
            use_cache      = req.use_cache if req.use_cache is not None else True,
            detect_faces   = req.detect_faces if req.detect_faces is not None else True,
            return_heatmap = req.return_heatmap or False,
            source         = req.source or "web",
        )
    except Exception as e:
        logger.error(f"[UnifiedAPI] /analyze-unified error: {e}")
        raise HTTPException(500, f"Orchestration error: {str(e)[:200]}")

    return result


# ─── 2. Live Unified (webcam frames) ─────────────────────────────────────────

@router.post("/live-unified")
async def live_unified(req: LiveFrameRequest):
    """
    Low-latency unified endpoint for live webcam frames.
    Skips face extraction and heatmap for speed.
    Uses 'balanced' tier (ViT + frequency) by default.
    """
    if not req.frame:
        raise HTTPException(400, "Frame data required")

    try:
        result = await orchestrator.analyze_live_frame(
            base64_frame = req.frame,
            models       = req.models,
            source       = req.source or "webcam",
        )
    except Exception as e:
        logger.error(f"[UnifiedAPI] /live-unified error: {e}")
        raise HTTPException(500, str(e)[:200])

    result["frame_idx"] = req.frame_idx
    return result


# ─── 3. Extension Scan ────────────────────────────────────────────────────────

@router.post("/extension-scan")
async def extension_scan(req: ExtensionScanRequest):
    """
    Browser extension primary endpoint.
    Accepts either base64 image data OR a URL to download + analyse.
    Returns compact JSON optimized for extension popup display.
    """
    image_input = None

    if req.data:
        image_input = req.data
    elif req.url:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get(req.url, headers={"User-Agent": "DeepShieldExtension/2.0"})
                resp.raise_for_status()
                image_input = "data:image/jpeg;base64," + base64.b64encode(resp.content).decode()
        except Exception as e:
            raise HTTPException(502, f"Cannot fetch image URL: {e}")
    else:
        raise HTTPException(400, "Either 'data' (base64) or 'url' must be provided")

    try:
        result = await orchestrator.analyze(
            image_input  = image_input,
            models       = req.models,
            tier         = req.tier or "balanced",
            use_cache    = True,
            detect_faces = True,
            source       = "extension",
        )
    except Exception as e:
        raise HTTPException(500, str(e)[:200])

    # Compact response for extension popup
    return {
        "verdict":         result["verdict"],
        "risk_score":      result["risk_score"],
        "confidence":      result["confidence"],
        "agreement_score": result["agreement_score"],
        "reliability":     result["reliability_index"],
        "message":         result["message"],
        "latency_ms":      result["latency_ms"],
        "models_used":     result["models_used"],
        "page_url":        req.page_url or "",
        "source":          "extension",
        "risk_label":      _risk_label(result["risk_score"]),
        "risk_color":      _risk_color(result["risk_score"]),
    }


# ─── 4. Video Analysis (unified) ─────────────────────────────────────────────

@router.post("/analyze-unified-video")
async def analyze_unified_video(
    file:         UploadFile = File(...),
    tier:         str        = Form("balanced"),
    sample_fps:   float      = Form(2.0),
    max_frames:   int        = Form(20),
    detect_faces: bool       = Form(True),
):
    """
    Orchestrated video deepfake analysis with temporal aggregation.
    Samples frames, runs all models per frame, aggregates temporally.
    """
    import tempfile
    from pathlib import Path
    from backend.ai.video_analyzer import analyze_video_advanced
    from backend.orchestrator.ensemble_aggregator import EnsembleResult, aggregate_temporal

    if not file.content_type.startswith("video/"):
        raise HTTPException(415, "Only video files accepted")

    _validate_tier(tier)
    models_list = TIER_PRESETS.get(tier, TIER_PRESETS["balanced"])

    data = await file.read()
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = await analyze_video_advanced(
            video_path      = tmp_path,
            sample_fps      = sample_fps,
            max_frames      = max_frames,
            models          = models_list,
            detect_faces    = detect_faces,
        )
    except Exception as e:
        raise HTTPException(500, str(e)[:200])
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    result["filename"] = file.filename
    result["tier"]     = tier
    result["endpoint"] = "analyze-unified-video"
    return result


# ─── 5. Health Report ────────────────────────────────────────────────────────

@router.get("/orchestrator-health")
async def orchestrator_health():
    """Full system health report including model health, cache stats, device."""
    import torch

    models = list_all()
    healthy_count = sum(1 for m in models if m["healthy"])
    enabled_count = sum(1 for m in models if m.get("enabled", True))

    return {
        "status":         "ok" if healthy_count > 0 else "degraded",
        "healthy_models": healthy_count,
        "enabled_models": enabled_count,
        "total_models":   len(models),
        "device":         "cuda" if torch.cuda.is_available() else "cpu",
        "cache":          cache_service.stats(),
        "tier_presets":   {k: v for k, v in TIER_PRESETS.items()},
        "models":         models,
    }


# ─── 6. Model Registry Status ────────────────────────────────────────────────

@router.get("/orchestrator-status")
async def orchestrator_status():
    """Compact model status for dashboard widget."""
    models = list_all()
    return {
        "models":       models,
        "total":        len(models),
        "healthy":      sum(1 for m in models if m["healthy"]),
        "enabled":      enabled_healthy(),
        "tier_presets": list(TIER_PRESETS.keys()),
        "cache":        cache_service.stats(),
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _validate_tier(tier: Optional[str]):
    if tier and tier not in TIER_PRESETS:
        raise HTTPException(400, f"Unknown tier '{tier}'. Available: {list(TIER_PRESETS)}")


def _risk_label(score: int) -> str:
    if score >= 65:  return "HIGH RISK"
    if score >= 30:  return "SUSPICIOUS"
    return "LOW RISK"


def _risk_color(score: int) -> str:
    if score >= 65:  return "#ef4444"
    if score >= 30:  return "#f59e0b"
    return "#22c55e"
