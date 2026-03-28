"""
DeepShield AI — Unified API Endpoints (Orchestration Layer)

Endpoints:
  POST /analyze-unified         → multipart file upload (image/video) or JSON base64
  POST /live-unified             → optimized webcam frame analysis
  POST /extension-scan           → browser extension primary endpoint
  POST /analyze-unified-video     → temporal video orchestration
  GET  /orchestrator-health       → full system health report
  GET  /orchestrator-status      → model registry status
"""

import io
import base64
import hashlib
import time
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from loguru import logger

from backend.orchestrator import orchestrator, cache_service
from backend.orchestrator.model_registry import REGISTRY, list_all, enabled_healthy, TIER_PRESETS
from backend.database import get_db
from backend.crud import create_detection
from backend.api.rate_limit import limiter
from backend.metrics import DETECTIONS_TOTAL, ACTIVE_SCANS, CONFIDENCE_HISTOGRAM

router = APIRouter()

ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_VIDEO = {"video/mp4", "video/webm", "video/quicktime"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".mp4", ".webm", ".mov"}


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


# ─── 1. Unified Image Analysis (JSON body with base64) ────────────────────────

@router.post("/analyze-unified")
@limiter.limit("20/minute")
async def analyze_unified(
    request: Request,
    req: Optional[UnifiedImageRequest] = None,
    file: Optional[UploadFile] = File(None),
    tier: str = Query("balanced"),
    return_heatmap: bool = Query(False),
    detect_faces: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    """
    Full orchestrated analysis. Accepts either:
    - JSON body: { "data": "base64...", "tier": "balanced", ... }
    - Multipart: file= (image or video), tier, return_heatmap, detect_faces.

    Validates file type (jpg, png, mp4, webm, mov). Saves result to DB. Returns real response.
    """
    ACTIVE_SCANS.inc()
    try:
        t0 = time.perf_counter()
        image_input = None
        file_bytes = None
        filename = ""
        media_type = "image"

        if file is not None and file.filename and file.filename.strip():
            filename = file.filename
            ext = filename[filename.rfind(".") :].lower() if "." in filename else ""
            if ext not in ALLOWED_EXT:
                raise HTTPException(400, f"File type not allowed. Use: {', '.join(ALLOWED_EXT)}")
            file_bytes = await file.read()
            if not file_bytes:
                raise HTTPException(400, "Empty file")
            if ext in (".mp4", ".webm", ".mov"):
                media_type = "video"
            else:
                image_input = base64.b64encode(file_bytes).decode()
                if ext in (".png",):
                    image_input = "data:image/png;base64," + image_input
                else:
                    image_input = "data:image/jpeg;base64," + image_input
        elif req and req.data:
            image_input = req.data
            tier = req.tier or tier
            return_heatmap = req.return_heatmap or return_heatmap
            detect_faces = req.detect_faces if req.detect_faces is not None else detect_faces
        else:
            raise HTTPException(400, "Either send JSON with 'data' (base64) or multipart 'file'")

        _validate_tier(tier)

        if media_type == "video" and file_bytes:
            from pathlib import Path
            import tempfile
            suffix = Path(filename).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                from backend.ai.video_analyzer import analyze_video_advanced
                result = await analyze_video_advanced(
                    video_path=tmp_path,
                    sample_fps=2.0,
                    max_frames=20,
                    models=TIER_PRESETS.get(tier, TIER_PRESETS["balanced"]),
                    detect_faces=detect_faces,
                )
                result["face_count"] = result.get("frame_count", 0)
                result["processing_time_ms"] = (time.perf_counter() - t0) * 1000
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            try:
                result = await orchestrator.analyze(
                    image_input=image_input,
                    models=None,
                    tier=tier,
                    use_cache=True,
                    detect_faces=detect_faces,
                    return_heatmap=return_heatmap,
                    source="web",
                )
            except Exception as e:
                logger.error(f"[UnifiedAPI] /analyze-unified error: {e}")
                raise HTTPException(500, f"Orchestration error: {str(e)[:200]}")

        processing_time_ms = (time.perf_counter() - t0) * 1000
        result["processing_time_ms"] = round(processing_time_ms, 1)
        risk_score = result.get("risk_score", 0.0)
        if isinstance(risk_score, (int, float)) and risk_score > 1:
            risk_score = risk_score / 100.0
        result["risk_score"] = round(float(risk_score), 4)
        model_breakdown = {}
        for p in result.get("per_model", []):
            name = p.get("model") or p.get("model_name")
            if name:
                model_breakdown[name] = round(float(p.get("cal_fake_prob", p.get("raw_fake_prob", 0.5))), 4)
        if not model_breakdown and result.get("per_model"):
            for p in result["per_model"]:
                name = p.get("model")
                if name:
                    model_breakdown[name] = round(float(p.get("raw_fake_prob", 0.5)), 4)
        result["model_breakdown"] = model_breakdown
        result["face_count"] = result.get("faces_found", 0)

        file_hash = None
        if file_bytes:
            file_hash = hashlib.sha256(file_bytes).hexdigest()
        try:
            scan = await create_detection(db, result, file_hash=file_hash, filename=filename or None)
            detection_id = scan.task_id
        except Exception as e:
            logger.warning(f"[UnifiedAPI] Could not save detection to DB: {e}")
            import uuid
            detection_id = str(uuid.uuid4())

        response_payload = {
            "detection_id": detection_id,
            "verdict": result.get("verdict", "REAL"),
            "risk_score": result["risk_score"],
            "confidence": round(float(result.get("confidence", 0)), 4),
            "model_breakdown": result["model_breakdown"],
            "heatmap_b64": result.get("heatmap_b64") or None,
            "processing_time_ms": result["processing_time_ms"],
            "face_count": result["face_count"],
        }

        DETECTIONS_TOTAL.labels(source="web", tier=tier).inc()
        CONFIDENCE_HISTOGRAM.labels(verdict=result.get("verdict", "REAL")).observe(float(result.get("confidence", 0)))
        
        return response_payload
    finally:
        ACTIVE_SCANS.dec()


# ─── 2. Live Unified (webcam frames) ─────────────────────────────────────────

@router.post("/live-unified")
@limiter.limit("60/minute")
async def live_unified(request: Request, req: LiveFrameRequest):
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
@limiter.limit("20/minute")
async def extension_scan(request: Request, req: ExtensionScanRequest):
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
@limiter.limit("5/minute")
async def analyze_unified_video(
    request: Request,
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
