"""
DeepShield AI — Models Management API

GET  /api/models           — list all available models with stats
GET  /api/models/active    — current active model info
POST /api/models/load      — switch active model
GET  /api/models/benchmark — run quick benchmark on current model
POST /api/models/export    — export active model to ONNX
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from loguru import logger
import time

from backend.detection.model_zoo import list_models, AVAILABLE_MODELS, export_onnx
from backend.detection.ml_pipeline import switch_model, get_active_model_name

router = APIRouter()


class LoadModelRequest(BaseModel):
    model_name: str


class ExportRequest(BaseModel):
    model_name: Optional[str] = None


@router.get("/")
async def get_models():
    """List all available deepfake detection models with capabilities."""
    models = list_models()
    active = get_active_model_name()
    for m in models:
        m["active"] = (m["name"] == active)
    return {
        "status": "ok",
        "active_model": active,
        "models": models,
        "device": _get_device_info(),
    }


@router.get("/active")
async def get_active_model():
    """Return currently loaded model details."""
    name = get_active_model_name()
    if name == "none":
        return {"status": "ok", "active": None, "message": "No model loaded yet — first inference will auto-load efficientnet_b4"}
    cfg = AVAILABLE_MODELS.get(name, {})
    return {"status": "ok", "active": name, **cfg}


@router.post("/load")
async def load_model(req: LoadModelRequest):
    """
    Load (or switch to) a specific model.
    First call may take 10–30 seconds to download pretrained weights.
    """
    if req.model_name not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model '{req.model_name}'. "
                            f"Available: {list(AVAILABLE_MODELS.keys())}")
    logger.info(f"[ModelsAPI] Loading model: {req.model_name}")
    result = switch_model(req.model_name)
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    return result


@router.get("/benchmark")
async def benchmark_model():
    """Run a quick timing benchmark on the active model using a blank 224×224 image."""
    import torch, base64, io
    import numpy as np
    from PIL import Image
    from backend.detection.ml_pipeline import ml_analyze_frame

    # Create a synthetic face-like test image (white noise)
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil = Image.fromarray(test_img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    start = time.perf_counter()
    result = ml_analyze_frame(b64, return_heatmap=False)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    return {
        "status": "ok",
        "model": get_active_model_name(),
        "latency_ms": elapsed_ms,
        "device": _get_device_info(),
        "verdict": result.get("verdict"),
    }


@router.post("/export")
async def export_model(req: ExportRequest, background_tasks: BackgroundTasks):
    """Export model to ONNX for faster CPU inference via onnxruntime."""
    name = req.model_name or get_active_model_name()
    if name == "none":
        raise HTTPException(400, "No model loaded. Call /api/models/load first.")

    def _do_export():
        path = export_onnx(name)
        logger.success(f"[ModelsAPI] ONNX export complete → {path}")

    background_tasks.add_task(_do_export)
    return {
        "status": "ok",
        "message": f"ONNX export started for '{name}' — will save to models/{name}_deepfake.onnx",
    }


def _get_device_info() -> dict:
    import torch
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        }
    return {
        "type": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": gpu,
    }
