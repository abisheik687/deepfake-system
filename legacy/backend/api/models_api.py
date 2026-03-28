"""
DeepShield AI — Models Management API
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from loguru import logger
import time
import torch

from backend.ai.hf_registry import get_model_info, load_hf_model
from backend.orchestrator.orchestrator import analyze

router = APIRouter()


class LoadModelRequest(BaseModel):
    model_name: str


class ExportRequest(BaseModel):
    model_name: Optional[str] = None


@router.get("/")
async def get_models():
    """List all available deepfake detection models with capabilities."""
    models = get_model_info()
    return {
        "status": "ok",
        "active_model": "ensemble",  # No single active model anymore
        "models": models,
        "device": _get_device_info(),
    }


@router.get("/active")
async def get_active_model():
    """Return currently loaded model details."""
    return {"status": "ok", "active": "ensemble", "message": "All models are orchestrated dynamically."}


@router.post("/load")
async def load_model(req: LoadModelRequest):
    """
    Pre-load a specific model into memory.
    """
    try:
        load_hf_model(req.model_name)
        return {"status": "ok", "message": f"{req.model_name} loaded successfully."}
    except Exception as e:
        logger.error(f"[ModelsAPI] Error loading model: {e}")
        raise HTTPException(500, str(e))


@router.get("/benchmark")
async def benchmark_model():
    """Run a quick timing benchmark on the ensemble orchestrator using a blank 224×224 image."""
    import base64, io
    import numpy as np
    from PIL import Image

    # Create a synthetic face-like test image (white noise)
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil = Image.fromarray(test_img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    start = time.perf_counter()
    result = await analyze(b64)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    return {
        "status": "ok",
        "model": "ensemble",
        "latency_ms": elapsed_ms,
        "device": _get_device_info(),
        "verdict": result.get("verdict"),
    }


@router.post("/export")
async def export_model(req: ExportRequest, background_tasks: BackgroundTasks):
    """Export model to ONNX for faster CPU inference via onnxruntime."""
    return {
        "status": "ok",
        "message": "ONNX export is now handled via the training pipeline directly. Run ./training/train.sh to generate .onnx files.",
    }


def _get_device_info() -> dict:
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
