"""Health check endpoint for KAVACH-AI."""
import time
import torch
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.database import get_db

health_router = APIRouter(prefix="/health", tags=["System"])
_START_TIME = time.time()


@health_router.get("/", summary="System health check")
async def health_check(db: Session = Depends(get_db)) -> dict:
    # DB connectivity
    try:
        db.execute("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False

    # GPU status
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return {
        "status": "ok" if db_ok else "degraded",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "database": "connected" if db_ok else "error",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
    }
