"""
ml_pipeline.py — DEPRECATED
================================================================================
This module has been superseded by the orchestrator/ package.
All functions below are compatibility shims that delegate to the canonical
implementations in backend/orchestrator/ and backend/ai/hf_registry.py.

DO NOT add new logic here. Update callers to import from orchestrator/ directly.
This file will be removed in a future release.
================================================================================
"""
import warnings
import functools
from typing import Any

_DEPRECATION_MSG = (
    "{name} in ml_pipeline is deprecated and will be removed. "
    "Use {replacement} instead."
)


def _deprecated(name: str, replacement: str):
    """Decorator: emit DeprecationWarning once per call site."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(
                _DEPRECATION_MSG.format(name=name, replacement=replacement),
                DeprecationWarning,
                stacklevel=2,
            )
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ── Shims ────────────────────────────────────────────────────────────────────

@_deprecated(
    name="ml_pipeline._frequency_score",
    replacement="backend.orchestrator.task_runner._infer_frequency",
)
def _frequency_score(face_bgr: Any) -> float:
    from backend.orchestrator.task_runner import _infer_frequency
    import cv2
    import numpy as np
    from PIL import Image
    
    # Needs to adapt to PIL Image if caller still passes BGR array
    if isinstance(face_bgr, np.ndarray):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
    else:
        pil_image = face_bgr
        
    fp, _ = _infer_frequency(pil_image, "combined")
    return fp


@_deprecated(
    name="ml_pipeline.ml_analyze_frame",
    replacement="backend.orchestrator.orchestrator.analyze",
)
async def ml_analyze_frame(*args, **kwargs) -> dict:
    from backend.orchestrator.orchestrator import analyze
    return await analyze(*args, **kwargs)


@_deprecated(
    name="ml_pipeline._load_model",
    replacement="backend.ai.hf_registry.load_hf_model",
)
def _load_model(name: str) -> Any:
    from backend.ai.hf_registry import load_hf_model
    return load_hf_model(name)
