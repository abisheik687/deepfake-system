"""
Internal trace:
- Wrong before: duplicate health endpoints disagreed on payload shape and pulled in database/GPU checks unrelated to the new upload pipeline.
- Fixed now: one fast health endpoint reports overall status and model slot readiness in the contract the frontend expects.
"""

from __future__ import annotations

from fastapi import APIRouter

from models.loader import get_model_registry


router = APIRouter(tags=['health'])


@router.get('/health')
async def health() -> dict[str, int | str]:
    registry = get_model_registry()
    return {'status': 'ok', 'models_loaded': registry.loaded_count}
