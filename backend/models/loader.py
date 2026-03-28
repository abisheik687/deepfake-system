"""
Internal trace:
- Wrong before: models were claimed to load on demand, which caused latency, inconsistent caches, and unpredictable request-time failures.
- Fixed now: startup builds one registry for the four image slots plus audio, records whether each slot is primary or fallback, and exposes it application-wide.
"""

from __future__ import annotations

import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable

from fastapi import FastAPI

from config import settings
from models.audio_model import AudioModelHandle, build_audio_model, build_fallback_audio_model
from models.image_models import ImageModelSlot, create_fallback_scorer, create_image_slots
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class LoadedImageModel:
    slot: ImageModelSlot
    infer: Callable
    mode: str
    warning: str | None = None


@dataclass
class ModelRegistry:
    image_models: list[LoadedImageModel] = field(default_factory=list)
    audio_model: AudioModelHandle | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def loaded_count(self) -> int:
        return len(self.image_models) + (1 if self.audio_model else 0)


_registry = ModelRegistry()


def _can_reach_huggingface() -> bool:
    if not settings.enable_remote_model_downloads:
        return False
    try:
        with socket.create_connection(('huggingface.co', 443), timeout=1):
            return True
    except OSError:
        return False


async def _load_models() -> None:
    _registry.image_models.clear()
    _registry.warnings.clear()

    remote_available = _can_reach_huggingface()
    if not remote_available:
        _registry.warnings.append('Remote model hub unavailable or disabled; using local forensic fallbacks')

    for slot in create_image_slots():
        if not remote_available:
            infer = create_fallback_scorer(slot.key)
            warning = f'{slot.label} primary model unavailable; using forensic fallback scorer'
            _registry.image_models.append(LoadedImageModel(slot=slot, infer=infer, mode='fallback', warning=warning))
            _registry.warnings.append(warning)
            continue
        try:
            infer, mode = slot.loader()
            _registry.image_models.append(LoadedImageModel(slot=slot, infer=infer, mode=mode))
        except Exception as exc:
            infer = create_fallback_scorer(slot.key)
            warning = f'{slot.label} primary model unavailable; using forensic fallback scorer'
            _registry.image_models.append(LoadedImageModel(slot=slot, infer=infer, mode='fallback', warning=warning))
            _registry.warnings.append(warning)
            logger.warning('image_model_fallback', extra={'model': slot.label, 'error': str(exc)})

    _registry.audio_model = build_audio_model() if remote_available else build_fallback_audio_model()
    if _registry.audio_model.mode == 'fallback':
        _registry.warnings.append('Audio primary model unavailable; using signal fallback scorer')


@asynccontextmanager
async def model_lifespan(_: FastAPI):
    await _load_models()
    yield
    _registry.image_models.clear()
    _registry.audio_model = None
    _registry.warnings.clear()


def get_model_registry() -> ModelRegistry:
    return _registry
