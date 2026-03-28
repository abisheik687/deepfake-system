"""
Internal trace:
- Wrong before: audio analysis mixed request parsing, feature visualization, temp file I/O, and heuristic scoring in the router itself.
- Fixed now: audio inference and waveform extraction happen here and return a clean schema payload.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import librosa
import numpy as np

from models.loader import ModelRegistry
from schemas.response import AnalysisResult, AudioResult


def _build_waveform(audio: np.ndarray, points: int = 96) -> list[float]:
    if audio.size == 0:
        return []
    chunks = np.array_split(audio, points)
    return [round(float(np.mean(np.abs(chunk))), 4) for chunk in chunks if chunk.size]


async def analyse_audio_file(file_path: Path, registry: ModelRegistry, validation) -> AnalysisResult:
    audio, sample_rate = await asyncio.to_thread(librosa.load, str(file_path), sr=16000, mono=True)
    handle = registry.audio_model
    fake_probability = await asyncio.to_thread(handle.infer, audio, sample_rate) if handle else 0.5
    verdict = 'FAKE' if fake_probability > 0.52 else 'REAL'
    confidence = max(fake_probability, 1.0 - fake_probability)
    warnings = list(registry.warnings)

    return AnalysisResult(
        file_type=validation.file_type,
        verdict=verdict,
        overall_confidence=round(confidence, 4),
        fake_probability=round(fake_probability, 4),
        audio_result=AudioResult(
            verdict=verdict,
            fake_probability=round(fake_probability, 4),
            waveform=_build_waveform(audio),
            mode=handle.mode if handle else 'missing',
        ),
        warnings=warnings,
    )
