"""
Internal trace:
- Wrong before: image analysis merged orchestration, face extraction, and response shaping across multiple services and dict payloads.
- Fixed now: image uploads flow through a single preprocessing and ensemble path that returns the new response model.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from models.ensemble import combine_weighted_scores
from models.loader import ModelRegistry
from models.image_models import prepare_image
from schemas.response import AnalysisResult, ModelScore


async def analyse_image_file(file_path: Path, registry: ModelRegistry, validation) -> AnalysisResult:
    image_bytes = file_path.read_bytes()
    prepared_image = await asyncio.to_thread(prepare_image, image_bytes)

    model_scores: list[ModelScore] = []
    warnings = list(registry.warnings)
    weighted_scores: list[tuple[float, float]] = []

    for loaded in registry.image_models:
        fake_probability = await asyncio.to_thread(loaded.infer, prepared_image)
        weighted_scores.append((fake_probability, loaded.slot.weight))
        model_scores.append(
            ModelScore(
                model=loaded.slot.label,
                fake_prob=round(fake_probability, 4),
                weight=loaded.slot.weight,
                mode=loaded.mode,
            )
        )
        if loaded.warning and loaded.warning not in warnings:
            warnings.append(loaded.warning)

    fake_probability, verdict, confidence = combine_weighted_scores(weighted_scores)
    return AnalysisResult(
        file_type=validation.file_type,
        verdict=verdict,
        overall_confidence=round(confidence, 4),
        fake_probability=round(fake_probability, 4),
        model_scores=model_scores,
        warnings=warnings,
    )
