"""
Internal trace:
- Wrong before: ensemble logic existed in several places with incompatible thresholds and verdict names such as SUSPICIOUS.
- Fixed now: one weighted-voting module implements the required fake threshold and uncertainty rule for image and video analysis.
"""

from __future__ import annotations

from statistics import mean

from config import settings


def combine_weighted_scores(scores: list[tuple[float, float]]) -> tuple[float, str, float]:
    if not scores:
        return 0.5, 'UNCERTAIN', 0.5

    total_weight = sum(weight for _, weight in scores)
    fake_probability = sum(prob * weight for prob, weight in scores) / max(total_weight, 1e-6)
    spread = max(prob for prob, _ in scores) - min(prob for prob, _ in scores)
    confidence = max(fake_probability, 1.0 - fake_probability)

    if spread > settings.disagreement_threshold:
        return fake_probability, 'UNCERTAIN', confidence
    if fake_probability > settings.default_image_threshold:
        return fake_probability, 'FAKE', fake_probability
    return fake_probability, 'REAL', 1.0 - fake_probability


def aggregate_video_scores(frame_scores: list[float]) -> tuple[float, str, float]:
    if not frame_scores:
        return 0.5, 'UNCERTAIN', 0.5
    fake_probability = (0.7 * mean(frame_scores)) + (0.3 * max(frame_scores))
    spread = max(frame_scores) - min(frame_scores)
    confidence = max(fake_probability, 1.0 - fake_probability)
    if spread > settings.disagreement_threshold:
        return fake_probability, 'UNCERTAIN', confidence
    if fake_probability > settings.default_image_threshold:
        return fake_probability, 'FAKE', fake_probability
    return fake_probability, 'REAL', 1.0 - fake_probability
