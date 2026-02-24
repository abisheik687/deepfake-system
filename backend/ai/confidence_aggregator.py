"""
DeepShield AI — Multi-Model Confidence Aggregator

Fuses confidence scores from multiple deepfake detection models
into a single final verdict using weighted ensemble logic.

Weighting strategy:
  • Fine-tuned HF models (ViT deepfake-specific): weight 1.0
  • Frequency analysis score:                     weight 0.15 (additive signal)
  • ImageNet-only timm models without fine-tuning: weight 0.5
"""

from __future__ import annotations
import numpy as np
from loguru import logger

# Model weights for ensemble fusion
MODEL_WEIGHTS: dict[str, float] = {
    "vit_deepfake_primary":    1.00,   # HF fine-tuned → highest trust
    "vit_deepfake_secondary":  0.90,   # HF fine-tuned → high trust
    "efficientnet_b4":         0.60,   # ImageNet only  → medium trust
    "xception":                0.60,   # ImageNet only  → medium trust
}
DEFAULT_WEIGHT = 0.50

FAKE_THRESHOLD = 0.50   # final fused score above this → FAKE


def aggregate(model_results: list[dict]) -> dict:
    """
    Weighted average fusion of individual model verdicts.

    Parameters
    ----------
    model_results : list of dicts from infer_single()
        Each must have: {"model": str, "verdict": str, "confidence": float}

    Returns
    -------
    {
        "verdict":     "FAKE" | "REAL",
        "confidence":  float 0–1,
        "fake_votes":  int,
        "real_votes":  int,
        "total_weight": float,
        "per_model":    list,
    }
    """
    if not model_results:
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "fake_votes": 0,
            "real_votes": 0,
            "total_weight": 0.0,
            "per_model": [],
        }

    weighted_fake_sum = 0.0
    total_weight      = 0.0
    fake_votes        = 0
    real_votes        = 0
    per_model         = []

    for r in model_results:
        name  = r.get("model", "unknown")
        conf  = float(r.get("confidence", 0.5))
        verdi = r.get("verdict", "REAL")
        weight = MODEL_WEIGHTS.get(name, DEFAULT_WEIGHT)

        # Convert verdict+confidence to a FAKE probability
        fake_prob = conf if verdi == "FAKE" else (1.0 - conf)

        weighted_fake_sum += weight * fake_prob
        total_weight      += weight

        if verdi == "FAKE":
            fake_votes += 1
        else:
            real_votes += 1

        per_model.append({
            "model":     name,
            "verdict":   verdi,
            "confidence": conf,
            "weight":    weight,
            "fake_prob": round(fake_prob, 4),
        })

    fused_fake_prob = weighted_fake_sum / total_weight if total_weight > 0 else 0.5
    is_fake         = fused_fake_prob >= FAKE_THRESHOLD
    confidence      = fused_fake_prob if is_fake else (1.0 - fused_fake_prob)

    return {
        "verdict":      "FAKE" if is_fake else "REAL",
        "confidence":   round(float(confidence), 4),
        "fake_prob":    round(float(fused_fake_prob), 4),
        "fake_votes":   fake_votes,
        "real_votes":   real_votes,
        "total_weight": round(total_weight, 2),
        "per_model":    per_model,
    }


def aggregate_temporal(frame_results: list[dict]) -> dict:
    """
    Temporal consistency fusion for video frame sequences.

    Weights recent frames higher (exponential decay with α=0.85).
    Returns overall verdict + temporal consistency score.
    """
    if not frame_results:
        return {"verdict": "ERROR", "confidence": 0.0, "temporal_consistency": 0.0}

    n      = len(frame_results)
    alpha  = 0.85
    weights = [alpha ** (n - 1 - i) for i in range(n)]

    fake_probs = []
    for r in frame_results:
        conf  = r.get("confidence", 0.5)
        is_f  = r.get("verdict", "REAL") == "FAKE"
        fake_p = conf if is_f else (1.0 - conf)
        fake_probs.append(fake_p)

    # Weighted average
    w_sum  = sum(w * p for w, p in zip(weights, fake_probs))
    w_norm = sum(weights)
    fused  = w_sum / w_norm if w_norm > 0 else 0.5

    # Temporal consistency: low std → consistent predictions
    std  = float(np.std(fake_probs))
    cons = round(max(0.0, 1.0 - 2 * std), 4)   # 0=chaotic, 1=perfectly consistent

    is_fake    = fused >= FAKE_THRESHOLD
    confidence = fused if is_fake else (1.0 - fused)

    return {
        "verdict":               "FAKE" if is_fake else "REAL",
        "confidence":            round(float(confidence), 4),
        "fake_prob":             round(float(fused), 4),
        "temporal_consistency":  cons,
        "frame_count":           n,
        "fake_frames":           sum(1 for r in frame_results if r.get("verdict") == "FAKE"),
        "real_frames":           sum(1 for r in frame_results if r.get("verdict") == "REAL"),
    }
