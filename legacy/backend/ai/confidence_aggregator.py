"""
DeepShield AI — Multi-Model Confidence Aggregator

Enhanced with weighted soft voting from backend.ml.ensemble
Follows requirements: ViT=0.30, EfficientNet-B4=0.25, Xception=0.20, ConvNeXt=0.15, Audio=0.10

Fuses confidence scores from multiple deepfake detection models
into a single final verdict using weighted ensemble logic.
"""

from __future__ import annotations
import numpy as np
from loguru import logger

# Import enhanced ensemble voting
try:
    from backend.ml.ensemble import (
        weighted_soft_voting,
        ModelPrediction,
        get_model_weight,
        calculate_uncertainty
    )
    ENHANCED_ENSEMBLE_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced ensemble module not available, using fallback weights")
    ENHANCED_ENSEMBLE_AVAILABLE = False

# Model weights for ensemble fusion (as per requirements)
MODEL_WEIGHTS: dict[str, float] = {
    "vit_deepfake_primary":    0.30,   # ViT → 30% weight
    "vit_deepfake_secondary":  0.30,   # ViT → 30% weight
    "efficientnet_b4":         0.25,   # EfficientNet-B4 → 25% weight
    "xception":                0.20,   # Xception → 20% weight
    "convnext_base":           0.15,   # ConvNeXt → 15% weight
    "audio_cnn":               0.10,   # Audio CNN → 10% weight
    "rawnet2":                 0.10,   # RawNet2 → 10% weight
    "lcnn":                    0.10,   # LCNN → 10% weight
    "frequency_dct":           0.05,   # Frequency analysis → 5% weight
    "frequency_fft":           0.05,   # Frequency analysis → 5% weight
}
DEFAULT_WEIGHT = 0.10

FAKE_THRESHOLD = 0.50   # final fused score above this → FAKE
ABSTAIN_THRESHOLD = 0.35  # top-2 delta threshold for abstaining


def aggregate(model_results: list[dict]) -> dict:
    """
    Enhanced weighted average fusion using backend.ml.ensemble if available.

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
        "abstain":     bool (if disagreement too high),
        "uncertainty": dict (variance, entropy, etc.)
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
            "abstain": False,
            "uncertainty": {}
        }

    # Try using enhanced ensemble voting
    if ENHANCED_ENSEMBLE_AVAILABLE:
        try:
            predictions = []
            for r in model_results:
                name = r.get("model", "unknown")
                conf = float(r.get("confidence", 0.5))
                verdi = r.get("verdict", "REAL")
                
                # Convert to fake probability
                fake_prob = conf if verdi == "FAKE" else (1.0 - conf)
                
                # Get weight from enhanced system
                weight = get_model_weight(name)
                
                # Create ModelPrediction
                pred = ModelPrediction(
                    model_name=name,
                    confidence=fake_prob,
                    raw_logits=np.array([1-fake_prob, fake_prob]),
                    latency_ms=r.get("latency_ms", 0.0),
                    weight=weight
                )
                predictions.append(pred)
            
            # Use enhanced ensemble voting
            result = weighted_soft_voting(predictions, abstain_threshold=ABSTAIN_THRESHOLD)
            
            # Calculate uncertainty metrics
            uncertainty = calculate_uncertainty(predictions)
            
            # Convert to expected format
            return {
                "verdict": result.verdict,
                "confidence": round(result.final_confidence, 4),
                "fake_prob": round(result.final_confidence if result.verdict == "FAKE" else 1.0 - result.final_confidence, 4),
                "fake_votes": sum(1 for p in predictions if p.confidence > 0.5),
                "real_votes": sum(1 for p in predictions if p.confidence <= 0.5),
                "total_weight": sum(p.weight for p in predictions),
                "per_model": [
                    {
                        "model": p.model_name,
                        "verdict": "FAKE" if p.confidence > 0.5 else "REAL",
                        "confidence": round(p.confidence, 4),
                        "weight": p.weight,
                        "fake_prob": round(p.confidence, 4)
                    }
                    for p in predictions
                ],
                "abstain": result.abstain,
                "agreement_score": round(result.agreement_score, 4),
                "uncertainty": uncertainty,
                "weighted_votes": result.weighted_votes
            }
        except Exception as e:
            logger.warning(f"Enhanced ensemble failed, using fallback: {e}")
            # Fall through to fallback implementation

    # Fallback implementation (original logic)
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

    # Check abstain condition
    confidences = [r.get("confidence", 0.5) for r in model_results]
    sorted_confs = sorted(confidences, reverse=True)
    abstain = False
    if len(sorted_confs) >= 2:
        top_2_delta = abs(sorted_confs[0] - sorted_confs[1])
        abstain = top_2_delta > ABSTAIN_THRESHOLD

    return {
        "verdict":      "FAKE" if is_fake else "REAL",
        "confidence":   round(float(confidence), 4),
        "fake_prob":    round(float(fused_fake_prob), 4),
        "fake_votes":   fake_votes,
        "real_votes":   real_votes,
        "total_weight": round(total_weight, 2),
        "per_model":    per_model,
        "abstain":      abstain,
        "agreement_score": round(1.0 - np.std(confidences), 4) if len(confidences) > 1 else 1.0,
        "uncertainty": {
            "variance": round(float(np.var(confidences)), 4),
            "std": round(float(np.std(confidences)), 4)
        }
    }


def aggregate_temporal(frame_results: list[dict]) -> dict:
    """
    Enhanced temporal consistency fusion for video frame sequences.
    
    Features:
    - Weights recent frames higher (exponential decay with α=0.85)
    - Detects temporal anomalies (variance > 0.15 threshold as per requirements)
    - Flags suspicious timestamps where score variance is high
    - Returns overall verdict + temporal consistency score
    """
    if not frame_results:
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "temporal_consistency": 0.0,
            "suspicious_timestamps": [],
            "frame_variance": 0.0,
            "temporal_anomaly_detected": False
        }

    n      = len(frame_results)
    alpha  = 0.85
    weights = [alpha ** (n - 1 - i) for i in range(n)]

    fake_probs = []
    timestamps = []
    for r in frame_results:
        conf  = r.get("confidence", 0.5)
        is_f  = r.get("verdict", "REAL") == "FAKE"
        fake_p = conf if is_f else (1.0 - conf)
        fake_probs.append(fake_p)
        timestamps.append(r.get("timestamp", 0.0))

    # Weighted average
    w_sum  = sum(w * p for w, p in zip(weights, fake_probs))
    w_norm = sum(weights)
    fused  = w_sum / w_norm if w_norm > 0 else 0.5

    # Temporal consistency: low std → consistent predictions
    std  = float(np.std(fake_probs))
    variance = float(np.var(fake_probs))
    cons = round(max(0.0, 1.0 - 2 * std), 4)   # 0=chaotic, 1=perfectly consistent

    # Detect temporal anomalies (variance > 0.15 as per Module B requirements)
    VARIANCE_THRESHOLD = 0.15
    temporal_anomaly = variance > VARIANCE_THRESHOLD

    # Find suspicious timestamps (frames where local variance is high)
    suspicious_timestamps = []
    if n >= 3:
        for i in range(1, n - 1):
            # Calculate local variance (3-frame window)
            local_probs = fake_probs[max(0, i-1):min(n, i+2)]
            local_var = float(np.var(local_probs))
            
            if local_var > VARIANCE_THRESHOLD:
                suspicious_timestamps.append({
                    "timestamp": round(timestamps[i], 2),
                    "frame_index": i,
                    "confidence": round(fake_probs[i], 4),
                    "local_variance": round(local_var, 4)
                })

    is_fake    = fused >= FAKE_THRESHOLD
    confidence = fused if is_fake else (1.0 - fused)

    result = {
        "verdict":               "FAKE" if is_fake else "REAL",
        "confidence":            round(float(confidence), 4),
        "fake_prob":             round(float(fused), 4),
        "temporal_consistency":  cons,
        "frame_count":           n,
        "fake_frames":           sum(1 for r in frame_results if r.get("verdict") == "FAKE"),
        "real_frames":           sum(1 for r in frame_results if r.get("verdict") == "REAL"),
        "frame_variance":        round(variance, 4),
        "temporal_anomaly_detected": temporal_anomaly,
        "suspicious_timestamps": suspicious_timestamps,
    }

    if temporal_anomaly:
        logger.warning(
            f"Temporal anomaly detected: variance={variance:.4f} > {VARIANCE_THRESHOLD}, "
            f"{len(suspicious_timestamps)} suspicious frames"
        )

    return result
