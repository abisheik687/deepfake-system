"""
DeepShield AI — Ensemble Confidence Aggregator

Combines calibrated fake probabilities from N models into a final verdict.

Algorithm:
  1. Temperature-scale each model's raw fake_prob
  2. Compute weighted average (soft voting) using registry weights
  3. Compute model agreement score (1 - normalized variance)
  4. Compute Deepfake Risk Score (0–100) with calibration corrections
  5. Apply verdict thresholds:
       REAL         →  risk_score < 30
       SUSPICIOUS   →  30 ≤ risk_score < 65
       FAKE         →  risk_score ≥ 65
  6. Compute reliability index from model agreement + sample coverage

Output schema (see EnsembleResult dataclass):
{
  "verdict":           "FAKE" | "SUSPICIOUS" | "REAL",
  "risk_score":        int 0–100,
  "confidence":        float 0–1,      ← P(verdict is correct)
  "fake_prob":         float 0–1,      ← weighted P(fake)
  "agreement_score":   float 0–1,      ← 1=all models agree, 0=split opinions
  "variance":          float           ← variance of fake_probs across models
  "reliability_index": float 0–1,      ← overall confidence in the verdict
  "models_used":       int,
  "models_timed_out":  int,
  "models_failed":     int,
  "per_model":         list[dict],
  "calibrated_probs":  list[float],
  "weights_used":      list[float],
}
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
from loguru import logger

from backend.orchestrator.temperature_scaler import calibrate_prob
from backend.orchestrator.model_registry     import ModelOutput   # canonical location

# Verdict thresholds on risk_score (float 0–1)
FAKE_THRESHOLD        = 0.75   # risk_score >= 0.75 → FAKE (HIGH)
SUSPICIOUS_THRESHOLD  = 0.50   # 0.50 <= risk_score < 0.75 → SUSPICIOUS (MEDIUM)
# risk_score < 0.50 → REAL (LOW)

# Default weights for weighted ensemble (vit, efficientnet, xception, convnext)
DEFAULT_WEIGHTS: dict[str, float] = {
    "vit_deepfake_primary":   0.30,
    "vit_deepfake_secondary": 0.20,
    "efficientnet_b4":       0.25,
    "xception":              0.25,
    "convnext_base":         0.20,
    "frequency_dct":         0.10,
    "frequency_fft":         0.10,
}


@dataclass
class EnsembleResult:
    verdict:           str
    risk_score:        float   # 0.0–1.0
    confidence:        float
    fake_prob:         float
    agreement_score:   float
    variance:          float
    reliability_index: float
    models_used:       int
    models_timed_out:  int
    models_failed:     int
    per_model:         list = field(default_factory=list)
    calibrated_probs:  list = field(default_factory=list)
    weights_used:      list = field(default_factory=list)
    message:           str  = ""

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate(model_outputs: List[ModelOutput]) -> EnsembleResult:
    """
    Aggregate outputs from multiple models into a single EnsembleResult.

    Handles partial failures: models that timed out or failed are excluded
    from the weighted average but counted in the output stats.
    """
    valid   = [o for o in model_outputs if not o.timed_out and not o.failed]
    failed  = [o for o in model_outputs if o.failed]
    timeouts= [o for o in model_outputs if o.timed_out]

    if not valid:
        logger.error("[Ensemble] All models failed or timed out!")
        return EnsembleResult(
            verdict="SUSPICIOUS",
            risk_score=0.5,
            confidence=0.0,
            fake_prob=0.5,
            agreement_score=0.0,
            variance=0.0,
            reliability_index=0.0,
            models_used=0,
            models_timed_out=len(timeouts),
            models_failed=len(failed),
            message="⚠️ All models unavailable — verdict unreliable",
        )

    # ── Step 1: Temperature-scale each model's fake_prob ────────────────────
    calibrated = []
    weights    = []
    per_model  = []

    for o in valid:
        cal_p  = calibrate_prob(o.fake_prob, o.model_name)
        w      = DEFAULT_WEIGHTS.get(o.model_name, o.weight)
        calibrated.append(cal_p)
        weights.append(w)
        per_model.append({
            "model":          o.model_name,
            "raw_fake_prob":  round(o.fake_prob, 4),
            "cal_fake_prob":  round(cal_p, 4),
            "verdict":        o.verdict,
            "confidence":     round(o.confidence, 4),
            "weight":         w,
            "latency_ms":     o.latency_ms,
        })

    # ── Step 2: Weighted average (soft voting) ───────────────────────────────
    total_weight  = sum(weights)
    weighted_sum  = sum(c * w for c, w in zip(calibrated, weights))
    fused_fake    = weighted_sum / total_weight if total_weight > 0 else 0.5

    # ── Step 3: Variance + agreement ────────────────────────────────────────
    if len(calibrated) >= 2:
        var  = float(statistics.variance(calibrated))
        std  = math.sqrt(var)
        # Agreement: 1 when all probs identical, 0 when maximally spread (0/1 split)
        # Normalize by max possible std (0.5 for binary)
        agreement = max(0.0, 1.0 - (std / 0.5))
    else:
        var       = 0.0
        agreement = 1.0   # only 1 model — no disagreement

    # ── Step 4: Risk score (float 0–1) ───────────────────────────────────────
    risk_score = float(fused_fake)
    # Agreement penalty: low agreement pulls risk toward 0.5
    risk_score = risk_score + (0.5 - risk_score) * (1 - agreement) * 0.20
    risk_score = max(0.0, min(1.0, round(risk_score, 4)))

    # ── Step 5: Verdict (thresholds on 0–1) ───────────────────────────────────
    if risk_score >= FAKE_THRESHOLD:
        verdict    = "FAKE"
        confidence = fused_fake
    elif risk_score >= SUSPICIOUS_THRESHOLD:
        verdict    = "SUSPICIOUS"
        mid        = (FAKE_THRESHOLD + SUSPICIOUS_THRESHOLD) / 2
        confidence = 0.5 + abs(risk_score - mid) / (FAKE_THRESHOLD - SUSPICIOUS_THRESHOLD)
        confidence = min(confidence, 0.99)
    else:
        verdict    = "REAL"
        confidence = 1.0 - fused_fake

    confidence = round(float(confidence), 4)

    # ── Step 6: Reliability index ─────────────────────────────────────────────
    coverage    = len(valid) / max(len(model_outputs), 1)
    thresh      = FAKE_THRESHOLD if verdict == "FAKE" else SUSPICIOUS_THRESHOLD
    dist_thresh = abs(risk_score - thresh)
    reliability = coverage * 0.4 + agreement * 0.4 + dist_thresh * 0.2
    reliability = round(max(0.0, min(1.0, float(reliability))), 4)

    icon = {"FAKE": "⚠️", "SUSPICIOUS": "🔶", "REAL": "✅"}[verdict]
    msg  = (f"{icon} {verdict} — Risk {risk_score:.2f} · "
            f"Agreement {int(agreement*100)}% · "
            f"{len(valid)}/{len(model_outputs)} models succeeded")

    return EnsembleResult(
        verdict           = verdict,
        risk_score        = risk_score,
        confidence        = confidence,
        fake_prob         = round(float(fused_fake), 4),
        agreement_score   = round(float(agreement), 4),
        variance          = round(float(var), 6),
        reliability_index = reliability,
        models_used       = len(valid),
        models_timed_out  = len(timeouts),
        models_failed     = len(failed),
        per_model         = per_model,
        calibrated_probs  = [round(c, 4) for c in calibrated],
        weights_used      = [round(w, 3) for w in weights],
        message           = msg,
    )


def aggregate_from_scores(model_scores: dict[str, float]) -> dict:
    """
    Weighted ensemble from a dict of model_name -> fake_probability (0–1).
    Applies temperature scaling, then weighted sum. Returns verdict, risk_score (float), confidence, model_breakdown.
    """
    if not model_scores:
        return {
            "verdict": "SUSPICIOUS",
            "risk_score": 0.5,
            "confidence": 0.0,
            "model_breakdown": {},
        }
    from backend.orchestrator.temperature_scaler import calibrate_prob
    scaled = {}
    weights_used = []
    for name, raw_p in model_scores.items():
        cal_p = calibrate_prob(float(raw_p), name)
        w = DEFAULT_WEIGHTS.get(name, 0.2)
        scaled[name] = cal_p
        weights_used.append(w)
    total_w = sum(weights_used)
    risk_score = sum(scaled[n] * DEFAULT_WEIGHTS.get(n, 0.2) for n in scaled) / total_w if total_w > 0 else 0.5
    risk_score = max(0.0, min(1.0, round(risk_score, 4)))
    if risk_score >= FAKE_THRESHOLD:
        verdict, confidence = "FAKE", risk_score
    elif risk_score >= SUSPICIOUS_THRESHOLD:
        verdict, confidence = "SUSPICIOUS", 0.5 + abs(risk_score - 0.625) / 0.25
    else:
        verdict, confidence = "REAL", 1.0 - risk_score
    return {
        "verdict": verdict,
        "risk_score": risk_score,
        "confidence": round(float(confidence), 4),
        "model_breakdown": {k: round(v, 4) for k, v in scaled.items()},
    }


def aggregate_temporal(
    frame_results: List[EnsembleResult],
    alpha:         float = 0.85,    # exponential recency weight
) -> dict:
    """
    Temporal aggregation for video: weight recent frames higher.
    Returns overall EnsembleResult stats + timeline. risk_score is float 0–1.
    """
    n = len(frame_results)
    if n == 0:
        return {"verdict": "ERROR", "risk_score": 0.5, "confidence": 0.0}

    decay_weights = [alpha ** (n - 1 - i) for i in range(n)]
    w_total       = sum(decay_weights)
    weighted_risk = sum(r.risk_score * w for r, w in zip(frame_results, decay_weights)) / w_total
    risk_score    = round(float(weighted_risk), 4)

    risks  = [r.risk_score for r in frame_results]
    std_r  = statistics.stdev(risks) if len(risks) > 1 else 0.0
    temporal_consistency = max(0.0, 1.0 - std_r)

    fake_frames = sum(1 for r in frame_results if r.verdict == "FAKE")
    susp_frames = sum(1 for r in frame_results if r.verdict == "SUSPICIOUS")
    real_frames = n - fake_frames - susp_frames

    if risk_score >= FAKE_THRESHOLD:
        verdict = "FAKE"
    elif risk_score >= SUSPICIOUS_THRESHOLD:
        verdict = "SUSPICIOUS"
    else:
        verdict = "REAL"

    return {
        "verdict":               verdict,
        "risk_score":            risk_score,
        "confidence":            round(max(r.confidence for r in frame_results), 4),
        "temporal_consistency":  round(temporal_consistency, 4),
        "frame_count":           n,
        "fake_frames":           fake_frames,
        "suspicious_frames":     susp_frames,
        "real_frames":           real_frames,
        "timeline":              [
            {"frame": i, "verdict": r.verdict, "risk_score": r.risk_score, "confidence": r.confidence}
            for i, r in enumerate(frame_results)
        ],
    }
