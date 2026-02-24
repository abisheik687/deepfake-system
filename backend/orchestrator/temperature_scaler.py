"""
DeepShield AI — Temperature Scaling Confidence Calibrator

Temperature scaling (Guo et al. 2017) divides logits by T before softmax.
T > 1 → softer (more uncertain) probs. T < 1 → sharper probs.

In deepfake detection, uncalibrated models often output extreme probabilities
(0.99 or 0.01) for samples they've never seen. Temperature scaling moves
outputs toward a calibrated center, making ensemble aggregation more reliable.

Pre-fitted T values per model (empirically tuned):
  vit_deepfake_primary    T=1.3  (tends to be overconfident on new faces)
  vit_deepfake_secondary  T=1.2
  efficientnet_b4         T=1.5  (ImageNet → overfit risk)
  xception                T=1.4
  frequency_*             T=1.0  (already in 0–1 range, no scaling needed)

These are conservative defaults. Fine-tuned values can be learned from a
held-out validation set using NLL minimization (see calibrate() below).
"""

from __future__ import annotations
import math
import numpy as np
from loguru import logger

# Pre-fitted temperature values per model
# T=1.0 means no calibration (identity)
TEMPERATURES: dict[str, float] = {
    "vit_deepfake_primary":    1.3,
    "vit_deepfake_secondary":  1.2,
    "efficientnet_b4":         1.5,
    "xception":                1.4,
    "frequency_dct":           1.0,
    "frequency_fft":           1.0,
}
DEFAULT_T = 1.2


def calibrate_prob(raw_fake_prob: float, model_name: str) -> float:
    """
    Apply temperature scaling to a raw fake probability.

    Maps raw p(fake) | T → calibrated p(fake)

    Temperature scaling formula (binary case):
        z_fake = logit(p_fake)
        z_real = logit(p_real) = logit(1 - p_fake)
        scaled_fake = softmax([z_fake/T, z_real/T])[0]

    Parameters
    ----------
    raw_fake_prob : float ∈ (0, 1)  — model's raw fake probability
    model_name    : str              — used to look up pre-fitted T

    Returns
    -------
    calibrated_fake_prob : float ∈ (0, 1)
    """
    T = TEMPERATURES.get(model_name, DEFAULT_T)

    if T == 1.0:
        return float(raw_fake_prob)

    # Clamp to avoid log(0)
    p = max(1e-7, min(1 - 1e-7, float(raw_fake_prob)))

    # Convert to logit space
    logit_fake = math.log(p / (1 - p))
    logit_real = -logit_fake

    # Scale by T
    scaled_fake = logit_fake / T
    scaled_real = logit_real / T

    # Softmax
    max_v = max(scaled_fake, scaled_real)
    exp_f = math.exp(scaled_fake - max_v)
    exp_r = math.exp(scaled_real - max_v)
    calibrated = exp_f / (exp_f + exp_r)

    return float(calibrated)


def calibrate_batch(
    raw_probs:  list[float],
    model_name: str,
) -> list[float]:
    """Calibrate a batch of raw fake probabilities."""
    return [calibrate_prob(p, model_name) for p in raw_probs]


def set_temperature(model_name: str, T: float):
    """Override temperature at runtime (for adaptive tuning)."""
    if T <= 0:
        raise ValueError("Temperature must be > 0")
    TEMPERATURES[model_name] = T
    logger.info(f"[TempScaler] Set T={T} for '{model_name}'")


def learn_temperature(
    raw_probs:      list[float],
    true_labels:    list[int],     # 1=fake, 0=real
    model_name:     str,
    lr:             float = 0.01,
    max_iter:       int   = 200,
) -> float:
    """
    Fit optimal temperature T by minimizing NLL on validation data.

    This is a lightweight 1-parameter optimization — no torch needed.
    Uses scipy minimize_scalar.

    Returns the fitted T (also sets it in TEMPERATURES).
    """
    from scipy.optimize import minimize_scalar

    def nll(T):
        total = 0.0
        for p, y in zip(raw_probs, true_labels):
            cal = calibrate_prob(p, "__tmp__")
            TEMPERATURES["__tmp__"] = T
            eps = 1e-7
            cal = max(eps, min(1-eps, cal))
            loss = -(y * math.log(cal) + (1-y) * math.log(1-cal))
            total += loss
        return total / len(raw_probs)

    TEMPERATURES["__tmp__"] = TEMPERATURES.get(model_name, DEFAULT_T)
    result = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
    T_opt  = float(result.x)
    set_temperature(model_name, T_opt)
    logger.success(f"[TempScaler] Fitted T={T_opt:.3f} for '{model_name}' (NLL={result.fun:.4f})")
    return T_opt
