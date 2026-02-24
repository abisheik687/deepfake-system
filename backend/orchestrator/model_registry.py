"""
DeepShield AI — Centralized Model Registry

Defines ALL detection models available in the orchestration layer.
Each model has:
  • name, kind, weight (for ensemble)
  • timeout_s (max seconds before fallback)
  • cpu_weight / gpu_weight (resource allocation)
  • health: tracked at runtime

Models are referenced by string key. The orchestrator loads them lazily.
Adding a new model = add one entry to REGISTRY dict below.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ModelKind = Literal["hf_pipeline", "timm", "frequency", "custom"]

@dataclass
class ModelSpec:
    name:        str
    kind:        ModelKind
    description: str
    weight:      float       = 1.0   # ensemble weight (higher = more trusted)
    timeout_s:   float       = 30.0  # per-inference timeout
    enabled:     bool        = True  # can disable without removing

    # HuggingFace-specific
    hf_repo:     Optional[str] = None
    fake_label:  str           = "Deepfake"
    real_label:  str           = "Real"

    # timm-specific
    timm_name:   Optional[str] = None

    # frequency analysis (no model weights needed)
    freq_method: Optional[str] = None  # "dct" | "fft" | "combined"

    # Runtime state (set during health monitoring)
    healthy:           bool  = True
    last_latency_ms:   float = 0.0
    total_calls:       int   = 0
    failed_calls:      int   = 0
    last_error:        str   = ""

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    def record_success(self, latency_ms: float):
        self.total_calls     += 1
        self.last_latency_ms  = latency_ms
        self.healthy          = True

    def record_failure(self, error: str):
        self.total_calls  += 1
        self.failed_calls += 1
        self.last_error    = error
        if self.failure_rate > 0.5 and self.total_calls >= 4:
            self.healthy = False


# ─── Global Model Registry ────────────────────────────────────────────────────
# Edit weights here to reconfigure ensemble without touching other code.
# Weights are normalized internally — no need to sum to 1.

REGISTRY: dict[str, ModelSpec] = {

    # ── Tier 1: Real deepfake-specific fine-tuned HF models ──────────────────
    "vit_deepfake_primary": ModelSpec(
        name="vit_deepfake_primary",
        kind="hf_pipeline",
        description="ViT-B/16 fine-tuned on deepfake datasets (prithivMLmods)",
        hf_repo="prithivMLmods/Deepfake-image-detect",
        weight=1.00,
        timeout_s=35.0,
        fake_label="Deepfake",
        real_label="Real",
    ),
    "vit_deepfake_secondary": ModelSpec(
        name="vit_deepfake_secondary",
        kind="hf_pipeline",
        description="ViT fine-tuned deepfake vs real (dima806)",
        hf_repo="dima806/deepfake_vs_real_image_detection",
        weight=0.90,
        timeout_s=35.0,
        fake_label="deepfake",
        real_label="real",
    ),

    # ── Tier 2: ImageNet pretrained architectures (fine-tune ready) ──────────
    "efficientnet_b4": ModelSpec(
        name="efficientnet_b4",
        kind="timm",
        description="EfficientNet-B4 — FaceForensics++ benchmark (timm)",
        timm_name="efficientnet_b4",
        weight=0.70,
        timeout_s=20.0,
    ),
    "xception": ModelSpec(
        name="xception",
        kind="timm",
        description="XceptionNet — original FaceForensics++ baseline (timm)",
        timm_name="xception",
        weight=0.65,
        timeout_s=25.0,
    ),

    # ── Tier 3: Frequency-domain analysis (always available, no GPU needed) ──
    "frequency_dct": ModelSpec(
        name="frequency_dct",
        kind="frequency",
        description="DCT frequency analysis — detects GAN artifacts in spectrum",
        freq_method="dct",
        weight=0.30,
        timeout_s=5.0,
    ),
    "frequency_fft": ModelSpec(
        name="frequency_fft",
        kind="frequency",
        description="FFT spectral analysis — periodic noise detection",
        freq_method="fft",
        weight=0.25,
        timeout_s=5.0,
    ),
}

TIER_PRESETS = {
    "fast": ["frequency_dct", "frequency_fft"],
    "balanced": ["vit_deepfake_primary", "efficientnet_b4", "frequency_dct"],
    "comprehensive": ["vit_deepfake_primary", "vit_deepfake_secondary", "efficientnet_b4", "xception", "frequency_dct", "frequency_fft"]
}


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"Model '{name}' not in registry. Available: {list(REGISTRY)}")
    return REGISTRY[name]


def list_all() -> list[dict]:
    return [
        {
            "name":          s.name,
            "kind":          s.kind,
            "description":   s.description,
            "weight":        s.weight,
            "enabled":       s.enabled,
            "healthy":       s.healthy,
            "timeout_s":     s.timeout_s,
            "calls":         s.total_calls,
            "failed":        s.failed_calls,
            "failure_rate":  round(s.failure_rate, 3),
            "last_latency_ms": s.last_latency_ms,
            "last_error":    s.last_error,
            "hf_repo":       s.hf_repo,
        }
        for s in REGISTRY.values()
    ]


def enabled_healthy() -> list[str]:
    """Return names of models that are both enabled and healthy."""
    return [name for name, spec in REGISTRY.items() if spec.enabled and spec.healthy]


def reset_health():
    """Reset all health counters (for testing)."""
    for s in REGISTRY.values():
        s.healthy     = True
        s.total_calls = 0
        s.failed_calls = 0
        s.last_error   = ""


# ─── ModelOutput (produced by task_runner, consumed by aggregator) ────────────
from dataclasses import dataclass as _dc, field as _field

@_dc
class ModelOutput:
    """Single model's raw output before ensemble aggregation."""
    model_name:  str
    fake_prob:   float         # raw P(fake) from model
    verdict:     str           # "FAKE" | "REAL"
    confidence:  float         # model's own confidence in its verdict
    latency_ms:  float = 0.0
    weight:      float = 1.0
    timed_out:   bool  = False
    failed:      bool  = False
    error:       str   = ""
