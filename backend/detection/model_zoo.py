"""
DeepShield AI — Model Zoo

Central registry for all deepfake detection models.
Handles: lazy loading, custom weight injection, ONNX export.

Available models (all free, CPU-compatible):
    efficientnet_b4    — EfficientNet-B4 (ImageNet pretrained → fine-tune ready)
    xception           — XceptionNet    (benchmark standard for FF++)
    vit_b16            — ViT-B/16       (transformer-based)
    mesonet4           — MesoNet-4      (lightweight, ~5MB, CPU-optimised)
    efficientnet_b0    — Lighter option for CPU

Fine-tuned weight files:
    Place .pth files in  models/<model_name>_deepfake.pth
    They are auto-loaded if found — otherwise falls back to ImageNet weights.

ONNX export:
    python -c "from backend.detection.model_zoo import export_onnx; export_onnx('efficientnet_b4')"
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from loguru import logger

import torch
import torch.nn as nn

# ─── Model Registry ───────────────────────────────────────────────────────────
AVAILABLE_MODELS: dict[str, dict] = {
    "efficientnet_b4": {
        "timm_name":   "efficientnet_b4",
        "num_classes": 2,
        "description": "EfficientNet-B4 — best accuracy/speed tradeoff (~85MB)",
        "auc_ff":      0.991,   # AUC on FaceForensics++ when fine-tuned
        "speed_cpu_ms": 2200,
        "speed_gpu_ms": 85,
    },
    "efficientnet_b0": {
        "timm_name":   "efficientnet_b0",
        "num_classes": 2,
        "description": "EfficientNet-B0 — lightweight, fast on CPU (~20MB)",
        "auc_ff":      0.962,
        "speed_cpu_ms": 800,
        "speed_gpu_ms": 30,
    },
    "xception": {
        "timm_name":   "xception",
        "num_classes": 2,
        "description": "XceptionNet — FaceForensics++ benchmark standard (~88MB)",
        "auc_ff":      0.990,
        "speed_cpu_ms": 3000,
        "speed_gpu_ms": 100,
    },
    "vit_b16": {
        "timm_name":   "vit_base_patch16_224",
        "num_classes": 2,
        "description": "Vision Transformer ViT-B/16 — transformer-based detection (~330MB)",
        "auc_ff":      0.988,
        "speed_cpu_ms": 4200,
        "speed_gpu_ms": 120,
    },
    "mesonet4": {
        "timm_name":   None,    # custom implementation
        "num_classes": 2,
        "description": "MesoNet-4 — ultra-lightweight deepfake detector (~5MB)",
        "auc_ff":      0.895,
        "speed_cpu_ms": 150,
        "speed_gpu_ms": 12,
    },
}

_MODELS_DIR = Path("models")
_model_cache: dict[str, nn.Module] = {}


# ─── MesoNet-4 (custom implementation, no timm needed) ───────────────────────

class MesoNet4(nn.Module):
    """
    MesoNet-4: "MesoNet: a Compact Facial Video Forgery Detection Network"
    Afchar et al. 2018 — https://arxiv.org/abs/1809.00888
    Very fast, runs well on CPU. AUC ~0.89 on FaceForensics++.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 112→56

            # Block 2
            nn.Conv2d(8, 8, 5, 1, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 56→28

            # Block 3
            nn.Conv2d(8, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 28→14

            # Block 4
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),   # 14→3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * 3 * 3, 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── Loader ───────────────────────────────────────────────────────────────────

def get_model(name: str, pretrained: bool = True) -> nn.Module:
    """
    Return a ready-to-use model for inference on DEVICE.

    1. Check cache
    2. Load architecture via timm (or custom for mesonet4)
    3. Adapt output head to 2 classes (binary: real vs fake)
    4. Try to load fine-tuned weights from models/<name>_deepfake.pth
    5. Fall back to ImageNet pretrained (if timm model) or random init
    """
    if name in _model_cache:
        return _model_cache[name]

    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {name!r}. Choose from: {list(AVAILABLE_MODELS)}")

    cfg = AVAILABLE_MODELS[name]

    # --- Build architecture ---
    if name == "mesonet4":
        model = MesoNet4(num_classes=cfg["num_classes"])
    else:
        import timm
        model = timm.create_model(
            cfg["timm_name"],
            pretrained=pretrained,
            num_classes=cfg["num_classes"],
        )

    # --- Load fine-tuned weights if available ---
    weights_path = _MODELS_DIR / f"{name}_deepfake.pth"
    if weights_path.exists():
        try:
            state = torch.load(weights_path, map_location="cpu")
            # Support both raw state_dict and checkpoint dicts
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.success(f"[ModelZoo] Loaded fine-tuned weights: {weights_path} "
                           f"(missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:
            logger.warning(f"[ModelZoo] Could not load {weights_path}: {e}  — using pretrained")
    else:
        logger.info(f"[ModelZoo] No fine-tuned weights at {weights_path}, using ImageNet weights")

    _model_cache[name] = model
    return model


def list_models() -> list[dict]:
    """Return serialisable list of available models with their info."""
    out = []
    for name, cfg in AVAILABLE_MODELS.items():
        weights_path = _MODELS_DIR / f"{name}_deepfake.pth"
        out.append({
            "name":           name,
            "description":    cfg["description"],
            "auc_ff":         cfg.get("auc_ff"),
            "speed_cpu_ms":   cfg.get("speed_cpu_ms"),
            "speed_gpu_ms":   cfg.get("speed_gpu_ms"),
            "finetuned":      weights_path.exists(),
            "weights_path":   str(weights_path) if weights_path.exists() else None,
            "active":         False,  # filled in by API layer
        })
    return out


def export_onnx(name: str, output_path: Optional[Path] = None) -> str:
    """
    Export a model to ONNX for faster CPU inference via onnxruntime.

    Returns path to saved .onnx file.
    """
    import torch
    model = get_model(name, pretrained=True)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    out_path = output_path or (_MODELS_DIR / f"{name}_deepfake.onnx")

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        verbose=False,
    )
    logger.success(f"[ModelZoo] ONNX exported → {out_path}")
    return str(out_path)
