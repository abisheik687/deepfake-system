"""
DeepShield AI — HuggingFace Real Deepfake Model Registry

Loads production-ready, deepfake-specific fine-tuned models from HuggingFace Hub.
All models are 100% free and open-source.

Primary models (real deepfake weights):
  • prithivMLmods/Deepfake-image-detect  — ViT-B/16 fine-tuned for deepfake detection
  • dima806/deepfake_vs_real_image_detection — ViT fine-tuned (deepfake vs real)

Fallback models (ImageNet → fine-tune ready):
  • EfficientNet-B4 via timm
  • XceptionNet via timm

Architecture:
  HFModelRegistry.get(name) → model + processor (cached, singleton per name)
  HFModelRegistry.infer(name, pil_image) → {"label": "fake"|"real", "score": 0.94}
"""

from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from loguru import logger

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[HFRegistry] Device: {DEVICE}")

# ─── Model catalogue ──────────────────────────────────────────────────────────
# All free, no API key required.  HuggingFace downloads to ~/.cache/huggingface/
HF_CATALOGUE: dict[str, dict] = {
    "vit_deepfake_primary": {
        "repo_id":     "prithivMLmods/Deepfake-image-detect",
        "kind":        "image-classification",
        "description": "ViT-B/16 fine-tuned for deepfake detection (primary HF model)",
        "real_label":  "Real",          # label string the model returns for real
        "fake_label":  "Deepfake",
        "size_mb":     330,
    },
    "vit_deepfake_secondary": {
        "repo_id":     "dima806/deepfake_vs_real_image_detection",
        "kind":        "image-classification",
        "description": "ViT fine-tuned deepfake vs real detection",
        "real_label":  "real",
        "fake_label":  "deepfake",
        "size_mb":     330,
    },
    "efficientnet_b4": {
        "repo_id":     None,            # loaded via timm (ImageNet pretrained)
        "kind":        "timm",
        "timm_name":   "efficientnet_b4",
        "description": "EfficientNet-B4 (ImageNet pretrained, fine-tune ready)",
        "size_mb":     85,
    },
    "xception": {
        "repo_id":     None,
        "kind":        "timm",
        "timm_name":   "xception",
        "description": "XceptionNet (FaceForensics++ benchmark arch)",
        "size_mb":     88,
    },
}

# In-memory cache: model_name → (pipeline or model, processor or None)
_cache: dict[str, tuple] = {}


def get_model_info() -> list[dict]:
    """Return serialisable catalogue for the /model-metadata endpoint."""
    items = []
    for name, cfg in HF_CATALOGUE.items():
        items.append({
            "name":        name,
            "repo_id":     cfg.get("repo_id"),
            "kind":        cfg["kind"],
            "description": cfg["description"],
            "size_mb":     cfg.get("size_mb"),
            "loaded":      name in _cache,
            "device":      str(DEVICE),
        })
    return items


def load_hf_model(name: str) -> tuple:
    """
    Load and cache a HuggingFace pipeline or timm model by registry name.

    Returns
    -------
    (pipeline_or_model, processor_or_none)
    """
    if name in _cache:
        return _cache[name]

    cfg = HF_CATALOGUE.get(name)
    if cfg is None:
        raise ValueError(f"Unknown model key: {name!r}. Available: {list(HF_CATALOGUE)}")

    logger.info(f"[HFRegistry] Loading '{name}' …")

    if cfg["kind"] == "image-classification":
        # Load via HuggingFace transformers pipeline
        try:
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline(
                "image-classification",
                model=cfg["repo_id"],
                device=0 if torch.cuda.is_available() else -1,
            )
            _cache[name] = (pipe, None)
            logger.success(f"[HFRegistry] '{name}' ready — {cfg['repo_id']}")
        except Exception as e:
            logger.error(f"[HFRegistry] Failed to load '{name}': {e}")
            raise

    elif cfg["kind"] == "timm":
        try:
            import timm
            model = timm.create_model(cfg["timm_name"], pretrained=True, num_classes=2)
            model.eval().to(DEVICE)

            # Check for fine-tuned weights in models/
            weights = Path("models") / f"{cfg['timm_name']}_deepfake.pth"
            if weights.exists():
                state = torch.load(weights, map_location=DEVICE)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state, strict=False)
                logger.success(f"[HFRegistry] Loaded fine-tuned weights: {weights}")
            else:
                logger.info(f"[HFRegistry] Using ImageNet weights for {cfg['timm_name']}")

            # Build timm data config for preprocessing
            data_cfg   = timm.data.resolve_model_data_config(model)
            transforms = timm.data.create_transform(**data_cfg, is_training=False)
            _cache[name] = (model, transforms)
            logger.success(f"[HFRegistry] '{name}' (timm) ready on {DEVICE}")
        except Exception as e:
            logger.error(f"[HFRegistry] timm load failed for '{name}': {e}")
            raise

    return _cache[name]


def infer_single(
    name: str,
    pil_image: Image.Image,
) -> dict:
    """
    Run inference on a single PIL image.

    Returns
    -------
    {
        "model":      str,
        "verdict":    "FAKE" | "REAL",
        "confidence": float 0–1,   ← probability of the predicted class
        "raw":        list[{label, score}]
    }
    """
    model_obj, processor = load_hf_model(name)
    cfg = HF_CATALOGUE[name]

    # ── HuggingFace pipeline path ──────────────────────────────────────────
    if cfg["kind"] == "image-classification":
        results = model_obj(pil_image, top_k=5)
        # results → [{"label": "Deepfake", "score": 0.94}, ...]
        fake_label = cfg["fake_label"].lower()
        real_label = cfg["real_label"].lower()

        fake_score = 0.0
        real_score = 0.0
        for r in results:
            lbl = r["label"].lower()
            if lbl == fake_label or "fake" in lbl or "deepfake" in lbl:
                fake_score = max(fake_score, r["score"])
            elif lbl == real_label or "real" in lbl or "authentic" in lbl:
                real_score = max(real_score, r["score"])

        if fake_score == 0.0 and real_score == 0.0:
            # Unknown labels — use top result
            top = results[0]["label"].lower()
            if "real" in top or "authentic" in top:
                real_score = results[0]["score"]
            else:
                fake_score = results[0]["score"]

        is_fake    = fake_score > real_score
        confidence = fake_score if is_fake else real_score

        return {
            "model":      name,
            "verdict":    "FAKE" if is_fake else "REAL",
            "confidence": round(float(confidence), 4),
            "raw":        results,
        }

    # ── timm model path ────────────────────────────────────────────────────
    elif cfg["kind"] == "timm":
        tensor = processor(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model_obj(tensor)
        probs    = torch.softmax(logits, dim=-1)[0]
        fake_p   = float(probs[1].item())
        real_p   = float(probs[0].item())
        is_fake  = fake_p >= 0.50

        return {
            "model":      name,
            "verdict":    "FAKE" if is_fake else "REAL",
            "confidence": round(fake_p if is_fake else real_p, 4),
            "raw":        [{"label": "REAL", "score": round(real_p, 4)},
                           {"label": "FAKE", "score": round(fake_p, 4)}],
        }

    raise ValueError(f"Unknown kind: {cfg['kind']}")


def preload_default_models():
    """
    Eagerly load the primary detection model on startup.
    Called once from lifespan() — makes first inference instant.
    """
    try:
        load_hf_model("vit_deepfake_primary")
        logger.success("[HFRegistry] Primary model pre-loaded ✓")
    except Exception as e:
        logger.warning(f"[HFRegistry] Pre-load failed (will lazy-load): {e}")
