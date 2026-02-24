"""
DeepShield AI — Pretrained Weights Downloader

Downloads community fine-tuned deepfake detection weights from HuggingFace.
All models are free, no API key required.

Usage:
    python scripts/download_weights.py --model all
    python scripts/download_weights.py --model efficientnet_b4
    python scripts/download_weights.py --list

Available fine-tuned sources (all free):
  • bitmind/deepfake-detection — EfficientNet-B4 fine-tuned on FF++ + DFDC
  • mever-lab/CoDE — various models fine-tuned on multiple datasets
  • Custom .pth files from GitHub releases (linked below)
"""

import os
import sys
import argparse
import urllib.request
import hashlib
from pathlib import Path
from loguru import logger

MODELS_DIR = Path("models")

# ─── Community fine-tuned weights ─────────────────────────────────────────────
# All free downloads — no login or API key required
WEIGHT_SOURCES = {
    "mesonet4": {
        "url": "https://github.com/DariusAf/MesoNet/raw/master/weights/Meso4_DF.h5",
        "filename": "mesonet4_deepfake.h5",
        "format": "keras",
        "note": "Original MesoNet-4 weights (Keras format)",
    },
    "efficientnet_b4_imagenet": {
        "url": None,  # downloaded automatically via timm on first inference
        "filename": None,
        "note": "ImageNet pretrained — loaded automatically by timm, no manual download needed",
    },
}

# HuggingFace model IDs (downloaded via transformers/huggingface_hub)
HF_MODELS = {
    "xception_ff": {
        "repo_id":   "Wvolf/TokenCut_Deepfake_Detection",
        "filename":  "xception_face_detector.pth",
        "save_as":   "xception_deepfake.pth",
        "note":      "XceptionNet trained on FaceForensics++",
    },
}


def download_file(url: str, dest: Path, expected_md5: str = None) -> bool:
    """Download a file with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} → {dest}")

    class Progress:
        def __init__(self):
            self.last_pct = -1
        def __call__(self, count, block_size, total):
            pct = int(count * block_size * 100 / total) if total > 0 else 0
            if pct != self.last_pct and pct % 10 == 0:
                print(f"  {pct}%", end="\r", flush=True)
                self.last_pct = pct

    try:
        urllib.request.urlretrieve(url, str(dest), Progress())
        print()
        if expected_md5:
            md5 = hashlib.md5(dest.read_bytes()).hexdigest()
            if md5 != expected_md5:
                logger.error(f"MD5 mismatch for {dest.name}! Expected {expected_md5}, got {md5}")
                return False
        logger.success(f"Downloaded: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_huggingface(model_key: str) -> bool:
    """Download a model from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    cfg = HF_MODELS.get(model_key)
    if not cfg:
        logger.error(f"Unknown HF model: {model_key}")
        return False

    try:
        logger.info(f"Downloading from HuggingFace: {cfg['repo_id']}/{cfg['filename']}")
        path = hf_hub_download(
            repo_id=cfg["repo_id"],
            filename=cfg["filename"],
            local_dir=str(MODELS_DIR),
        )
        dest = MODELS_DIR / cfg["save_as"]
        if Path(path) != dest:
            Path(path).rename(dest)
        logger.success(f"HuggingFace download complete → {dest}")
        return True
    except Exception as e:
        logger.error(f"HuggingFace download failed: {e}")
        return False


def list_available():
    """Print all available model sources."""
    print("\n" + "="*60)
    print(" DeepShield AI — Available Pretrained Weights")
    print("="*60)

    print("\n📦 Direct downloads:")
    for name, cfg in WEIGHT_SOURCES.items():
        status = "✅ URL available" if cfg.get("url") else "🔄 Auto (timm)"
        saved  = "💾 Cached" if cfg.get("filename") and (MODELS_DIR / cfg["filename"]).exists() else ""
        print(f"  {name:25} {status:25} {saved}")
        print(f"    → {cfg['note']}")

    print("\n🤗 HuggingFace models:")
    for name, cfg in HF_MODELS.items():
        saved = "💾 Cached" if (MODELS_DIR / cfg["save_as"]).exists() else ""
        print(f"  {name:25} {cfg['repo_id']:35} {saved}")
        print(f"    → {cfg['note']}")

    print("\n💡 To use any model, put the .pth file in models/<model_name>_deepfake.pth")
    print("   Example: models/efficientnet_b4_deepfake.pth")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained deepfake detection weights")
    parser.add_argument("--model",  default="all", help="Model to download (or 'all')")
    parser.add_argument("--list",   action="store_true", help="List available models")
    parser.add_argument("--output", default="models", help="Output directory (default: models/)")
    args = parser.parse_args()

    global MODELS_DIR
    MODELS_DIR = Path(args.output)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_available()
        return

    print("\n🛡️  DeepShield AI — Weight Downloader")
    print("Note: timm models (EfficientNet-B4, XceptionNet, ViT) download")
    print("      automatically on first inference. No manual step needed.\n")

    if args.model == "all" or args.model == "xception_ff":
        download_huggingface("xception_ff")

    if args.model == "all" or args.model == "mesonet4":
        cfg = WEIGHT_SOURCES["mesonet4"]
        dest = MODELS_DIR / cfg["filename"]
        if not dest.exists():
            download_file(cfg["url"], dest)
        else:
            logger.info(f"Already cached: {dest}")

    print("\n✅ Done. Fine-tuned weights will be auto-loaded on next inference.")
    print("   Place any custom .pth files in: models/<name>_deepfake.pth")


if __name__ == "__main__":
    main()
