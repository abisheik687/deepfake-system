"""
KAVACH-AI Hugging Face Model Fallback System
Auto-downloads pre-trained ONNX weights from Hugging Face Hub if no GPU available
Supports: hbenedek/efficientnet-deepfake, dima806/deepfake_vs_real_image_detection
"""

import os
from pathlib import Path
from typing import Optional, Dict
from loguru import logger
import requests
from tqdm import tqdm


# Hugging Face model registry for fallback
HF_MODEL_REGISTRY = {
    "efficientnet_b4": {
        "repo": "hbenedek/efficientnet-deepfake",
        "filename": "efficientnet_b4_deepfake.onnx",
        "url": "https://huggingface.co/hbenedek/efficientnet-deepfake/resolve/main/efficientnet_b4_deepfake.onnx",
        "size_mb": 75,
        "description": "EfficientNet-B4 trained on FaceForensics++",
    },
    "xception": {
        "repo": "hbenedek/xception-deepfake",
        "filename": "xception_deepfake.onnx",
        "url": "https://huggingface.co/hbenedek/xception-deepfake/resolve/main/xception_deepfake.onnx",
        "size_mb": 88,
        "description": "Xception trained on FaceForensics++",
    },
    "vit_deepfake": {
        "repo": "dima806/deepfake_vs_real_image_detection",
        "filename": "vit_deepfake.onnx",
        "url": "https://huggingface.co/dima806/deepfake_vs_real_image_detection/resolve/main/pytorch_model.bin",
        "size_mb": 340,
        "description": "ViT fine-tuned for deepfake detection",
    },
    "rawnet2": {
        "repo": "Jungjee/RawNet2",
        "filename": "rawnet2_audio.onnx",
        "url": "https://huggingface.co/Jungjee/RawNet2/resolve/main/model.pth",
        "size_mb": 12,
        "description": "RawNet2 for audio deepfake detection",
    },
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """
    Download file from URL with progress bar.
    
    Args:
        url: Download URL
        dest_path: Destination file path
        desc: Progress bar description
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.success(f"Downloaded {dest_path.name} ({total_size / 1024 / 1024:.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def check_model_exists(model_name: str, models_dir: Path) -> bool:
    """
    Check if model file exists locally.
    
    Args:
        model_name: Name of the model
        models_dir: Models directory path
        
    Returns:
        True if model exists, False otherwise
    """
    if model_name not in HF_MODEL_REGISTRY:
        return False
    
    filename = HF_MODEL_REGISTRY[model_name]["filename"]
    model_path = models_dir / filename
    
    return model_path.exists()


def download_pretrained_model(
    model_name: str,
    models_dir: Optional[Path] = None,
    force: bool = False
) -> Optional[Path]:
    """
    Download pre-trained model from Hugging Face Hub.
    
    Args:
        model_name: Name of the model (must be in HF_MODEL_REGISTRY)
        models_dir: Directory to save models (default: ./models)
        force: Force re-download even if file exists
        
    Returns:
        Path to downloaded model file, or None if failed
    """
    if model_name not in HF_MODEL_REGISTRY:
        logger.error(f"Model '{model_name}' not found in HF registry")
        logger.info(f"Available models: {list(HF_MODEL_REGISTRY.keys())}")
        return None
    
    if models_dir is None:
        models_dir = Path("models")
    
    model_info = HF_MODEL_REGISTRY[model_name]
    filename = model_info["filename"]
    model_path = models_dir / filename
    
    # Check if already exists
    if model_path.exists() and not force:
        logger.info(f"Model already exists: {model_path}")
        return model_path
    
    logger.info(f"Downloading {model_name} from Hugging Face Hub...")
    logger.info(f"Repository: {model_info['repo']}")
    logger.info(f"Size: ~{model_info['size_mb']} MB")
    logger.info(f"Description: {model_info['description']}")
    
    # Download
    success = download_file(
        url=model_info["url"],
        dest_path=model_path,
        desc=f"Downloading {model_name}"
    )
    
    if success:
        logger.success(f"Model saved to: {model_path}")
        return model_path
    else:
        return None


def download_all_models(
    models_dir: Optional[Path] = None,
    skip_existing: bool = True
) -> Dict[str, Optional[Path]]:
    """
    Download all available pre-trained models.
    
    Args:
        models_dir: Directory to save models
        skip_existing: Skip models that already exist
        
    Returns:
        Dictionary mapping model names to their paths (or None if failed)
    """
    if models_dir is None:
        models_dir = Path("models")
    
    results = {}
    
    logger.info(f"Downloading {len(HF_MODEL_REGISTRY)} models from Hugging Face Hub...")
    
    for model_name in HF_MODEL_REGISTRY:
        if skip_existing and check_model_exists(model_name, models_dir):
            logger.info(f"Skipping {model_name} (already exists)")
            results[model_name] = models_dir / HF_MODEL_REGISTRY[model_name]["filename"]
            continue
        
        path = download_pretrained_model(model_name, models_dir, force=not skip_existing)
        results[model_name] = path
    
    # Summary
    successful = sum(1 for p in results.values() if p is not None)
    logger.info(f"Downloaded {successful}/{len(HF_MODEL_REGISTRY)} models successfully")
    
    return results


def auto_download_if_missing(model_name: str, models_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Automatically download model if it doesn't exist locally.
    This is the main function to use in inference code.
    
    Args:
        model_name: Name of the model
        models_dir: Models directory
        
    Returns:
        Path to model file, or None if unavailable
    """
    if models_dir is None:
        models_dir = Path("models")
    
    # Check if exists
    if check_model_exists(model_name, models_dir):
        filename = HF_MODEL_REGISTRY[model_name]["filename"]
        return models_dir / filename
    
    # Try to download
    logger.warning(f"Model '{model_name}' not found locally. Attempting download from HF Hub...")
    return download_pretrained_model(model_name, models_dir)


def check_gpu_available() -> bool:
    """
    Check if GPU is available for inference.
    
    Returns:
        True if GPU available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_fallback_strategy() -> str:
    """
    Determine fallback strategy based on available hardware.
    
    Returns:
        "local" if GPU available, "hf_download" if CPU only
    """
    if check_gpu_available():
        logger.info("GPU detected - using local training pipeline")
        return "local"
    else:
        logger.warning("No GPU detected - will use pre-trained models from HF Hub")
        return "hf_download"


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download pre-trained models from Hugging Face Hub")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(HF_MODEL_REGISTRY.keys()) + ["all"],
        default="all",
        help="Model to download (default: all)"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save models (default: ./models)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Check GPU availability and exit"
    )
    
    args = parser.parse_args()
    
    if args.check_gpu:
        gpu_available = check_gpu_available()
        print(f"GPU Available: {gpu_available}")
        print(f"Fallback Strategy: {get_fallback_strategy()}")
        exit(0)
    
    if args.model == "all":
        results = download_all_models(args.models_dir, skip_existing=not args.force)
        
        print("\n=== Download Summary ===")
        for model_name, path in results.items():
            status = "✓" if path else "✗"
            print(f"{status} {model_name}: {path}")
    else:
        path = download_pretrained_model(args.model, args.models_dir, force=args.force)
        if path:
            print(f"✓ Successfully downloaded to: {path}")
        else:
            print(f"✗ Failed to download {args.model}")
            exit(1)

# Made with Bob
