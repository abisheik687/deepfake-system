"""
KAVACH-AI Enhanced Ensemble Voting System
Implements weighted soft voting with test-time augmentation (TTA)
Follows requirements: ViT=0.30, EfficientNet-B4=0.25, Xception=0.20, ConvNeXt=0.15, Audio CNN=0.10
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ModelPrediction:
    """Single model prediction result"""
    model_name: str
    confidence: float  # 0-1 probability of being fake
    raw_logits: np.ndarray
    latency_ms: float = 0.0
    weight: float = 1.0


@dataclass
class EnsembleResult:
    """Ensemble prediction result"""
    final_confidence: float  # Weighted average confidence
    verdict: str  # "REAL" or "FAKE"
    individual_predictions: List[ModelPrediction]
    agreement_score: float  # How much models agree (0-1)
    abstain: bool  # True if top-2 confidence delta > 0.35
    weighted_votes: Dict[str, float]  # Per-model weighted contribution


# Model weights as specified in requirements
MODEL_WEIGHTS = {
    "vit": 0.30,
    "vit_deepfake_primary": 0.30,
    "vit_deepfake_secondary": 0.30,
    "efficientnet_b4": 0.25,
    "efficientnet": 0.25,
    "xception": 0.20,
    "convnext": 0.15,
    "convnext_base": 0.15,
    "audio_cnn": 0.10,
    "rawnet2": 0.10,
    "lcnn": 0.10,
    # Frequency models get lower weights
    "frequency_dct": 0.05,
    "frequency_fft": 0.05,
}


def get_model_weight(model_name: str) -> float:
    """
    Get ensemble weight for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Weight value (0-1)
    """
    # Try exact match first
    if model_name in MODEL_WEIGHTS:
        return MODEL_WEIGHTS[model_name]
    
    # Try partial match (e.g., "efficientnet_b4_v2" matches "efficientnet")
    for key, weight in MODEL_WEIGHTS.items():
        if key in model_name.lower():
            return weight
    
    # Default weight for unknown models
    return 0.10


def weighted_soft_voting(
    predictions: List[ModelPrediction],
    abstain_threshold: float = 0.35
) -> EnsembleResult:
    """
    Perform weighted soft voting across model predictions.
    
    Args:
        predictions: List of model predictions
        abstain_threshold: If top-2 model confidence delta > this, abstain
        
    Returns:
        EnsembleResult with final verdict and metadata
    """
    if not predictions:
        raise ValueError("No predictions provided for ensemble voting")
    
    # Normalize weights
    total_weight = sum(p.weight for p in predictions)
    if total_weight == 0:
        total_weight = len(predictions)
    
    # Calculate weighted average confidence
    weighted_sum = sum(p.confidence * p.weight for p in predictions)
    final_confidence = weighted_sum / total_weight
    
    # Calculate agreement score (inverse of variance)
    confidences = [p.confidence for p in predictions]
    variance = np.var(confidences)
    agreement_score = 1.0 / (1.0 + variance)  # Higher = more agreement
    
    # Check abstain condition (top-2 delta > threshold)
    sorted_confs = sorted(confidences, reverse=True)
    abstain = False
    if len(sorted_confs) >= 2:
        top_2_delta = abs(sorted_confs[0] - sorted_confs[1])
        if top_2_delta > abstain_threshold:
            abstain = True
            logger.warning(
                f"Abstaining from verdict: top-2 delta = {top_2_delta:.3f} "
                f"(threshold = {abstain_threshold})"
            )
    
    # Determine verdict
    verdict = "FAKE" if final_confidence > 0.5 else "REAL"
    
    # Calculate per-model weighted contributions
    weighted_votes = {
        p.model_name: (p.confidence * p.weight) / total_weight
        for p in predictions
    }
    
    result = EnsembleResult(
        final_confidence=float(final_confidence),
        verdict=verdict,
        individual_predictions=predictions,
        agreement_score=float(agreement_score),
        abstain=abstain,
        weighted_votes=weighted_votes
    )
    
    logger.info(
        f"Ensemble result: {verdict} (conf={final_confidence:.3f}, "
        f"agreement={agreement_score:.3f}, abstain={abstain})"
    )
    
    return result


def test_time_augmentation(
    image: np.ndarray,
    predict_fn: callable,
    num_crops: int = 5
) -> Tuple[float, List[float]]:
    """
    Perform test-time augmentation (TTA) with multiple crops.
    
    Args:
        image: Input image (H, W, C)
        predict_fn: Function that takes image and returns confidence
        num_crops: Number of crops to average (default: 5)
        
    Returns:
        (average_confidence, individual_confidences)
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")
    
    h, w, c = image.shape
    confidences = []
    
    # Center crop
    center_crop = _center_crop(image, 224, 224)
    conf = predict_fn(center_crop)
    confidences.append(conf)
    
    # Four corner crops
    if num_crops >= 5:
        corners = [
            (0, 0),  # Top-left
            (0, w - 224),  # Top-right
            (h - 224, 0),  # Bottom-left
            (h - 224, w - 224),  # Bottom-right
        ]
        
        for y, x in corners:
            crop = image[y:y+224, x:x+224]
            if crop.shape[:2] == (224, 224):
                conf = predict_fn(crop)
                confidences.append(conf)
    
    # Average all predictions
    avg_confidence = np.mean(confidences)
    
    logger.debug(
        f"TTA: {len(confidences)} crops, "
        f"avg={avg_confidence:.3f}, "
        f"std={np.std(confidences):.3f}"
    )
    
    return float(avg_confidence), confidences


def _center_crop(image: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """Extract center crop from image"""
    h, w = image.shape[:2]
    
    if h < crop_h or w < crop_w:
        # Pad if image is smaller than crop size
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        image = np.pad(
            image,
            ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2), (0, 0)),
            mode='reflect'
        )
        h, w = image.shape[:2]
    
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    
    return image[start_y:start_y+crop_h, start_x:start_x+crop_w]


def apply_temperature_scaling(
    logits: np.ndarray,
    temperature: float = 1.5
) -> np.ndarray:
    """
    Apply temperature scaling to calibrate confidence.
    
    Args:
        logits: Raw model logits
        temperature: Temperature parameter (>1 = less confident, <1 = more confident)
        
    Returns:
        Calibrated probabilities
    """
    scaled_logits = logits / temperature
    
    # Apply softmax
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)
    
    return probs


def calculate_uncertainty(predictions: List[ModelPrediction]) -> Dict[str, float]:
    """
    Calculate uncertainty metrics for ensemble.
    
    Args:
        predictions: List of model predictions
        
    Returns:
        Dictionary with uncertainty metrics
    """
    confidences = np.array([p.confidence for p in predictions])
    
    # Variance-based uncertainty
    variance = np.var(confidences)
    
    # Entropy-based uncertainty
    # Convert confidences to binary probabilities
    probs = np.column_stack([1 - confidences, confidences])
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1).mean()
    
    # Disagreement rate (how many models disagree with majority)
    majority_vote = 1 if np.mean(confidences) > 0.5 else 0
    disagreement_rate = np.mean([
        1 if (conf > 0.5) != majority_vote else 0
        for conf in confidences
    ])
    
    return {
        "variance": float(variance),
        "entropy": float(entropy),
        "disagreement_rate": float(disagreement_rate),
        "confidence_std": float(np.std(confidences)),
        "confidence_range": float(np.max(confidences) - np.min(confidences))
    }


# Example usage
if __name__ == "__main__":
    # Test ensemble voting
    predictions = [
        ModelPrediction("vit_deepfake_primary", 0.85, np.array([0.15, 0.85]), 120.5, 0.30),
        ModelPrediction("efficientnet_b4", 0.78, np.array([0.22, 0.78]), 95.2, 0.25),
        ModelPrediction("xception", 0.82, np.array([0.18, 0.82]), 110.3, 0.20),
        ModelPrediction("convnext_base", 0.75, np.array([0.25, 0.75]), 105.8, 0.15),
        ModelPrediction("audio_cnn", 0.65, np.array([0.35, 0.65]), 80.1, 0.10),
    ]
    
    result = weighted_soft_voting(predictions)
    print(f"Final verdict: {result.verdict}")
    print(f"Confidence: {result.final_confidence:.3f}")
    print(f"Agreement: {result.agreement_score:.3f}")
    print(f"Abstain: {result.abstain}")
    print(f"Weighted votes: {result.weighted_votes}")
    
    # Test uncertainty calculation
    uncertainty = calculate_uncertainty(predictions)
    print(f"Uncertainty metrics: {uncertainty}")

# Made with Bob
