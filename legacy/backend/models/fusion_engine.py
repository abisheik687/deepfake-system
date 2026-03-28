
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class ForensicReport:
    final_score: float
    verdict: str
    confidence: str
    breakdown: Dict[str, float]

class FusionEngine:
    """
    KAVACH-AI Day 6: Multimodal Fusion Engine
    Combines scores from Video, Audio, and Temporal models.
    """
    def __init__(self, weights: Dict[str, float] = None):
        if weights is None:
            # Default weights: Video is usually most reliable, but Audio is strong for cloning
            self.weights = {
                "video_spatial": 0.4,
                "audio_spectral": 0.3,
                "temporal_lstm": 0.3
            }
        else:
            self.weights = weights

    def fuse(self, video_score: float, audio_score: float, temporal_score: float) -> ForensicReport:
        """
        Fuse individual modality scores into a final verdict.
        Scores are expected to be probabilities [0.0 - 1.0] where 1.0 = FAKE.
        """
        
        # 1. Weighted Average
        weighted_sum = (
            (video_score * self.weights["video_spatial"]) +
            (audio_score * self.weights["audio_spectral"]) +
            (temporal_score * self.weights["temporal_lstm"])
        )
        # Normalize in case weights don't sum to 1
        total_weight = sum(self.weights.values())
        final_score = weighted_sum / total_weight

        # 2. Rule-based Overrides ("Veto" power)
        # If any single modality is extremely confident (>0.95), it pulls the score up
        # This prevents a poor video model from hiding a blatant audio fake
        if max(video_score, audio_score, temporal_score) > 0.95:
            final_score = max(final_score, 0.95)

        # 3. Determine Verdict
        if final_score > 0.85:
            verdict = "DEEPFAKE DETECTED"
            confidence = "HIGH"
        elif final_score > 0.50:
            verdict = "SUSPICIOUS activity"
            confidence = "MEDIUM"
        else:
            verdict = "REAL / AUTHENTIC"
            confidence = "HIGH" if final_score < 0.15 else "MEDIUM"

        return ForensicReport(
            final_score=float(final_score),
            verdict=verdict,
            confidence=confidence,
            breakdown={
                "video_spatial": video_score,
                "audio_spectral": audio_score,
                "temporal_lstm": temporal_score
            }
        )
