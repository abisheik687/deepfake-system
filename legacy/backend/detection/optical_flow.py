"""
KAVACH-AI — Optical Flow Forensic Module
Detects temporal inconsistencies and 'ghosting' artifacts in deepfake videos.
"""

import cv2
import numpy as np
from loguru import logger
from typing import List, Tuple

class OpticalFlowAnalyzer:
    """
    Measures the smoothness of motion across frames.
    Deepfakes often show 'jitter' or 'jumpy' motion in the face-swap region.
    """
    def __init__(self):
        logger.info("[OpticalFlow] Initialized")

    def analyze_motion_consistency(self, frames: List[np.ndarray]) -> float:
        """
        Calculates the average motion magnitude and variance.
        
        Returns:
            consistency_score: 1.0 (smooth, natural motion) to 0.0 (erratic, artificial motion)
        """
        if len(frames) < 2:
            return 1.0

        magnitudes = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate Farneback Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(np.mean(mag))
            prev_gray = curr_gray

        # A high standard deviation in motion magnitude suggests jitter (artificial)
        mag_std = np.std(magnitudes)
        
        # Normalize: a std > 5.0 is considered highly erratic for a stable face
        consistency_score = max(0.0, 1.0 - (mag_std / 5.0))
        
        return round(float(consistency_score), 4)

    def detect_warp_artifacts(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """
        Specifically looks for warping around the edges of the face-swap mask.
        """
        # Placeholder for mask-based warping detection
        # In a full implementation, we would pass a face mask here
        return 0.0
