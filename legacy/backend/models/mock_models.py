"""
Mock Model Implementation for Demo
Simulates deepfake detection without actual ML models
"""

import numpy as np
from typing import Tuple, Optional
from loguru import logger


class MockSpatialDetector:
    """Mock spatial detection model"""
    
    def __init__(self):
        logger.info("MockSpatialDetector initialized (demo mode)")
    
    def predict(self, frame: np.ndarray) -> Tuple[float, dict]:
        """
        Simulate spatial detection.
        
        Args:
            frame: Input frame
        
        Returns:
            (confidence, features_dict)
        """
        # Generate random confidence (biased toward real for demo)
        confidence = np.random.beta(2, 5)  # More likely to be < 0.5 (real)
        
        features = {
            'spatial_anomaly': float(confidence + np.random.normal(0, 0.1)),
            'artifact_score': float(np.random.uniform(0, 1)),
            'consistency_score': float(1.0 - confidence)
        }
        
        return confidence, features


class MockTemporalDetector:
    """Mock temporal detection model"""
    
    def __init__(self):
        self.sequence_buffer = []
        logger.info("MockTemporalDetector initialized (demo mode)")
    
    def predict(self, features: np.ndarray) -> Tuple[float, dict]:
        """
        Simulate temporal detection.
        
        Args:
            features: Temporal features
        
        Returns:
            (confidence, temporal_info)
        """
        confidence = np.random.beta(2, 5)
        
        temporal_info = {
            'temporal_anomaly_score': float(confidence),
            'blink_rate': float(np.random.uniform(5, 15)),  # blinks/min
            'landmark_jitter': float(np.random.uniform(0, 20))  # pixels
        }
        
        return confidence, temporal_info


class MockAudioDetector:
    """Mock audio detection model"""
    
    def __init__(self):
        logger.info("MockAudioDetector initialized (demo mode)")
    
    def predict(self, audio_features: np.ndarray) -> Tuple[float, dict]:
        """
        Simulate audio detection.
        
        Args:
            audio_features: MFCC features
        
        Returns:
            (confidence, audio_info)
        """
        confidence = np.random.beta(2, 5)
        
        audio_info = {
            'voice_anomaly': float(confidence),
            'pitch_variance': float(np.random.uniform(0, 1)),
            'spectral_anomaly': float(np.random.uniform(0, 1))
        }
        
        return confidence, audio_info


# Global instances
_spatial_detector: Optional[MockSpatialDetector] = None
_temporal_detector: Optional[MockTemporalDetector] = None
_audio_detector: Optional[MockAudioDetector] = None


def get_spatial_detector() -> MockSpatialDetector:
    """Get global spatial detector instance"""
    global _spatial_detector
    if _spatial_detector is None:
        _spatial_detector = MockSpatialDetector()
    return _spatial_detector


def get_temporal_detector() -> MockTemporalDetector:
    """Get global temporal detector instance"""
    global _temporal_detector
    if _temporal_detector is None:
        _temporal_detector = MockTemporalDetector()
    return _temporal_detector


def get_audio_detector() -> MockAudioDetector:
    """Get global audio detector instance"""
    global _audio_detector
    if _audio_detector is None:
        _audio_detector = MockAudioDetector()
    return _audio_detector
