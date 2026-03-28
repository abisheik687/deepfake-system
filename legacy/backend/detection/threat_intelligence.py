"""
KAVACH-AI Threat Intelligence
Advanced threat detection, classification, and prioritization
NO API KEYS REQUIRED - All processing is local
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from backend.config import settings


class AttackType(Enum):
    """Deepfake attack types"""
    FACE_SWAP = "face_swap"              # Face replacement
    FACE_REENACTMENT = "face_reenactment"  # Expression transfer
    VOICE_CLONE = "voice_clone"          # Synthetic voice
    LIP_SYNC = "lip_sync"                # Audio-driven animation
    FULL_SYNTHESIS = "full_synthesis"    # Fully generated
    HYBRID = "hybrid"                    # Multiple techniques
    UNKNOWN = "unknown"


class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"              # <0.6 confidence
    MEDIUM = "medium"        # 0.6-0.75
    HIGH = "high"            # 0.75-0.9
    CRITICAL = "critical"    # >0.9


@dataclass
class ThreatAssessment:
    """Comprehensive threat assessment"""
    # Core detection
    confidence: float
    calibrated_confidence: float  # After Platt scaling
    
    # Temporal accumulation
    temporal_score: float         # EMA over time
    consistency_score: float      # Across frames
    
    # Classification
    attack_type: AttackType
    threat_level: ThreatLevel
    
    # Multi-source correlation
    num_detections: int
    detection_rate: float         # Detections / total frames
    
    # Priority
    priority_score: float         # [0, 1] for alert queue
    
    # Metadata
    first_detected: datetime
    last_detected: datetime
    
    # Evidence
    indicators: List[str]         # Why it's classified as threat


class ConfidenceCalibrator:
    """
    Calibrate raw model outputs to probabilities using Platt scaling.
    
    Platt scaling: Fit a sigmoid to map scores to calibrated probabilities.
    """
    
    def __init__(self):
        """Initialize calibrator"""
        # Platt scaling parameters (will be learned from data)
        # For now, use identity (no calibration)
        self.a = 1.0
        self.b = 0.0
        
        logger.info("Confidence calibrator initialized (identity mapping)")
    
    def calibrate(self, raw_score: float) -> float:
        """
        Calibrate raw confidence score.
        
        Args:
            raw_score: Raw model output [0, 1]
        
        Returns:
            Calibrated probability [0, 1]
        """
        # Platt scaling: P(y=1|x) = 1 / (1 + exp(A*x + B))
        # For now, identity mapping (A=1, B=0)
        calibrated = 1.0 / (1.0 + np.exp(-(self.a * raw_score + self.b)))
        return float(calibrated)
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit Platt scaling parameters.
        
        Args:
            scores: Raw model scores
            labels: Ground truth labels (0 or 1)
        """
        # TODO: Implement Platt scaling parameter fitting
        # Would use maximum likelihood estimation
        logger.warning("Platt scaling fitting not yet implemented")
        pass


class TemporalAccumulator:
    """
    Accumulate detections over time using exponential moving average.
    
    Helps distinguish persistent threats from transient noise.
    """
    
    def __init__(self, alpha: float = 0.3, halflife_seconds: int = 10):
        """
        Initialize temporal accumulator.
        
        Args:
            alpha: EMA smoothing factor (higher = more reactive)
            halflife_seconds: Time for weight to halve
        """
        self.alpha = alpha
        self.halflife_seconds = halflife_seconds
        
        # Track per stream
        self.stream_scores: Dict[int, float] = {}
        self.stream_timestamps: Dict[int, datetime] = {}
        
        logger.info(
            f"Temporal accumulator initialized "
            f"(alpha={alpha}, halflife={halflife_seconds}s)"
        )
    
    def update(
        self,
        stream_id: int,
        confidence: float,
        timestamp: datetime
    ) -> float:
        """
        Update temporal score for stream.
        
        Args:
            stream_id: Stream ID
            confidence: Current detection confidence
            timestamp: Detection timestamp
        
        Returns:
            Accumulated temporal score
        """
        # Initialize if first detection
        if stream_id not in self.stream_scores:
            self.stream_scores[stream_id] = confidence
            self.stream_timestamps[stream_id] = timestamp
            return confidence
        
        # Calculate time decay
        prev_timestamp = self.stream_timestamps[stream_id]
        time_delta = (timestamp - prev_timestamp).total_seconds()
        
        decay_factor = np.exp(-time_delta / self.halflife_seconds)
        
        # EMA update with decay
        prev_score = self.stream_scores[stream_id]
        new_score = decay_factor * (self.alpha * confidence + (1 - self.alpha) * prev_score)
        
        self.stream_scores[stream_id] = new_score
        self.stream_timestamps[stream_id] = timestamp
        
        logger.debug(
            f"Temporal accumulation (stream {stream_id}): "
            f"{prev_score:.3f} → {new_score:.3f} "
            f"(conf={confidence:.3f}, decay={decay_factor:.3f})"
        )
        
        return float(new_score)
    
    def get_score(self, stream_id: int) -> Optional[float]:
        """Get current temporal score for stream"""
        return self.stream_scores.get(stream_id)
    
    def reset(self, stream_id: int):
        """Reset temporal accumulation for stream"""
        self.stream_scores.pop(stream_id, None)
        self.stream_timestamps.pop(stream_id, None)


class ThreatClassifier:
    """
    Classify attack type and threat level.
    
    Uses rule-based logic and confidence thresholds.
    """
    
    def __init__(self):
        """Initialize threat classifier"""
        logger.info("Threat classifier initialized")
    
    def classify_attack_type(
        self,
        spatial_conf: float,
        temporal_conf: Optional[float],
        audio_conf: Optional[float],
        temporal_features: Optional[Dict] = None,
        audio_features: Optional[Dict] = None
    ) -> AttackType:
        """
        Classify deepfake attack type.
        
        Args:
            spatial_conf: Spatial detection confidence
            temporal_conf: Temporal detection confidence
            audio_conf: Audio detection confidence
            temporal_features: Temporal feature dict
            audio_features: Audio feature dict
        
        Returns:
            Attack type classification
        """
        # Rule-based classification
        
        # Both video and audio high → Full synthesis or hybrid
        if spatial_conf > 0.7 and audio_conf and audio_conf > 0.7:
            return AttackType.FULL_SYNTHESIS
        
        # High spatial, low temporal → Face swap
        if spatial_conf > 0.7 and temporal_conf and temporal_conf < 0.5:
            return AttackType.FACE_SWAP
        
        # High temporal anomaly (blink issues) → Face reenactment
        if temporal_features and temporal_features.get('temporal_anomaly_score', 0) > 0.6:
            if temporal_features.get('blink_rate', 20) < 10:
                return AttackType.FACE_REENACTMENT
        
        # Audio only → Voice clone
        if audio_conf and audio_conf > 0.7 and spatial_conf < 0.5:
            return AttackType.VOICE_CLONE
        
        # Mixed signals → Hybrid
        if spatial_conf > 0.5 and audio_conf and audio_conf > 0.5:
            return AttackType.HYBRID
        
        # Default
        return AttackType.UNKNOWN
    
    def classify_threat_level(self, confidence: float) -> ThreatLevel:
        """
        Classify threat level based on confidence.
        
        Args:
            confidence: Detection confidence
        
        Returns:
            Threat level
        """
        if confidence >= settings.HIGH_CONFIDENCE_THRESHOLD:  # 0.9
            return ThreatLevel.CRITICAL
        elif confidence >= settings.ALERT_THRESHOLD:  # 0.75
            return ThreatLevel.HIGH
        elif confidence >= settings.DETECTION_CONFIDENCE_THRESHOLD:  # 0.6
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def calculate_priority(
        self,
        confidence: float,
        temporal_score: float,
        attack_type: AttackType,
        threat_level: ThreatLevel
    ) -> float:
        """
        Calculate alert priority score.
        
        Higher priority = more urgent.
        
        Args:
            confidence: Detection confidence
            temporal_score: Temporal accumulation score
            attack_type: Attack type
            threat_level: Threat level
        
        Returns:
            Priority score [0, 1]
        """
        # Base priority from confidence
        priority = confidence
        
        # Boost from temporal consistency
        priority += 0.2 * temporal_score
        
        # Boost from threat level
        level_boost = {
            ThreatLevel.CRITICAL: 0.3,
            ThreatLevel.HIGH: 0.2,
            ThreatLevel.MEDIUM: 0.1,
            ThreatLevel.LOW: 0.0
        }
        priority += level_boost[threat_level]
        
        # Boost from attack type (some are more dangerous)
        type_boost = {
            AttackType.FULL_SYNTHESIS: 0.2,
            AttackType.HYBRID: 0.15,
            AttackType.FACE_SWAP: 0.1,
            AttackType.FACE_REENACTMENT: 0.1,
            AttackType.VOICE_CLONE: 0.05,
            AttackType.LIP_SYNC: 0.05,
            AttackType.UNKNOWN: 0.0
        }
        priority += type_boost.get(attack_type, 0.0)
        
        # Clamp to [0, 1]
        priority = min(1.0, max(0.0, priority))
        
        return float(priority)
    
    def generate_indicators(
        self,
        spatial_conf: float,
        temporal_conf: Optional[float],
        audio_conf: Optional[float],
        temporal_features: Optional[Dict],
        audio_features: Optional[Dict],
        uncertainty: float
    ) -> List[str]:
        """
        Generate threat indicators (evidence).
        
        Args:
            spatial_conf: Spatial confidence
            temporal_conf: Temporal confidence
            audio_conf: Audio confidence
            temporal_features: Temporal features
            audio_features: Audio features
            uncertainty: Fusion uncertainty
        
        Returns:
            List of indicator strings
        """
        indicators = []
        
        # Spatial indicators
        if spatial_conf > 0.7:
            indicators.append(f"High spatial anomaly ({spatial_conf:.1%})")
        
        # Temporal indicators
        if temporal_conf and temporal_conf > 0.7:
            indicators.append(f"High temporal anomaly ({temporal_conf:.1%})")
        
        if temporal_features:
            blink_rate = temporal_features.get('blink_rate', 0)
            if blink_rate < 10:
                indicators.append(f"Abnormally low blink rate ({blink_rate:.1f}/min)")
            elif blink_rate > 30:
                indicators.append(f"Abnormally high blink rate ({blink_rate:.1f}/min)")
            
            jitter = temporal_features.get('landmark_jitter', 0)
            if jitter > 10:
                indicators.append(f"High facial landmark jitter ({jitter:.1f}px)")
        
        # Audio indicators
        if audio_conf and audio_conf > 0.7:
            indicators.append(f"High audio anomaly ({audio_conf:.1%})")
        
        if audio_features:
            anomalies = audio_features.get('anomalies', {})
            if anomalies.get('pitch_stability', 0) > 0.5:
                indicators.append("Unnatural pitch stability")
            if anomalies.get('spectral_artifacts', 0) > 0.5:
                indicators.append("Spectral artifacts detected")
        
        # Uncertainty
        if uncertainty > 0.5:
            indicators.append(f"High prediction uncertainty ({uncertainty:.1%})")
        
        return indicators


class ThreatIntelligence:
    """
    Main threat intelligence system.
    
    Combines calibration, accumulation, and classification.
    """
    
    def __init__(self):
        """Initialize threat intelligence"""
        self.calibrator = ConfidenceCalibrator()
        self.accumulator = TemporalAccumulator(
            alpha=settings.EMA_ALPHA,
            halflife_seconds=settings.TEMPORAL_HALFLIFE_SECONDS
        )
        self.classifier = ThreatClassifier()
        
        logger.info("Threat intelligence system initialized")
    
    def assess_threat(
        self,
        stream_id: int,
        confidence: float,
        spatial_conf: float,
        temporal_conf: Optional[float],
        audio_conf: Optional[float],
        temporal_features: Optional[Dict],
        audio_features: Optional[Dict],
        uncertainty: float,
        timestamp: datetime
    ) -> ThreatAssessment:
        """
        Complete threat assessment.
        
        Args:
            stream_id: Stream ID
            confidence: Raw detection confidence
            spatial_conf: Spatial confidence
            temporal_conf: Temporal confidence
            audio_conf: Audio confidence
            temporal_features: Temporal features
            audio_features: Audio features
            uncertainty: Fusion uncertainty
            timestamp: Detection timestamp
        
        Returns:
            ThreatAssessment
        """
        # 1. Calibrate confidence
        calibrated_conf = self.calibrator.calibrate(confidence)
        
        # 2. Temporal accumulation
        temporal_score = self.accumulator.update(
            stream_id,
            calibrated_conf,
            timestamp
        )
        
        # 3. Classify attack type
        attack_type = self.classifier.classify_attack_type(
            spatial_conf,
            temporal_conf,
            audio_conf,
            temporal_features,
            audio_features
        )
        
        # 4. Classify threat level
        threat_level = self.classifier.classify_threat_level(calibrated_conf)
        
        # 5. Calculate priority
        priority = self.classifier.calculate_priority(
            calibrated_conf,
            temporal_score,
            attack_type,
            threat_level
        )
        
        # 6. Generate indicators
        indicators = self.classifier.generate_indicators(
            spatial_conf,
            temporal_conf,
            audio_conf,
            temporal_features,
            audio_features,
            uncertainty
        )
        
        return ThreatAssessment(
            confidence=confidence,
            calibrated_confidence=calibrated_conf,
            temporal_score=temporal_score,
            consistency_score=1.0 - uncertainty,  # Inverse of uncertainty
            attack_type=attack_type,
            threat_level=threat_level,
            num_detections=1,  # Per-frame
            detection_rate=1.0,  # Would track over window
            priority_score=priority,
            first_detected=timestamp,
            last_detected=timestamp,
            indicators=indicators
        )


# Global threat intelligence instance
_threat_intelligence: Optional[ThreatIntelligence] = None


def get_threat_intelligence() -> ThreatIntelligence:
    """Get global threat intelligence instance"""
    global _threat_intelligence
    if _threat_intelligence is None:
        _threat_intelligence = ThreatIntelligence()
    return _threat_intelligence
