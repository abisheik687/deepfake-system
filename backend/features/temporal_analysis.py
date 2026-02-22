"""
KAVACH-AI Temporal Analysis
Temporal consistency detection for deepfakes using physiological signals
NO API KEYS REQUIRED - All processing is local
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from collections import deque
from loguru import logger

from backend.features.face_detection import FaceDetection


@dataclass
class BlinkEvent:
    """Blink detection event"""
    timestamp: float
    duration: float  # seconds
    eye_aspect_ratio: float


@dataclass
class TemporalFeatures:
    """Temporal features extracted from frame sequence"""
    # Blink analysis
    blink_rate: float  # blinks per minute
    blink_events: List[BlinkEvent]
    avg_blink_duration: float
    
    # Head movement
    head_movement_variance: float
    avg_head_speed: float  # degrees per second
    
    # Consistency metrics
    landmark_jitter: float  # pixel variance across frames
    expression_stability: float
    
    # Temporal score
    temporal_anomaly_score: float  # [0, 1] - higher = more anomalous


class TemporalAnalyzer:
    """
    Temporal analysis for deepfake detection.
    
    Analyzes physiological signals and temporal consistency:
    - Blink detection (Eye Aspect Ratio)
    - Head pose tracking
    - Landmark stability
    - Expression consistency
    """
    
    def __init__(
        self,
        buffer_size: int = 32,
        blink_ear_threshold: float = 0.21,
        min_blink_frames: int = 2
    ):
        """
        Initialize temporal analyzer.
        
        Args:
            buffer_size: Number of frames to buffer for analysis
            blink_ear_threshold: Eye Aspect Ratio threshold for blink detection
            min_blink_frames: Minimum consecutive frames for blink
        """
        self.buffer_size = buffer_size
        self.blink_ear_threshold = blink_ear_threshold
        self.min_blink_frames = min_blink_frames
        
        # Frame buffers: stream_id -> deque of FaceDetection
        self.face_buffers: Dict[int, deque] = {}
        
        # Blink tracking: stream_id -> track_id -> list of EAR values
        self.ear_history: Dict[int, Dict[int, deque]] = {}
        
        logger.info(
            f"Temporal analyzer initialized "
            f"(buffer: {buffer_size}, EAR threshold: {blink_ear_threshold})"
        )
    
    def add_detection(self, stream_id: int, face_detection: FaceDetection):
        """
        Add face detection to temporal buffer.
        
        Args:
            stream_id: Stream ID
            face_detection: Face detection result
        """
        # Initialize buffer if needed
        if stream_id not in self.face_buffers:
            self.face_buffers[stream_id] = deque(maxlen=self.buffer_size)
            self.ear_history[stream_id] = {}
        
        # Add to buffer
        self.face_buffers[stream_id].append(face_detection)
        
        # Track EAR for this face
        track_id = face_detection.track_id
        if track_id is not None:
            if track_id not in self.ear_history[stream_id]:
                self.ear_history[stream_id][track_id] = deque(maxlen=100)
            
            ear = self._calculate_ear(face_detection.landmarks)
            self.ear_history[stream_id][track_id].append(ear)
    
    def analyze(self, stream_id: int, track_id: int) -> Optional[TemporalFeatures]:
        """
        Analyze temporal features for a tracked face.
        
        Args:
            stream_id: Stream ID
            track_id: Face track ID
        
        Returns:
            TemporalFeatures or None if insufficient data
        """
        if stream_id not in self.face_buffers:
            return None
        
        # Get face detections for this track
        face_sequence = [
            f for f in self.face_buffers[stream_id]
            if f.track_id == track_id
        ]
        
        if len(face_sequence) < 5:
            logger.debug(f"Insufficient frames for temporal analysis (need 5, have {len(face_sequence)})")
            return None
        
        # Analyze blinks
        blink_rate, blink_events, avg_blink_dur = self._analyze_blinks(stream_id, track_id)
        
        # Analyze head movement
        head_variance, head_speed = self._analyze_head_movement(face_sequence)
        
        # Analyze landmark stability
        landmark_jitter = self._calculate_landmark_jitter(face_sequence)
        
        # Analyze expression stability
        expression_stability = self._calculate_expression_stability(face_sequence)
        
        # Calculate temporal anomaly score
        anomaly_score = self._calculate_anomaly_score(
            blink_rate=blink_rate,
            head_variance=head_variance,
            landmark_jitter=landmark_jitter,
            expression_stability=expression_stability
        )
        
        return TemporalFeatures(
            blink_rate=blink_rate,
            blink_events=blink_events,
            avg_blink_duration=avg_blink_dur,
            head_movement_variance=head_variance,
            avg_head_speed=head_speed,
            landmark_jitter=landmark_jitter,
            expression_stability=expression_stability,
            temporal_anomaly_score=anomaly_score
        )
    
    def _calculate_ear(self, landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        MediaPipe landmark indices:
        - Right eye: 33, 160, 158, 133, 153, 144
        - Left eye: 362, 385, 387, 263, 373, 380
        
        Args:
            landmarks: Facial landmarks (468, 3)
        
        Returns:
            Eye Aspect Ratio (lower = more closed)
        """
        # Right eye landmarks
        right_eye_indices = [33, 160, 158, 133, 153, 144]
        # Left eye landmarks
        left_eye_indices = [362, 385, 387, 263, 373, 380]
        
        def eye_ear(eye_landmarks):
            # Vertical distances
            v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            # Horizontal distance
            h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # EAR formula
            ear = (v1 + v2) / (2.0 * h)
            return ear
        
        # Calculate EAR for both eyes
        right_ear = eye_ear(landmarks[right_eye_indices])
        left_ear = eye_ear(landmarks[left_eye_indices])
        
        # Average EAR
        avg_ear = (right_ear + left_ear) / 2.0
        
        return float(avg_ear)
    
    def _analyze_blinks(
        self,
        stream_id: int,
        track_id: int
    ) -> Tuple[float, List[BlinkEvent], float]:
        """
        Analyze blink patterns from EAR history.
        
        Returns:
            (blink_rate, blink_events, avg_duration)
        """
        if stream_id not in self.ear_history or track_id not in self.ear_history[stream_id]:
            return (0.0, [], 0.0)
        
        ear_values = list(self.ear_history[stream_id][track_id])
        
        if len(ear_values) < 10:
            return (0.0, [], 0.0)
        
        # Detect blinks (EAR drops below threshold)
        blink_events = []
        in_blink = False
        blink_start = 0
        
        for i, ear in enumerate(ear_values):
            if ear < self.blink_ear_threshold and not in_blink:
                # Blink started
                in_blink = True
                blink_start = i
            elif ear >= self.blink_ear_threshold and in_blink:
                # Blink ended
                blink_duration = (i - blink_start) / 30.0  # Assume 30 fps
                
                if i - blink_start >= self.min_blink_frames:
                    blink_events.append(BlinkEvent(
                        timestamp=i / 30.0,
                        duration=blink_duration,
                        eye_aspect_ratio=min(ear_values[blink_start:i])
                    ))
                
                in_blink = False
        
        # Calculate blink rate (per minute)
        total_duration = len(ear_values) / 30.0  # seconds
        blink_rate = (len(blink_events) / total_duration) * 60 if total_duration > 0 else 0.0
        
        # Average blink duration
        avg_duration = np.mean([b.duration for b in blink_events]) if blink_events else 0.0
        
        return (float(blink_rate), blink_events, float(avg_duration))
    
    def _analyze_head_movement(
        self,
        face_sequence: List[FaceDetection]
    ) -> Tuple[float, float]:
        """
        Analyze head movement from pose angles.
        
        Returns:
            (variance, avg_speed)
        """
        if len(face_sequence) < 2:
            return (0.0, 0.0)
        
        # Extract pose angles
        pitches = [f.pitch for f in face_sequence if f.pitch is not None]
        yaws = [f.yaw for f in face_sequence if f.yaw is not None]
        rolls = [f.roll for f in face_sequence if f.roll is not None]
        
        if not pitches or not yaws or not rolls:
            return (0.0, 0.0)
        
        # Calculate variance across all angles
        all_angles = pitches + yaws + rolls
        variance = float(np.var(all_angles))
        
        # Calculate average angular speed
        speeds = []
        for i in range(1, len(pitches)):
            pitch_speed = abs(pitches[i] - pitches[i-1])
            yaw_speed = abs(yaws[i] - yaws[i-1])
            roll_speed = abs(rolls[i] - rolls[i-1])
            speeds.append(pitch_speed + yaw_speed + roll_speed)
        
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        
        return (variance, avg_speed)
    
    def _calculate_landmark_jitter(
        self,
        face_sequence: List[FaceDetection]
    ) -> float:
        """
        Calculate landmark position jitter across frames.
        
        High jitter indicates unstable tracking or synthetic generation.
        
        Returns:
            Jitter score (pixels)
        """
        if len(face_sequence) < 2:
            return 0.0
        
        # Stack landmarks from all frames
        landmark_sequences = [f.landmarks for f in face_sequence]
        
        # Calculate variance for each landmark point
        variances = []
        for lm_idx in range(468):
            positions = np.array([lm[lm_idx, :2] for lm in landmark_sequences])
            variance = np.var(positions)
            variances.append(variance)
        
        # Average variance across all landmarks
        jitter = float(np.mean(variances))
        
        return jitter
    
    def _calculate_expression_stability(
        self,
        face_sequence: List[FaceDetection]
    ) -> float:
        """
        Calculate expression stability score.
        
        Uses mouth and eyebrow landmark distances.
        
        Returns:
            Stability score [0, 1] (higher = more stable)
        """
        if len(face_sequence) < 2:
            return 1.0
        
        # Track mouth opening (distance between upper/lower lip)
        mouth_openings = []
        for face in face_sequence:
            # Upper lip center: 13, Lower lip center: 14
            mouth_opening = np.linalg.norm(
                face.landmarks[13, :2] - face.landmarks[14, :2]
            )
            mouth_openings.append(mouth_opening)
        
        # Calculate coefficient of variation
        mouth_cv = np.std(mouth_openings) / (np.mean(mouth_openings) + 1e-6)
        
        # Stability score (inverse of variation)
        stability = 1.0 / (1.0 + mouth_cv)
        
        return float(stability)
    
    def _calculate_anomaly_score(
        self,
        blink_rate: float,
        head_variance: float,
        landmark_jitter: float,
        expression_stability: float
    ) -> float:
        """
        Calculate overall temporal anomaly score.
        
        Combines multiple temporal signals into single score.
        
        Returns:
            Anomaly score [0, 1] (higher = more anomalous/likely deepfake)
        """
        anomaly = 0.0
        
        # Abnormal blink rate (normal: 15-20 blinks/min)
        if blink_rate < 10 or blink_rate > 30:
            anomaly += 0.3
        
        # Excessive head movement variance (unrealistic in deepfakes)
        if head_variance < 5.0:  # Too stable
            anomaly += 0.2
        elif head_variance > 100.0:  # Too jittery
            anomaly += 0.3
        
        # High landmark jitter (unstable generation)
        if landmark_jitter > 10.0:
            anomaly += 0.3
        
        # Low expression stability (unnatural)
        if expression_stability < 0.5:
            anomaly += 0.2
        
        # Clip to [0, 1]
        anomaly = min(1.0, anomaly)
        
        return float(anomaly)
    
    def clear_buffer(self, stream_id: int):
        """Clear temporal buffer for stream"""
        if stream_id in self.face_buffers:
            self.face_buffers[stream_id].clear()
        if stream_id in self.ear_history:
            self.ear_history[stream_id].clear()


# Global temporal analyzer instance
_temporal_analyzer: Optional[TemporalAnalyzer] = None


def get_temporal_analyzer() -> TemporalAnalyzer:
    """Get global temporal analyzer instance"""
    global _temporal_analyzer
    if _temporal_analyzer is None:
        _temporal_analyzer = TemporalAnalyzer()
    return _temporal_analyzer
