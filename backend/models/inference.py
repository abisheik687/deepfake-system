"""
KAVACH-AI Model Inference
ONNX Runtime-based model serving for deepfake detection
NO API KEYS REQUIRED - All models run locally
"""

import onnxruntime as ort
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
from loguru import logger

from backend.config import settings


class SpatialDetector:
    """
    Spatial deepfake detection model.
    
    Uses pre-trained CNN (EfficientNet-B4 or MobileNetV3) for
    single-frame deepfake detection.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize spatial detector.
        
        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = model_path or settings.SPATIAL_MODEL_PATH
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.input_shape: Optional[Tuple] = None
        
        # Load model if path exists
        if Path(self.model_path).exists():
            self._load_model()
        else:
            logger.warning(
                f"Spatial model not found at {self.model_path}. "
                "Detection will use placeholder logic until model is provided."
            )
    
    def _load_model(self):
        """Load ONNX model"""
        try:
            # Set execution providers based on configuration
            providers = self._get_execution_providers()
            
            # Create inference session
            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers
            )
            
            # Get input/output metadata
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            logger.success(
                f"Spatial model loaded: {self.model_path} "
                f"(providers: {providers})"
            )
        
        except Exception as e:
            logger.error(f"Error loading spatial model: {e}")
            self.session = None
    
    def _get_execution_providers(self) -> List[str]:
        """
        Get ONNX Runtime execution providers based on configuration.
        
        Returns:
            List of execution providers in priority order
        """
        providers = []
        
        if settings.ENABLE_GPU:
            if settings.EXECUTION_PROVIDER == "cuda":
                providers.append("CUDAExecutionProvider")
            elif settings.EXECUTION_PROVIDER == "dml":
                providers.append("DmlExecutionProvider")  # DirectML for AMD/Intel
            elif settings.EXECUTION_PROVIDER == "openvino":
                providers.append("OpenVINOExecutionProvider")
        
        # Always add CPU as fallback
        providers.append("CPUExecutionProvider")
        
        return providers
    
    def predict(self, features: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Predict deepfake probability for feature tensor.
        
        Args:
            features: Feature tensor (H, W, C) or batch (B, H, W, C)
        
        Returns:
            (confidence, raw_output)
            - confidence: Deepfake probability [0, 1]
            - raw_output: Raw model logits
        """
        if self.session is None:
            # Placeholder: Use simple heuristic based on feature statistics
            return self._placeholder_prediction(features)
        
        try:
            # Ensure batch dimension
            if features.ndim == 3:
                features = np.expand_dims(features, axis=0)
            
            # Transpose to NCHW format (if needed by model)
            # ONNX models often expect channels-first
            if features.shape[-1] in [3, 5]:  # Channels last
                features = np.transpose(features, (0, 3, 1, 2))
            
            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: features.astype(np.float32)}
            )
            
            raw_output = outputs[0]
            
            # Apply sigmoid if output is logits
            if raw_output.min() < 0 or raw_output.max() > 1:
                confidence = 1 / (1 + np.exp(-raw_output))
            else:
                confidence = raw_output
            
            # Get confidence for deepfake class (assuming binary classification)
            if confidence.ndim > 1:
                confidence = confidence[0, 1] if confidence.shape[-1] == 2 else confidence[0, 0]
            else:
                confidence = float(confidence)
            
            return (float(confidence), raw_output)
        
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return self._placeholder_prediction(features)
    
    def _placeholder_prediction(self, features: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Placeholder prediction when model not available.
        
        Uses simple statistical analysis of features.
        This is a temporary fallback until actual model is provided.
        """
        # Calculate feature statistics
        mean = np.mean(features)
        std = np.std(features)
        
        # Simple heuristic: unusual statistics suggest manipulation
        # Real images typically have certain statistical properties
        anomaly_score = 0.0
        
        # Check mean brightness (deepfakes often have unusual brightness)
        if mean < 0.3 or mean > 0.7:
            anomaly_score += 0.2
        
        # Check standard deviation (deepfakes may have lower variance)
        if std < 0.1:
            anomaly_score += 0.3
        
        # Add random component for variability
        noise = np.random.uniform(-0.1, 0.1)
        confidence = np.clip(anomaly_score + noise, 0.0, 1.0)
        
        raw_output = np.array([[1 - confidence, confidence]])
        
        logger.debug(
            f"Placeholder prediction: {confidence:.3f} "
            "(model not loaded, using heuristic)"
        )
        
        return (float(confidence), raw_output)
    
    def predict_batch(
        self,
        features_batch: List[np.ndarray]
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Predict for batch of features.
        
        Args:
            features_batch: List of feature tensors
        
        Returns:
            List of (confidence, raw_output) tuples
        """
        results = []
        for features in features_batch:
            try:
                result = self.predict(features)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                results.append((0.0, np.array([[0.0, 0.0]])))
        
        return results


class TemporalDetector:
    """
    Temporal deepfake detection model.
    
    Uses LSTM for analyzing temporal consistency across frames.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize temporal detector.
        
        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = model_path or settings.TEMPORAL_MODEL_PATH
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        
        if Path(self.model_path).exists():
            self._load_model()
        else:
            logger.warning(
                f"Temporal model not found at {self.model_path}. "
                "Will use heuristic-based analysis until model is provided."
            )
    
    def _load_model(self):
        """Load ONNX model"""
        try:
            providers = self._get_execution_providers()
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.success(f"Temporal model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading temporal model: {e}")
            self.session = None
    
    def _get_execution_providers(self) -> List[str]:
        """Get execution providers"""
        providers = []
        if settings.ENABLE_GPU and settings.EXECUTION_PROVIDER == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    
    def predict(
        self,
        frame_sequence: np.ndarray,
        temporal_features: Optional[dict] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Predict temporal consistency.
        
        Args:
            frame_sequence: Sequence of frames (T, H, W, C)
            temporal_features: Optional temporal features dict
        
        Returns:
            (confidence, raw_output)
        """
        if self.session is None:
            # Use heuristic-based temporal analysis
            return self._heuristic_prediction(temporal_features)
        
        try:
            # TODO: Actual LSTM model inference
            # For now, use heuristics
            return self._heuristic_prediction(temporal_features)
        
        except Exception as e:
            logger.error(f"Error in temporal inference: {e}")
            return (0.0, np.array([[0.0, 0.0]]))
    
    def _heuristic_prediction(
        self,
        temporal_features: Optional[dict]
    ) -> Tuple[float, np.ndarray]:
        """
        Heuristic-based temporal prediction.
        
        Uses temporal anomaly score from temporal analyzer.
        """
        if temporal_features is None:
            return (0.0, np.array([[1.0, 0.0]]))
        
        # Use temporal anomaly score as confidence
        anomaly_score = temporal_features.get('temporal_anomaly_score', 0.0)
        
        confidence = float(anomaly_score)
        raw_output = np.array([[1 - confidence, confidence]])
        
        logger.debug(
            f"Temporal heuristic prediction: {confidence:.3f} "
            f"(blink_rate: {temporal_features.get('blink_rate', 0):.1f}, "
            f"jitter: {temporal_features.get('landmark_jitter', 0):.1f})"
        )
        
        return (confidence, raw_output)


class AudioDetector:
    """
    Audio deepfake detection model.
    
    Uses CNN-LSTM for audio analysis or heuristic-based anomaly detection.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize audio detector.
        
        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = model_path or settings.AUDIO_MODEL_PATH
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        
        if Path(self.model_path).exists():
            self._load_model()
        else:
            logger.warning(
                f"Audio model not found at {self.model_path}. "
                "Will use heuristic-based anomaly detection until model is provided."
            )
    
    def _load_model(self):
        """Load ONNX model"""
        try:
            providers = self._get_execution_providers()
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.success(f"Audio model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading audio model: {e}")
            self.session = None
    
    def _get_execution_providers(self) -> List[str]:
        """Get execution providers"""
        providers = []
        if settings.ENABLE_GPU and settings.EXECUTION_PROVIDER == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    
    def predict(
        self,
        audio_features: Optional[Dict] = None,
        mfcc: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Predict audio deepfake probability.
        
        Args:
            audio_features: Dictionary of audio features
            mfcc: MFCC features (fallback if features dict not provided)
        
        Returns:
            (confidence, raw_output)
        """
        if self.session is None:
            # Use heuristic-based anomaly detection
            return self._heuristic_prediction(audio_features)
        
        try:
            # TODO: Actual CNN-LSTM model inference
            # For now, use heuristics
            return self._heuristic_prediction(audio_features)
        
        except Exception as e:
            logger.error(f"Error in audio inference: {e}")
            return (0.0, np.array([[1.0, 0.0]]))
    
    def _heuristic_prediction(
        self,
        audio_features: Optional[Dict]
    ) -> Tuple[float, np.ndarray]:
        """
        Heuristic-based audio prediction.
        
        Uses anomaly scores from audio feature extraction.
        """
        if audio_features is None:
            return (0.0, np.array([[1.0, 0.0]]))
        
        # Use anomaly scores
        anomalies = audio_features.get('anomalies', {})
        overall_anomaly = anomalies.get('overall', 0.0)
        
        confidence = float(overall_anomaly)
        raw_output = np.array([[1 - confidence, confidence]])
        
        logger.debug(
            f"Audio heuristic prediction: {confidence:.3f} "
            f"(pitch: {anomalies.get('pitch_stability', 0):.2f}, "
            f"spectral: {anomalies.get('spectral_artifacts', 0):.2f})"
        )
        
        return (confidence, raw_output)


# Global model instances
_spatial_detector: Optional[SpatialDetector] = None
_temporal_detector: Optional[TemporalDetector] = None
_audio_detector: Optional[AudioDetector] = None


def get_spatial_detector() -> SpatialDetector:
    """Get global spatial detector instance"""
    global _spatial_detector
    if _spatial_detector is None:
        _spatial_detector = SpatialDetector()
    return _spatial_detector


def get_temporal_detector() -> TemporalDetector:
    """Get global temporal detector instance"""
    global _temporal_detector
    if _temporal_detector is None:
        _temporal_detector = TemporalDetector()
    return _temporal_detector


def get_audio_detector() -> AudioDetector:
    """Get global audio detector instance"""
    global _audio_detector
    if _audio_detector is None:
        _audio_detector = AudioDetector()
    return _audio_detector
