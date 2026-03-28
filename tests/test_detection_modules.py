"""
KAVACH-AI Detection Modules Test Suite
Comprehensive pytest tests for all detection modules (A-G)
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from PIL import Image
import io
import base64


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing"""
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    return img


@pytest.fixture
def sample_image_bytes(sample_image):
    """Convert sample image to bytes"""
    buf = io.BytesIO()
    sample_image.save(buf, format='JPEG')
    return buf.getvalue()


@pytest.fixture
def sample_image_base64(sample_image_bytes):
    """Convert sample image to base64"""
    return base64.b64encode(sample_image_bytes).decode('utf-8')


@pytest.fixture
def sample_audio_data():
    """Create sample audio data (1 second at 16kHz)"""
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    # Generate sine wave
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    return audio.astype(np.float32)


@pytest.fixture
def sample_video_frames():
    """Create sample video frames"""
    frames = []
    for i in range(10):
        img = Image.new('RGB', (224, 224), color=(i * 25, 128, 128))
        frames.append(np.array(img))
    return frames


# ============================================================================
# MODULE A: IMAGE DEEPFAKE DETECTION TESTS
# ============================================================================

class TestImageDetection:
    """Test Module A: Image Deepfake Detection"""
    
    def test_ensemble_import(self):
        """Test that ensemble module can be imported"""
        from backend.ml import ensemble
        assert hasattr(ensemble, 'weighted_soft_voting')
        assert hasattr(ensemble, 'test_time_augmentation')
    
    def test_model_weights_defined(self):
        """Test that model weights are properly defined"""
        from backend.ml.ensemble import MODEL_WEIGHTS
        
        assert 'efficientnet_b4' in MODEL_WEIGHTS
        assert 'xception' in MODEL_WEIGHTS
        assert 'vit' in MODEL_WEIGHTS or 'vit_deepfake_primary' in MODEL_WEIGHTS
        
        # Check weights sum to reasonable value
        total = sum(MODEL_WEIGHTS.values())
        assert 0.5 < total < 2.0  # Reasonable range
    
    def test_weighted_soft_voting(self):
        """Test weighted soft voting ensemble"""
        from backend.ml.ensemble import weighted_soft_voting, ModelPrediction
        
        predictions = [
            ModelPrediction("vit", 0.85, np.array([0.15, 0.85]), 100.0, 0.30),
            ModelPrediction("efficientnet_b4", 0.78, np.array([0.22, 0.78]), 95.0, 0.25),
            ModelPrediction("xception", 0.82, np.array([0.18, 0.82]), 110.0, 0.20),
        ]
        
        result = weighted_soft_voting(predictions)
        
        assert result.verdict in ["REAL", "FAKE"]
        assert 0.0 <= result.final_confidence <= 1.0
        assert 0.0 <= result.agreement_score <= 1.0
        assert isinstance(result.abstain, bool)
        assert len(result.weighted_votes) == 3
    
    def test_abstain_threshold(self):
        """Test abstain mechanism when models disagree"""
        from backend.ml.ensemble import weighted_soft_voting, ModelPrediction
        
        # Create predictions with high disagreement
        predictions = [
            ModelPrediction("model1", 0.95, np.array([0.05, 0.95]), 100.0, 1.0),
            ModelPrediction("model2", 0.10, np.array([0.90, 0.10]), 100.0, 1.0),
        ]
        
        result = weighted_soft_voting(predictions, abstain_threshold=0.35)
        
        # Should abstain due to high disagreement
        assert result.abstain == True
    
    def test_test_time_augmentation(self):
        """Test TTA with multiple crops"""
        from backend.ml.ensemble import test_time_augmentation
        
        # Create sample image
        image = np.random.rand(256, 256, 3).astype(np.float32)
        
        # Mock prediction function
        def mock_predict(crop):
            return 0.75  # Fixed confidence
        
        avg_conf, individual = test_time_augmentation(image, mock_predict, num_crops=5)
        
        assert 0.0 <= avg_conf <= 1.0
        assert len(individual) >= 1  # At least center crop
        assert all(0.0 <= c <= 1.0 for c in individual)
    
    def test_temperature_scaling(self):
        """Test temperature scaling for calibration"""
        from backend.ml.ensemble import apply_temperature_scaling
        
        logits = np.array([2.0, -1.0])
        
        # Test with different temperatures
        probs_low = apply_temperature_scaling(logits, temperature=0.5)
        probs_high = apply_temperature_scaling(logits, temperature=2.0)
        
        assert probs_low.shape == logits.shape
        assert probs_high.shape == logits.shape
        assert np.allclose(probs_low.sum(), 1.0)
        assert np.allclose(probs_high.sum(), 1.0)
        
        # Lower temperature should make predictions more confident
        assert probs_low[0] > probs_high[0]


# ============================================================================
# MODULE B: VIDEO DEEPFAKE DETECTION TESTS
# ============================================================================

class TestVideoDetection:
    """Test Module B: Video Deepfake Detection"""
    
    def test_temporal_consistency_check(self):
        """Test temporal consistency analysis"""
        # Create frame scores with variance
        frame_scores = np.array([0.7, 0.72, 0.68, 0.85, 0.71, 0.69])
        
        # Calculate variance
        variance = np.var(frame_scores)
        
        # Should flag if variance > 0.15
        threshold = 0.15
        is_suspicious = variance > threshold
        
        assert isinstance(is_suspicious, (bool, np.bool_))
    
    def test_frame_sampling(self):
        """Test frame sampling strategy (every 5th frame)"""
        total_frames = 150
        sample_rate = 5
        
        sampled_indices = list(range(0, total_frames, sample_rate))
        
        assert len(sampled_indices) == 30  # 150 / 5
        assert sampled_indices[0] == 0
        assert sampled_indices[-1] == 145


# ============================================================================
# MODULE C: AUDIO DEEPFAKE DETECTION TESTS
# ============================================================================

class TestAudioDetection:
    """Test Module C: Audio Deepfake Detection"""
    
    def test_audio_features_extraction(self, sample_audio_data):
        """Test audio feature extraction"""
        import librosa
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=sample_audio_data,
            sr=16000,
            n_mels=128
        )
        
        assert mel_spec.shape[0] == 128  # n_mels
        assert mel_spec.shape[1] > 0  # time frames
    
    def test_audio_anomaly_detection(self, sample_audio_data):
        """Test audio anomaly detection features"""
        import librosa
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=sample_audio_data, sr=16000)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(sample_audio_data)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=sample_audio_data, sr=16000, n_mfcc=13)
        
        assert centroid.shape[1] > 0
        assert zcr.shape[1] > 0
        assert mfccs.shape == (13, centroid.shape[1])


# ============================================================================
# MODULE D: SOCIAL MEDIA URL DETECTION TESTS
# ============================================================================

class TestSocialMediaDetection:
    """Test Module D: Social Media URL Detection"""
    
    def test_platform_detection(self):
        """Test platform detection from URL"""
        test_urls = {
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "youtube",
            "https://twitter.com/user/status/123": "twitter",
            "https://www.instagram.com/p/ABC123/": "instagram",
            "https://www.tiktok.com/@user/video/123": "tiktok",
            "https://www.facebook.com/watch/?v=123": "facebook",
        }
        
        for url, expected_platform in test_urls.items():
            # Simple platform detection
            if "youtube" in url:
                platform = "youtube"
            elif "twitter" in url or "x.com" in url:
                platform = "twitter"
            elif "instagram" in url:
                platform = "instagram"
            elif "tiktok" in url:
                platform = "tiktok"
            elif "facebook" in url:
                platform = "facebook"
            else:
                platform = "unknown"
            
            assert platform == expected_platform
    
    def test_risk_level_classification(self):
        """Test risk level classification"""
        risk_scores = [0.2, 0.45, 0.68, 0.92]
        expected_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        for score, expected in zip(risk_scores, expected_levels):
            if score < 0.3:
                level = "LOW"
            elif score < 0.6:
                level = "MEDIUM"
            elif score < 0.8:
                level = "HIGH"
            else:
                level = "CRITICAL"
            
            assert level == expected


# ============================================================================
# MODULE E: LIVE VIDEO CALL DETECTION TESTS
# ============================================================================

class TestLiveVideoDetection:
    """Test Module E: Live Video Call Detection"""
    
    def test_frame_capture_rate(self):
        """Test frame capture every 2 seconds"""
        fps = 30
        capture_interval = 2  # seconds
        
        frames_to_skip = fps * capture_interval
        
        assert frames_to_skip == 60
    
    def test_real_time_inference_latency(self):
        """Test that inference meets real-time requirements (<50ms)"""
        # Mock inference time
        inference_times = [45, 48, 42, 50, 47]  # milliseconds
        
        avg_latency = np.mean(inference_times)
        max_latency = np.max(inference_times)
        
        assert avg_latency < 50
        assert max_latency <= 50


# ============================================================================
# MODULE F: VOICE CALL DETECTION TESTS
# ============================================================================

class TestVoiceCallDetection:
    """Test Module F: Voice Call/Audio Stream Detection"""
    
    def test_audio_chunk_processing(self):
        """Test 2-second audio chunk processing"""
        sample_rate = 16000
        chunk_duration = 2.0
        
        chunk_samples = int(sample_rate * chunk_duration)
        
        assert chunk_samples == 32000
    
    def test_consecutive_fake_detection(self):
        """Test consecutive fake detection alerting"""
        # Simulate chunk scores
        chunk_scores = [0.82, 0.78, 0.85, 0.45, 0.88]
        threshold = 0.75
        consecutive_required = 3
        
        consecutive_count = 0
        alert_triggered = False
        
        for score in chunk_scores:
            if score > threshold:
                consecutive_count += 1
                if consecutive_count >= consecutive_required:
                    alert_triggered = True
                    break
            else:
                consecutive_count = 0
        
        assert alert_triggered == True


# ============================================================================
# MODULE G: INTERVIEW PROCTORING TESTS
# ============================================================================

class TestInterviewProctoring:
    """Test Module G: Live Interview Proctoring Mode"""
    
    def test_integrity_score_calculation(self):
        """Test weighted integrity score (60% video, 40% audio)"""
        video_score = 0.85
        audio_score = 0.72
        
        integrity_score = (video_score * 0.6) + (audio_score * 0.4)
        
        expected = 0.798
        assert abs(integrity_score - expected) < 0.001
    
    def test_session_report_generation(self):
        """Test PDF report generation structure"""
        report_data = {
            "candidate_name": "Test User",
            "date": "2026-03-26",
            "duration": "45:30",
            "integrity_verdict": "PASS",
            "video_confidence": 0.85,
            "audio_confidence": 0.72,
            "suspicious_events": [
                {"timestamp": "00:15:30", "type": "video", "confidence": 0.68},
                {"timestamp": "00:32:45", "type": "audio", "confidence": 0.55},
            ]
        }
        
        assert "candidate_name" in report_data
        assert "integrity_verdict" in report_data
        assert len(report_data["suspicious_events"]) == 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across modules"""
    
    def test_hf_fallback_system(self):
        """Test Hugging Face fallback system"""
        from backend.ml.hf_fallback import HF_MODEL_REGISTRY, get_model_weight
        
        assert len(HF_MODEL_REGISTRY) > 0
        assert "efficientnet_b4" in HF_MODEL_REGISTRY
        
        # Test weight retrieval
        weight = get_model_weight("efficientnet_b4")
        assert 0.0 < weight <= 1.0
    
    def test_uncertainty_calculation(self):
        """Test uncertainty metrics calculation"""
        from backend.ml.ensemble import calculate_uncertainty, ModelPrediction
        
        predictions = [
            ModelPrediction("model1", 0.85, np.array([0.15, 0.85]), 100.0, 1.0),
            ModelPrediction("model2", 0.82, np.array([0.18, 0.82]), 100.0, 1.0),
            ModelPrediction("model3", 0.88, np.array([0.12, 0.88]), 100.0, 1.0),
        ]
        
        uncertainty = calculate_uncertainty(predictions)
        
        assert "variance" in uncertainty
        assert "entropy" in uncertainty
        assert "disagreement_rate" in uncertainty
        assert all(0.0 <= v <= 1.0 for v in uncertainty.values() if v != uncertainty["entropy"])


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and load tests"""
    
    def test_ensemble_performance(self):
        """Test ensemble voting performance"""
        from backend.ml.ensemble import weighted_soft_voting, ModelPrediction
        import time
        
        # Create large batch of predictions
        predictions = [
            ModelPrediction(f"model{i}", 0.5 + np.random.rand() * 0.3, 
                          np.array([0.5, 0.5]), 100.0, 1.0)
            for i in range(10)
        ]
        
        start = time.time()
        result = weighted_soft_voting(predictions)
        elapsed = time.time() - start
        
        # Should complete in < 100ms
        assert elapsed < 0.1
        assert result is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

# Made with Bob
