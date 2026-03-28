import pytest
import numpy as np
from backend.models.fusion_engine import FusionEngine

def test_fusion_agreement():
    engine = FusionEngine()
    # All models agree it's a deepfake
    report = engine.fuse(video_score=0.9, audio_score=0.9, temporal_score=0.9)
    assert report.verdict == "DEEPFAKE DETECTED"
    assert report.final_score >= 0.9

def test_fusion_disagreement_veto():
    engine = FusionEngine()
    # Video and Temporal say REAL, but Audio is extremely confident it's FAKE
    report = engine.fuse(video_score=0.1, audio_score=0.98, temporal_score=0.1)
    # The 0.95 override should trigger
    assert report.verdict == "DEEPFAKE DETECTED"
    assert report.final_score >= 0.95

def test_fusion_weights():
    # Custom weights
    weights = {"video_spatial": 1.0, "audio_spectral": 0.0, "temporal_lstm": 0.0}
    engine = FusionEngine(weights=weights)
    report = engine.fuse(video_score=0.7, audio_score=0.1, temporal_score=0.1)
    # Result should strictly follow video
    assert report.final_score == 0.7
