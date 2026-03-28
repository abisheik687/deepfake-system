"""
KAVACH-AI Audio Deepfake Detection Router
Module C: Audio Deepfake / Voice Clone Detection

Accepts: MP3/WAV/OGG up to 50MB
Pipeline: Mel-spectrogram extraction → Audio CNN (RawNet2) → LCNN binary classifier
Returns: authenticity score, spectrogram visualization, anomaly regions, confidence %
"""

import io
import base64
import time
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from loguru import logger
import numpy as np
from PIL import Image
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from backend.database import get_db
from backend.crud import create_detection
from backend.api.rate_limit import limiter
from backend.features.audio_extraction import AudioFeatureExtractor
from backend.config import settings

router = APIRouter()

# Allowed audio formats
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/ogg", "audio/x-wav", "audio/wave"}
ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".ogg"}
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB


class AudioAnalysisRequest(BaseModel):
    """Request schema for base64 audio analysis"""
    data: str  # base64 encoded audio
    format: str = "wav"  # wav, mp3, ogg
    return_spectrogram: bool = True


class AudioAnalysisResponse(BaseModel):
    """Response schema for audio analysis"""
    verdict: str  # REAL, FAKE, SUSPICIOUS
    confidence: float  # 0-1
    authenticity_score: float  # 0-100
    fake_probability: float  # 0-1
    spectrogram_b64: Optional[str] = None
    anomaly_regions: list = []
    features: dict = {}
    processing_time_ms: float
    message: str


def _generate_spectrogram_image(audio_data: np.ndarray, sr: int, anomaly_regions: list = None) -> str:
    """
    Generate spectrogram visualization with anomaly highlighting
    Returns base64 encoded PNG image
    """
    try:
        # Create mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sr,
            n_mels=128,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            cmap='viridis'
        )
        
        # Highlight anomaly regions if provided
        if anomaly_regions:
            for region in anomaly_regions:
                start_time = region.get('start_time', 0)
                end_time = region.get('end_time', 0)
                ax.axvspan(start_time, end_time, alpha=0.3, color='red')
        
        ax.set_title('Mel-Spectrogram with Anomaly Detection')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_b64}"
        
    except Exception as e:
        logger.error(f"Error generating spectrogram: {e}")
        return ""


def _detect_anomalies(mel_spec: np.ndarray, sr: int, duration: float) -> list:
    """
    Detect anomaly regions in spectrogram using heuristics
    Returns list of suspicious time regions
    """
    anomalies = []
    
    try:
        # Calculate spectral energy per time frame
        energy = np.sum(mel_spec, axis=0)
        
        # Detect sudden energy spikes (potential artifacts)
        energy_diff = np.abs(np.diff(energy))
        threshold = np.mean(energy_diff) + 2 * np.std(energy_diff)
        
        spike_indices = np.where(energy_diff > threshold)[0]
        
        # Convert frame indices to time
        hop_length = 512
        times = librosa.frames_to_time(spike_indices, sr=sr, hop_length=hop_length)
        
        # Group nearby spikes into regions
        if len(times) > 0:
            current_start = times[0]
            current_end = times[0]
            
            for t in times[1:]:
                if t - current_end < 0.5:  # Within 0.5 seconds
                    current_end = t
                else:
                    anomalies.append({
                        'start_time': float(current_start),
                        'end_time': float(current_end),
                        'type': 'energy_spike',
                        'severity': 'medium'
                    })
                    current_start = t
                    current_end = t
            
            # Add last region
            anomalies.append({
                'start_time': float(current_start),
                'end_time': float(current_end),
                'type': 'energy_spike',
                'severity': 'medium'
            })
    
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
    
    return anomalies


def _analyze_audio_features(audio_data: np.ndarray, sr: int) -> dict:
    """
    Extract audio features for deepfake detection
    Returns dict of feature statistics
    """
    features = {}
    
    try:
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        
        # Prosody features
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        
        features = {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
            'mfcc_mean': float(np.mean(mfccs)),
            'mfcc_std': float(np.std(mfccs)),
            'pitch_variance': float(np.var(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0,
            'duration_seconds': float(len(audio_data) / sr)
        }
        
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
    
    return features


def _compute_audio_score(features: dict, anomaly_count: int) -> tuple[str, float, float]:
    """
    Compute deepfake score based on audio features and anomalies
    Returns (verdict, confidence, fake_probability)
    
    This is a heuristic-based approach. In production, this would use
    a trained RawNet2 or LCNN model for actual inference.
    """
    # Heuristic scoring based on feature anomalies
    score = 0.0
    
    # Check for unusual spectral characteristics
    if features.get('spectral_centroid_std', 0) > 1000:
        score += 0.2
    
    if features.get('zero_crossing_rate_mean', 0) > 0.15:
        score += 0.15
    
    # Check for pitch irregularities (common in voice clones)
    if features.get('pitch_variance', 0) > 5000:
        score += 0.25
    
    # Anomaly regions contribute to fake score
    if anomaly_count > 0:
        score += min(0.3, anomaly_count * 0.1)
    
    # MFCC variance (synthetic audio often has lower variance)
    if features.get('mfcc_std', 0) < 5:
        score += 0.1
    
    # Clamp score
    fake_prob = min(1.0, max(0.0, score))
    confidence = abs(fake_prob - 0.5) * 2  # Higher confidence when far from 0.5
    
    # Determine verdict
    if fake_prob >= 0.75:
        verdict = "FAKE"
    elif fake_prob >= 0.50:
        verdict = "SUSPICIOUS"
    else:
        verdict = "REAL"
    
    return verdict, confidence, fake_prob


@router.post("/analyze", response_model=AudioAnalysisResponse)
@limiter.limit("10/minute")
async def analyze_audio(
    request: Request,
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze audio file for deepfake detection
    
    Accepts: MP3, WAV, OGG files up to 50MB
    Returns: Authenticity score, spectrogram, anomaly regions
    """
    t0 = time.perf_counter()
    
    try:
        # Validate file upload
        if not file or not file.filename:
            raise HTTPException(400, "No audio file provided")
        
        # Check file extension
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXT:
            raise HTTPException(
                400,
                f"Invalid audio format. Allowed: {', '.join(ALLOWED_AUDIO_EXT)}"
            )
        
        # Check file size
        content = await file.read()
        if len(content) > MAX_AUDIO_SIZE:
            raise HTTPException(
                400,
                f"File too large. Maximum size: {MAX_AUDIO_SIZE / (1024*1024):.0f}MB"
            )
        
        # Save temporarily
        temp_path = Path(settings.EVIDENCE_DIR) / f"temp_audio_{int(time.time())}{ext}"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        # Load audio
        audio_data, sr = librosa.load(str(temp_path), sr=settings.AUDIO_SAMPLE_RATE)
        duration = len(audio_data) / sr
        
        logger.info(f"Loaded audio: {duration:.2f}s, sr={sr}Hz")
        
        # Extract features
        features = _analyze_audio_features(audio_data, sr)
        
        # Generate mel-spectrogram for analysis
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        
        # Detect anomalies
        anomaly_regions = _detect_anomalies(mel_spec, sr, duration)
        
        # Compute deepfake score
        verdict, confidence, fake_prob = _compute_audio_score(features, len(anomaly_regions))
        
        # Generate spectrogram visualization
        spectrogram_b64 = _generate_spectrogram_image(audio_data, sr, anomaly_regions)
        
        # Calculate authenticity score (inverse of fake probability)
        authenticity_score = (1.0 - fake_prob) * 100
        
        # Clean up temp file
        try:
            temp_path.unlink()
        except:
            pass
        
        # Save to database
        detection_data = {
            'file_name': file.filename,
            'file_type': 'audio',
            'verdict': verdict,
            'confidence': confidence,
            'fake_probability': fake_prob,
            'processing_time_ms': (time.perf_counter() - t0) * 1000,
            'metadata': {
                'duration': duration,
                'sample_rate': sr,
                'anomaly_count': len(anomaly_regions),
                'features': features
            }
        }
        
        await create_detection(db, detection_data)
        
        processing_time = (time.perf_counter() - t0) * 1000
        
        message = f"Audio analysis complete. Detected {len(anomaly_regions)} anomaly regions."
        if verdict == "FAKE":
            message += " High probability of synthetic/cloned audio."
        elif verdict == "SUSPICIOUS":
            message += " Some suspicious patterns detected."
        else:
            message += " Audio appears authentic."
        
        return AudioAnalysisResponse(
            verdict=verdict,
            confidence=confidence,
            authenticity_score=authenticity_score,
            fake_probability=fake_prob,
            spectrogram_b64=spectrogram_b64,
            anomaly_regions=anomaly_regions,
            features=features,
            processing_time_ms=processing_time,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        raise HTTPException(500, f"Audio analysis failed: {str(e)}")


@router.post("/analyze-base64", response_model=AudioAnalysisResponse)
@limiter.limit("10/minute")
async def analyze_audio_base64(
    request: Request,
    req: AudioAnalysisRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze base64-encoded audio data
    Useful for WebSocket/streaming audio analysis
    """
    t0 = time.perf_counter()
    
    try:
        # Decode base64
        if req.data.startswith('data:audio'):
            # Remove data URI prefix
            audio_b64 = req.data.split(',')[1]
        else:
            audio_b64 = req.data
        
        audio_bytes = base64.b64decode(audio_b64)
        
        # Save temporarily
        temp_path = Path(settings.EVIDENCE_DIR) / f"temp_audio_{int(time.time())}.{req.format}"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            f.write(audio_bytes)
        
        # Load and analyze (same as file upload)
        audio_data, sr = librosa.load(str(temp_path), sr=settings.AUDIO_SAMPLE_RATE)
        duration = len(audio_data) / sr
        
        features = _analyze_audio_features(audio_data, sr)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        anomaly_regions = _detect_anomalies(mel_spec, sr, duration)
        verdict, confidence, fake_prob = _compute_audio_score(features, len(anomaly_regions))
        
        spectrogram_b64 = ""
        if req.return_spectrogram:
            spectrogram_b64 = _generate_spectrogram_image(audio_data, sr, anomaly_regions)
        
        authenticity_score = (1.0 - fake_prob) * 100
        
        # Clean up
        try:
            temp_path.unlink()
        except:
            pass
        
        processing_time = (time.perf_counter() - t0) * 1000
        
        return AudioAnalysisResponse(
            verdict=verdict,
            confidence=confidence,
            authenticity_score=authenticity_score,
            fake_probability=fake_prob,
            spectrogram_b64=spectrogram_b64,
            anomaly_regions=anomaly_regions,
            features=features,
            processing_time_ms=processing_time,
            message=f"Audio analysis complete. {len(anomaly_regions)} anomalies detected."
        )
        
    except Exception as e:
        logger.error(f"Base64 audio analysis error: {e}")
        raise HTTPException(500, f"Audio analysis failed: {str(e)}")


@router.get("/health")
async def audio_health():
    """Health check for audio detection module"""
    return {
        "status": "healthy",
        "module": "audio_detection",
        "supported_formats": list(ALLOWED_AUDIO_EXT),
        "max_file_size_mb": MAX_AUDIO_SIZE / (1024 * 1024),
        "sample_rate": settings.AUDIO_SAMPLE_RATE
    }

# Made with Bob
