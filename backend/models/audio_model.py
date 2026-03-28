"""
Internal trace:
- Wrong before: audio analysis was purely heuristic while claiming a RawNet/LCNN pipeline, and it mixed visualization logic with inference.
- Fixed now: startup loads a Wav2Vec2-based detector when available and falls back to a transparent signal-based scorer when weights are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from config import settings
from utils.file_utils import clamp


def signal_fallback(audio: np.ndarray, sample_rate: int) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-4:
        return 0.5
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    zero_crossing = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    contrast = float(np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate)))
    score = (spectral_flatness * 2.0) + (zero_crossing * 0.9) + ((contrast / 30.0) * 0.55)
    return clamp(score)


@dataclass
class AudioModelHandle:
    infer: Callable[[np.ndarray, int], float]
    mode: str


def build_fallback_audio_model() -> AudioModelHandle:
    return AudioModelHandle(infer=signal_fallback, mode='fallback')


def build_audio_model() -> AudioModelHandle:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(settings.model_audio_repo)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(settings.model_audio_repo)
        model.eval().to(device)

        def infer(audio: np.ndarray, sample_rate: int) -> float:
            inputs = extractor(audio, sampling_rate=sample_rate, return_tensors='pt', padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            id2label = {int(key): value.lower() for key, value in model.config.id2label.items()}
            fake_indices = [index for index, label in id2label.items() if 'fake' in label or 'spoof' in label]
            if fake_indices:
                return clamp(float(sum(probs[index] for index in fake_indices)))
            return clamp(float(probs[-1]))

        return AudioModelHandle(infer=infer, mode='primary')
    except Exception:
        return build_fallback_audio_model()
