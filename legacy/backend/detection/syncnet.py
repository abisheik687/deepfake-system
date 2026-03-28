"""
KAVACH-AI — SyncNet Lip-Audio Synchronization
Detects deepfakes by measuring the correlation between lip movements and audio features.
"""

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from typing import List, Tuple

class SyncNet:
    """
    World-Class Lip-Audio Synchronization Detector.
    Calculates the 'Sync Score' — high correlation = REAL, low correlation = FAKE.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        logger.info(f"[SyncNet] Initialized on {device}")

    def calculate_sync_score(self, lip_embeddings: np.ndarray, audio_embeddings: np.ndarray) -> float:
        """
        Calculates the cross-correlation between lip and audio embedding sequences.
        
        Parameters:
            lip_embeddings: (N, D) sequence of facial visual features
            audio_embeddings: (N, D) sequence of audio features (e.g. Wav2Vec2)
            
        Returns:
            sync_score: 0.0 (totally out of sync) to 1.0 (perfectly synced)
        """
        if len(lip_embeddings) != len(audio_embeddings):
            # Trim to match
            min_len = min(len(lip_embeddings), len(audio_embeddings))
            lip_embeddings = lip_embeddings[:min_len]
            audio_embeddings = audio_embeddings[:min_len]

        if len(lip_embeddings) < 5:
            return 0.5 # Not enough data

        # Normalize across the temporal dimension
        v = torch.tensor(lip_embeddings).to(self.device).T # (D, N)
        a = torch.tensor(audio_embeddings).to(self.device).T # (D, N)

        # Compute cosine similarity per frame
        v_norm = F.normalize(v, p=2, dim=0)
        a_norm = F.normalize(a, p=2, dim=0)
        
        # Mean cosine similarity across the sequence
        scores = (v_norm * a_norm).sum(dim=0)
        sync_score = scores.mean().item()

        # Scale from [-1, 1] to [0, 1]
        sync_score = (sync_score + 1) / 2
        return round(sync_score, 4)

    def detect_offset(self, lip_embeddings: np.ndarray, audio_embeddings: np.ndarray, max_offset: int = 15) -> Tuple[int, float]:
        """
        Shifts the audio relative to video to find the best sync point.
        If the best sync point is far from 0, it's a strong indicator of a deepfake.
        """
        best_score = -1.0
        best_offset = 0

        for offset in range(-max_offset, max_offset + 1):
            if offset < 0:
                v_slice = lip_embeddings[-offset:]
                a_slice = audio_embeddings[:len(v_slice)]
            elif offset > 0:
                a_slice = audio_embeddings[offset:]
                v_slice = lip_embeddings[:len(a_slice)]
            else:
                v_slice, a_slice = lip_embeddings, audio_embeddings
            
            score = self.calculate_sync_score(v_slice, a_slice)
            if score > best_score:
                best_score = score
                best_offset = offset

        return best_offset, best_score
