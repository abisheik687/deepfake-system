
import numpy as np
import librosa
import torch
import torch.nn as nn

class AudioFeatureExtractor:
    """
    KAVACH-AI Day 3: Audio Feature Extractor
    Extracts Mel-Spectrograms and MFCCs for Deepfake Audio Detection.
    """
    
    def __init__(self, sample_rate=16000, duration=3.0, n_mels=128):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.target_length = int(sample_rate * duration)

    def load_audio(self, audio_path):
        """Load audio and pad/trim to fixed duration"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or trim
            if len(y) < self.target_length:
                padding = self.target_length - len(y)
                y = np.pad(y, (0, padding), mode='constant')
            else:
                y = y[:self.target_length]
                
            return y
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return np.zeros(self.target_length)

    def extract_mel_spectrogram(self, audio_data):
        """
        Convert waveform to Mel-Spectrogram.
        Returns tensor of shape (1, n_mels, time_steps)
        """
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=self.sample_rate, 
            n_mels=self.n_mels
        )
        
        # Log-Mel
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-1
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Convert to tensor
        spec_tensor = torch.from_numpy(mel_spec_norm).float()
        
        # Add channel dimension (1 channel for grayscale-like spectrogram)
        spec_tensor = spec_tensor.unsqueeze(0)
        
        return spec_tensor

    def extract_mfcc(self, audio_data, n_mfcc=40):
        """Extract MFCC features (alternative or complementary)"""
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=n_mfcc)
        return torch.from_numpy(mfcc).float().unsqueeze(0)

    def get_features(self, audio_input):
        """Main entry point: accepts path or numpy array"""
        if isinstance(audio_input, str):
            audio_data = self.load_audio(audio_input)
        else:
            audio_data = audio_input
            
        return self.extract_mel_spectrogram(audio_data)
