
import cv2
import numpy as np
import torch
from torchvision import transforms

class FeatureExtractor:
    """
    KAVACH-AI Day 2: Multi-channel Feature Extractor
    Extracts RGB (Spatial) and FFT (Frequency) features.
    """
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_rgb(self, face_crop):
        """Standard RGB preprocessing for CNNs"""
        if isinstance(face_crop, str):
            face_crop = cv2.imread(face_crop)
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
        return self.transform(face_crop)

    def extract_fft(self, face_crop):
        """
        Extract Frequency Domain features using FFT.
        Deepfakes often have artifacts in high-frequency domains.
        """
        if isinstance(face_crop, str):
            face_crop = cv2.imread(face_crop, cv2.IMREAD_GRAYSCALE)
        elif len(face_crop.shape) == 3:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            
        # Resize to fixed size
        img = cv2.resize(face_crop, (self.image_size, self.image_size))
        
        # 2D FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        # Normalize to 0-1
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 1, cv2.NORM_MINMAX)
        
        # Convert to tensor and stack to 3 channels (to match CNN input)
        fft_tensor = torch.from_numpy(magnitude_spectrum).float()
        fft_tensor = fft_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        return fft_tensor

    def get_dual_stream_input(self, face_crop):
        """Returns stacked RGB and FFT features"""
        rgb = self.preprocess_rgb(face_crop)
        fft_feat = self.extract_fft(face_crop)
        return rgb, fft_feat
