import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger

class FaceExtractor:
    """
    World-Class Face Extraction Pipeline using MTCNN.
    Features: Multi-face tracking, padding, alignment, and high-performance batch processing.
    """
    
    def __init__(self, 
                 image_size: int = 224, 
                 margin: int = 20, 
                 post_process: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.image_size = image_size
        
        # Initialize MTCNN
        # keep_all=True allows detecting multiple faces per frame
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=True,
            post_process=post_process,
            device=self.device
        )
        logger.info(f"FaceExtractor initialized on {self.device}")

    def extract_faces_from_image(self, image_path: Path) -> List[np.ndarray]:
        """Extracts all faces from a single image file."""
        try:
            img = Image.open(image_path).convert('RGB')
            # mtcnn returns a list of tensors [3, image_size, image_size]
            faces = self.mtcnn(img)
            
            if faces is None:
                return []
                
            # Convert tensors to numpy arrays for easier handling downstream
            return [face.permute(1, 2, 0).cpu().numpy() for face in faces]
        except Exception as e:
            logger.error(f"Failed to extract face from {image_path}: {e}")
            return []

    def process_video_frames(self, frames: List[np.ndarray], batch_size: int = 32) -> List[List[np.ndarray]]:
        """
        Processes a sequence of frames in batches for high throughput.
        Returns a list of lists (faces per frame).
        """
        all_faces = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            # Convert numpy frames to PIL
            pil_batch = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch]
            
            # Batch inference
            batch_faces = self.mtcnn(pil_batch)
            
            # mtcnn returns a list where each element is a tensor (multiple faces) or None
            for faces in batch_faces:
                if faces is None:
                    all_faces.append([])
                else:
                    all_faces.append([f.permute(1, 2, 0).cpu().numpy() for f in faces])
                    
        return all_faces

    def save_face(self, face_array: np.ndarray, output_path: Path):
        """Saves a single extracted face to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Rescale if needed (mtcnn post_process=True returns values between -1 and 1)
        if face_array.min() < 0:
            face_array = (face_array + 1) / 2
            face_array = (face_array * 255).astype(np.uint8)
        else:
            face_array = face_array.astype(np.uint8)
            
        img = Image.fromarray(face_array)
        img.save(output_path)
