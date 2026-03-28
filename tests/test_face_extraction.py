import pytest
import numpy as np
import cv2
from pathlib import Path
from backend.ml.mtcnn_extractor import FaceExtractor

def test_extractor_initialization():
    extractor = FaceExtractor(image_size=224, device="cpu")
    assert extractor is not None
    assert extractor.image_size == 224

def test_extract_from_blank_image():
    # Create a blank black image
    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    temp_path = Path("test_blank.jpg")
    cv2.imwrite(str(temp_path), blank_img)
    
    extractor = FaceExtractor(device="cpu")
    faces = extractor.extract_faces_from_image(temp_path)
    
    # Clean up
    if temp_path.exists():
        temp_path.unlink()
        
    # Should detect 0 faces in a black image
    assert len(faces) == 0

def test_batch_processing_empty():
    extractor = FaceExtractor(device="cpu")
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    results = extractor.process_video_frames(frames, batch_size=2)
    
    assert len(results) == 5
    for r in results:
        assert isinstance(r, list)
        assert len(r) == 0
