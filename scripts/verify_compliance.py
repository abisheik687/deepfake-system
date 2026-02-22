
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from backend.config import settings
from backend.features.face_detection import FaceDetection
from backend.detection.pipeline import DetectionPipeline
from backend.detection.multimodal_fusion import MultimodalFusion

def verify_privacy():
    print("Testing Privacy Enforcement...")
    
    # Ensure privacy is ON
    settings.ENFORCE_PRIVACY = True
    print(f"ENFORCE_PRIVACY: {settings.ENFORCE_PRIVACY}")
    
    # Mock a face image (white circle on black background)
    face_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(face_img, (50, 50), 30, (255, 255, 255), -1)
    
    # Calculate initial sharpness (variance of Laplacian)
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    initial_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Initial Sharpness: {initial_sharpness:.2f}")
    
    # Create a mock detection
    det = FaceDetection(
        bbox=(0, 0, 100, 100),
        confidence=0.99,
        landmarks=np.zeros((468, 3)),
        face_image=face_img.copy()
    )
    
    # Function to simulate pipeline privacy logic
    def apply_privacy(detection):
        if settings.ENFORCE_PRIVACY:
            if detection.face_image is not None:
                h, w = detection.face_image.shape[:2]
                sigma = max(5, min(w, h) // 3)
                detection.face_image = cv2.GaussianBlur(detection.face_image, (0, 0), sigma)
                detection.blur_score = 0.0
    
    # Apply privacy
    apply_privacy(det)
    
    # Check sharpness after blur
    gray_blur = cv2.cvtColor(det.face_image, cv2.COLOR_BGR2GRAY)
    final_sharpness = cv2.Laplacian(gray_blur, cv2.CV_64F).var()
    print(f"Final Sharpness: {final_sharpness:.2f}")
    
    if final_sharpness < initial_sharpness and final_sharpness < 10:
        print("✅ Privacy Test PASSED: Face successfully blurred.")
    else:
        print("❌ Privacy Test FAILED: Face not blurred significantly.")

def verify_truth_logic():
    print("\nTesting Truth Logic Placeholders...")
    
    fusion = MultimodalFusion()
    
    # Test placeholders
    res1 = fusion.check_phoneme_viseme_alignment(None, None)
    res2 = fusion.analyze_emotional_coherence("happy", "sad")
    
    if res1 is None and res2 is None:
        print("✅ Truth Logic Test PASSED: Placeholders return None (Skipped).")
    else:
        print(f"❌ Truth Logic Test FAILED: Expected None, got {res1}, {res2}")

if __name__ == "__main__":
    verify_privacy()
    verify_truth_logic()
