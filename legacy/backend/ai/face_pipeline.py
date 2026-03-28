import numpy as np
import cv2
from PIL import Image
from loguru import logger

class FacePipeline:
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            # model_selection=1 is for far faces (full range), 0 is for close faces. 
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.enabled = True
            logger.info("[FacePipeline] mediapipe ready")
        except ImportError:
            logger.warning("mediapipe not installed. Face detection disabled.")
            self.enabled = False

    def extract_faces(self, pil_image: Image.Image) -> list[Image.Image]:
        """Extract all faces from a PIL Image using Mediapipe."""
        if not self.enabled:
            return [pil_image]
            
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        results = self.detector.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        if not results or not getattr(results, 'detections', None):
            return []
            
        faces = []
        h, w, _ = cv_image.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ymin = int(bboxC.ymin * h)
            xmin = int(bboxC.xmin * w)
            height = int(bboxC.height * h)
            width = int(bboxC.width * w)
            
            # Add ~20% margin around face
            margin_y = int(height * 0.2)
            margin_x = int(width * 0.2)
            
            y1 = max(0, ymin - margin_y)
            y2 = min(h, ymin + height + margin_y)
            x1 = max(0, xmin - margin_x)
            x2 = min(w, xmin + width + margin_x)
            
            if y2 > y1 and x2 > x1:
                face_crop = cv_image[y1:y2, x1:x2]
                face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                faces.append(face_pil)
                
        return faces

_pipeline = None

def get_face_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = FacePipeline()
    return _pipeline
