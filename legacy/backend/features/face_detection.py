"""
KAVACH-AI Face Detection and Tracking
MediaPipe-based face detection and tracking for deepfake analysis
NO API KEYS REQUIRED - All processing is local
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from loguru import logger


@dataclass
class FaceDetection:
    """Face detection result"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    landmarks: np.ndarray  # 468 facial landmarks (x, y, z)
    track_id: Optional[int] = None
    
    # Face region
    face_image: Optional[np.ndarray] = None
    
    # Quality metrics
    blur_score: Optional[float] = None
    brightness_score: Optional[float] = None
    occlusion_score: Optional[float] = None
    
    # Head pose
    pitch: Optional[float] = None  # Up/down rotation
    yaw: Optional[float] = None    # Left/right rotation
    roll: Optional[float] = None   # Tilt


class FaceDetector:
    """
    Face detection and tracking using MediaPipe Face Mesh.
    
    Features:
    - Real-time face detection
    - 468-point facial landmarks
    - Face tracking with IDs
    - Quality assessment
    - Head pose estimation
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 5
    ):
        """
        Initialize face detector.
        
        Args:
            min_detection_confidence: Minimum confidence for initial detection
            min_tracking_confidence: Minimum confidence for tracking
            max_num_faces: Maximum number of faces to detect
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_faces = max_num_faces
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Face tracker
        self.next_track_id = 0
        self.tracked_faces: Dict[int, np.ndarray] = {}  # track_id -> last position
        self.track_id_threshold = 50  # pixels - threshold for track assignment
        
        logger.info(
            f"Face detector initialized "
            f"(detection: {min_detection_confidence}, tracking: {min_tracking_confidence})"
        )
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            List of detected faces with landmarks
        """
        if frame is None or frame.size == 0:
            return []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return []
        
        detections = []
        height, width = frame.shape[:2]
        
        for face_landmarks in results.multi_face_landmarks:
            # Convert landmarks to numpy array
            landmarks = np.array([
                [lm.x * width, lm.y * height, lm.z * width]
                for lm in face_landmarks.landmark
            ])
            
            # Calculate bounding box
            bbox = self._calculate_bbox(landmarks, width, height)
            
            # Extract face region
            x, y, w, h = bbox
            face_image = frame[y:y+h, x:x+w].copy() if self._valid_bbox(bbox, width, height) else None
            
            # Assign track ID
            track_id = self._assign_track_id(landmarks)
            
            # Calculate quality metrics
            blur_score = self._calculate_blur(face_image) if face_image is not None else None
            brightness = self._calculate_brightness(face_image) if face_image is not None else None
            occlusion = self._calculate_occlusion(landmarks)
            
            # Estimate head pose
            pitch, yaw, roll = self._estimate_head_pose(landmarks, width, height)
            
            detection = FaceDetection(
                bbox=bbox,
                confidence=1.0,  # MediaPipe doesn't provide per-face confidence
                landmarks=landmarks,
                track_id=track_id,
                face_image=face_image,
                blur_score=blur_score,
                brightness_score=brightness,
                occlusion_score=occlusion,
                pitch=pitch,
                yaw=yaw,
                roll=roll
            )
            
            detections.append(detection)
        
        return detections
    
    def _calculate_bbox(
        self,
        landmarks: np.ndarray,
        width: int,
        height: int
    ) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box from landmarks.
        
        Returns:
            (x, y, width, height)
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min = int(np.min(x_coords))
        y_min = int(np.min(y_coords))
        x_max = int(np.max(x_coords))
        y_max = int(np.max(y_coords))
        
        # Add padding
        padding = 0.1
        w = x_max - x_min
        h = y_max - y_min
        
        x_min = max(0, int(x_min - w * padding))
        y_min = max(0, int(y_min - h * padding))
        x_max = min(width, int(x_max + w * padding))
        y_max = min(height, int(y_max + h * padding))
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _valid_bbox(self, bbox: Tuple[int, int, int, int], width: int, height: int) -> bool:
        """Check if bounding box is valid"""
        x, y, w, h = bbox
        return (
            x >= 0 and y >= 0 and
            x + w <= width and y + h <= height and
            w > 0 and h > 0
        )
    
    def _assign_track_id(self, landmarks: np.ndarray) -> int:
        """
        Assign track ID to face based on position.
        
        Uses simple distance-based tracking.
        """
        # Calculate face center
        center = np.mean(landmarks[:, :2], axis=0)
        
        # Find closest tracked face
        min_distance = float('inf')
        assigned_id = None
        
        for track_id, last_center in self.tracked_faces.items():
            distance = np.linalg.norm(center - last_center)
            if distance < min_distance and distance < self.track_id_threshold:
                min_distance = distance
                assigned_id = track_id
        
        # Create new track if no match found
        if assigned_id is None:
            assigned_id = self.next_track_id
            self.next_track_id += 1
        
        # Update tracked position
        self.tracked_faces[assigned_id] = center
        
        return assigned_id
    
    def _calculate_blur(self, face_image: np.ndarray) -> float:
        """
        Calculate blur score using Laplacian variance.
        
        Returns:
            Blur score (higher = sharper, lower = blurrier)
        """
        if face_image is None or face_image.size == 0:
            return 0.0
        
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            return float(variance)
        except Exception as e:
            logger.warning(f"Error calculating blur: {e}")
            return 0.0
    
    def _calculate_brightness(self, face_image: np.ndarray) -> float:
        """
        Calculate average brightness.
        
        Returns:
            Brightness score [0, 255]
        """
        if face_image is None or face_image.size == 0:
            return 0.0
        
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray))
        except Exception as e:
            logger.warning(f"Error calculating brightness: {e}")
            return 0.0
    
    def _calculate_occlusion(self, landmarks: np.ndarray) -> float:
        """
        Estimate face occlusion based on landmark visibility.
        
        Returns:
            Occlusion score [0, 1] (0 = no occlusion, 1 = fully occluded)
        """
        # Simple heuristic: check landmark depth variance
        # Higher variance suggests occlusion or profile view
        z_coords = landmarks[:, 2]
        variance = np.var(z_coords)
        
        # Normalize to [0, 1]
        occlusion_score = min(1.0, variance / 100.0)
        
        return float(occlusion_score)
    
    def _estimate_head_pose(
        self,
        landmarks: np.ndarray,
        width: int,
        height: int
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose (pitch, yaw, roll) from landmarks.
        
        Uses simplified PnP-based approach with key facial points.
        
        Returns:
            (pitch, yaw, roll) in degrees
        """
        try:
            # 3D model points (nose tip, chin, left eye, right eye, left mouth, right mouth)
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left Mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])
            
            # 2D image points from MediaPipe landmarks
            # MediaPipe indices: nose tip=1, chin=152, left eye=33, right eye=263, left mouth=61, right mouth=291
            image_points = np.array([
                landmarks[1][:2],     # Nose tip
                landmarks[152][:2],   # Chin
                landmarks[33][:2],    # Left eye
                landmarks[263][:2],   # Right eye
                landmarks[61][:2],    # Left mouth
                landmarks[291][:2]    # Right mouth
            ], dtype=np.float64)
            
            # Camera matrix (simplified)
            focal_length = width
            center = (width / 2, height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Assume no lens distortion
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return (0.0, 0.0, 0.0)
            
            # Convert rotation vector to rotation matrix
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            
            # Calculate Euler angles
            # pitch, yaw, roll
            sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
            
            pitch = np.arctan2(-rotation_mat[2, 0], sy)
            yaw = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
            roll = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            
            # Convert to degrees
            pitch = np.degrees(pitch)
            yaw = np.degrees(yaw)
            roll = np.degrees(roll)
            
            return (float(pitch), float(yaw), float(roll))
        
        except Exception as e:
            logger.warning(f"Error estimating head pose: {e}")
            return (0.0, 0.0, 0.0)
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[FaceDetection],
        draw_landmarks: bool = True,
        draw_bbox: bool = True
    ) -> np.ndarray:
        """
        Draw face detections on frame for visualization.
        
        Args:
            frame: Input frame
            detections: List of face detections
            draw_landmarks: Draw facial landmarks
            draw_bbox: Draw bounding box
        
        Returns:
            Frame with visualizations
        """
        output = frame.copy()
        
        for detection in detections:
            # Draw bounding box
            if draw_bbox:
                x, y, w, h = detection.bbox
                color = (0, 255, 0) if detection.blur_score and detection.blur_score > 100 else (0, 255, 255)
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                
                # Draw track ID
                if detection.track_id is not None:
                    cv2.putText(
                        output,
                        f"ID: {detection.track_id}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
            
            # Draw landmarks
            if draw_landmarks and detection.landmarks is not None:
                for landmark in detection.landmarks:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(output, (x, y), 1, (255, 0, 0), -1)
        
        return output
    
    def blur_faces(self, frame: np.ndarray, detections: List[FaceDetection]) -> np.ndarray:
        """
        Blur detected faces for privacy.
        
        Args:
            frame: Input frame
            detections: List of face detections
        
        Returns:
            Frame with faces blurred
        """
        if frame is None or not detections:
            return frame
            
        output = frame.copy()
        h, w = frame.shape[:2]
        
        for detection in detections:
            x, y, box_w, box_h = detection.bbox
            
            # Create ROI (Region of Interest)
            roi = output[y:y+box_h, x:x+box_w]
            
            if roi.size == 0:
                continue
                
            # Apply Gaussian blur
            # Sigma depends on face size
            sigma = max(15, min(box_w, box_h) // 4)
            blurred_roi = cv2.GaussianBlur(roi, (0, 0), sigma)
            
            # Update output frame
            output[y:y+box_h, x:x+box_w] = blurred_roi
            
        return output

    def cleanup(self):
        """Clean up resources"""
        self.face_mesh.close()
        logger.info("Face detector closed")


# Global face detector instance (lazy initialization)
_face_detector: Optional[FaceDetector] = None


def get_face_detector() -> FaceDetector:
    """Get global face detector instance"""
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceDetector()
    return _face_detector
