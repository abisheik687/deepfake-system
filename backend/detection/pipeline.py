"""
KAVACH-AI Detection Pipeline
Handles both file-based analysis and real-time frame analysis for live camera scanning.
"""

import os
import cv2
import base64
import numpy as np
from loguru import logger
from backend.features.feature_extraction import FeatureExtractor
from backend.features.audio_extraction import AudioFeatureExtractor
from backend.models.fusion_engine import FusionEngine, ForensicReport


# ─── Haar cascade for face detection ───────────────────────────────────────────
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def _compute_frame_score(gray: np.ndarray, faces) -> float:
    """
    Lightweight heuristics to produce a deepfake confidence score (0–1).

    Signals used (no GPU / no external model required):
    1.  Laplacian sharpness — real camera frames tend to have natural blur/depth;
        synthetically generated faces are often over-sharp or under-sharp.
    2.  Edge density — GAN-generated imagery often has unnatural edge distributions.
    3.  Face-region colour channel variance — cloned faces exhibit reduced local variance.
    4.  Detected face count — multiple overlapping or absent faces are suspicious.

    These are simple heuristics that flag anomalies; the output is a probability
    *estimate* suitable for a demo system without a trained ML model.
    """
    if len(faces) == 0:
        return 0.0  # No face → can't make a determination

    # --- Signal 1: Laplacian sharpness variance ---
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Typical real-cam frames: 50–600; GANs are often very high (>700) or very low (<30)
    if lap_var < 30 or lap_var > 800:
        sharpness_score = 0.75
    else:
        sharpness_score = 0.15

    # --- Signal 2: Edge density anomaly ---
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    # Unusually low (<0.02) or high (>0.25) edge density is suspicious
    if edge_density < 0.02 or edge_density > 0.25:
        edge_score = 0.70
    else:
        edge_score = 0.12

    # --- Signal 3: Colour variance inside the face bounding box ---
    face_scores = []
    h_img, w_img = gray.shape
    for (x, y, w, h) in faces:
        # Clamp ROI to image bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        variance = roi.var()
        # Very low intra-face variance → possible blended/cloned face
        face_scores.append(0.65 if variance < 200 else 0.10)

    colour_score = float(np.mean(face_scores)) if face_scores else 0.15

    # --- Signal 4: Multiple or no faces ---
    face_count_score = 0.60 if len(faces) > 2 else 0.10

    # Weighted blend
    confidence = (
        0.35 * sharpness_score +
        0.25 * edge_score +
        0.25 * colour_score +
        0.15 * face_count_score
    )
    return round(float(np.clip(confidence, 0.0, 1.0)), 3)


# ─── Public API ────────────────────────────────────────────────────────────────

def analyze_frame(base64_image: str) -> dict:
    """
    Analyse a single camera frame sent as a base64-encoded PNG/JPEG string.

    Parameters
    ----------
    base64_image : str
        Raw base64 string (with or without the data-URI prefix).

    Returns
    -------
    dict with keys:
        verdict     – "FAKE" | "REAL" | "NO_FACE"
        confidence  – float 0–1
        faces       – list of {x, y, w, h} dicts (relative to frame dimensions)
        message     – human-readable summary string
    """
    try:
        # Strip data-URI prefix if present (e.g. "data:image/png;base64,...")
        if "," in base64_image:
            base64_image = base64_image.split(",", 1)[1]

        img_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.warning("analyze_frame: failed to decode image bytes")
            return {
                "verdict": "ERROR",
                "confidence": 0.0,
                "faces": [],
                "message": "Could not decode frame"
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Detect faces
        faces_raw = _face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        faces = []
        if len(faces_raw) > 0:
            for (fx, fy, fw, fh) in faces_raw:
                faces.append({
                    "x": int(fx),
                    "y": int(fy),
                    "w": int(fw),
                    "h": int(fh),
                    # Normalised coords (0–1) for easy canvas scaling on the frontend
                    "xn": round(fx / w, 4),
                    "yn": round(fy / h, 4),
                    "wn": round(fw / w, 4),
                    "hn": round(fh / h, 4),
                })

        if not faces:
            return {
                "verdict": "NO_FACE",
                "confidence": 0.0,
                "faces": [],
                "message": "No face detected in frame"
            }

        score = _compute_frame_score(gray, faces_raw)
        FAKE_THRESHOLD = 0.40
        is_fake = score >= FAKE_THRESHOLD

        verdict = "FAKE" if is_fake else "REAL"
        pct = int(score * 100) if is_fake else int((1 - score) * 100)
        message = f"{'⚠️ FAKE DETECTED' if is_fake else '✅ REAL'} ({pct}% confidence)"

        return {
            "verdict": verdict,
            "confidence": score if is_fake else round(1.0 - score, 3),
            "faces": faces,
            "message": message,
        }

    except Exception as exc:
        logger.error(f"analyze_frame error: {exc}")
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "faces": [],
            "message": f"Analysis error: {str(exc)}"
        }


# ─── Legacy file-based pipeline (unchanged) ───────────────────────────────────

class DetectionPipeline:
    """
    KAVACH-AI Day 9: Core Detection Controller
    Orchestrates Day 2 (Video), Day 3 (Audio), and Day 6 (Fusion) logic.
    """
    def __init__(self):
        self.video_extractor = FeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        self.fusion = FusionEngine()
        logger.info("Detection Pipeline Initialized")

    def process_media(self, file_path: str) -> ForensicReport:
        logger.info(f"Starting analysis for: {file_path}")
        video_score = self._analyze_video(file_path)
        audio_score = self._analyze_audio(file_path)
        temporal_score = 0.5
        report = self.fusion.fuse(video_score, audio_score, temporal_score)
        logger.info(f"Analysis Complete. Verdict: {report.verdict}")
        return report

    def _analyze_video(self, path: str) -> float:
        if not os.path.exists(path):
            return 0.0
        return 0.75

    def _analyze_audio(self, path: str) -> float:
        return 0.2


# Global Instance
monitor_pipeline = DetectionPipeline()
