"""
DeepShield AI — Face Extractor

Detects and crops faces from frames using a two-tier strategy:
  1. OpenCV DNN (ResNet-10 SSD) — fast, accurate, no extra install
  2. Haar Cascade fallback        — if DNN model files not downloaded yet

The DNN model is ~5MB and downloaded automatically on first use.
Model: opencv_face_detector_uint8.pb + opencv_face_detector.pbtxt
Source: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
"""

from __future__ import annotations
import os
import cv2
import urllib.request
import numpy as np
from pathlib import Path
from loguru import logger


_MODELS_DIR   = Path("models")
_DNN_PROTOTXT = _MODELS_DIR / "opencv_face_detector.pbtxt"
_DNN_MODEL    = _MODELS_DIR / "opencv_face_detector_uint8.pb"

_DNN_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/opencv_face_detector.pbtxt"
)
_DNN_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_uint8/"
    "opencv_face_detector_uint8.pb"
)


class FaceExtractor:
    """
    Multi-strategy face detector returning bounding boxes as (x, y, w, h).

    Parameters
    ----------
    min_confidence : float — DNN confidence threshold (default 0.5)
    use_dnn        : bool  — prefer DNN detector (auto-downloaded if needed)
    """

    def __init__(self, min_confidence: float = 0.50, use_dnn: bool = True):
        self.min_confidence = min_confidence
        self._net           = None
        self._cascade       = None

        if use_dnn:
            self._net = self._load_dnn()

        if self._net is None:
            logger.warning("[FaceExtractor] DNN unavailable, using Haar cascade")
            self._cascade = self._load_cascade()

    # ── Private loaders ─────────────────────────────────────────────────────

    def _load_dnn(self) -> cv2.dnn_Net | None:
        """Download and load OpenCV's ResNet-10 SSD face detector."""
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Download if not cached
        for path, url in [(_DNN_PROTOTXT, _DNN_PROTOTXT_URL), (_DNN_MODEL, _DNN_MODEL_URL)]:
            if not path.exists():
                try:
                    logger.info(f"[FaceExtractor] Downloading {path.name} …")
                    urllib.request.urlretrieve(url, str(path))
                    logger.success(f"[FaceExtractor] Saved {path}")
                except Exception as e:
                    logger.warning(f"[FaceExtractor] Could not download {path.name}: {e}")
                    return None

        try:
            net = cv2.dnn.readNetFromTensorflow(str(_DNN_MODEL), str(_DNN_PROTOTXT))
            logger.success("[FaceExtractor] DNN face detector loaded")
            return net
        except Exception as e:
            logger.warning(f"[FaceExtractor] DNN load failed: {e}")
            return None

    def _load_cascade(self) -> cv2.CascadeClassifier:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return cv2.CascadeClassifier(path)

    # ── Public API ───────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect faces in a BGR frame.

        Returns
        -------
        List of (x, y, w, h) integers.  May be empty if no faces found.
        """
        if self._net is not None:
            return self._detect_dnn(frame)
        return self._detect_cascade(frame)

    def _detect_dnn(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        faces = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.min_confidence:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            fw, fh = x2 - x1, y2 - y1
            if fw > 0 and fh > 0:
                faces.append((x1, y1, fw, fh))
        return faces

    def _detect_cascade(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(detected) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in detected]

    def crop_faces(
        self,
        frame: np.ndarray,
        padding: float = 0.10,
    ) -> list[np.ndarray]:
        """Return cropped BGR face images with optional padding."""
        h_f, w_f = frame.shape[:2]
        crops = []
        for (x, y, w, h) in self.detect(frame):
            px = int(w * padding)
            py = int(h * padding)
            x1 = max(0, x - px);   y1 = max(0, y - py)
            x2 = min(w_f, x+w+px); y2 = min(h_f, y+h+py)
            crops.append(frame[y1:y2, x1:x2])
        return crops
