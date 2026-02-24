"""
DeepShield AI — Face Extraction Pipeline

Two-tier face detector:
  1. MTCNN from facenet-pytorch (best accuracy, returns aligned faces)
  2. OpenCV DNN SSD (our existing FaceExtractor — always available)

Auto-fallback: if facenet-pytorch is not installed, uses DNN silently.

Usage:
    pipeline = FacePipeline()
    faces = pipeline.extract_faces(pil_image)
    # → list of PIL.Image crops (aligned, 160×160 for MTCNN)
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from loguru import logger

# Try MTCNN (facenet-pytorch) — optional
try:
    from facenet_pytorch import MTCNN
    import torch
    _HAS_MTCNN = True
    logger.info("[FacePipeline] facenet-pytorch MTCNN available")
except ImportError:
    _HAS_MTCNN = False
    logger.info("[FacePipeline] facenet-pytorch not installed — using DNN fallback")


class FacePipeline:
    """
    Extracts and aligns face crops from images.

    Parameters
    ----------
    min_face_size : int   — ignore detections smaller than this (pixels)
    margin        : float — fractional padding around detected box
    prefer_mtcnn  : bool  — prefer MTCNN over DNN (if available)
    """

    def __init__(
        self,
        min_face_size: int   = 80,
        margin:        float = 0.15,
        prefer_mtcnn:  bool  = True,
    ):
        self.min_face_size = min_face_size
        self.margin        = margin
        self._mtcnn        = None
        self._dnn          = None

        if prefer_mtcnn and _HAS_MTCNN:
            try:
                device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
                self._mtcnn = MTCNN(
                    image_size=224,
                    margin=int(margin * 100),
                    min_face_size=min_face_size,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=False,
                    device=device,
                    keep_all=True,
                )
                logger.success("[FacePipeline] MTCNN ready")
            except Exception as e:
                logger.warning(f"[FacePipeline] MTCNN init failed: {e}")
                self._mtcnn = None

        if self._mtcnn is None:
            self._init_dnn()

    def _init_dnn(self):
        """Load DNN face detector as fallback."""
        try:
            from backend.detection.face_extractor import FaceExtractor
            self._dnn = FaceExtractor(min_confidence=0.5)
            logger.info("[FacePipeline] DNN face extractor initialised as fallback")
        except Exception as e:
            logger.warning(f"[FacePipeline] DNN fallback also failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_faces(self, image: "Image.Image | np.ndarray") -> List[Image.Image]:
        """
        Detect and crop faces from an image.

        Parameters
        ----------
        image : PIL.Image (RGB) or BGR numpy array

        Returns
        -------
        List of PIL.Image face crops (RGB, ready for model preprocessing)
        """
        pil = self._to_pil(image)

        if self._mtcnn is not None:
            return self._extract_mtcnn(pil)
        elif self._dnn is not None:
            return self._extract_dnn(pil)
        else:
            logger.warning("[FacePipeline] No face detector available — returning full image")
            return [pil.resize((224, 224))]

    def extract_boxes(self, pil: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Return face bounding boxes as (x1,y1,x2,y2) tuples."""
        if self._mtcnn is not None:
            import torch
            boxes, _ = self._mtcnn.detect(pil)
            if boxes is None:
                return []
            return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes]
        elif self._dnn is not None:
            bgr = self._to_bgr(pil)
            rects = self._dnn.detect(bgr)
            return [(x, y, x+w, y+h) for x, y, w, h in rects]
        return []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_mtcnn(self, pil: Image.Image) -> List[Image.Image]:
        """MTCNN extracts aligned face tensors — convert back to PIL."""
        try:
            import torch
            boxes, probs = self._mtcnn.detect(pil)
            if boxes is None:
                return []
            w_img, h_img = pil.size
            crops = []
            for box, prob in zip(boxes, probs):
                if prob < 0.70:
                    continue
                x1, y1, x2, y2 = box
                # Add margin
                mx = (x2 - x1) * self.margin
                my = (y2 - y1) * self.margin
                x1 = max(0, int(x1 - mx)); y1 = max(0, int(y1 - my))
                x2 = min(w_img, int(x2 + mx)); y2 = min(h_img, int(y2 + my))
                crop = pil.crop((x1, y1, x2, y2))
                if crop.width < self.min_face_size or crop.height < self.min_face_size:
                    continue
                crops.append(crop.resize((224, 224)))
            return crops
        except Exception as e:
            logger.debug(f"[FacePipeline] MTCNN extract failed: {e}")
            return []

    def _extract_dnn(self, pil: Image.Image) -> List[Image.Image]:
        """DNN face extractor path."""
        import cv2, numpy as np
        bgr    = self._to_bgr(pil)
        rects  = self._dnn.detect(bgr)
        crops  = []
        h_img, w_img = bgr.shape[:2]
        for (x, y, w, h) in rects:
            if w < self.min_face_size or h < self.min_face_size:
                continue
            mx = int(w * self.margin); my = int(h * self.margin)
            x1 = max(0, x-mx); y1 = max(0, y-my)
            x2 = min(w_img, x+w+mx); y2 = min(h_img, y+h+my)
            crop_bgr = bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb).resize((224, 224))
            crops.append(crop_pil)
        return crops

    @staticmethod
    def _to_pil(img) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        import cv2
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _to_bgr(pil: Image.Image) -> np.ndarray:
        import cv2
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# Module-level singleton
_pipeline: Optional[FacePipeline] = None

def get_face_pipeline() -> FacePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FacePipeline()
    return _pipeline
