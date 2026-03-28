"""
KAVACH-AI Detection Pipeline
Handles both file-based analysis and real-time frame analysis for live camera scanning.
Uses real model inference (inference_service) when available; falls back to heuristics otherwise.
"""

import os
import cv2
import base64
import numpy as np
from typing import Optional
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

def analyze_frame(
    base64_image: str,
    include_heatmap: bool = False,
    use_ml: bool = True,
) -> dict:
    """
    Analyse a single frame (base64). Uses real model inference when use_ml=True.

    Steps: decode image → face detection (MTCNN/face_pipeline or Haar) →
    if no face return NO_FACE; else run inference_service for each registered model,
    optionally GradCAM heatmap. Returns model_scores, face_detected, heatmap_b64.
    """
    try:
        if "," in base64_image:
            base64_image = base64_image.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {
                "verdict": "ERROR",
                "confidence": 0.0,
                "model_scores": {},
                "face_detected": False,
                "heatmap_b64": None,
                "faces": [],
                "message": "Could not decode frame",
            }

        import asyncio
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Face detection: try MTCNN/face_pipeline first, else Haar
        face_detected = False
        analysis_image = pil_image
        faces = []
        if use_ml:
            try:
                from backend.ai.face_pipeline import get_face_pipeline
                pipe = get_face_pipeline()
                crops = pipe.extract_faces(pil_image)
                if crops:
                    face_detected = True
                    analysis_image = crops[0]
            except Exception as e:
                logger.debug(f"[pipeline] Face pipeline failed: {e}")
        if not face_detected:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            faces_raw = _face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces_raw) == 0:
                return {
                    "verdict": "NO_FACE",
                    "confidence": 0.0,
                    "model_scores": {},
                    "face_detected": False,
                    "heatmap_b64": None,
                    "faces": [],
                    "message": "No face detected in frame",
                }
            face_detected = True
            for (fx, fy, fw, fh) in faces_raw:
                faces.append({"x": int(fx), "y": int(fy), "w": int(fw), "h": int(fh)})

        # Real model inference (run in fresh loop when called from sync context)
        if use_ml:
            try:
                from backend.ai.inference_service import analyze_image_async
                from backend.orchestrator.model_registry import get_models_by_tier
                models = get_models_by_tier("balanced")
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(
                            asyncio.run,
                            analyze_image_async(analysis_image, models=models, return_heatmap=include_heatmap),
                        )
                        result = future.result(timeout=120)
                else:
                    result = asyncio.run(
                        analyze_image_async(analysis_image, models=models, return_heatmap=include_heatmap)
                    )
                model_scores = {}
                for r in result.get("model_results", []):
                    if r.get("failed"):
                        continue
                    name = r.get("model", "")
                    ver = r.get("verdict", "REAL")
                    conf = float(r.get("confidence", 0.5))
                    model_scores[name] = conf if ver == "FAKE" else (1.0 - conf)
                heatmap_b64 = result.get("heatmap_b64") or None
                if heatmap_b64 and heatmap_b64.startswith("data:"):
                    heatmap_b64 = heatmap_b64.split(",", 1)[-1] if "," in heatmap_b64 else heatmap_b64
                return {
                    "verdict": result.get("verdict", "REAL"),
                    "confidence": float(result.get("confidence", 0.0)),
                    "model_scores": model_scores,
                    "face_detected": True,
                    "heatmap_b64": heatmap_b64,
                    "faces": faces,
                    "message": result.get("message", ""),
                }
            except Exception as e:
                logger.warning(f"[pipeline] ML inference failed, falling back to heuristics: {e}")

        # Fallback: heuristic score
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_raw = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces_raw) == 0:
            return {
                "verdict": "NO_FACE",
                "confidence": 0.0,
                "model_scores": {},
                "face_detected": False,
                "heatmap_b64": None,
                "faces": [],
                "message": "No face detected in frame",
            }
        score = _compute_frame_score(gray, faces_raw)
        is_fake = score >= 0.40
        verdict = "FAKE" if is_fake else "REAL"
        confidence = score if is_fake else round(1.0 - score, 3)
        faces = [{"x": int(fx), "y": int(fy), "w": int(fw), "h": int(fh)} for (fx, fy, fw, fh) in faces_raw]
        return {
            "verdict": verdict,
            "confidence": confidence,
            "model_scores": {},
            "face_detected": True,
            "heatmap_b64": None,
            "faces": faces,
            "message": f"{verdict} ({int(confidence*100)}% confidence)",
        }

    except Exception as exc:
        logger.error(f"analyze_frame error: {exc}")
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "model_scores": {},
            "face_detected": False,
            "heatmap_b64": None,
            "faces": [],
            "message": str(exc),
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

    async def process_media(self, file_path: str) -> ForensicReport:
        logger.info(f"Starting analysis for: {file_path}")
        # Run video and audio analysis
        frame_probs = await self._analyze_video(file_path)
        audio_score = await self._analyze_audio(file_path)
        
        video_score = float(np.mean(frame_probs)) if frame_probs else 0.0
        
        # Temporal score: variance of fake_probs across frames
        import statistics
        temporal_score = 0.5
        if len(frame_probs) > 1:
            stdev = statistics.stdev(frame_probs)
            # High variance -> high temporal anomaly -> suspicious
            temporal_score = min(1.0, stdev * 2.0 + 0.2)
            
        report = self.fusion.fuse(video_score, audio_score, temporal_score)
        logger.info(f"Analysis Complete. Verdict: {report.verdict}")
        return report

    async def _analyze_video(self, path: str) -> list[float]:
        if not os.path.exists(path):
            return []
            
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = int(round(fps)) # 1 fps
        
        frames_to_process = []
        count = 0
        while cap.isOpened() and len(frames_to_process) < 30:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                from PIL import Image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames_to_process.append(pil_image)
            count += 1
        cap.release()
        
        from backend.orchestrator.orchestrator import analyze
        fake_probs = []
        for img in frames_to_process:
            try:
                res = await analyze(img, tier="fast", use_cache=False, return_heatmap=False)
                fake_probs.append(res.get("fake_prob", 0.0))
            except Exception as e:
                logger.error(f"Error analyzing frame: {e}")
                
        return fake_probs

    async def _analyze_audio(self, path: str) -> float:
        if not os.path.exists(path):
            return 0.0
        try:
            import ffmpeg
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav = tmp.name
            
            try:
                (
                    ffmpeg.input(path)
                    .output(temp_wav, acodec='pcm_s16le', ac=1, ar='16k')
                    .overwrite_output()
                    .run(quiet=True)
                )
                # If generated successfully, we'd run AudioFeatureExtractor here
                # For now returning a placeholder as model may not be loaded
                score = 0.2
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            return score
        except Exception as e:
            logger.warning(f"Audio extraction skipped/failed: {e}")
            return 0.0


# Global Instance
monitor_pipeline = DetectionPipeline()
