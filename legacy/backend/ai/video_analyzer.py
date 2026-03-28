"""
DeepShield AI — Video Analysis Service

Advanced frame-based deepfake detection for video files and webcam streams.

Features:
  • Smart frame sampling (configurable FPS)
  • Keyframe extraction (scene change detection)
  • Per-frame face extraction via FacePipeline
  • Temporal consistency scoring
  • Returns JSON with per-frame results + aggregated verdict
"""

from __future__ import annotations
import asyncio
import base64
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from backend.ai.face_pipeline import get_face_pipeline
from backend.ai.inference_service import analyze_image_async
from backend.ai.confidence_aggregator import aggregate_temporal


async def analyze_video_advanced(
    video_path: str | Path,
    sample_fps:      float = 2.0,    # sample rate
    max_frames:      int   = 30,     # hard cap to prevent OOM
    models:          List[str] = None,
    return_heatmaps: bool  = False,
    detect_faces:    bool  = True,
) -> dict:
    """
    Full deepfake analysis of a video file.

    Parameters
    ----------
    video_path     : path to video file (mp4, avi, mov, etc.)
    sample_fps     : sample N frames per second of video
    max_frames     : maximum frames to analyse (cap for speed/memory)
    models         : HF model keys to use
    return_heatmaps: include Grad-CAM heatmaps (slow for video)
    detect_faces   : crop to faces before inference (usually better accuracy)

    Returns
    -------
    {
        "verdict": "FAKE"|"REAL", "confidence": float,
        "temporal_consistency": float,
        "frames_analysed": int, "fake_frames": int,
        "frame_results": [...],
        "timeline": [...],   ← [{t_secs, confidence, verdict}, ...]
        "latency_ms": float,
    }
    """
    if models is None:
        models = ["vit_deepfake_primary"]

    t0  = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    vid_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step       = max(1, int(vid_fps / sample_fps))
    duration_s = total_fr / vid_fps

    logger.info(f"[VideoAnalyzer] {Path(video_path).name}: {total_fr} frames @ {vid_fps:.0f}fps, "
                f"sampling every {step} frames (~{sample_fps}fps)")

    # ── Collect frame samples ──────────────────────────────────────────────
    frames_to_analyze: List[Tuple[int, float, Image.Image]] = []  # (idx, t_secs, pil)
    frame_idx = 0

    while len(frames_to_analyze) < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb)
            t_s  = round(frame_idx / vid_fps, 2)
            frames_to_analyze.append((frame_idx, t_s, pil))
        frame_idx += 1

    cap.release()

    if not frames_to_analyze:
        return {"verdict": "ERROR", "message": "No frames extracted", "latency_ms": _ms(t0)}

    logger.info(f"[VideoAnalyzer] Analysing {len(frames_to_analyze)} frames …")

    # ── Per-frame inference ────────────────────────────────────────────────
    face_pipe    = get_face_pipeline() if detect_faces else None
    frame_results = []
    timeline      = []

    for (idx, t_s, pil) in frames_to_analyze:
        # Face extraction
        analysis_images = [pil]
        if face_pipe:
            faces = face_pipe.extract_faces(pil)
            if faces:
                analysis_images = faces[:2]   # at most 2 faces per frame

        # Analyse primary image (first face, or full frame)
        result = await analyze_image_async(
            analysis_images[0],
            models=models,
            return_heatmap=return_heatmaps,
        )
        result["frame_idx"] = idx
        result["timestamp"] = t_s
        result["faces_found"] = len(face_pipe.extract_boxes(pil)) if face_pipe else 0

        frame_results.append(result)
        timeline.append({
            "t":          t_s,
            "verdict":    result["verdict"],
            "confidence": result["confidence"],
            "fake_prob":  result.get("fused_scores", {}).get("fake_prob", 0.5),
        })

    # ── Temporal aggregation ───────────────────────────────────────────────
    temporal = aggregate_temporal(frame_results)

    return {
        **temporal,
        "total_frames":    total_fr,
        "frames_analysed": len(frame_results),
        "duration_seconds": round(duration_s, 1),
        "sample_fps":       sample_fps,
        "models":           models,
        "frame_results":    frame_results,
        "timeline":         timeline,
        "latency_ms":       _ms(t0),
        "message": (f"{'FAKE' if temporal['verdict']=='FAKE' else 'REAL'} — "
                    f"{temporal['fake_frames']}/{temporal['frame_count']} frames suspicious, "
                    f"temporal consistency={temporal['temporal_consistency']:.2f}"),
    }


def extract_keyframes(
    video_path: str | Path,
    max_frames: int = 15,
    threshold:  float = 25.0,   # frame difference threshold (SSIM)
) -> List[Tuple[int, float, Image.Image]]:
    """
    Extract keyframes using frame-difference scene detection.
    Good for detecting edits/splices that indicate manipulation.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    prev       = None
    keyframes  = []
    frame_idx  = 0

    while len(keyframes) < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff = cv2.absdiff(prev, gray).mean()
            if diff > threshold:
                pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                keyframes.append((frame_idx, round(frame_idx / fps, 2), pil))
        prev       = gray
        frame_idx += 1

    cap.release()
    return keyframes


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
