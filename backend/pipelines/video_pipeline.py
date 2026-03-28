"""
Internal trace:
- Wrong before: video analysis returned clip-level verdicts from all sampled frames but exposed per-model scores only from the last processed frame, hardcoded frame stride/max-frame limits instead of using runtime settings, and treated unreadable videos like uncertain analyses.
- Fixed now: frame sampling obeys config, per-model scores are averaged across the whole clip, unreadable/corrupt videos fail with a clear 422 error, and audio extraction remains optional but isolated.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import librosa

from config import settings
from models.ensemble import aggregate_video_scores
from models.loader import ModelRegistry
from pipelines.audio_pipeline import _build_waveform
from schemas.response import AnalysisResult, AudioResult, ModelScore, VideoFramePreview
from utils.file_utils import AppError, cleanup_path, find_ffmpeg_binary, image_to_base64


def _extract_frames(video_path: Path) -> list[tuple[int, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    frames: list[tuple[int, object]] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % settings.video_frame_stride == 0:
            frames.append((index, frame))
        if len(frames) >= settings.max_video_frames:
            break
        index += 1

    capture.release()
    return frames


def _analyse_frame(frame, registry: ModelRegistry) -> tuple[float, list[ModelScore], VideoFramePreview | None]:
    from PIL import Image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    weighted_scores: list[tuple[float, float]] = []
    model_scores: list[ModelScore] = []

    for loaded in registry.image_models:
        fake_probability = float(loaded.infer(pil_image))
        weighted_scores.append((fake_probability, loaded.slot.weight))
        model_scores.append(
            ModelScore(
                model=loaded.slot.label,
                fake_prob=round(fake_probability, 4),
                weight=loaded.slot.weight,
                mode=loaded.mode,
            )
        )

    total_weight = sum(weight for _, weight in weighted_scores)
    frame_probability = sum(prob * weight for prob, weight in weighted_scores) / max(total_weight, 1e-6)
    preview = VideoFramePreview(
        index=0,
        fake_probability=round(frame_probability, 4),
        image_base64=image_to_base64(cv2.resize(frame, (256, 144))),
    )
    return frame_probability, model_scores, preview


def _extract_audio_track(video_path: Path, target_dir: Path) -> Path | None:
    ffmpeg_binary = find_ffmpeg_binary()
    if not ffmpeg_binary:
        return None

    audio_path = target_dir / 'video_audio.wav'
    command = [
        ffmpeg_binary,
        '-y',
        '-i',
        str(video_path),
        '-vn',
        '-acodec',
        'pcm_s16le',
        '-ar',
        '16000',
        '-ac',
        '1',
        str(audio_path),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0 or not audio_path.exists():
        return None
    return audio_path


def _average_model_scores(model_scores_per_frame: list[list[ModelScore]]) -> list[ModelScore]:
    if not model_scores_per_frame:
        return []

    totals: dict[str, dict[str, float | str]] = defaultdict(lambda: {'sum': 0.0, 'weight': 0.0, 'mode': 'fallback'})
    counts: dict[str, int] = defaultdict(int)

    for frame_scores in model_scores_per_frame:
        for score in frame_scores:
            totals[score.model]['sum'] += score.fake_prob
            totals[score.model]['weight'] = score.weight
            totals[score.model]['mode'] = score.mode
            counts[score.model] += 1

    averaged: list[ModelScore] = []
    for score in model_scores_per_frame[0]:
        total = totals[score.model]
        averaged.append(
            ModelScore(
                model=score.model,
                fake_prob=round(float(total['sum']) / max(counts[score.model], 1), 4),
                weight=float(total['weight']),
                mode=str(total['mode']),
            )
        )
    return averaged


async def analyse_video_file(file_path: Path, registry: ModelRegistry, validation, background_tasks) -> AnalysisResult:
    frames = await asyncio.to_thread(_extract_frames, file_path)
    warnings = list(dict.fromkeys(registry.warnings))

    if not frames:
        raise AppError(422, 'Video could not be decoded. Upload a valid MP4 or WEBM file.', 'INVALID_VIDEO_FILE')

    frame_scores: list[float] = []
    previews: list[VideoFramePreview] = []
    model_scores_per_frame: list[list[ModelScore]] = []

    for index, frame in frames:
        frame_probability, model_scores, preview = await asyncio.to_thread(_analyse_frame, frame, registry)
        frame_scores.append(round(frame_probability, 4))
        model_scores_per_frame.append(model_scores)
        if preview:
            preview.index = index
            previews.append(preview)

    fake_probability, verdict, confidence = aggregate_video_scores(frame_scores)
    averaged_model_scores = _average_model_scores(model_scores_per_frame)

    audio_result = None
    temp_audio_dir = Path(tempfile.mkdtemp(prefix='kavach_video_audio_', dir=file_path.parent))
    background_tasks.add_task(cleanup_path, temp_audio_dir)
    audio_path = await asyncio.to_thread(_extract_audio_track, file_path, temp_audio_dir)
    if audio_path:
        audio, sample_rate = await asyncio.to_thread(librosa.load, str(audio_path), sr=16000, mono=True)
        handle = registry.audio_model
        audio_probability = await asyncio.to_thread(handle.infer, audio, sample_rate) if handle else 0.5
        audio_verdict = 'FAKE' if audio_probability > settings.default_image_threshold else 'REAL'
        audio_result = AudioResult(
            verdict=audio_verdict,
            fake_probability=round(audio_probability, 4),
            waveform=_build_waveform(audio),
            mode=handle.mode if handle else 'missing',
        )
    else:
        warnings.append('Video audio track could not be extracted in the current environment')

    return AnalysisResult(
        file_type=validation.file_type,
        verdict=verdict,
        overall_confidence=round(confidence, 4),
        fake_probability=round(fake_probability, 4),
        model_scores=averaged_model_scores,
        video_frame_scores=frame_scores,
        video_frame_previews=previews,
        audio_result=audio_result,
        warnings=list(dict.fromkeys(warnings)),
    )
