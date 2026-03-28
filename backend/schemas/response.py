"""
Internal trace:
- Wrong before: response shapes varied across endpoints and mixed generic dicts with partially documented keys.
- Fixed now: the analyse contract, per-model scores, audio payloads, and video frame metadata are explicit Pydantic models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Verdict = Literal['REAL', 'FAKE', 'UNCERTAIN']


class ModelScore(BaseModel):
    model: str
    fake_prob: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    mode: str = 'primary'


class AudioResult(BaseModel):
    verdict: Verdict
    fake_probability: float = Field(ge=0.0, le=1.0)
    waveform: list[float] = Field(default_factory=list)
    mode: str = 'primary'


class VideoFramePreview(BaseModel):
    index: int
    fake_probability: float = Field(ge=0.0, le=1.0)
    image_base64: str


class AnalysisResult(BaseModel):
    file_type: Literal['image', 'video', 'audio']
    verdict: Verdict
    overall_confidence: float = Field(ge=0.0, le=1.0)
    fake_probability: float = Field(ge=0.0, le=1.0)
    model_scores: list[ModelScore] = Field(default_factory=list)
    video_frame_scores: list[float] = Field(default_factory=list)
    video_frame_previews: list[VideoFramePreview] = Field(default_factory=list)
    audio_result: AudioResult | None = None
    processing_time_ms: int = 0
    warnings: list[str] = Field(default_factory=list)
