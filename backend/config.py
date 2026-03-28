"""
Internal trace:
- Wrong before: duplicated CORS settings, unrelated legacy infrastructure knobs, and environment variables that did not reflect the actual runtime.
- Fixed now: minimal pydantic-settings config for the upload pipeline, documented model slots, file limits, and predictable local defaults.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = 'KAVACH-AI'
    app_version: str = '2.0.0'
    environment: str = 'development'
    host: str = '0.0.0.0'
    port: int = 8000
    log_level: str = 'INFO'

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            'http://localhost:5173',
            'http://127.0.0.1:5173',
            'http://localhost:4173',
            'http://127.0.0.1:4173',
        ]
    )

    model_cache_dir: Path = Path(tempfile.gettempdir()) / 'kavach-ai' / 'models'
    temp_dir: Path = Path(tempfile.gettempdir()) / 'kavach-ai' / 'uploads'
    max_image_audio_bytes: int = 20 * 1024 * 1024
    max_video_bytes: int = 100 * 1024 * 1024
    max_video_frames: int = 30
    video_frame_stride: int = 10
    analysis_timeout_seconds: int = 120
    allow_fallback_models: bool = True
    enable_remote_model_downloads: bool = False
    default_image_threshold: float = 0.52
    disagreement_threshold: float = 0.4

    model_vit_repo: str = 'prithivMLmods/Deep-Fake-Detector-Model'
    model_efficientnet_repo: str = 'Wvolf/EfficientNet_Deepfake'
    model_xception_repo: str = 'not-lain/xception-deepfake'
    model_convnext_repo: str = 'facebook/convnext-base-224'
    model_audio_repo: str = 'mo-thecreator/deepfake-audio-detector'

    ffmpeg_binary: str | None = None

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')


settings = Settings()
settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
settings.temp_dir.mkdir(parents=True, exist_ok=True)
