"""
Internal trace:
- Wrong before: uploads had no single request schema, weak MIME checking, and conflicting size limits across endpoints.
- Fixed now: validation is centralized here with one Pydantic metadata model and shared file-type rules.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import File, UploadFile
from pydantic import BaseModel

from config import settings
from utils.file_utils import AppError, sniff_file_type


SUPPORTED_TYPES: dict[str, str] = {
    'image/jpeg': 'image',
    'image/png': 'image',
    'image/webp': 'image',
    'video/mp4': 'video',
    'video/webm': 'video',
    'audio/wav': 'audio',
    'audio/x-wav': 'audio',
    'audio/mpeg': 'audio',
    'audio/ogg': 'audio',
}


class UploadValidationInfo(BaseModel):
    file_name: str
    content_type: str
    file_type: str
    suffix: str
    size_bytes: int


async def validate_upload(file: UploadFile = File(...)) -> UploadValidationInfo:
    if file is None or not file.filename:
        raise AppError(422, 'No file uploaded', 'MISSING_FILE')

    content = await file.read()
    await file.seek(0)

    if not content:
        raise AppError(422, 'Uploaded file is empty', 'EMPTY_FILE')

    content_type = (file.content_type or '').lower()
    sniffed_type = sniff_file_type(content)
    effective_type = sniffed_type or content_type

    if effective_type not in SUPPORTED_TYPES:
        raise AppError(
            422,
            'Unsupported file type. Allowed: JPEG, PNG, WEBP, MP4, WEBM, WAV, MP3, OGG.',
            'INVALID_FILE_TYPE',
        )

    file_type = SUPPORTED_TYPES[effective_type]
    max_size = settings.max_video_bytes if file_type == 'video' else settings.max_image_audio_bytes
    if len(content) > max_size:
        raise AppError(
            413,
            f'File exceeds the {max_size // (1024 * 1024)} MB limit for {file_type} uploads.',
            'FILE_TOO_LARGE',
        )

    suffix = (Path(file.filename).suffix or '').lower()
    if not suffix:
        suffix = {'image': '.jpg', 'video': '.mp4', 'audio': '.wav'}[file_type]

    return UploadValidationInfo(
        file_name=file.filename,
        content_type=effective_type,
        file_type=file_type,
        suffix=suffix,
        size_bytes=len(content),
    )
