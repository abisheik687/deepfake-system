"""
Internal trace:
- Wrong before: temporary media handling was duplicated, MIME checks trusted client headers too much, and cleanup paths were easy to miss.
- Fixed now: uploads are persisted once, cleaned up centrally, and invalid media returns explicit AppError codes.
"""

from __future__ import annotations

import base64
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

from config import settings


class AppError(Exception):
    def __init__(self, status_code: int, message: str, code: str):
        self.status_code = status_code
        self.message = message
        self.code = code
        super().__init__(message)


def sniff_file_type(content: bytes) -> str | None:
    header = content[:16]
    if header.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
        return 'audio/wav'
    if header.startswith(b'OggS'):
        return 'audio/ogg'
    if header.startswith(b'ID3') or header[:2] == b'\xff\xfb':
        return 'audio/mpeg'
    if len(header) >= 12 and header[4:8] == b'ftyp':
        brand = header[8:12]
        if brand in {b'isom', b'iso2', b'mp41', b'mp42', b'avc1'}:
            return 'video/mp4'
    if header.startswith(b'\x1aE\xdf\xa3'):
        return 'video/webm'
    return None


async def persist_upload_to_temp(file: UploadFile, validation) -> Path:
    content = await file.read()
    await file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=validation.suffix) as handle:
        handle.write(content)
        return Path(handle.name)


def cleanup_path(path: str | Path) -> None:
    target = Path(path)
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        return
    if target.exists():
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        parent = target.parent
        if parent != settings.temp_dir and parent.exists() and parent.name.startswith('kavach_'):
            shutil.rmtree(parent, ignore_errors=True)


def image_to_base64(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode('.jpg', frame)
    if not ok:
        return ''
    return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def find_ffmpeg_binary() -> str | None:
    if settings.ffmpeg_binary and Path(settings.ffmpeg_binary).exists():
        return settings.ffmpeg_binary
    return shutil.which('ffmpeg')
