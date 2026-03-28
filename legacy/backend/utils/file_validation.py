"""
KAVACH-AI File Upload Validation
Validates file type, size, and performs basic security checks
"""

import magic
from pathlib import Path
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException
from loguru import logger


# File size limits (in bytes)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50 MB

# Allowed MIME types
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/quicktime",  # .mov
    "video/x-msvideo",  # .avi
    "video/webm",
}

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",  # .mp3
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/webm",
}

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".webm"}


def validate_file_size(file_size: int, max_size: int, file_type: str) -> None:
    """
    Validate file size against maximum allowed.
    
    Args:
        file_size: Size of file in bytes
        max_size: Maximum allowed size in bytes
        file_type: Type of file (for error message)
        
    Raises:
        HTTPException: If file exceeds size limit
    """
    if file_size > max_size:
        size_mb = file_size / 1024 / 1024
        max_mb = max_size / 1024 / 1024
        raise HTTPException(
            status_code=413,
            detail=f"{file_type} file too large: {size_mb:.1f} MB (max: {max_mb:.1f} MB)"
        )


def validate_file_extension(filename: str, allowed_extensions: set) -> None:
    """
    Validate file extension.
    
    Args:
        filename: Name of the file
        allowed_extensions: Set of allowed extensions
        
    Raises:
        HTTPException: If extension not allowed
    """
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension: {ext}. Allowed: {', '.join(allowed_extensions)}"
        )


def detect_mime_type(file_content: bytes) -> str:
    """
    Detect MIME type from file content using python-magic.
    
    Args:
        file_content: Raw file bytes
        
    Returns:
        MIME type string
    """
    try:
        mime = magic.from_buffer(file_content, mime=True)
        return mime
    except Exception as e:
        logger.warning(f"Failed to detect MIME type: {e}")
        return "application/octet-stream"


def validate_mime_type(mime_type: str, allowed_types: set, file_type: str) -> None:
    """
    Validate MIME type against allowed types.
    
    Args:
        mime_type: Detected MIME type
        allowed_types: Set of allowed MIME types
        file_type: Type of file (for error message)
        
    Raises:
        HTTPException: If MIME type not allowed
    """
    if mime_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {file_type} file type: {mime_type}. Allowed: {', '.join(allowed_types)}"
        )


def virus_scan_stub(file_content: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """
    Stub for virus scanning. In production, integrate with ClamAV or similar.
    
    Args:
        file_content: Raw file bytes
        filename: Name of the file
        
    Returns:
        (is_safe, threat_name) - (True, None) if safe, (False, threat) if infected
    """
    # Stub implementation - always returns safe
    # TODO: Integrate with actual antivirus scanner (ClamAV, VirusTotal API, etc.)
    
    # Basic heuristics for demonstration
    suspicious_patterns = [
        b"<script>",  # JavaScript injection
        b"<?php",     # PHP code
        b"eval(",     # Code execution
    ]
    
    for pattern in suspicious_patterns:
        if pattern in file_content[:1024]:  # Check first 1KB
            logger.warning(f"Suspicious pattern detected in {filename}: {pattern}")
            return False, f"Suspicious pattern: {pattern.decode('utf-8', errors='ignore')}"
    
    return True, None


async def validate_image_upload(file: UploadFile) -> bytes:
    """
    Validate image file upload.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        File content as bytes
        
    Raises:
        HTTPException: If validation fails
    """
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Validate size
    validate_file_size(file_size, MAX_IMAGE_SIZE, "Image")
    
    # Validate extension
    validate_file_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS)
    
    # Detect and validate MIME type
    mime_type = detect_mime_type(content)
    validate_mime_type(mime_type, ALLOWED_IMAGE_TYPES, "image")
    
    # Virus scan
    is_safe, threat = virus_scan_stub(content, file.filename)
    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"File failed security scan: {threat}"
        )
    
    logger.info(f"Image upload validated: {file.filename} ({file_size / 1024:.1f} KB, {mime_type})")
    
    return content


async def validate_video_upload(file: UploadFile) -> bytes:
    """
    Validate video file upload.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        File content as bytes
        
    Raises:
        HTTPException: If validation fails
    """
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Validate size
    validate_file_size(file_size, MAX_VIDEO_SIZE, "Video")
    
    # Validate extension
    validate_file_extension(file.filename, ALLOWED_VIDEO_EXTENSIONS)
    
    # Detect and validate MIME type
    mime_type = detect_mime_type(content)
    validate_mime_type(mime_type, ALLOWED_VIDEO_TYPES, "video")
    
    # Virus scan
    is_safe, threat = virus_scan_stub(content, file.filename)
    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"File failed security scan: {threat}"
        )
    
    logger.info(f"Video upload validated: {file.filename} ({file_size / 1024 / 1024:.1f} MB, {mime_type})")
    
    return content


async def validate_audio_upload(file: UploadFile) -> bytes:
    """
    Validate audio file upload.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        File content as bytes
        
    Raises:
        HTTPException: If validation fails
    """
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Validate size
    validate_file_size(file_size, MAX_AUDIO_SIZE, "Audio")
    
    # Validate extension
    validate_file_extension(file.filename, ALLOWED_AUDIO_EXTENSIONS)
    
    # Detect and validate MIME type
    mime_type = detect_mime_type(content)
    validate_mime_type(mime_type, ALLOWED_AUDIO_TYPES, "audio")
    
    # Virus scan
    is_safe, threat = virus_scan_stub(content, file.filename)
    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"File failed security scan: {threat}"
        )
    
    logger.info(f"Audio upload validated: {file.filename} ({file_size / 1024 / 1024:.1f} MB, {mime_type})")
    
    return content


def validate_url(url: str) -> None:
    """
    Validate social media URL.
    
    Args:
        url: URL string
        
    Raises:
        HTTPException: If URL is invalid
    """
    if not url or not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL: must start with http:// or https://"
        )
    
    # Check for supported platforms
    supported_platforms = [
        "youtube.com",
        "youtu.be",
        "twitter.com",
        "x.com",
        "instagram.com",
        "tiktok.com",
        "facebook.com",
    ]
    
    if not any(platform in url.lower() for platform in supported_platforms):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported platform. Supported: {', '.join(supported_platforms)}"
        )
    
    logger.info(f"URL validated: {url}")


# Example usage
if __name__ == "__main__":
    # Test file size validation
    try:
        validate_file_size(15 * 1024 * 1024, MAX_IMAGE_SIZE, "Image")
    except HTTPException as e:
        print(f"Size validation failed: {e.detail}")
    
    # Test extension validation
    try:
        validate_file_extension("test.exe", ALLOWED_IMAGE_EXTENSIONS)
    except HTTPException as e:
        print(f"Extension validation failed: {e.detail}")
    
    # Test URL validation
    try:
        validate_url("https://www.youtube.com/watch?v=test")
        print("URL validation passed")
    except HTTPException as e:
        print(f"URL validation failed: {e.detail}")

# Made with Bob
