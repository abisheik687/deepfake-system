"""
KAVACH-AI — Image Analysis Endpoints
  POST /api/analyze-url    — backend downloads image from a URL
  POST /api/analyze-frame  — accepts raw base64 image from the Chrome extension
The extension uses /api/analyze-frame so that the browser's cookie-keyed CDN
URLs (Instagram, Facebook, etc.) are fetched by the extension itself, not the
server, avoiding CDN 403 errors entirely.
"""

import base64
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from loguru import logger
from backend.detection.pipeline import analyze_frame

router = APIRouter()


class ImageURLRequest(BaseModel):
    url: str


class FrameRequest(BaseModel):
    data: str                  # base64 encoded image (data-URI prefix optional)
    source: Optional[str] = "" # e.g. "instagram.com" for logging


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


@router.post("/")
async def analyze_image_url(request: ImageURLRequest):
    """
    Download an image from the given URL and run deepfake analysis on it.
    Returns the same structure as the WebSocket frame_result messages.
    """
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    logger.info(f"Analyzing image URL: {url[:80]}...")

    try:
        async with httpx.AsyncClient(
            timeout=12,
            follow_redirects=True,
            headers=BROWSER_HEADERS,
        ) as client:
            resp = await client.get(url)

        if resp.status_code != 200:
            logger.warning(f"Failed to fetch image: HTTP {resp.status_code}")
            return {
                "verdict": "UNAVAILABLE",
                "confidence": 0.0,
                "faces": [],
                "message": f"Could not download image (HTTP {resp.status_code})",
                "status": "error",
            }

        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            return {
                "verdict": "UNAVAILABLE",
                "confidence": 0.0,
                "faces": [],
                "message": "URL does not point to an image",
                "status": "error",
            }

        img_b64 = base64.b64encode(resp.content).decode("utf-8")
        result = analyze_frame(img_b64)
        result["status"] = "ok"
        result["url"] = url
        logger.info(f"URL analysis result: {result['verdict']} ({result['confidence']:.2%})")
        return result

    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching image: {url[:80]}")
        return {
            "verdict": "UNAVAILABLE",
            "confidence": 0.0,
            "faces": [],
            "message": "Image download timed out",
            "status": "error",
        }
    except Exception as exc:
        logger.error(f"analyze_image_url error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Direct base64 endpoint (used by Chrome extension) ─────────────────────────

@router.post("/frame")
async def analyze_frame_base64(request: FrameRequest):
    """
    Accepts a raw base64-encoded image (sent by the Chrome extension).
    The extension fetches CDN images using the browser's own cookie jar,
    converts them to base64, and POSTs here — no CDN auth issues.
    """
    if not request.data:
        raise HTTPException(status_code=400, detail="Image data required")

    logger.info(f"Extension frame analysis from: {request.source or 'unknown'}")
    try:
        result = analyze_frame(request.data)
        result["status"] = "ok"
        result["source"] = request.source
        logger.info(f"Frame result: {result['verdict']} ({result['confidence']:.2%})")
        return result
    except Exception as exc:
        logger.error(f"analyze_frame_base64 error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

