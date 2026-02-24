"""
KAVACH-AI — AI Explanation Endpoint
Calls Gemini or OpenAI with live web search to explain WHY detected content is fake.
The API key is supplied per-request by the Chrome extension (never stored on server).
"""

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
from loguru import logger

router = APIRouter()


# ─── Request / Response schemas ────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    image_url: str
    source_page: str                           # e.g. "instagram.com"
    caption: Optional[str] = ""               # Post caption if available
    verdict: str                               # "FAKE" | "REAL"
    confidence: float
    provider: Literal["gemini", "openai"]     # User's chosen AI provider
    api_key: str


# ─── Gemini Helper ─────────────────────────────────────────────────────────────

async def _ask_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini with Google Search grounding for live web search."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 400,
        },
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, json=payload)
    
    if resp.status_code != 200:
        raise ValueError(f"Gemini API error {resp.status_code}: {resp.text[:200]}")
    
    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip()
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response structure: {e}")


# ─── OpenAI Helper ────────────────────────────────────────────────────────────

async def _ask_openai(prompt: str, api_key: str) -> str:
    """Call OpenAI chat completions (uses model's training knowledge)."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a deepfake and misinformation analyst. "
                    "Your job is to explain concisely why a piece of media may be synthetic or manipulated. "
                    "Be factual, objective, and limit your response to 3 sentences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, json=payload, headers=headers)
    
    if resp.status_code != 200:
        raise ValueError(f"OpenAI API error {resp.status_code}: {resp.text[:200]}")
    
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected OpenAI response structure: {e}")


# ─── Main Endpoint ─────────────────────────────────────────────────────────────

@router.post("/")
async def explain_detection(request: ExplainRequest):
    """
    Use Gemini or OpenAI to explain why a detected image may be fake.
    Gemini uses Google Search grounding for live web results.
    OpenAI uses its training knowledge.
    """
    if not request.api_key or len(request.api_key) < 10:
        raise HTTPException(status_code=400, detail="Valid API key required")

    confidence_pct = int(request.confidence * 100)
    caption_snippet = f' Caption: "{request.caption[:120]}"' if request.caption else ""

    prompt = (
        f"A deepfake detection AI flagged an image from {request.source_page} "
        f"as {request.verdict} with {confidence_pct}% confidence.{caption_snippet} "
        f"Image source URL: {request.image_url[:120]}\n\n"
        "Search the web and explain in 2–3 sentences:\n"
        "1. Does this person/face appear to be real or synthetic?\n"
        "2. Are there any known deepfake incidents or misinformation related to this person or topic?\n"
        "3. What visual or contextual signals suggest this may be manipulated?\n"
        "Be concise and factual."
    )

    logger.info(f"Explain request: provider={request.provider}, verdict={request.verdict}, source={request.source_page}")

    try:
        if request.provider == "gemini":
            explanation = await _ask_gemini(prompt, request.api_key)
        else:
            explanation = await _ask_openai(prompt, request.api_key)

        logger.success(f"Explanation generated ({len(explanation)} chars)")
        return {
            "status": "ok",
            "provider": request.provider,
            "verdict": request.verdict,
            "explanation": explanation,
        }

    except ValueError as ve:
        logger.warning(f"AI API error: {ve}")
        raise HTTPException(status_code=502, detail=str(ve))
    except Exception as exc:
        logger.error(f"Explain endpoint error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
