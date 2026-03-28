"""
DeepShield AI — Cache Service

Two-tier caching for inference results:
  Tier 1: In-memory dict (always available, fast)
  Tier 2: Redis (optional, shared across workers)

Cache key = SHA256 hash of image bytes + model selection string.
TTL default: 600s (10 minutes).

Redis connection is attempted on import; falls back to in-memory silently.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Optional, Any

from loguru import logger

# ─── In-memory fallback ────────────────────────────────────────────────────────
_MEMORY_CACHE: dict[str, tuple[Any, float]] = {}  # {key: (value, expires_at)}
DEFAULT_TTL  = 600   # seconds


def _mem_get(key: str) -> Optional[Any]:
    entry = _MEMORY_CACHE.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if time.time() > expires_at:
        del _MEMORY_CACHE[key]
        return None
    return value


def _mem_set(key: str, value: Any, ttl: int = DEFAULT_TTL):
    _MEMORY_CACHE[key] = (value, time.time() + ttl)


# ─── Redis (optional) ──────────────────────────────────────────────────────────
_redis = None

def _init_redis():
    global _redis
    try:
        import redis as redis_lib
        import os
        r = redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
        )
        r.ping()
        _redis = r
        logger.success("[Cache] Redis connected ✓")
    except Exception as e:
        logger.info(f"[Cache] Redis unavailable ({e}) — using in-memory cache")
        _redis = None

_init_redis()


# ─── Public API ────────────────────────────────────────────────────────────────

def make_key(image_bytes: bytes, model_list: list[str], extra: str = "") -> str:
    """Generate a deterministic cache key from image content + model selection."""
    model_str = ",".join(sorted(model_list))
    raw       = image_bytes + model_str.encode() + extra.encode()
    return "ds:" + hashlib.sha256(raw).hexdigest()[:32]


def get(key: str) -> Optional[dict]:
    """Fetch cached result. Returns None on miss or error."""
    # Try Redis first
    if _redis is not None:
        try:
            raw = _redis.get(key)
            if raw is not None:
                logger.debug(f"[Cache] HIT Redis: {key[:16]}…")
                return json.loads(raw)
        except Exception:
            pass   # Redis hiccup → fall through to memory

    # Memory fallback
    val = _mem_get(key)
    if val is not None:
        logger.debug(f"[Cache] HIT memory: {key[:16]}…")
        return val

    logger.debug(f"[Cache] MISS: {key[:16]}…")
    return None


def set(key: str, value: dict, ttl: int = DEFAULT_TTL):
    """Store result in cache (Redis + memory)."""
    _mem_set(key, value, ttl)

    if _redis is not None:
        try:
            _redis.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.debug(f"[Cache] Redis write failed: {e}")


def invalidate(key: str):
    """Remove a single key from both caches."""
    _MEMORY_CACHE.pop(key, None)
    if _redis is not None:
        try:
            _redis.delete(key)
        except Exception:
            pass


def clear_all():
    """Clear in-memory cache (Redis is NOT cleared — admin action needed)."""
    _MEMORY_CACHE.clear()
    logger.info("[Cache] In-memory cache cleared")


def stats() -> dict:
    """Return cache statistics."""
    return {
        "backend":    "redis+memory" if _redis else "memory",
        "redis_ok":   _redis is not None,
        "memory_keys": len(_MEMORY_CACHE),
    }
