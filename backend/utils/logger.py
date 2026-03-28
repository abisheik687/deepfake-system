"""
Internal trace:
- Wrong before: logging depended on partially imported third-party setup and crashed the main app on startup.
- Fixed now: the API uses a safe standard-library logger with one predictable formatter.
"""

from __future__ import annotations

import logging

from config import settings


_configured = False


def _configure_logging() -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_logging()
    return logging.getLogger(name)
