"""
KAVACH-AI — Detections & Statistics API
Provides scan history and aggregate statistics for the dashboard.
Routes consumed by the frontend Dashboard and Reports pages.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime
from loguru import logger

from backend.database import get_db, ScanResult, Alert

router = APIRouter()


# ─── 1. Aggregate Statistics (Dashboard) ──────────────────────────────────────

@router.get("/stats/summary")
async def get_detection_stats(db: Session = Depends(get_db)):
    """
    Aggregate statistics for the dashboard command center.
    Returns total scans, fakes detected, average confidence, and alert count.
    """
    try:
        total_scans = db.query(func.count(ScanResult.id)).scalar() or 0
        total_fakes = (
            db.query(func.count(ScanResult.id))
            .filter(ScanResult.verdict == "FAKE")
            .scalar() or 0
        )
        avg_confidence = (
            db.query(func.avg(ScanResult.final_score))
            .filter(ScanResult.final_score.isnot(None))
            .scalar()
        )
        total_alerts = db.query(func.count(Alert.id)).scalar() or 0

        return {
            "total_detections": total_scans,
            "total_fakes": total_fakes,
            "total_alerts": total_alerts,
            "average_confidence": round(float(avg_confidence), 4) if avg_confidence else 0.0,
            "severity_distribution": {
                "low":      total_scans - total_fakes,
                "high":     total_fakes,
            },
        }
    except Exception as e:
        logger.error(f"[DetectionsAPI] stats/summary error: {e}")
        return {
            "total_detections": 0,
            "total_fakes": 0,
            "total_alerts": 0,
            "average_confidence": 0.0,
            "severity_distribution": {"low": 0, "high": 0},
        }


# ─── 2. Scan History (Reports Page) ───────────────────────────────────────────

@router.get("/")
async def get_detection_history(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    verdict: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Paginated scan/detection history for the Reports page.
    Optionally filter by verdict: FAKE | REAL | SUSPICIOUS
    """
    try:
        query = db.query(ScanResult)
        if verdict:
            query = query.filter(ScanResult.verdict == verdict.upper())
        results = (
            query.order_by(ScanResult.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "task_id": r.task_id,
                "filename": r.filename or "Unknown",
                "status": r.status,
                "verdict": r.verdict or "PENDING",
                "confidence": r.final_score or 0.0,
                "video_score": r.video_score,
                "audio_score": r.audio_score,
                "temporal_score": r.temporal_score,
                "timestamp": r.created_at.isoformat() if r.created_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"[DetectionsAPI] history error: {e}")
        return []


# ─── 3. Single Detection Detail ────────────────────────────────────────────────

@router.get("/{detection_id}")
async def get_detection(detection_id: int, db: Session = Depends(get_db)):
    """Get full detail for a single scan result by its integer ID."""
    result = db.query(ScanResult).filter(ScanResult.id == detection_id).first()
    if not result:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")
    return {
        "id": result.id,
        "task_id": result.task_id,
        "filename": result.filename or "Unknown",
        "status": result.status,
        "verdict": result.verdict or "PENDING",
        "confidence": result.final_score or 0.0,
        "video_score": result.video_score,
        "audio_score": result.audio_score,
        "temporal_score": result.temporal_score,
        "meta_data": result.meta_data,
        "timestamp": result.created_at.isoformat() if result.created_at else None,
        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
    }
