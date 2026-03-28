"""
KAVACH-AI — Detections & Statistics API
Provides scan history and aggregate statistics for the dashboard.
Routes consumed by the frontend Dashboard and Reports pages.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime
from loguru import logger

from backend.database import get_db, ScanResult, Alert
from backend.crud import get_detection_stats as crud_detection_stats

router = APIRouter()


# ─── 1. Aggregate Statistics (Dashboard) ──────────────────────────────────────

@router.get("/stats")
async def get_detection_stats_live(db: AsyncSession = Depends(get_db)):
    """Real-time stats for dashboard: today's scans, fake/real counts, avg risk (from DB)."""
    try:
        return await crud_detection_stats(db)
    except Exception as e:
        logger.error(f"[DetectionsAPI] /stats error: {e}")
        return {
            "total_scans_today": 0,
            "fake_detections_today": 0,
            "real_verdicts_today": 0,
            "avg_risk_score_today": 0.0,
        }


@router.get("/stats/summary")
async def get_detection_stats(db: AsyncSession = Depends(get_db)):
    """
    Aggregate statistics for the dashboard command center.
    Returns total scans, fakes detected, average confidence, and alert count.
    """
    try:
        total_scans_result = await db.execute(select(func.count(ScanResult.id)))
        total_scans = total_scans_result.scalar() or 0
        
        total_fakes_result = await db.execute(select(func.count(ScanResult.id)).filter(ScanResult.verdict == "FAKE"))
        total_fakes = total_fakes_result.scalar() or 0
        
        avg_conf_result = await db.execute(select(func.avg(ScanResult.final_score)).filter(ScanResult.final_score.isnot(None)))
        avg_confidence = avg_conf_result.scalar()
        
        total_alerts_result = await db.execute(select(func.count(Alert.id)))
        total_alerts = total_alerts_result.scalar() or 0

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
    db: AsyncSession = Depends(get_db),
):
    """
    Paginated scan/detection history for the Reports page.
    Optionally filter by verdict: FAKE | REAL | SUSPICIOUS
    """
    try:
        query = select(ScanResult)
        if verdict:
            query = query.filter(ScanResult.verdict == verdict.upper())
            
        result = await db.execute(
            query.order_by(ScanResult.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        results = result.scalars().all()
        
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

@router.get("/by-id/{detection_id}")
async def get_detection_by_int_id(detection_id: int, db: AsyncSession = Depends(get_db)):
    """Get full detail for a single scan result by its integer ID."""
    result = await db.execute(select(ScanResult).filter(ScanResult.id == detection_id))
    db_result = result.scalars().first()
    
    if not db_result:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")
        
    return {
        "id": db_result.id,
        "task_id": db_result.task_id,
        "filename": db_result.filename or "Unknown",
        "status": db_result.status,
        "verdict": db_result.verdict or "PENDING",
        "confidence": db_result.final_score or 0.0,
        "video_score": db_result.video_score,
        "audio_score": db_result.audio_score,
        "temporal_score": db_result.temporal_score,
        "meta_data": db_result.meta_data,
        "timestamp": db_result.created_at.isoformat() if db_result.created_at else None,
        "completed_at": db_result.completed_at.isoformat() if db_result.completed_at else None,
    }


@router.get("/{detection_id}")
async def get_detection(detection_id: str, db: AsyncSession = Depends(get_db)):
    """Get full detail by task_id (UUID string) or integer ID."""
    from backend.crud import get_detection_by_id
    if detection_id.isdigit():
        result = await db.execute(select(ScanResult).filter(ScanResult.id == int(detection_id)))
        db_result = result.scalars().first()
    else:
        db_result = await get_detection_by_id(db, detection_id)
    if not db_result:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")
    meta = db_result.meta_data or {}
    return {
        "id": db_result.id,
        "detection_id": db_result.task_id,
        "task_id": db_result.task_id,
        "filename": db_result.filename or "Unknown",
        "status": db_result.status,
        "verdict": db_result.verdict or "PENDING",
        "confidence": float(db_result.confidence) if db_result.confidence is not None else (db_result.final_score or 0.0),
        "risk_score": meta.get("risk_score", db_result.final_score or 0.0),
        "model_breakdown": meta.get("model_breakdown", {}),
        "heatmap_b64": meta.get("heatmap_b64"),
        "processing_time_ms": meta.get("processing_time_ms"),
        "face_count": meta.get("face_count", 0),
        "video_score": db_result.video_score,
        "audio_score": db_result.audio_score,
        "temporal_score": db_result.temporal_score,
        "meta_data": meta,
        "timestamp": db_result.created_at.isoformat() if db_result.created_at else None,
        "completed_at": db_result.completed_at.isoformat() if db_result.completed_at else None,
    }
