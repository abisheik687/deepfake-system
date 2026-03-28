from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_
from datetime import datetime, date, timedelta
from typing import Optional
from backend.database import User, ScanResult, AuditLog, Alert, Detection
from backend.api.auth import get_password_hash

# KAVACH-AI: CRUD for scans, detections, alerts

async def create_detection(db: AsyncSession, result_dict: dict, file_hash: Optional[str] = None, filename: Optional[str] = None, owner_id: Optional[int] = None) -> ScanResult:
    """Save orchestration result to scan_results table. Returns the created ORM object."""
    import uuid
    task_id = str(uuid.uuid4())
    verdict = result_dict.get("verdict", "REAL")
    confidence = result_dict.get("confidence", 0.0)
    risk_score = result_dict.get("risk_score", 0.0)
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.0
    meta = {
        "model_breakdown": result_dict.get("model_breakdown") or result_dict.get("per_model") or {},
        "heatmap_b64": result_dict.get("heatmap_b64"),
        "processing_time_ms": result_dict.get("processing_time_ms") or result_dict.get("latency_ms"),
        "face_count": result_dict.get("face_count") or result_dict.get("faces_found", 0),
        "risk_score": risk_score,
    }
    scan = ScanResult(
        task_id=task_id,
        filename=filename or "",
        file_hash=file_hash,
        owner_id=owner_id,
        status="completed",
        final_score=float(risk_score) if isinstance(risk_score, (int, float)) else 0.0,
        verdict=verdict,
        confidence=float(confidence),
        meta_data=meta,
        completed_at=datetime.utcnow(),
    )
    db.add(scan)
    await db.commit()
    await db.refresh(scan)
    return scan


async def get_detections(db: AsyncSession, skip: int = 0, limit: int = 50) -> list:
    """Paginated list of scan results (ScanResult), most recent first."""
    result = await db.execute(
        select(ScanResult)
        .order_by(ScanResult.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_detection_stats(db: AsyncSession) -> dict:
    """Aggregate counts and avg risk for today (scan_results)."""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    count_result = await db.execute(
        select(func.count(ScanResult.id)).where(
            and_(ScanResult.created_at >= today_start, ScanResult.status == "completed")
        )
    )
    total_today = count_result.scalar() or 0
    fake_result = await db.execute(
        select(func.count(ScanResult.id)).where(
            and_(ScanResult.created_at >= today_start, ScanResult.verdict == "FAKE", ScanResult.status == "completed")
        )
    )
    real_result = await db.execute(
        select(func.count(ScanResult.id)).where(
            and_(ScanResult.created_at >= today_start, ScanResult.verdict == "REAL", ScanResult.status == "completed")
        )
    )
    avg_result = await db.execute(
        select(func.avg(ScanResult.final_score)).where(
            and_(ScanResult.created_at >= today_start, ScanResult.status == "completed")
        )
    )
    return {
        "total_scans_today": total_today,
        "fake_detections_today": fake_result.scalar() or 0,
        "real_verdicts_today": real_result.scalar() or 0,
        "avg_risk_score_today": round(float(avg_result.scalar() or 0), 4),
    }


async def get_detection_by_id(db: AsyncSession, detection_id: str) -> Optional[ScanResult]:
    """Get a single scan result by task_id (detection_id)."""
    result = await db.execute(select(ScanResult).where(ScanResult.task_id == detection_id))
    return result.scalars().first()


async def create_alert(db: AsyncSession, detection_id: str, severity: str = "medium") -> Optional[Alert]:
    """Create an alert linked to a scan result. Requires Detection record; we use ScanResult.task_id as reference."""
    scan = await get_detection_by_id(db, detection_id)
    if not scan:
        return None
    alert = Alert(
        severity=severity,
        attack_type="unspecified_synthetic_media",
        confidence=float(scan.confidence or 0),
        status="unacknowledged",
        start_time=scan.created_at or datetime.utcnow(),
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return alert


async def get_alerts(db: AsyncSession, skip: int = 0, limit: int = 50, reviewed: Optional[bool] = None) -> list:
    """Filtered list of alerts. reviewed=True -> acknowledged/resolved/false_positive."""
    q = select(Alert).order_by(Alert.created_at.desc()).offset(skip).limit(limit)
    if reviewed is not None:
        if reviewed:
            q = q.where(Alert.status.in_(["acknowledged", "resolved", "false_positive"]))
        else:
            q = q.where(Alert.status == "unacknowledged")
    result = await db.execute(q)
    return list(result.scalars().all())


async def create_scan_result(db: AsyncSession, task_id: str, filename: str, owner_id: Optional[int] = None):
    db_scan = ScanResult(
        task_id=task_id,
        filename=filename,
        owner_id=owner_id,
        status="queued",
        created_at=datetime.utcnow(),
    )
    db.add(db_scan)
    await db.commit()
    await db.refresh(db_scan)
    return db_scan

async def update_scan_result(db: AsyncSession, task_id: str, report: dict, status="completed"):
    result = await db.execute(select(ScanResult).filter(ScanResult.task_id == task_id))
    db_scan = result.scalars().first()
    if db_scan:
        db_scan.status = status
        db_scan.final_score = report.get("final_score")
        db_scan.verdict = report.get("verdict")
        db_scan.confidence = report.get("confidence")
        
        # Parse breakdown if available
        breakdown = report.get("breakdown", {})
        db_scan.video_score = breakdown.get("video_spatial")
        db_scan.audio_score = breakdown.get("audio_spectral")
        db_scan.temporal_score = breakdown.get("temporal_lstm")
        
        db_scan.completed_at = datetime.utcnow()
        await db.commit()
        await db.refresh(db_scan)
    return db_scan

async def log_action(db: AsyncSession, user_id: int, action: str, details: str):
    log = AuditLog(
        user_id=user_id,
        action=action,
        user=str(user_id) # For backward compatibility with 'user' column
    )
    log.details_json = {"message": details} 
    
    db.add(log)
    await db.commit()
    return log

async def get_scan_result(db: AsyncSession, task_id: str):
    result = await db.execute(select(ScanResult).filter(ScanResult.task_id == task_id))
    return result.scalars().first()

async def get_user_scans(db: AsyncSession, user_id: int, skip: int = 0, limit: int = 100):
    result = await db.execute(
        select(ScanResult)
        .filter(ScanResult.owner_id == user_id)
        .order_by(ScanResult.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()
