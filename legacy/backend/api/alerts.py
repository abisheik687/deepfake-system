"""
KAVACH-AI Alerts API
Manage threat alerts and forensic evidence
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_
from typing import List, Optional
import datetime
import json
from loguru import logger

import httpx
from backend.database import get_db, Alert, EvidenceChain, ScanResult
from backend.schemas import AlertResponse, AlertAcknowledge, EvidenceChainResponse
from backend.config import settings

router = APIRouter()


@router.get("/", response_model=List[AlertResponse])
async def query_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    attack_type: Optional[str] = None,
    verdict: Optional[str] = None,
    start_time: Optional[datetime.datetime] = None,
    end_time: Optional[datetime.datetime] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    Query alert history with filters.
    
    Filters:
    - status: unacknowledged, acknowledged, resolved, false_positive
    - severity: low, medium, high, critical, emergency
    - verdict: FAKE, REAL, SUSPICIOUS (via ScanResult)
    - attack_type: Attack classification
    - start_time / end_time: Time range (UTC)
    - limit / offset: Pagination
    """
    try:
        query = select(Alert)
        
        # Apply filters
        filters = []
        
        if status:
            filters.append(Alert.status == status)
        
        if severity:
            filters.append(Alert.severity == severity)
        
        if attack_type:
            filters.append(Alert.attack_type == attack_type)
            
        if verdict and verdict != "All":
            query = query.join(ScanResult, ScanResult.alert_id == Alert.id)
            filters.append(ScanResult.verdict == verdict)
        
        if start_time:
            filters.append(Alert.created_at >= start_time)
        
        if end_time:
            filters.append(Alert.created_at <= end_time)
        
        if filters:
            query = query.filter(and_(*filters))
        
        # Order and paginate
        result = await db.execute(
            query.order_by(Alert.created_at.desc()).offset(offset).limit(limit)
        )
        
        return result.scalars().all()
    
    except Exception as e:
        logger.error(f"Error querying alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error querying alerts: {str(e)}"
        )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific alert"""
    result = await db.execute(select(Alert).filter(Alert.id == alert_id))
    alert = result.scalars().first()
    
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found"
        )
    
    return alert


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: int,
    ack: AlertAcknowledge,
    db: AsyncSession = Depends(get_db)
):
    """
    Acknowledge an alert.
    
    This marks the alert as acknowledged and records who acknowledged it.
    """
    result = await db.execute(select(Alert).filter(Alert.id == alert_id))
    alert = result.scalars().first()
    
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found"
        )
    
    try:
        alert.status = "acknowledged"
        alert.acknowledged_at = datetime.datetime.utcnow()
        alert.acknowledged_by = ack.acknowledged_by
        
        # Add notes to context if provided
        if ack.notes:
            if not alert.context_json:
                alert.context_json = {}
            alert.context_json["acknowledgment_notes"] = ack.notes
        
        await db.commit()
        await db.refresh(alert)
        
        logger.info(f"Alert {alert_id} acknowledged by {ack.acknowledged_by}")
        
        return alert
    
    except Exception as e:
        await db.rollback()
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error acknowledging alert: {str(e)}"
        )


@router.get("/{alert_id}/evidence", response_model=List[EvidenceChainResponse])
async def get_alert_evidence(
    alert_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get forensic evidence chain for an alert.
    
    Returns the complete Merkle tree evidence chain with cryptographic hashes.
    """
    result = await db.execute(select(Alert).filter(Alert.id == alert_id))
    alert = result.scalars().first()
    
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found"
        )
    
    # Get evidence chain
    chain_result = await db.execute(
        select(EvidenceChain)
        .filter(EvidenceChain.alert_id == alert_id)
        .order_by(EvidenceChain.timestamp)
    )
    return chain_result.scalars().all()


@router.get("/{alert_id}/evidence/export", response_class=Response)
async def export_alert_evidence(
    alert_id: int,
    db: AsyncSession = Depends(get_db)
) -> Response:
    """
    Build and return a downloadable evidence bundle for the given alert.
    The bundle contains: alert metadata, scan result scores per model,
    GradCAM path, and the full evidence chain log.
    """
    result = await db.execute(select(Alert).filter(Alert.id == alert_id))
    alert = result.scalars().first()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "alert_not_found",
                    "message": f"No alert found with id={alert_id}"},
        )

    scan_result = await db.execute(select(ScanResult).filter(ScanResult.alert_id == alert_id))
    scan = scan_result.scalars().first()

    chain_result = await db.execute(
        select(EvidenceChain)
        .filter(EvidenceChain.alert_id == alert_id)
        .order_by(EvidenceChain.timestamp.asc())
    )
    chain_entries = chain_result.scalars().all()

    model_scores: dict = {}
    if scan and getattr(scan, 'model_scores', None):
        try:
            model_scores = json.loads(scan.model_scores)
        except (TypeError, json.JSONDecodeError):
            model_scores = {}

    bundle = {
        "schema_version": "1.0",
        "exported_at": datetime.datetime.utcnow().isoformat() + "Z",
        "alert": {
            "id": alert.id,
            "created_at": alert.created_at.isoformat() if alert.created_at else None,
            "severity": alert.severity,
            "source_type": getattr(alert, 'source_type', None),
            "source_url": getattr(alert, 'source_url', None),
        },
        "verdict": {
            "label": scan.verdict if scan else "unknown",
            "confidence": scan.confidence if scan else None,
            "faces_detected": getattr(scan, 'faces_detected', 0),
            "processing_time_ms": getattr(scan, 'processing_time_ms', None),
        },
        "model_scores": model_scores,
        "explainability": {
            "gradcam_path": getattr(scan, 'gradcam_path', None),
            "gradcam_available": bool(scan and getattr(scan, 'gradcam_path', None)),
        },
        "evidence_chain": [
            {
                "step": i + 1,
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                "stage": getattr(entry, 'stage', str(entry.id)),
                "detail": getattr(entry, 'detail', str(entry.id)),
                "model": getattr(entry, 'model_name', None),
                "score": getattr(entry, 'score', None),
            }
            for i, entry in enumerate(chain_entries)
        ],
    }

    payload = json.dumps(bundle, indent=2, default=str)
    filename = f"kavach_evidence_alert_{alert_id}.json"

    return Response(
        content=payload,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Evidence-Alert-ID": str(alert_id),
            "X-Export-Timestamp": bundle["exported_at"],
        },
    )

@router.post("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert(
    alert_id: int,
    resolution_notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Mark an alert as resolved.
    """
    result = await db.execute(select(Alert).filter(Alert.id == alert_id))
    alert = result.scalars().first()
    
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found"
        )
    
    try:
        alert.status = "resolved"
        alert.end_time = datetime.datetime.utcnow()
        
        if resolution_notes:
            if not alert.context_json:
                alert.context_json = {}
            alert.context_json["resolution_notes"] = resolution_notes
        
        await db.commit()
        await db.refresh(alert)
        
        logger.info(f"Alert {alert_id} resolved")
        
        return alert
    
    except Exception as e:
        await db.rollback()
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resolving alert: {str(e)}"
        )


@router.post("/{alert_id}/false-positive", response_model=AlertResponse)
async def mark_false_positive(
    alert_id: int,
    notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Mark an alert as a false positive.
    
    This is important for model improvement and calibration.
    """
    result = await db.execute(select(Alert).filter(Alert.id == alert_id))
    alert = result.scalars().first()
    
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found"
        )
    
    try:
        alert.status = "false_positive"
        alert.end_time = datetime.datetime.utcnow()
        
        if notes:
            if not alert.context_json:
                alert.context_json = {}
            alert.context_json["false_positive_notes"] = notes
        
        await db.commit()
        await db.refresh(alert)
        
        logger.info(f"Alert {alert_id} marked as false positive")
        
        return alert
    
    except Exception as e:
        await db.rollback()
        logger.error(f"Error marking false positive: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error marking false positive: {str(e)}"
        )

# ─── Webhook Dispatcher ──────────────────────────────────────────────────────

async def dispatch_webhook(alert_id: int, db: AsyncSession):
    """
    KAVACH-AI World-Class Webhook System.
    Dispatches alert data to configured external endpoints for real-time security.
    """
    result = await db.execute(select(Alert).filter(Alert.id == alert_id))
    alert = result.scalars().first()
    
    if not alert or not settings.WEBHOOK_URL:
        return

    payload = {
        "event": "kavach.alert.detected",
        "alert_id": alert.id,
        "severity": alert.severity,
        "attack_type": alert.attack_type,
        "timestamp": alert.created_at.isoformat(),
        "details": alert.context_json
    }

    import hmac, hashlib, json
    
    sig = hmac.new(
        (settings.WEBHOOK_SECRET or "").encode(),
        json.dumps(payload).encode(),
        hashlib.sha256
    ).hexdigest()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                settings.WEBHOOK_URL,
                json=payload,
                headers={"X-KAVACH-Signature": f"sha256={sig}"}
            )
            logger.info(f"[Alerts] Webhook dispatched to {settings.WEBHOOK_URL}: {resp.status_code}")
    except Exception as e:
        logger.error(f"[Alerts] Webhook dispatch failed: {e}")
