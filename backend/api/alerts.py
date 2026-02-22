"""
KAVACH-AI Alerts API
Manage threat alerts and forensic evidence
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Optional
from datetime import datetime
from loguru import logger

from backend.database import get_db, Alert, EvidenceChain
from backend.schemas import AlertResponse, AlertAcknowledge, EvidenceChainResponse
from backend.config import settings

router = APIRouter()


@router.get("/", response_model=List[AlertResponse])
async def query_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    attack_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Query alert history with filters.
    
    Filters:
    - status: unacknowledged, acknowledged, resolved, false_positive
    - severity: low, medium, high, critical, emergency
    - attack_type: Attack classification
    - start_time / end_time: Time range (UTC)
    - limit / offset: Pagination
    """
    try:
        query = db.query(Alert)
        
        # Apply filters
        filters = []
        
        if status:
            filters.append(Alert.status == status)
        
        if severity:
            filters.append(Alert.severity == severity)
        
        if attack_type:
            filters.append(Alert.attack_type == attack_type)
        
        if start_time:
            filters.append(Alert.created_at >= start_time)
        
        if end_time:
            filters.append(Alert.created_at <= end_time)
        
        if filters:
            query = query.filter(and_(*filters))
        
        # Order and paginate
        alerts = query.order_by(
            Alert.created_at.desc()
        ).offset(offset).limit(limit).all()
        
        return alerts
    
    except Exception as e:
        logger.error(f"Error querying alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error querying alerts: {str(e)}"
        )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific alert"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
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
    db: Session = Depends(get_db)
):
    """
    Acknowledge an alert.
    
    This marks the alert as acknowledged and records who acknowledged it.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found"
        )
    
    try:
        alert.status = "acknowledged"
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = ack.acknowledged_by
        
        # Add notes to context if provided
        if ack.notes:
            if not alert.context_json:
                alert.context_json = {}
            alert.context_json["acknowledgment_notes"] = ack.notes
        
        db.commit()
        db.refresh(alert)
        
        logger.info(f"Alert {alert_id} acknowledged by {ack.acknowledged_by}")
        
        return alert
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error acknowledging alert: {str(e)}"
        )


@router.get("/{alert_id}/evidence", response_model=List[EvidenceChainResponse])
async def get_alert_evidence(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """
    Get forensic evidence chain for an alert.
    
    Returns the complete Merkle tree evidence chain with cryptographic hashes.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found"
        )
    
    # Get evidence chain
    evidence_chain = db.query(EvidenceChain).filter(
        EvidenceChain.alert_id == alert_id
    ).order_by(EvidenceChain.timestamp).all()
    
    return evidence_chain


@router.get("/{alert_id}/evidence/export")
async def export_alert_evidence(
    alert_id: int,
    export_format: str = Query("json", regex="^(json|cef|stix|pdf)$"),
    db: Session = Depends(get_db)
):
    """
    Export forensic evidence package for an alert.
    
    Supported formats:
    - json: Structured JSON export
    - cef: Common Event Format (SIEM integration)
    - stix: Structured Threat Information eXpression
    - pdf: Human-readable PDF report (future implementation)
    
    Returns a downloadable evidence package.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found"
        )
    
    # TODO: Implement evidence export logic
    # This would package all evidence, generate the appropriate format,
    # and return as a downloadable file
    
    logger.warning(f"Evidence export not yet implemented for alert {alert_id}")
    
    return {
        "alert_id": alert_id,
        "format": export_format,
        "status": "not_implemented",
        "message": "Evidence export will be implemented in Phase 9"
    }


@router.post("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert(
    alert_id: int,
    resolution_notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Mark an alert as resolved.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found"
        )
    
    try:
        alert.status = "resolved"
        alert.end_time = datetime.utcnow()
        
        if resolution_notes:
            if not alert.context_json:
                alert.context_json = {}
            alert.context_json["resolution_notes"] = resolution_notes
        
        db.commit()
        db.refresh(alert)
        
        logger.info(f"Alert {alert_id} resolved")
        
        return alert
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resolving alert: {str(e)}"
        )


@router.post("/{alert_id}/false-positive", response_model=AlertResponse)
async def mark_false_positive(
    alert_id: int,
    notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Mark an alert as a false positive.
    
    This is important for model improvement and calibration.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found"
        )
    
    try:
        alert.status = "false_positive"
        alert.end_time = datetime.utcnow()
        
        if notes:
            if not alert.context_json:
                alert.context_json = {}
            alert.context_json["false_positive_notes"] = notes
        
        db.commit()
        db.refresh(alert)
        
        logger.info(f"Alert {alert_id} marked as false positive")
        
        # TODO: Use this feedback for model calibration
        
        return alert
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error marking false positive: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error marking false positive: {str(e)}"
        )
