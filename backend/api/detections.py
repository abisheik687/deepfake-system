"""
KAVACH-AI Detections API
Query and analyze detection history
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional
from datetime import datetime
from loguru import logger

from backend.database import get_db, Detection
from backend.schemas import DetectionResponse, DetectionStatistics

router = APIRouter()


@router.get("/", response_model=List[DetectionResponse])
async def query_detections(
    stream_id: Optional[int] = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    max_confidence: float = Query(1.0, ge=0.0, le=1.0),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    severity: Optional[str] = None,
    attack_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Query detection history with filters.
    
    Filters:
    - stream_id: Specific stream
    - min_confidence / max_confidence: Confidence range [0, 1]
    - start_time / end_time: Time range (UTC)
    - severity: Severity level (low, medium, high, critical, emergency)
    - attack_type: Attack classification
    - limit / offset: Pagination
    """
    try:
        query = db.query(Detection)
        
        # Apply filters
        filters = []
        
        if stream_id:
            filters.append(Detection.stream_id == stream_id)
        
        filters.append(Detection.confidence >= min_confidence)
        filters.append(Detection.confidence <= max_confidence)
        
        if start_time:
            filters.append(Detection.timestamp >= start_time)
        
        if end_time:
            filters.append(Detection.timestamp <= end_time)
        
        if severity:
            filters.append(Detection.severity == severity)
        
        if attack_type:
            filters.append(Detection.attack_type == attack_type)
        
        if filters:
            query = query.filter(and_(*filters))
        
        # Order and paginate
        detections = query.order_by(
            Detection.timestamp.desc()
        ).offset(offset).limit(limit).all()
        
        return detections
    
    except Exception as e:
        logger.error(f"Error querying detections: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error querying detections: {str(e)}"
        )


@router.get("/stats/summary", response_model=DetectionStatistics)
async def get_detection_statistics(
    stream_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Get aggregated detection statistics.
    
    Returns:
    - Total detections and alerts
    - Distribution by severity and attack type
    - Average confidence
    - Time range
    """
    try:
        query = db.query(Detection)
        
        # Apply filters
        filters = []
        if stream_id:
            filters.append(Detection.stream_id == stream_id)
        if start_time:
            filters.append(Detection.timestamp >= start_time)
        if end_time:
            filters.append(Detection.timestamp <= end_time)
        
        if filters:
            query = query.filter(and_(*filters))
        
        # Get all detections (might be expensive for large datasets)
        detections = query.all()
        
        if not detections:
            return DetectionStatistics(
                total_detections=0,
                total_alerts=0,
                severity_distribution={},
                attack_type_distribution={},
                average_confidence=0.0,
                time_range={"start": None, "end": None}
            )
        
        # Calculate statistics
        total_detections = len(detections)
        
        # Severity distribution
        severity_dist = {}
        for d in detections:
            if d.severity:
                severity_dist[d.severity] = severity_dist.get(d.severity, 0) + 1
        
        # Attack type distribution
        attack_type_dist = {}
        for d in detections:
            if d.attack_type:
                attack_type_dist[d.attack_type] = attack_type_dist.get(d.attack_type, 0) + 1
        
        # Average confidence
        avg_confidence = sum(d.confidence for d in detections) / total_detections
        
        # Time range
        timestamps = [d.timestamp for d in detections]
        time_range = {
            "start": min(timestamps),
            "end": max(timestamps)
        }
        
        # Count alerts (high severity detections)
        total_alerts = sum(
            1 for d in detections 
            if d.severity in ['high', 'critical', 'emergency']
        )
        
        return DetectionStatistics(
            total_detections=total_detections,
            total_alerts=total_alerts,
            severity_distribution=severity_dist,
            attack_type_distribution=attack_type_dist,
            average_confidence=round(avg_confidence, 3),
            time_range=time_range
        )
    
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating statistics: {str(e)}"
        )


@router.get("/{detection_id}", response_model=DetectionResponse)
async def get_detection(
    detection_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific detection"""
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    
    if not detection:
        raise HTTPException(
            status_code=404,
            detail=f"Detection {detection_id} not found"
        )
    
    return detection



