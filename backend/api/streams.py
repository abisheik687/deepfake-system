"""
KAVACH-AI Streams API
Manage live stream sources for monitoring
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from loguru import logger

from backend.database import get_db, Stream
from backend.schemas import StreamCreate, StreamResponse
from backend.config import settings

router = APIRouter()


@router.post("/", response_model=StreamResponse, status_code=status.HTTP_201_CREATED)
async def create_stream(
    stream: StreamCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new stream source for monitoring.
    
    Supported source types:
    - youtube_live: YouTube Live streams (via yt-dlp)
    - rtsp: RTSP surveillance feeds
    - rtmp: RTMP streams
    - http_stream: HTTP progressive streams
    - audio: Audio-only streams
    
    **NO API KEYS REQUIRED** - All sources are publicly accessible
    """
    try:
        # Check concurrent stream limit
        active_streams = db.query(Stream).filter(Stream.active == True).count()
        if active_streams >= settings.MAX_CONCURRENT_STREAMS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Maximum concurrent streams ({settings.MAX_CONCURRENT_STREAMS}) reached"
            )
        
        # Create stream record
        db_stream = Stream(
            url=stream.url,
            source_type=stream.source_type.value,
            sampling_interval=stream.sampling_interval,
            metadata_json=stream.metadata,
            active=True
        )
        
        db.add(db_stream)
        db.commit()
        db.refresh(db_stream)
        
        logger.info(f"Created stream {db_stream.id}: {stream.source_type} - {stream.url}")
        
        # Trigger background task to start ingestion
        try:
            from backend.tasks import start_stream_ingestion
            task = start_stream_ingestion.delay(db_stream.id)
            logger.info(f"Started ingestion task {task.id} for stream {db_stream.id}")
        except Exception as task_error:
            logger.warning(f"Could not start ingestion task: {task_error}")
            # Stream still created, can be started manually
        
        return db_stream
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating stream: {str(e)}"
        )


@router.get("/", response_model=List[StreamResponse])
async def list_streams(
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    List all stream sources.
    
    Query parameters:
    - active_only: Filter to only active streams (default: True)
    """
    try:
        query = db.query(Stream)
        
        if active_only:
            query = query.filter(Stream.active == True)
        
        streams = query.order_by(Stream.created_at.desc()).all()
        return streams
    
    except Exception as e:
        logger.error(f"Error listing streams: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing streams: {str(e)}"
        )


@router.get("/{stream_id}", response_model=StreamResponse)
async def get_stream(
    stream_id: int,
    db: Session = Depends(get_db)
):
    """Get details of a specific stream"""
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    
    if not stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream {stream_id} not found"
        )
    
    return stream


@router.delete("/{stream_id}", status_code=status.HTTP_204_NO_CONTENT)
async def stop_stream(
    stream_id: int,
    db: Session = Depends(get_db)
):
    """
    Stop monitoring a stream.
    
    This marks the stream as inactive and stops ingestion.
    Detections and alerts are preserved.
    """
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    
    if not stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream {stream_id} not found"
        )
    
    try:
        stream.active = False
        db.commit()
        
        logger.info(f"Stopped stream {stream_id}")
        
        # Signal background task to stop ingestion
        try:
            from backend.tasks import stop_stream_ingestion
            task = stop_stream_ingestion.delay(stream_id)
            logger.info(f"Started stop task {task.id} for stream {stream_id}")
        except Exception as task_error:
            logger.warning(f"Could not trigger stop task: {task_error}")
        
        return None
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error stopping stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error stopping stream: {str(e)}"
        )


@router.get("/{stream_id}/status")
async def get_stream_status(
    stream_id: int,
    db: Session = Depends(get_db)
):
    """
    Get real-time status of a stream.
    
    Returns:
    - Connection health
    - Processing metrics
    - Recent detection count
    """
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    
    if not stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream {stream_id} not found"
        )
    
    # Get recent detection count
    from backend.database import Detection
    recent_detections = db.query(Detection).filter(
        Detection.stream_id == stream_id
    ).count()
    
    return {
        "stream_id": stream_id,
        "active": stream.active,
        "url": stream.url,
        "source_type": stream.source_type,
        "sampling_interval": stream.sampling_interval,
        "total_detections": recent_detections,
        "status": "active" if stream.active else "stopped",
        # TODO: Add real-time metrics from ingestion engine
        "health": "unknown"
    }
