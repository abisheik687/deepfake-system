"""
KAVACH-AI Pydantic Models for API Request/Response Validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================
# ENUMS
# ============================================

class SourceType(str, Enum):
    """Supported stream source types"""
    YOUTUBE_LIVE = "youtube_live"
    RTSP = "rtsp"
    RTMP = "rtmp"
    HTTP_STREAM = "http_stream"
    AUDIO = "audio"


class SeverityLevel(str, Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AttackType(str, Enum):
    """Attack classification taxonomy"""
    IDENTITY_IMPERSONATION = "identity_impersonation"
    VISHING = "vishing"
    DISINFORMATION = "disinformation"
    EVIDENCE_TAMPERING = "evidence_tampering"
    UNSPECIFIED = "unspecified_synthetic_media"


class AlertStatus(str, Enum):
    """Alert lifecycle status"""
    UNACKNOWLEDGED = "unacknowledged"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


# ============================================
# STREAM MODELS
# ============================================

class StreamCreate(BaseModel):
    """Request model for creating a new stream"""
    url: str = Field(..., description="Stream URL (YouTube Live, RTSP, etc.)")
    source_type: SourceType = Field(..., description="Type of stream source")
    sampling_interval: int = Field(2000, ge=500, le=10000, description="Frame sampling interval in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v or len(v) < 10:
            raise ValueError('Invalid URL')
        return v


class StreamResponse(BaseModel):
    """Response model for stream data"""
    id: int
    url: str
    source_type: str
    active: bool
    sampling_interval: int
    created_at: datetime
    updated_at: datetime
    metadata_json: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


# ============================================
# DETECTION MODELS
# ============================================

class DetectionResponse(BaseModel):
    """Response model for detection data"""
    id: int
    stream_id: int
    timestamp: datetime
    confidence: float
    spatial_confidence: Optional[float] = None
    temporal_confidence: Optional[float] = None
    audio_confidence: Optional[float] = None
    severity: Optional[str] = None
    attack_type: Optional[str] = None
    track_id: Optional[str] = None
    face_bbox: Optional[Dict[str, float]] = None
    
    class Config:
        from_attributes = True


class DetectionQuery(BaseModel):
    """Query parameters for detection history"""
    stream_id: Optional[int] = None
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)
    max_confidence: float = Field(1.0, ge=0.0, le=1.0)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    severity: Optional[List[SeverityLevel]] = None
    attack_type: Optional[List[AttackType]] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


# ============================================
# ALERT MODELS
# ============================================

class AlertResponse(BaseModel):
    """Response model for alert data"""
    id: int
    created_at: datetime
    updated_at: datetime
    severity: str
    attack_type: str
    confidence: float
    status: str
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    evidence_hash: Optional[str] = None
    evidence_path: Optional[str] = None
    notifications_sent: Optional[List[str]] = None
    context_json: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class AlertAcknowledge(BaseModel):
    """Request model for acknowledging an alert"""
    acknowledged_by: str = Field(..., description="User/system acknowledging the alert")
    notes: Optional[str] = Field(None, description="Optional notes")


class AlertQuery(BaseModel):
    """Query parameters for alert history"""
    status: Optional[List[AlertStatus]] = None
    severity: Optional[List[SeverityLevel]] = None
    attack_type: Optional[List[AttackType]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)


# ============================================
# WEBSOCKET MODELS
# ============================================

class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type: detection_update, alert, system_status, etc.")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(..., description="Message payload")


class DetectionUpdateMessage(BaseModel):
    """Real-time detection update via WebSocket"""
    stream_id: int
    detection_id: int
    confidence: float
    severity: Optional[str] = None
    attack_type: Optional[str] = None
    frame_preview_base64: Optional[str] = Field(None, description="Base64 encoded preview frame")


# ============================================
# EVIDENCE MODELS
# ============================================

class EvidenceExportRequest(BaseModel):
    """Request model for evidence export"""
    alert_id: int
    export_format: str = Field("json", pattern="^(json|cef|stix|pdf)$")
    include_media: bool = Field(True, description="Include encrypted media files")


class EvidenceChainResponse(BaseModel):
    """Response model for evidence chain data"""
    id: int
    alert_id: int
    timestamp: datetime
    content_hash: str
    previous_hash: Optional[str] = None
    block_hash: str
    evidence_type: str
    processing_metadata: Optional[Dict[str, Any]] = None
    model_versions: Optional[Dict[str, str]] = None
    
    class Config:
        from_attributes = True


# ============================================
# STATISTICS MODELS
# ============================================

class DetectionStatistics(BaseModel):
    """Aggregated detection statistics"""
    total_detections: int
    total_alerts: int
    severity_distribution: Dict[str, int]
    attack_type_distribution: Dict[str, int]
    average_confidence: float
    time_range: Dict[str, datetime]


class SystemHealth(BaseModel):
    """System health status"""
    status: str = Field(..., description="overall, degraded, critical")
    active_streams: int
    cpu_usage_percent: float
    memory_usage_percent: float
    processing_latency_ms: float
    queue_depth: int
    uptime_seconds: int
