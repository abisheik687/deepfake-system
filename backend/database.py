"""
KAVACH-AI Database Models and Setup
SQLAlchemy ORM for SQLite/PostgreSQL
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from backend.config import settings

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
    echo=settings.DEBUG
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# ============================================
# MODELS
# ============================================

class Stream(Base):
    """Live stream sources being monitored"""
    __tablename__ = "streams"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    source_type = Column(String, nullable=False)  # youtube_live, rtsp, audio, etc.
    active = Column(Boolean, default=True)
    sampling_interval = Column(Integer, default=2000)  # milliseconds
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    metadata_json = Column(JSON, nullable=True)
    
    # Relationships
    detections = relationship("Detection", back_populates="stream", cascade="all, delete-orphan")


class Detection(Base):
    """Individual detection events"""
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    stream_id = Column(Integer, ForeignKey("streams.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Detection results
    confidence = Column(Float, nullable=False)  # Overall confidence (0-1)
    spatial_confidence = Column(Float, nullable=True)
    temporal_confidence = Column(Float, nullable=True)
    audio_confidence = Column(Float, nullable=True)
    
    # Threat analysis
    severity = Column(String, nullable=True)  # low, medium, high, critical, emergency
    attack_type = Column(String, nullable=True)  # identity_impersonation, vishing, etc.
    
    # Face/tracking info
    track_id = Column(String, nullable=True)
    face_bbox = Column(JSON, nullable=True)  # Bounding box coordinates
    
    # Evidence reference
    frame_hash = Column(String, nullable=True)  # Cryptographic hash of frame
    audio_hash = Column(String, nullable=True)
    
    # Additional data
    features_json = Column(JSON, nullable=True)
    
    # Relationships
    stream = relationship("Stream", back_populates="detections")
    alerts = relationship("Alert", secondary="alert_detections", back_populates="detections")


class Alert(Base):
    """Aggregated threat alerts"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Alert classification
    severity = Column(String, nullable=False)
    attack_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Status
    status = Column(String, default="unacknowledged")  # unacknowledged, acknowledged, resolved, false_positive
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String, nullable=True)
    
    # Time range
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    
    # Evidence
    evidence_hash = Column(String, nullable=True)  # Reference to forensic evidence package
    evidence_path = Column(String, nullable=True)
    
    # Notification tracking
    notifications_sent = Column(JSON, nullable=True)  # List of notification channels/recipients
    
    # Additional context
    context_json = Column(JSON, nullable=True)
    
    # Relationships
    detections = relationship("Detection", secondary="alert_detections", back_populates="alerts")
    evidence_chain = relationship("EvidenceChain", back_populates="alert")


class AlertDetection(Base):
    """Many-to-many relationship between alerts and detections"""
    __tablename__ = "alert_detections"
    
    alert_id = Column(Integer, ForeignKey("alerts.id"), primary_key=True)
    detection_id = Column(Integer, ForeignKey("detections.id"), primary_key=True)


class EvidenceChain(Base):
    """Immutable forensic evidence chain (Merkle tree)"""
    __tablename__ = "evidence_chain"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Chain integrity
    content_hash = Column(String, nullable=False)  # SHA-256 of evidence content
    previous_hash = Column(String, nullable=True)  # Hash of previous block
    block_hash = Column(String, nullable=False)  # Hash of this block (content + previous)
    
    # Evidence data
    evidence_type = Column(String, nullable=False)  # frame, audio, metadata, etc.
    encrypted_path = Column(String, nullable=True)  # Path to encrypted evidence file
    
    # Processing metadata
    processing_metadata = Column(JSON, nullable=True)
    model_versions = Column(JSON, nullable=True)
    
    # Relationships
    alert = relationship("Alert", back_populates="evidence_chain")



class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    role = Column(String, default="user") # user, admin, forensic_analyst
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    scans = relationship("ScanResult", back_populates="owner")
    # logs relationship omitted to avoid conflict with existing AuditLog structure

class ScanResult(Base):
    __tablename__ = "scan_results"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    filename = Column(String)
    file_hash = Column(String, nullable=True) # SHA-256 for evidence chain
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    status = Column(String) # queued, processing, completed, failed
    final_score = Column(Float, nullable=True)
    verdict = Column(String, nullable=True)
    confidence = Column(String, nullable=True)
    
    # Detailed breakdown stored as JSON
    video_score = Column(Float, nullable=True)
    audio_score = Column(Float, nullable=True)
    temporal_score = Column(Float, nullable=True)
    meta_data = Column(JSON, nullable=True) # Technical metadata (resolution, codec, etc.)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    owner = relationship("User", back_populates="scans")

class AuditLog(Base):
    """Immutable activity audit log"""
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Action details
    action = Column(String, nullable=False)  # stream_started, detection_triggered, alert_acknowledged, etc.
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Added FK
    user = Column(String, nullable=True) # Kept for backward compat or denormalization
    
    # Related entities
    stream_id = Column(Integer, nullable=True)
    detection_id = Column(Integer, nullable=True)
    alert_id = Column(Integer, nullable=True)
    
    # Details
    details_json = Column(JSON, nullable=True)
    
    # IP and metadata
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)


# ============================================
# DATABASE INITIALIZATION
# ============================================

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for FastAPI to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
