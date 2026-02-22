
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database.core import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    role = Column(String, default="user") # user, admin, forensic_analyst
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    scans = relationship("ScanResult", back_populates="owner")
    logs = relationship("AuditLog", back_populates="user")

class ScanResult(Base):
    __tablename__ = "scan_results"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    filename = Column(String)
    file_hash = Column(String) # SHA-256 for evidence chain
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    status = Column(String) # queued, processing, completed, failed
    final_score = Column(Float)
    verdict = Column(String)
    confidence = Column(String)
    
    # Detailed breakdown stored as JSON
    video_score = Column(Float)
    audio_score = Column(Float)
    temporal_score = Column(Float)
    meta_data = Column(JSON) # Technical metadata (resolution, codec, etc.)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    owner = relationship("User", back_populates="scans")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String) # LOGIN, SCAN_UPLOAD, REPORT_GENERATED
    details = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="logs")
