"""
KAVACH-AI Backend Configuration
Zero API Key Configuration - All Local Processing
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    NO API KEYS REQUIRED - All processing is local.
    """
    
    # ============================================
    # APPLICATION
    # ============================================
    APP_NAME: str = "KAVACH-AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # ============================================
    # SERVER
    # ============================================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # ============================================
    # DATABASE
    # ============================================
    DATABASE_URL: str = "sqlite:///./data/kavach.db"
    
    # ============================================
    # REDIS
    # ============================================
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # ============================================
    # MODEL PATHS (Local Models)
    # ============================================
    MODELS_DIR: Path = Path("./models")
    SPATIAL_MODEL_PATH: Optional[str] = None
    TEMPORAL_MODEL_PATH: Optional[str] = None
    AUDIO_MODEL_PATH: Optional[str] = None
    
    # ============================================
    # PROCESSING CONFIGURATION
    # ============================================
    # Video Settings
    BASE_SAMPLING_INTERVAL: int = 2000  # ms
    MAX_SAMPLING_INTERVAL: int = 4000
    MIN_SAMPLING_INTERVAL: int = 500
    TARGET_LATENCY_MS: int = 5000
    FRAME_BUFFER_SIZE: int = 32
    VIDEO_RESOLUTION: int = 384
    
    # Audio Settings
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_SEGMENT_DURATION: int = 5  # seconds - Duration of each audio segment for processing
    AUDIO_WINDOW_SECONDS: float = 4.0
    AUDIO_OVERLAP_SECONDS: float = 2.0
    MEL_BINS: int = 128
    MFCC_COEFFICIENTS: int = 13
    
    # Ring Buffer
    RING_BUFFER_SECONDS: int = 30
    RING_BUFFER_DURATION: int = 30  # Alias for capture engine compatibility
    
    # ============================================
    # DETECTION THRESHOLDS
    # ============================================
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.6
    ALERT_THRESHOLD: float = 0.7
    HIGH_CONFIDENCE_THRESHOLD: float = 0.9
    
    # Temporal Accumulation
    EMA_ALPHA: float = 0.3
    TEMPORAL_HALFLIFE_SECONDS: int = 10
    TEMPORAL_DECAY_RATE: float = 0.95
    
    # ============================================
    # HARDWARE ACCELERATION
    # ============================================
    EXECUTION_PROVIDER: str = "cpu"
    ENABLE_GPU: bool = False
    CUDA_DEVICE_ID: int = 0
    
    # ============================================
    # STREAM SOURCES
    # ============================================
    MAX_CONCURRENT_STREAMS: int = 5
    ENABLE_YOUTUBE_LIVE: bool = True
    ENABLE_RTSP_STREAMS: bool = True
    ENABLE_AUDIO_STREAMS: bool = True
    
    # ============================================
    # THREAT INTELLIGENCE
    # ============================================
    DEFAULT_SEVERITY: str = "medium"
    CAMPAIGN_CORRELATION_THRESHOLD: float = 0.7
    VIRAL_VELOCITY_THRESHOLD: int = 1000
    SUSTAINED_HIGH_DURATION_SECONDS: int = 5
    
    # ============================================
    # FORENSIC EVIDENCE
    # ============================================
    EVIDENCE_DIR: Path = Path("./evidence")
    EVIDENCE_RETENTION_DAYS: int = 90
    ENABLE_CRYPTOGRAPHIC_CHAIN: bool = True
    EXPORT_FORMATS: List[str] = ["json", "cef", "pdf"]
    
    # ============================================
    # ALERTS & NOTIFICATIONS
    # ============================================
    # WebSocket
    ENABLE_WEBSOCKET: bool = True
    WEBSOCKET_HEARTBEAT_SECONDS: int = 30
    
    # SIEM Integration (No auth for local)
    ENABLE_SIEM: bool = False
    SIEM_WEBHOOK_URL: Optional[str] = None
    
    # Email Alerts (Local SMTP - optional)
    ENABLE_EMAIL_ALERTS: bool = False
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 1025
    SMTP_TLS: bool = False
    ALERT_EMAIL_FROM: str = "kavach@localhost"
    ALERT_EMAIL_TO: str = "security@localhost"
    
    # ============================================
    # LOGGING
    # ============================================
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("./logs/kavach.log")
    LOG_ROTATION: str = "500 MB"
    LOG_RETENTION: str = "10 days"
    
    # ============================================
    # FRONTEND
    # ============================================
    FRONTEND_URL: str = "http://localhost:3000"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]

    
    # ============================================
    # CALIBRATION
    # ============================================
    PLATT_TEMPERATURE: float = 1.0
    PLATT_BIAS: float = 0.0
    TARGET_ECE: float = 0.05
    
    # ============================================
    # ADAPTIVE RESOURCE MANAGEMENT
    # ============================================
    ENABLE_ADAPTIVE_SAMPLING: bool = True
    ENABLE_FAST_PATH: bool = True
    CPU_THRESHOLD_PERCENT: int = 80
    MEMORY_THRESHOLD_PERCENT: int = 85
    
    # ============================================
    # PRIVACY & ETHICS
    # ============================================
    ENFORCE_PRIVACY: bool = True  # Automatically blur faces in stored evidence
    ENABLE_ETHICAL_WATERMARK: bool = True  # Add "AI PROCESSED" watermark
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Set default model paths if not provided
        if not self.SPATIAL_MODEL_PATH:
            self.SPATIAL_MODEL_PATH = str(self.MODELS_DIR / "efficientnet_b4_deepfake.onnx")
        if not self.TEMPORAL_MODEL_PATH:
            self.TEMPORAL_MODEL_PATH = str(self.MODELS_DIR / "lstm_temporal.onnx")
        if not self.AUDIO_MODEL_PATH:
            self.AUDIO_MODEL_PATH = str(self.MODELS_DIR / "audio_classifier.onnx")


# Global settings instance
settings = Settings()
