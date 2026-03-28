"""
Simple test script to verify KAVACH-AI setup
"""

from backend.database import init_db, SessionLocal, Stream
from backend.config import settings
from loguru import logger

def test_database():
    """Test database connection and initialization"""
    logger.info("Testing database initialization...")
    
    # Initialize database
    init_db()
    logger.success("✅ Database initialized successfully")
    
    # Test connection
    db = SessionLocal()
    try:
        # Create test stream
        test_stream = Stream(
            name="Test Stream",
            source_url="https://www.youtube.com/watch?v=test",
            source_type="youtube_live",
            status="active"
        )
        db.add(test_stream)
        db.commit()
        db.refresh(test_stream)
        
        logger.success(f"✅ Created test stream with ID: {test_stream.id}")
        
        # Query streams
        streams = db.query(Stream).all()
        logger.info(f"Total streams in database: {len(streams)}")
        
        # Clean up
        db.delete(test_stream)
        db.commit()
        logger.success("✅ Test stream deleted")
        
    finally:
        db.close()


def test_models():
    """Test mock models"""
    logger.info("Testing mock models...")
    
    from backend.models.mock_models import get_spatial_detector, get_temporal_detector, get_audio_detector
    import numpy as np
    
    # Test spatial detector
    spatial = get_spatial_detector()
    fake_frame = np.random.rand(224, 224, 3)
    conf, features = spatial.predict(fake_frame)
    logger.success(f"✅ Spatial detector: confidence={conf:.3f}")
    
    # Test temporal detector
    temporal = get_temporal_detector()
    fake_features = np.random.rand(10, 128)
    conf, info = temporal.predict(fake_features)
    logger.success(f"✅ Temporal detector: confidence={conf:.3f}")
    
    # Test audio detector
    audio = get_audio_detector()
    fake_audio = np.random.rand(13, 100)
    conf, info = audio.predict(fake_audio)
    logger.success(f"✅ Audio detector: confidence={conf:.3f}")


def test_config():
    """Test configuration"""
    logger.info("Testing configuration...")
    logger.info(f"App Name: {settings.APP_NAME}")
    logger.info(f"Version: {settings.APP_VERSION}")
    logger.info(f"Database: {settings.DATABASE_URL}")
    logger.info(f"Models Dir: {settings.MODELS_DIR}")
    logger.success("✅ Configuration loaded")


if __name__ == "__main__":
    logger.info("🛡️  KAVACH-AI System Test")
    logger.info("=" * 50)
    
    test_config()
    logger.info("")
    
    test_database()
    logger.info("")
    
    test_models()
    logger.info("")
    
    logger.success("🎉 All tests passed!")
    logger.info("=" * 50)
    logger.info("Ready to start server: python -m uvicorn backend.main:app --reload")
