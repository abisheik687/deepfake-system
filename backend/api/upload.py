from fastapi import APIRouter, File, UploadFile, Header, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import random
from loguru import logger

from backend.database import get_db, Detection, Stream

router = APIRouter()

UPLOAD_STREAM_URL = "file://upload"

def get_or_create_upload_stream(db: Session) -> int:
    """Get the dedicated upload-type Stream, or create it if missing."""
    stream = db.query(Stream).filter(Stream.url == UPLOAD_STREAM_URL).first()
    if not stream:
        stream = Stream(
            url=UPLOAD_STREAM_URL,
            source_type="file_upload",
            active=False,
            metadata_json={"description": "Auto-created for uploaded file scans"}
        )
        db.add(stream)
        db.commit()
        db.refresh(stream)
        logger.info(f"Created upload stream with ID {stream.id}")
    return stream.id

@router.post("/chunked")
async def upload_chunked(
    file: UploadFile = File(...),
    file_id: str = Header(..., alias="file-id"),
    chunk_index: int = Query(...),
    total_chunks: int = Query(...),
    db: Session = Depends(get_db)
):
    """
    Handle chunked video uploads for deepfake forensic analysis.
    Simulates ML processing — generates a mock Detection record
    with confidence scores when the final chunk is received.
    """
    try:
        # When final chunk arrives, trigger mock forensic analysis
        if chunk_index == total_chunks - 1:
            logger.info(f"Final chunk received for '{file.filename}'. Running mock analysis...")

            # Get or create the upload stream (satisfies FK constraint)
            upload_stream_id = get_or_create_upload_stream(db)

            # Simulate ML scoring
            is_fake = random.choice([True, False, True])  # Skewed toward fake for demo
            confidence = round(random.uniform(0.85, 0.99) if is_fake else random.uniform(0.05, 0.30), 3)
            severity = "high" if confidence > 0.85 else "low"
            attack_type = "identity_impersonation" if is_fake else "none"

            detection = Detection(
                stream_id=upload_stream_id,
                timestamp=datetime.utcnow(),
                confidence=confidence,
                spatial_confidence=min(1.0, confidence + random.uniform(-0.1, 0.1)),
                temporal_confidence=min(1.0, confidence + random.uniform(-0.1, 0.1)),
                audio_confidence=min(1.0, confidence + random.uniform(-0.1, 0.1)),
                severity=severity,
                attack_type=attack_type,
                features_json={"filename": file.filename, "file_id": file_id}
            )

            db.add(detection)
            db.commit()
            db.refresh(detection)

            logger.success(f"Analysis complete — Detection ID: {detection.id} | Verdict: {'FAKE' if is_fake else 'REAL'} ({confidence:.1%})")
            return {
                "status": "completed",
                "file_id": file_id,
                "detection_id": detection.id,
                "verdict": "FAKE" if is_fake else "REAL",
                "confidence": confidence
            }

        # Intermediate chunk — just acknowledge
        return {"status": "uploading", "chunk": chunk_index}

    except Exception as e:
        logger.error(f"Error in chunked upload: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

