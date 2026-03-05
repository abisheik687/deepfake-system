
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
import os
import shutil

from backend.models.fusion_engine import FusionEngine
from backend.database import get_db, engine, Base
from backend.database import User
from backend.api.auth import auth_router, get_current_user

# KAVACH-AI Day 10: API Expansion
# Added: Auth Router, Chunked Uploads

# Initialize generic DB if not exists (for dev convenience)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="KAVACH-AI Deepfake Detection API",
    description="Multimodal Deepfake Detection System (Video + Audio + Temporal)",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

fusion = FusionEngine()
UPLOAD_DIR = "data/raw/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ScanResponse(BaseModel):
    filename: str
    status: str
    task_id: Optional[str] = None

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
@limiter.limit("60/minute")
def health_check(request: Request):
    return {"status": "active", "system": "KAVACH-AI", "version": "1.0.0"}

from backend.crud import create_scan_result, log_action

# ... imports ...

@app.post("/scan/upload", response_model=ScanResponse)
@limiter.limit("5/minute")
async def upload_for_analysis(
    request: Request,
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Standard Upload (Small files < 100MB)
    Protected by JWT.
    """
    file_path = os.path.join(UPLOAD_DIR, f"{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    task_id = "job_" + file.filename # Mock ID
    
    # Create DB Entry
    create_scan_result(db, task_id, file.filename, current_user.id)
    log_action(db, current_user.id, "UPLOAD", f"Uploaded {file.filename}")

    # Trigger Celery Task
    # from backend.worker import process_video_task
    # task = process_video_task.delay(file_path, task_id)
    
    return {
        "filename": file.filename,
        "status": "queued",
        "task_id": task_id
    }

@app.post("/scan/upload/chunked")
async def upload_chunk(
    file: UploadFile = File(...),
    chunk_index: int = 0,
    total_chunks: int = 1,
    file_id: str = Header(...),
    current_user: User = Depends(get_current_user)
):
    """
    Chunked Upload Handler for Large Files.
    Appends chunks to a temp file.
    """
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}.part")
    
    # Append mode
    with open(temp_path, "ab") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    if chunk_index == total_chunks - 1:
        # Final chunk received, rename to final file
        final_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
        os.rename(temp_path, final_path)
        return {"status": "complete", "path": final_path}
    
    return {"status": "chunk_received", "index": chunk_index}

# Caching Utility
import json
import redis
from functools import wraps

# Redis Connection
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

def cache_response(expiration=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from endpoint + params
            # Note: This is a simplified cache key strategy for the Blitz
            request = kwargs.get('request')
            user = kwargs.get('current_user')
            
            # Simple key based on function name and user ID (if auth) or basic args
            key_parts = [func.__name__]
            if user:
                key_parts.append(str(user.id))
            for k, v in kwargs.items():
                if k not in ['request', 'current_user', 'db']:
                     key_parts.append(f"{k}:{v}")
            cache_key = ":".join(key_parts)
            
            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Run function
            result = func(*args, **kwargs)
            
            # Cache result (ensure serializable)
            # Pydantic models need .dict() or compatible serialization
            # For this quick impl, assuming result is dict/list
            if result:
                # Convert datetime objects if any (simplified)
                # In production, use a custom JSON encoder
                redis_client.setex(cache_key, expiration, json.dumps(result, default=str))
                
            return result
        return wrapper
    return decorator

@app.get("/scan/history")
# @cache_response(expiration=30) # Disabled for now to ensure fresh data during dev
def get_history(
    skip: int = 0, 
    limit: int = 50, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from backend.database.crud import get_user_scans
    scans = get_user_scans(db, current_user.id, skip, limit)
    return scans

@app.get("/scan/history")
def get_history(
    skip: int = 0, 
    limit: int = 50, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from backend.database.crud import get_user_scans
    scans = get_user_scans(db, current_user.id, skip, limit)
    return scans

@app.get("/scan/{task_id}")
def get_result(
    task_id: str, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from backend.database.crud import get_scan_result
    scan = get_scan_result(db, task_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    # Check ownership
    if scan.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this scan")
        
    return {
        "id": scan.task_id,
        "filename": scan.filename,
        "status": scan.status,
        "verdict": scan.verdict,
        "confidence_score": scan.final_score,
        "timestamp": scan.created_at,
        "model_breakdown": {
            "video_score": scan.video_score,
            "audio_score": scan.audio_score,
            "temporal_score": scan.temporal_score
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
