"""
KAVACH-AI FastAPI Application Entry Point
Real-Time Deepfake Detection and Threat Intelligence System
NO API KEYS REQUIRED - All processing is local
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import sys
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse



from backend.config import settings
from backend.database import init_db, engine
from backend.api import alerts
from backend.api.auth import auth_router
from backend.api import scan            # Phase ORCH — Unified Orchestration Scanner
from backend.api import models_api      # Models and training info
from backend.api import detections      # Detection history & stats
from backend.api import agency          # AI Agency Mission Control
from backend.detection.pipeline import analyze_frame
from backend.api.rate_limit import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator




# ============================================
# LOGGING SETUP
# ============================================

logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL
)
logger.add(
    str(settings.LOG_FILE),
    rotation=settings.LOG_ROTATION,
    retention=settings.LOG_RETENTION,
    level=settings.LOG_LEVEL
)


# ============================================
# LIFESPAN CONTEXT MANAGER
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info(f"🛡️  Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    if not settings.SECRET_KEY or settings.SECRET_KEY == "supersecretkey_change_me_in_prod":
        raise RuntimeError("FATAL: SECRET_KEY is not set. Generate one via: python -c \"import secrets; print(secrets.token_urlsafe(64))\" and set it in .env")
        
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Database: {settings.DATABASE_URL}")
    
    # Initialize database
    logger.info("Initializing database...")
    await init_db()
    logger.success("✓ Database initialized")
    
    # Create directories
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    settings.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    logger.success("✓ Directories created")
    
    # TODO: Load ML models
    logger.info("ML models will be loaded on-demand")
    
    logger.success(f"✓ {settings.APP_NAME} started successfully!")
    logger.info(f"API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}...")
    # Cleanup tasks here
    logger.success("✓ Shutdown complete")


# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-Time Deepfake Detection and Threat Intelligence System",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Instrumentation for Prometheus metrics
Instrumentator().instrument(app).expose(app)


# ============================================
# MIDDLEWARE
# ============================================

from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings

def _parse_origins(raw: str) -> list[str]:
    """
    Parse a comma-separated ALLOWED_ORIGINS string into a clean list.
    """
    return [o.strip() for o in raw.split(",") if o.strip()]

allowed_origins = _parse_origins(settings.ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Request-ID",
        "Accept",
    ],
    expose_headers=[
        "Content-Disposition",
        "X-Evidence-Alert-ID",
        "X-Export-Timestamp",
    ],
)


# ============================================
# INCLUDE ROUTERS
# ============================================

from backend.api.health import health_router
from backend.routers import audio, social, live_video, live_audio, interview

app.include_router(auth_router)
app.include_router(health_router)
app.include_router(alerts.router,      prefix="/api/alerts",      tags=["Alerts"])
# ── DeepShield AI — Config and Models ────────────────────────────────────────
app.include_router(models_api.router,  prefix="/api/models",      tags=["Models"])
# ── DeepShield AI — Detection History & Stats ────────────────────────────────
app.include_router(detections.router,  prefix="/api/detections",  tags=["Detections"])
# ── DeepShield AI — Consolidated Scanner (Orchestration Layer) ───────────────
app.include_router(scan.router,        prefix="/api/scan",        tags=["Scanner"])
# ── DeepShield AI — AI Agency Mission Control ────────────────────────────────
app.include_router(agency.router,      prefix="/api/agency",      tags=["Agency"])
# ── DeepShield AI — Audio Deepfake Detection (Module C) ──────────────────────
app.include_router(audio.router,       prefix="/api/audio",       tags=["Audio Detection"])
# ── DeepShield AI — Social Media Detection (Module D) ────────────────────────
app.include_router(social.router,      prefix="/api/social",      tags=["Social Media"])
# ── DeepShield AI — Live Video Detection (Module E) ──────────────────────────
app.include_router(live_video.router,  prefix="/api/live-video",  tags=["Live Video"])
# ── DeepShield AI — Live Audio Detection (Module F) ──────────────────────────
app.include_router(live_audio.router,  prefix="/api/live-audio",  tags=["Live Audio"])
# ── DeepShield AI — Interview Proctoring (Module G) ──────────────────────────
app.include_router(interview.router,   prefix="/api/interview",   tags=["Interview Proctoring"])



# ============================================
# SELF-RESTART ENDPOINT (Workaround)
# ============================================

@app.post("/api/admin/restart")
async def restart_server():
    """Restarts the server to apply changes."""
    logger.warning("🛡️  Self-restart triggered via API")
    
    def restart():
        os.execv(sys.executable, [sys.executable] + sys.argv)
    
    # We use a background task or just call it after a short delay
    from threading import Timer
    Timer(1.0, restart).start()
    
    return {"message": "Server is restarting... Please refresh in 5 seconds."}


# ============================================
# ROOT ENDPOINTS
# ============================================

# ============================================
# FRONTEND SERVING (SPA Support)
# ============================================

# Define potential frontend paths
BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

def get_frontend_file(filename: str = "index.html"):
    """Helper to get frontend file path if exists"""
    # 1. Try relative to this file
    path1 = (Path(__file__).parent.parent / "frontend" / "dist" / filename).resolve()
    if path1.exists():
        return path1
    
    # 2. Try relative to current working directory
    path2 = (Path.cwd() / "frontend" / "dist" / filename).resolve()
    if path2.exists():
        return path2
        
    # 3. Try hardcoded path if on Windows (common for this user)
    path3 = Path(r"e:\Users\Abisheik\downloads\deepfake system\deepfake system\frontend\dist") / filename
    if path3.exists():
        return path3
        
    return None

# Mount static files for built frontend
frontend_assets = get_frontend_file("assets")
if frontend_assets:
    app.mount("/assets", StaticFiles(directory=str(frontend_assets)), name="assets")
    logger.success(f"✓ Frontend assets mounted from {frontend_assets}")
else:
    logger.warning("⚠️ Frontend assets directory not found")

@app.get("/")
async def root():
    """Root endpoint serving the frontend or API info"""
    index_path = get_frontend_file("index.html")
    if index_path:
        return FileResponse(str(index_path))
    
    # Debug info if not found
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Real-Time Deepfake Detection System - Dashboard not found",
        "debug": {
            "cwd": str(Path.cwd()),
            "file": __file__,
            "expected_dist": str((Path(__file__).parent.parent / "frontend" / "dist").resolve()),
            "exists": (Path(__file__).parent.parent / "frontend" / "dist" / "index.html").exists()
        },
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    """Catch-all for SPA routing"""
    # Skip API calls
    if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("redoc") or full_path.startswith("ws"):
        # This will normally fall through to FastAPI's 404, but we handle it just in case
        return JSONResponse(status_code=404, content={"error": "Not Found", "message": f"Path '{full_path}' not found"})
    
    # Try to serve as a static file if it exists in dist
    static_file = get_frontend_file(full_path)
    if static_file and static_file.is_file():
        return FileResponse(str(static_file))
    
    # Otherwise fallback to index.html for SPA frontend routing
    index_path = get_frontend_file("index.html")
    if index_path:
        return FileResponse(str(index_path))
    
    return JSONResponse(status_code=404, content={"error": "Frontend not found"})



@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }


# ============================================
# WEBSOCKET ENDPOINT
# ============================================

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates
    
    Clients receive:
    - Detection updates
    - Alert notifications
    - System status updates
    """
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "connection",
            "message": f"Connected to {settings.APP_NAME}",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

        
        while True:
            # Receive messages from client (heartbeat, commands, etc.)
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get("type")
            
            if message_type == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }, websocket)

            
            elif message_type == "frame":
                # Real-time camera frame analysis
                frame_data = data.get("data", "")
                if frame_data:
                    result = analyze_frame(frame_data)
                    is_alert = result["verdict"] == "FAKE"
                    await manager.send_personal_message({
                        "type": "frame_result",
                        "verdict": result["verdict"],
                        "confidence": result["confidence"],
                        "faces": result["faces"],
                        "message": result["message"],
                        "data": {
                            "severity": "high" if is_alert else "low",
                        }
                    }, websocket)
            
            elif message_type == "subscribe":
                # Subscribe to specific streams or alert types
                logger.info(f"Client subscribed: {data}")
                await manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "data": data
                }, websocket)
            
            else:
                logger.warning(f"Unknown WebSocket message type: {message_type}")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# ============================================
# STARTUP MESSAGE
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
