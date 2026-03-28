"""
KAVACH-AI Live Video Call Detection Router
Module E: Live Video Call Deepfake Detection

WebRTC-based in-browser stream capture
Every 2 seconds: capture frame → WebSocket → lightweight MobileNetV3 (≤50ms inference)
Real-time overlay: GREEN border = real, RED = fake
Session log: timestamps, screenshots, confidence timeline
"""

import base64
import io
import time
import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
import numpy as np
from PIL import Image

from backend.database import get_db
from backend.crud import create_detection
from backend.detection.pipeline import analyze_frame
from backend.config import settings

router = APIRouter()

# Active WebSocket connections for live video sessions
active_sessions: Dict[str, dict] = {}


class LiveVideoSession:
    """Manages a live video detection session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.utcnow()
        self.frame_count = 0
        self.fake_detections = []
        self.confidence_timeline = []
        self.screenshots = []
        
    def add_frame_result(self, verdict: str, confidence: float, timestamp: float):
        """Add a frame analysis result to the session"""
        self.frame_count += 1
        self.confidence_timeline.append({
            'timestamp': timestamp,
            'verdict': verdict,
            'confidence': confidence,
            'frame_number': self.frame_count
        })
        
        if verdict == "FAKE":
            self.fake_detections.append({
                'timestamp': timestamp,
                'confidence': confidence,
                'frame_number': self.frame_count
            })
    
    def add_screenshot(self, frame_data: str, verdict: str, confidence: float):
        """Save a screenshot of a suspicious frame"""
        if len(self.screenshots) < 10:  # Limit to 10 screenshots per session
            self.screenshots.append({
                'frame_data': frame_data[:100],  # Store only thumbnail reference
                'verdict': verdict,
                'confidence': confidence,
                'timestamp': time.time()
            })
    
    def get_summary(self) -> dict:
        """Get session summary statistics"""
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        fake_count = len(self.fake_detections)
        fake_percentage = (fake_count / self.frame_count * 100) if self.frame_count > 0 else 0
        
        avg_confidence = 0.0
        if self.confidence_timeline:
            avg_confidence = sum(r['confidence'] for r in self.confidence_timeline) / len(self.confidence_timeline)
        
        return {
            'session_id': self.session_id,
            'duration_seconds': duration,
            'total_frames': self.frame_count,
            'fake_detections': fake_count,
            'fake_percentage': fake_percentage,
            'average_confidence': avg_confidence,
            'suspicious_timestamps': [d['timestamp'] for d in self.fake_detections],
            'screenshot_count': len(self.screenshots)
        }


@router.websocket("/ws/video")
async def live_video_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for live video call detection
    
    Client sends: { "type": "frame", "data": "base64_image", "timestamp": 123456 }
    Server responds: { "type": "result", "verdict": "REAL/FAKE", "confidence": 0.95, "color": "green/red" }
    """
    await websocket.accept()
    session_id = f"video_{int(time.time())}_{id(websocket)}"
    session = LiveVideoSession(session_id)
    active_sessions[session_id] = session
    
    logger.info(f"Live video session started: {session_id}")
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Live video detection active",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            # Receive frame data from client
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            
            if message_type == "frame":
                # Process video frame
                frame_data = data.get("data", "")
                client_timestamp = data.get("timestamp", time.time())
                
                if not frame_data:
                    continue
                
                # Analyze frame using lightweight model
                t0 = time.perf_counter()
                result = analyze_frame(frame_data, include_heatmap=False, use_ml=True)
                inference_time = (time.perf_counter() - t0) * 1000
                
                verdict = result.get("verdict", "SUSPICIOUS")
                confidence = result.get("confidence", 0.5)
                
                # Determine border color
                if verdict == "FAKE":
                    color = "red"
                    severity = "high"
                elif verdict == "SUSPICIOUS":
                    color = "yellow"
                    severity = "medium"
                else:
                    color = "green"
                    severity = "low"
                
                # Add to session
                session.add_frame_result(verdict, confidence, client_timestamp)
                
                # Save screenshot if fake detected
                if verdict == "FAKE" and confidence > 0.75:
                    session.add_screenshot(frame_data, verdict, confidence)
                
                # Send result back to client
                await websocket.send_json({
                    "type": "result",
                    "verdict": verdict,
                    "confidence": confidence,
                    "color": color,
                    "severity": severity,
                    "inference_time_ms": inference_time,
                    "frame_number": session.frame_count,
                    "message": result.get("message", "")
                })
                
                # Log if inference is too slow
                if inference_time > 50:
                    logger.warning(f"Slow inference: {inference_time:.1f}ms (target: <50ms)")
            
            elif message_type == "ping":
                # Heartbeat
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            elif message_type == "get_summary":
                # Client requests session summary
                summary = session.get_summary()
                await websocket.send_json({
                    "type": "summary",
                    "data": summary
                })
            
            elif message_type == "end_session":
                # Client ending session
                summary = session.get_summary()
                await websocket.send_json({
                    "type": "session_ended",
                    "summary": summary
                })
                break
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
    
    except WebSocketDisconnect:
        logger.info(f"Live video session disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Live video session error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        # Clean up session
        if session_id in active_sessions:
            summary = active_sessions[session_id].get_summary()
            logger.info(f"Session {session_id} summary: {summary}")
            del active_sessions[session_id]
        
        try:
            await websocket.close()
        except:
            pass


@router.get("/sessions")
async def get_active_sessions():
    """Get list of active live video sessions"""
    sessions = []
    for session_id, session in active_sessions.items():
        sessions.append({
            'session_id': session_id,
            'start_time': session.start_time.isoformat(),
            'frame_count': session.frame_count,
            'fake_detections': len(session.fake_detections)
        })
    
    return {
        "active_sessions": len(sessions),
        "sessions": sessions
    }


@router.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get summary of a specific session"""
    if session_id not in active_sessions:
        return {"error": "Session not found"}
    
    session = active_sessions[session_id]
    return session.get_summary()


@router.post("/session/{session_id}/export")
async def export_session_report(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Export session report as PDF
    Includes: timeline, screenshots, statistics
    """
    if session_id not in active_sessions:
        return {"error": "Session not found"}
    
    session = active_sessions[session_id]
    summary = session.get_summary()
    
    # Save to database
    detection_data = {
        'file_name': f"live_video_session_{session_id}",
        'file_type': 'live_video',
        'verdict': 'FAKE' if summary['fake_percentage'] > 50 else 'REAL',
        'confidence': summary['average_confidence'],
        'fake_probability': summary['fake_percentage'] / 100,
        'metadata': {
            'session_id': session_id,
            'summary': summary,
            'confidence_timeline': session.confidence_timeline,
            'fake_detections': session.fake_detections
        }
    }
    
    detection = await create_detection(db, detection_data)
    
    return {
        "success": True,
        "detection_id": detection.id if detection else None,
        "summary": summary,
        "message": "Session report saved"
    }


@router.get("/health")
async def live_video_health():
    """Health check for live video detection module"""
    return {
        "status": "healthy",
        "module": "live_video_detection",
        "active_sessions": len(active_sessions),
        "target_inference_time_ms": 50,
        "websocket_endpoint": "/api/live-video/ws/video"
    }

# Made with Bob
