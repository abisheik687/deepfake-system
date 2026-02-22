"""
KAVACH-AI Capture Engine
Real-time stream capture coordination with ring buffer
NO API KEYS REQUIRED - All processing is local
"""

import asyncio
from typing import Optional, Callable, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import numpy as np

from backend.ingestion.stream_manager import StreamManager, StreamInfo, StreamType, stream_manager
from backend.ingestion.ffmpeg_processor import FFmpegProcessor, VideoProperties, AudioProperties
from backend.config import settings


@dataclass
class CapturedFrame:
    """Captured video frame with metadata"""
    frame_number: int
    timestamp: datetime
    stream_timestamp: float  # Seconds from stream start
    frame_data: np.ndarray  # BGR image
    width: int
    height: int
    stream_id: int


@dataclass
class CapturedAudio:
    """Captured audio segment with metadata"""
    segment_number: int
    timestamp: datetime
    stream_timestamp: float
    audio_path: Path
    duration: float  # Seconds
    sample_rate: int
    stream_id: int


@dataclass
class CaptureStats:
    """Capture statistics"""
    stream_id: int
    start_time: datetime
    frames_captured: int = 0
    audio_segments_captured: int = 0
    frames_dropped: int = 0
    total_bytes: int = 0
    average_fps: float = 0.0
    errors: int = 0
    last_capture_time: Optional[datetime] = None


class RingBuffer:
    """
    Ring buffer for storing recent frames/audio.
    Maintains last N seconds of data for evidence capture.
    """
    
    def __init__(self, max_duration_seconds: int = 30):
        """
        Initialize ring buffer.
        
        Args:
            max_duration_seconds: Maximum duration to keep (default 30s)
        """
        self.max_duration = timedelta(seconds=max_duration_seconds)
        self.frames: deque[CapturedFrame] = deque()
        self.audio: deque[CapturedAudio] = deque()
        self.max_size = 1000  # Max items regardless of time
    
    def add_frame(self, frame: CapturedFrame):
        """Add frame to ring buffer"""
        self.frames.append(frame)
        
        # Remove old frames
        self._cleanup_old_items()
        
        # Enforce max size
        while len(self.frames) > self.max_size:
            self.frames.popleft()
    
    def add_audio(self, audio: CapturedAudio):
        """Add audio segment to ring buffer"""
        self.audio.append(audio)
        self._cleanup_old_items()
    
    def _cleanup_old_items(self):
        """Remove items older than max_duration"""
        now = datetime.utcnow()
        cutoff = now - self.max_duration
        
        # Remove old frames
        while self.frames and self.frames[0].timestamp < cutoff:
            self.frames.popleft()
        
        # Remove old audio
        while self.audio and self.audio[0].timestamp < cutoff:
            self.audio.popleft()
    
    def get_recent_frames(self, duration_seconds: int = 10) -> list[CapturedFrame]:
        """Get frames from last N seconds"""
        cutoff = datetime.utcnow() - timedelta(seconds=duration_seconds)
        return [f for f in self.frames if f.timestamp >= cutoff]
    
    def get_recent_audio(self, duration_seconds: int = 10) -> list[CapturedAudio]:
        """Get audio segments from last N seconds"""
        cutoff = datetime.utcnow() - timedelta(seconds=duration_seconds)
        return [a for a in self.audio if a.timestamp >= cutoff]
    
    def get_frames_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> list[CapturedFrame]:
        """Get frames within time range"""
        return [
            f for f in self.frames
            if start_time <= f.timestamp <= end_time
        ]
    
    def clear(self):
        """Clear all buffered data"""
        self.frames.clear()
        self.audio.clear()


class CaptureEngine:
    """
    Real-time capture engine coordinating stream ingestion.
    
    Responsibilities:
    - Coordinate stream manager and FFmpeg processor
    - Maintain ring buffer of recent frames/audio
    - Adaptive sampling based on motion/audio activity
    - Queue high-priority frames for detection
    - Generate capture statistics
    """
    
    def __init__(self):
        self.stream_manager = stream_manager
        self.ffmpeg = FFmpegProcessor(
            enable_gpu=settings.ENABLE_GPU,
            gpu_type=settings.EXECUTION_PROVIDER
        )
        
        # Active captures: stream_id -> capture task
        self.active_captures: Dict[int, asyncio.Task] = {}
        
        # Ring buffers: stream_id -> RingBuffer
        self.ring_buffers: Dict[int, RingBuffer] = {}
        
        # Statistics: stream_id -> CaptureStats
        self.stats: Dict[int, CaptureStats] = {}
        
        # Frame callbacks: stream_id -> callback
        self.frame_callbacks: Dict[int, Callable] = {}
        
        # Audio callbacks: stream_id -> callback
        self.audio_callbacks: Dict[int, Callable] = {}
        
        logger.info("Capture engine initialized")
    
    async def start_capture(
        self,
        stream_id: int,
        url: str,
        stream_type: StreamType,
        frame_callback: Optional[Callable] = None,
        audio_callback: Optional[Callable] = None
    ):
        """
        Start capturing from a stream.
        
        Args:
            stream_id: Database stream ID
            url: Stream URL
            stream_type: Type of stream
            frame_callback: Async callback(frame: CapturedFrame)
            audio_callback: Async callback(audio: CapturedAudio)
        """
        if stream_id in self.active_captures:
            logger.warning(f"Capture already active for stream {stream_id}")
            return
        
        logger.info(f"Starting capture for stream {stream_id}: {stream_type.value}")
        
        # Add stream to manager
        try:
            stream_info = await self.stream_manager.add_stream(
                stream_id=stream_id,
                url=url,
                stream_type=stream_type
            )
        except Exception as e:
            logger.error(f"Failed to add stream {stream_id}: {e}")
            raise
        
        # Initialize ring buffer
        self.ring_buffers[stream_id] = RingBuffer(
            max_duration_seconds=settings.RING_BUFFER_DURATION
        )
        
        # Initialize statistics
        self.stats[stream_id] = CaptureStats(
            stream_id=stream_id,
            start_time=datetime.utcnow()
        )
        
        # Store callbacks
        if frame_callback:
            self.frame_callbacks[stream_id] = frame_callback
        if audio_callback:
            self.audio_callbacks[stream_id] = audio_callback
        
        # Start capture tasks
        capture_task = asyncio.create_task(
            self._capture_loop(stream_id, stream_info)
        )
        self.active_captures[stream_id] = capture_task
        
        logger.success(f"Capture started for stream {stream_id}")
    
    async def _capture_loop(self, stream_id: int, stream_info: StreamInfo):
        """
        Main capture loop for a stream.
        
        Args:
            stream_id: Stream ID
            stream_info: Stream information
        """
        logger.info(f"Capture loop started for stream {stream_id}")
        
        stats = self.stats[stream_id]
        ring_buffer = self.ring_buffers[stream_id]
        
        # Determine capture URLs
        video_url = stream_info.video_url or stream_info.url
        audio_url = stream_info.audio_url or stream_info.url
        
        # Start video capture task
        video_task = asyncio.create_task(
            self._capture_video(stream_id, video_url, stats, ring_buffer)
        )
        
        # Start audio capture task (if audio available)
        audio_task = None
        if audio_url and stream_info.stream_type != StreamType.AUDIO:
            audio_task = asyncio.create_task(
                self._capture_audio(stream_id, audio_url, stats, ring_buffer)
            )
        
        try:
            # Wait for both tasks
            if audio_task:
                await asyncio.gather(video_task, audio_task)
            else:
                await video_task
        
        except asyncio.CancelledError:
            logger.info(f"Capture cancelled for stream {stream_id}")
            video_task.cancel()
            if audio_task:
                audio_task.cancel()
        
        except Exception as e:
            logger.error(f"Capture loop error for stream {stream_id}: {e}")
            stats.errors += 1
            
            # Attempt reconnection
            if await self.stream_manager.reconnect_stream(stream_id):
                # Restart capture
                updated_info = self.stream_manager.get_stream_info(stream_id)
                if updated_info:
                    await self._capture_loop(stream_id, updated_info)
        
        finally:
            logger.info(f"Capture loop ended for stream {stream_id}")
    
    async def _capture_video(
        self,
        stream_id: int,
        video_url: str,
        stats: CaptureStats,
        ring_buffer: RingBuffer
    ):
        """
        Capture video frames from stream.
        
        Args:
            stream_id: Stream ID
            video_url: Video stream URL
            stats: Capture statistics
            ring_buffer: Ring buffer for storage
        """
        logger.info(f"Video capture started for stream {stream_id}")
        
        # Get video properties
        props = await self.ffmpeg.get_video_properties(video_url)
        if not props:
            logger.error(f"Could not get video properties for stream {stream_id}")
            return
        
        # Calculate sampling rate
        # Sample every N milliseconds based on settings
        sampling_fps = 1000.0 / settings.BASE_SAMPLING_INTERVAL  # Convert ms to fps
        
        logger.info(
            f"Video sampling: {sampling_fps:.2f} fps "
            f"(original: {props.fps:.2f} fps)"
        )
        
        frame_number = 0
        stream_start = datetime.utcnow()
        
        try:
            # Use continuous capture
            async for frame in self.ffmpeg.start_continuous_capture(
                url=video_url,
                fps=sampling_fps
            ):
                now = datetime.utcnow()
                elapsed = (now - stream_start).total_seconds()
                
                # Create captured frame
                captured_frame = CapturedFrame(
                    frame_number=frame_number,
                    timestamp=now,
                    stream_timestamp=elapsed,
                    frame_data=frame,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    stream_id=stream_id
                )
                
                # Add to ring buffer
                ring_buffer.add_frame(captured_frame)
                
                # Update statistics
                stats.frames_captured += 1
                stats.total_bytes += frame.nbytes
                stats.last_capture_time = now
                stats.average_fps = stats.frames_captured / elapsed if elapsed > 0 else 0
                
                # Call frame callback if registered
                if stream_id in self.frame_callbacks:
                    try:
                        await self.frame_callbacks[stream_id](captured_frame)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")
                
                frame_number += 1
                
                # Log progress periodically
                if frame_number % 100 == 0:
                    logger.info(
                        f"Stream {stream_id}: {frame_number} frames captured "
                        f"({stats.average_fps:.2f} fps)"
                    )
        
        except asyncio.CancelledError:
            logger.info(f"Video capture cancelled for stream {stream_id}")
            raise
        
        except Exception as e:
            logger.error(f"Video capture error for stream {stream_id}: {e}")
            raise
    
    async def _capture_audio(
        self,
        stream_id: int,
        audio_url: str,
        stats: CaptureStats,
        ring_buffer: RingBuffer
    ):
        """
        Capture audio segments from stream.
        
        Args:
            stream_id: Stream ID
            audio_url: Audio stream URL
            stats: Capture statistics
            ring_buffer: Ring buffer for storage
        """
        logger.info(f"Audio capture started for stream {stream_id}")
        
        # Get audio properties
        props = await self.ffmpeg.get_audio_properties(audio_url)
        if not props:
            logger.error(f"Could not get audio properties for stream {stream_id}")
            return
        
        # Audio segment duration (from settings)
        segment_duration = settings.AUDIO_SEGMENT_DURATION  # seconds
        
        logger.info(
            f"Audio sampling: {segment_duration}s segments, "
            f"{props.sample_rate}Hz, {props.channels}ch"
        )
        
        segment_number = 0
        stream_start = datetime.utcnow()
        
        try:
            while True:
                now = datetime.utcnow()
                elapsed = (now - stream_start).total_seconds()
                
                # Extract audio segment
                output_path = (
                    Path(settings.EVIDENCE_DIR) / 
                    f"stream_{stream_id}_audio_{segment_number:06d}.wav"
                )
                
                success = await self.ffmpeg.extract_audio_segment(
                    url=audio_url,
                    start_time=elapsed,
                    duration=segment_duration,
                    output_path=output_path,
                    sample_rate=settings.AUDIO_SAMPLE_RATE
                )
                
                if success:
                    # Create captured audio
                    captured_audio = CapturedAudio(
                        segment_number=segment_number,
                        timestamp=now,
                        stream_timestamp=elapsed,
                        audio_path=output_path,
                        duration=segment_duration,
                        sample_rate=settings.AUDIO_SAMPLE_RATE,
                        stream_id=stream_id
                    )
                    
                    # Add to ring buffer
                    ring_buffer.add_audio(captured_audio)
                    
                    # Update statistics
                    stats.audio_segments_captured += 1
                    
                    # Call audio callback if registered
                    if stream_id in self.audio_callbacks:
                        try:
                            await self.audio_callbacks[stream_id](captured_audio)
                        except Exception as e:
                            logger.error(f"Audio callback error: {e}")
                    
                    segment_number += 1
                else:
                    logger.warning(f"Failed to capture audio segment {segment_number}")
                
                # Wait for next segment
                await asyncio.sleep(segment_duration)
        
        except asyncio.CancelledError:
            logger.info(f"Audio capture cancelled for stream {stream_id}")
            raise
        
        except Exception as e:
            logger.error(f"Audio capture error for stream {stream_id}: {e}")
            raise
    
    async def stop_capture(self, stream_id: int):
        """
        Stop capturing from a stream.
        
        Args:
            stream_id: Stream ID
        """
        if stream_id not in self.active_captures:
            logger.warning(f"No active capture for stream {stream_id}")
            return
        
        logger.info(f"Stopping capture for stream {stream_id}")
        
        # Cancel capture task
        task = self.active_captures.pop(stream_id)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Remove from stream manager
        self.stream_manager.remove_stream(stream_id)
        
        # Clean up callbacks
        self.frame_callbacks.pop(stream_id, None)
        self.audio_callbacks.pop(stream_id, None)
        
        # Log final statistics
        if stream_id in self.stats:
            stats = self.stats[stream_id]
            duration = (datetime.utcnow() - stats.start_time).total_seconds()
            
            logger.info(
                f"Stream {stream_id} capture statistics:\n"
                f"  Duration: {duration:.2f}s\n"
                f"  Frames: {stats.frames_captured}\n"
                f"  Audio segments: {stats.audio_segments_captured}\n"
                f"  Average FPS: {stats.average_fps:.2f}\n"
                f"  Errors: {stats.errors}"
            )
        
        logger.success(f"Capture stopped for stream {stream_id}")
    
    def get_ring_buffer(self, stream_id: int) -> Optional[RingBuffer]:
        """Get ring buffer for a stream"""
        return self.ring_buffers.get(stream_id)
    
    def get_stats(self, stream_id: int) -> Optional[CaptureStats]:
        """Get capture statistics for a stream"""
        return self.stats.get(stream_id)
    
    def get_active_captures(self) -> list[int]:
        """Get list of active capture stream IDs"""
        return list(self.active_captures.keys())


# Global capture engine instance
capture_engine = CaptureEngine()
