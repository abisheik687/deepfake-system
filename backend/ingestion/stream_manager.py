"""
KAVACH-AI Stream Manager
Unified interface for managing multiple live stream sources
NO API KEYS REQUIRED - All sources are publicly accessible
"""

import asyncio
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
from loguru import logger
import yt_dlp


class StreamType(Enum):
    """Supported stream types"""
    YOUTUBE_LIVE = "youtube_live"
    RTSP = "rtsp"
    RTMP = "rtmp"
    HTTP_STREAM = "http_stream"
    AUDIO = "audio"


class StreamStatus(Enum):
    """Stream status"""
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    STREAMING = "streaming"
    PAUSED = "paused"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamInfo:
    """Stream information and metadata"""
    stream_id: int
    url: str
    stream_type: StreamType
    status: StreamStatus
    
    # Stream properties
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    format_id: Optional[str] = None
    
    # Quality information
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    bitrate: Optional[int] = None
    
    # Metadata
    title: Optional[str] = None
    description: Optional[str] = None
    uploader: Optional[str] = None
    
    # State
    reconnect_attempts: int = 0
    error_message: Optional[str] = None


class StreamManager:
    """
    Unified stream manager for all source types.
    
    Responsibilities:
    - URL extraction and validation
    - Stream format selection
    - Connection management
    - Auto-reconnection
    - Quality adaptation
    """
    
    def __init__(self):
        self.active_streams: Dict[int, StreamInfo] = {}
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
    
    async def add_stream(
        self,
        stream_id: int,
        url: str,
        stream_type: StreamType,
        preferred_quality: str = "best"
    ) -> StreamInfo:
        """
        Add a new stream source.
        
        Args:
            stream_id: Database stream ID
            url: Stream URL
            stream_type: Type of stream
            preferred_quality: Quality preference (best, 720p, 480p, etc.)
        
        Returns:
            StreamInfo with extracted metadata
        """
        logger.info(f"Adding stream {stream_id}: {stream_type.value} - {url}")
        
        stream_info = StreamInfo(
            stream_id=stream_id,
            url=url,
            stream_type=stream_type,
            status=StreamStatus.INITIALIZING
        )
        
        try:
            if stream_type == StreamType.YOUTUBE_LIVE:
                stream_info = await self._extract_youtube_live(stream_info, preferred_quality)
            
            elif stream_type in [StreamType.RTSP, StreamType.RTMP]:
                stream_info = await self._validate_rtsp_rtmp(stream_info)
            
            elif stream_type == StreamType.HTTP_STREAM:
                stream_info = await self._validate_http_stream(stream_info)
            
            elif stream_type == StreamType.AUDIO:
                stream_info = await self._validate_audio_stream(stream_info)
            
            stream_info.status = StreamStatus.CONNECTED
            self.active_streams[stream_id] = stream_info
            
            logger.success(f"Stream {stream_id} added successfully")
            return stream_info
        
        except Exception as e:
            logger.error(f"Error adding stream {stream_id}: {e}")
            stream_info.status = StreamStatus.ERROR
            stream_info.error_message = str(e)
            raise
    
    async def _extract_youtube_live(
        self,
        stream_info: StreamInfo,
        preferred_quality: str
    ) -> StreamInfo:
        """
        Extract YouTube Live stream URLs using yt-dlp.
        NO API KEY REQUIRED - Uses public extraction
        """
        logger.info(f"Extracting YouTube Live stream: {stream_info.url}")
        
        ydl_opts = {
            'format': self._get_youtube_format(preferred_quality),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
            # NO API KEY NEEDED!
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, stream_info.url, download=False)
                
                if not info:
                    raise ValueError("Could not extract stream information")
                
                # Get best video and audio URLs
                if 'url' in info:
                    stream_info.video_url = info['url']
                
                if 'formats' in info:
                    # Try to get separate video/audio streams for better control
                    video_formats = [f for f in info['formats'] if f.get('vcodec') != 'none']
                    audio_formats = [f for f in info['formats'] if f.get('acodec') != 'none']
                    
                    if video_formats:
                        best_video = max(video_formats, key=lambda x: x.get('height', 0) or 0)
                        stream_info.video_url = best_video.get('url')
                        stream_info.width = best_video.get('width')
                        stream_info.height = best_video.get('height')
                        stream_info.fps = best_video.get('fps')
                        stream_info.format_id = best_video.get('format_id')
                    
                    if audio_formats:
                        best_audio = max(audio_formats, key=lambda x: x.get('abr', 0) or 0)
                        stream_info.audio_url = best_audio.get('url')
                
                # Extract metadata
                stream_info.title = info.get('title')
                stream_info.description = info.get('description')
                stream_info.uploader = info.get('uploader')
                
                logger.success(f"YouTube Live extraction complete: {stream_info.title}")
                logger.info(f"Video URL: {stream_info.video_url[:100]}..." if stream_info.video_url else "No video URL")
                logger.info(f"Audio URL: {stream_info.audio_url[:100]}..." if stream_info.audio_url else "No audio URL")
                
                return stream_info
        
        except Exception as e:
            logger.error(f"YouTube extraction error: {e}")
            raise ValueError(f"Failed to extract YouTube Live stream: {e}")
    
    def _get_youtube_format(self, preferred_quality: str) -> str:
        """
        Get yt-dlp format string based on quality preference.
        
        Format selection prioritizes:
        1. Live streams (no download)
        2. Specified quality
        3. Best available quality
        """
        quality_map = {
            'best': 'best',
            '1080p': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            '720p': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
            '480p': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
            '360p': 'bestvideo[height<=360]+bestaudio/best[height<=360]',
        }
        
        return quality_map.get(preferred_quality, 'best')
    
    async def _validate_rtsp_rtmp(self, stream_info: StreamInfo) -> StreamInfo:
        """
        Validate RTSP/RTMP stream URL.
        Direct protocol access - NO API KEYS
        """
        logger.info(f"Validating {stream_info.stream_type.value} stream: {stream_info.url}")
        
        # For RTSP/RTMP, the URL is used directly by FFmpeg
        # We just validate the URL format
        url_lower = stream_info.url.lower()
        
        if stream_info.stream_type == StreamType.RTSP:
            if not url_lower.startswith('rtsp://'):
                raise ValueError("Invalid RTSP URL - must start with rtsp://")
        
        elif stream_info.stream_type == StreamType.RTMP:
            if not url_lower.startswith('rtmp://'):
                raise ValueError("Invalid RTMP URL - must start with rtmp://")
        
        # Direct URL usage - no extraction needed
        stream_info.video_url = stream_info.url
        
        # Basic metadata (can be enhanced with FFprobe)
        stream_info.title = f"{stream_info.stream_type.value.upper()} Stream"
        
        logger.success(f"{stream_info.stream_type.value.upper()} stream validated")
        return stream_info
    
    async def _validate_http_stream(self, stream_info: StreamInfo) -> StreamInfo:
        """
        Validate HTTP progressive stream URL.
        Direct HTTP access - NO API KEYS
        """
        logger.info(f"Validating HTTP stream: {stream_info.url}")
        
        url_lower = stream_info.url.lower()
        
        # Accept common streaming formats
        valid_extensions = ['.m3u8', '.mpd', '.mp4', '.webm', '.flv']
        if not any(ext in url_lower for ext in valid_extensions):
            logger.warning(f"URL may not be a valid stream format: {stream_info.url}")
        
        stream_info.video_url = stream_info.url
        stream_info.title = "HTTP Stream"
        
        logger.success("HTTP stream validated")
        return stream_info
    
    async def _validate_audio_stream(self, stream_info: StreamInfo) -> StreamInfo:
        """
        Validate audio-only stream URL.
        Direct access - NO API KEYS
        """
        logger.info(f"Validating audio stream: {stream_info.url}")
        
        url_lower = stream_info.url.lower()
        
        # Accept common audio formats
        valid_extensions = ['.mp3', '.m4a', '.aac', '.opus', '.ogg', '.wav', '.flac']
        if not any(ext in url_lower for ext in valid_extensions):
            logger.warning(f"URL may not be a valid audio format: {stream_info.url}")
        
        stream_info.audio_url = stream_info.url
        stream_info.title = "Audio Stream"
        
        logger.success("Audio stream validated")
        return stream_info
    
    async def reconnect_stream(self, stream_id: int) -> bool:
        """
        Attempt to reconnect a failed stream.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        if stream_id not in self.active_streams:
            logger.error(f"Stream {stream_id} not found in active streams")
            return False
        
        stream_info = self.active_streams[stream_id]
        
        if stream_info.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnect attempts reached for stream {stream_id}")
            stream_info.status = StreamStatus.ERROR
            return False
        
        stream_info.reconnect_attempts += 1
        stream_info.status = StreamStatus.RECONNECTING
        
        logger.info(
            f"Reconnecting stream {stream_id} "
            f"(attempt {stream_info.reconnect_attempts}/{self.max_reconnect_attempts})"
        )
        
        try:
            # Wait before reconnecting
            await asyncio.sleep(self.reconnect_delay)
            
            # Re-extract stream information
            updated_info = await self.add_stream(
                stream_id=stream_id,
                url=stream_info.url,
                stream_type=stream_info.stream_type
            )
            
            # Reset reconnect counter on success
            updated_info.reconnect_attempts = 0
            updated_info.status = StreamStatus.CONNECTED
            
            logger.success(f"Stream {stream_id} reconnected successfully")
            return True
        
        except Exception as e:
            logger.error(f"Reconnection failed for stream {stream_id}: {e}")
            stream_info.error_message = str(e)
            return False
    
    def get_stream_info(self, stream_id: int) -> Optional[StreamInfo]:
        """Get stream information by ID"""
        return self.active_streams.get(stream_id)
    
    def remove_stream(self, stream_id: int) -> bool:
        """Remove a stream from active streams"""
        if stream_id in self.active_streams:
            stream_info = self.active_streams.pop(stream_id)
            stream_info.status = StreamStatus.STOPPED
            logger.info(f"Stream {stream_id} removed")
            return True
        return False
    
    def get_active_streams(self) -> List[StreamInfo]:
        """Get list of all active streams"""
        return list(self.active_streams.values())
    
    def get_stream_count(self) -> int:
        """Get count of active streams"""
        return len(self.active_streams)


# Global stream manager instance
stream_manager = StreamManager()
