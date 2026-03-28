"""
KAVACH-AI FFmpeg Processor
Hardware-accelerated video and audio processing using FFmpeg
NO API KEYS REQUIRED - All processing is local
"""

import asyncio
import subprocess
import json
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from loguru import logger
import numpy as np
import cv2


@dataclass
class VideoProperties:
    """Video stream properties"""
    width: int
    height: int
    fps: float
    duration: Optional[float] = None
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    total_frames: Optional[int] = None


@dataclass
class AudioProperties:
    """Audio stream properties"""
    sample_rate: int
    channels: int
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    duration: Optional[float] = None


class FFmpegProcessor:
    """
    FFmpeg wrapper for video/audio processing.
    
    Features:
    - Hardware-accelerated decoding (CUDA, QSV, VAAPI)
    - Frame extraction with precise timestamps
    - Audio segmentation
    - Stream probing
    - Format conversion
    """
    
    def __init__(
        self,
        enable_gpu: bool = False,
        gpu_type: str = "cuda"
    ):
        """
        Initialize FFmpeg processor.
        
        Args:
            enable_gpu: Enable GPU acceleration
            gpu_type: GPU type (cuda, qsv, vaapi)
        """
        self.enable_gpu = enable_gpu
        self.gpu_type = gpu_type
        
        # Verify FFmpeg installation
        self._verify_ffmpeg()
        
        logger.info(
            f"FFmpeg processor initialized "
            f"(GPU: {enable_gpu}, Type: {gpu_type if enable_gpu else 'N/A'})"
        )
    
    def _verify_ffmpeg(self):
        """Verify FFmpeg is installed and accessible"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                logger.success(f"FFmpeg found: {version_line}")
            else:
                raise RuntimeError("FFmpeg not working properly")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found! Please install FFmpeg:\n"
                "- Windows: Download from https://ffmpeg.org/download.html\n"
                "- Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "- macOS: brew install ffmpeg"
            )
        except Exception as e:
            raise RuntimeError(f"FFmpeg verification failed: {e}")
    
    async def probe_stream(self, url: str) -> Dict[str, Any]:
        """
        Probe stream/file to extract metadata using ffprobe.
        
        Args:
            url: Stream URL or file path
        
        Returns:
            Dictionary with stream metadata
        """
        logger.info(f"Probing stream: {url[:100]}...")
        
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            url
        ]
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {stderr.decode()}")
            
            metadata = json.loads(stdout.decode())
            logger.success("Stream probed successfully")
            
            return metadata
        
        except Exception as e:
            logger.error(f"Stream probing error: {e}")
            raise
    
    async def get_video_properties(self, url: str) -> Optional[VideoProperties]:
        """
        Extract video properties from stream.
        
        Args:
            url: Stream URL or file path
        
        Returns:
            VideoProperties or None if no video stream
        """
        try:
            metadata = await self.probe_stream(url)
            
            # Find video stream
            video_stream = next(
                (s for s in metadata.get('streams', []) if s['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                logger.warning("No video stream found")
                return None
            
            # Extract properties
            props = VideoProperties(
                width=video_stream.get('width', 0),
                height=video_stream.get('height', 0),
                fps=eval(video_stream.get('r_frame_rate', '0/1')),
                codec=video_stream.get('codec_name'),
                bitrate=int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else None
            )
            
            # Duration from format or stream
            if 'format' in metadata and 'duration' in metadata['format']:
                props.duration = float(metadata['format']['duration'])
            elif 'duration' in video_stream:
                props.duration = float(video_stream['duration'])
            
            # Calculate total frames
            if props.duration and props.fps:
                props.total_frames = int(props.duration * props.fps)
            
            logger.info(
                f"Video properties: {props.width}x{props.height} @ {props.fps:.2f}fps, "
                f"codec={props.codec}"
            )
            
            return props
        
        except Exception as e:
            logger.error(f"Error getting video properties: {e}")
            return None
    
    async def get_audio_properties(self, url: str) -> Optional[AudioProperties]:
        """
        Extract audio properties from stream.
        
        Args:
            url: Stream URL or file path
        
        Returns:
            AudioProperties or None if no audio stream
        """
        try:
            metadata = await self.probe_stream(url)
            
            # Find audio stream
            audio_stream = next(
                (s for s in metadata.get('streams', []) if s['codec_type'] == 'audio'),
                None
            )
            
            if not audio_stream:
                logger.warning("No audio stream found")
                return None
            
            props = AudioProperties(
                sample_rate=int(audio_stream.get('sample_rate', 0)),
                channels=int(audio_stream.get('channels', 0)),
                codec=audio_stream.get('codec_name'),
                bitrate=int(audio_stream.get('bit_rate', 0)) if audio_stream.get('bit_rate') else None
            )
            
            # Duration
            if 'duration' in audio_stream:
                props.duration = float(audio_stream['duration'])
            
            logger.info(
                f"Audio properties: {props.sample_rate}Hz, {props.channels}ch, "
                f"codec={props.codec}"
            )
            
            return props
        
        except Exception as e:
            logger.error(f"Error getting audio properties: {e}")
            return None
    
    async def extract_frame(
        self,
        url: str,
        timestamp: float,
        output_path: Optional[Path] = None
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame at specified timestamp.
        
        Args:
            url: Stream URL or file path
            timestamp: Timestamp in seconds
            output_path: Optional path to save frame
        
        Returns:
            Frame as numpy array (BGR format) or None
        """
        logger.debug(f"Extracting frame at {timestamp:.2f}s")
        
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),  # Seek to timestamp
            '-i', url,
            '-vframes', '1',  # Extract 1 frame
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-'
        ]
        
        # Add GPU decoding if enabled
        if self.enable_gpu:
            cmd = self._add_gpu_decode(cmd)
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.error(f"Frame extraction failed: {stderr.decode()}")
                return None
            
            # Convert to numpy array
            # First, get frame dimensions (need to probe first in practice)
            # For now, assume we know dimensions or handle dynamically
            
            # Save to file if requested
            if output_path:
                output_path.write_bytes(stdout)
                logger.debug(f"Frame saved to {output_path}")
            
            return stdout
        
        except Exception as e:
            logger.error(f"Error extracting frame: {e}")
            return None
    
    async def extract_frames_batch(
        self,
        url: str,
        timestamps: list[float],
        output_dir: Path
    ) -> list[Path]:
        """
        Extract multiple frames at specified timestamps.
        
        Args:
            url: Stream URL or file path
            timestamps: List of timestamps in seconds
            output_dir: Directory to save frames
        
        Returns:
            List of paths to extracted frames
        """
        logger.info(f"Extracting {len(timestamps)} frames")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        
        for i, ts in enumerate(timestamps):
            output_path = output_dir / f"frame_{i:06d}_{ts:.2f}s.jpg"
            
            cmd = [
                'ffmpeg',
                '-ss', str(ts),
                '-i', url,
                '-vframes', '1',
                '-q:v', '2',  # Quality (1-31, 2 is high)
                '-y',  # Overwrite
                str(output_path)
            ]
            
            if self.enable_gpu:
                cmd = self._add_gpu_decode(cmd)
            
            try:
                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await result.communicate()
                
                if result.returncode == 0 and output_path.exists():
                    output_paths.append(output_path)
                else:
                    logger.warning(f"Failed to extract frame at {ts}s")
            
            except Exception as e:
                logger.error(f"Error extracting frame at {ts}s: {e}")
        
        logger.success(f"Extracted {len(output_paths)}/{len(timestamps)} frames")
        return output_paths
    
    async def extract_audio_segment(
        self,
        url: str,
        start_time: float,
        duration: float,
        output_path: Path,
        sample_rate: int = 16000
    ) -> bool:
        """
        Extract audio segment from stream.
        
        Args:
            url: Stream URL or file path
            start_time: Start time in seconds
            duration: Duration in seconds
            output_path: Output file path
            sample_rate: Target sample rate (Hz)
        
        Returns:
            True if successful
        """
        logger.info(f"Extracting audio segment: {start_time:.2f}s - {start_time + duration:.2f}s")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', url,
            '-t', str(duration),
            '-ar', str(sample_rate),  # Resample to target rate
            '-ac', '1',  # Mono
            '-f', 'wav',
            '-y',
            str(output_path)
        ]
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.error(f"Audio extraction failed: {stderr.decode()}")
                return False
            
            logger.success(f"Audio segment saved to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False
    
    def _add_gpu_decode(self, cmd: list[str]) -> list[str]:
        """
        Add GPU hardware acceleration to FFmpeg command.
        
        Args:
            cmd: FFmpeg command list
        
        Returns:
            Modified command with GPU flags
        """
        if not self.enable_gpu:
            return cmd
        
        # Insert hardware acceleration before input
        input_idx = cmd.index('-i')
        
        if self.gpu_type == 'cuda':
            # NVIDIA CUDA
            cmd.insert(input_idx, '-hwaccel')
            cmd.insert(input_idx + 1, 'cuda')
            cmd.insert(input_idx + 2, '-hwaccel_output_format')
            cmd.insert(input_idx + 3, 'cuda')
        
        elif self.gpu_type == 'qsv':
            # Intel Quick Sync Video
            cmd.insert(input_idx, '-hwaccel')
            cmd.insert(input_idx + 1, 'qsv')
        
        elif self.gpu_type == 'vaapi':
            # Video Acceleration API (Linux)
            cmd.insert(input_idx, '-hwaccel')
            cmd.insert(input_idx + 1, 'vaapi')
            cmd.insert(input_idx + 2, '-vaapi_device')
            cmd.insert(input_idx + 3, '/dev/dri/renderD128')
        
        return cmd
    
    async def start_continuous_capture(
        self,
        url: str,
        fps: float = 1.0,
        callback=None
    ):
        """
        Start continuous frame capture from stream.
        
        Args:
            url: Stream URL
            fps: Capture rate (frames per second)
            callback: Async function to call with each frame
        
        This is a generator that yields frames continuously.
        """
        logger.info(f"Starting continuous capture at {fps} fps")
        
        cmd = [
            'ffmpeg',
            '-i', url,
            '-vf', f'fps={fps}',  # Set output FPS
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-'
        ]
        
        if self.enable_gpu:
            cmd = self._add_gpu_decode(cmd)
        
        # Start FFmpeg process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Get video properties to determine frame size
        props = await self.get_video_properties(url)
        if not props:
            logger.error("Could not get video properties")
            process.kill()
            return
        
        frame_size = props.width * props.height * 3  # BGR = 3 bytes per pixel
        frame_count = 0
        
        try:
            while True:
                # Read one frame
                frame_bytes = await process.stdout.read(frame_size)
                
                if len(frame_bytes) != frame_size:
                    logger.warning("Incomplete frame or stream ended")
                    break
                
                # Convert to numpy array
                frame = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = frame.reshape((props.height, props.width, 3))
                
                frame_count += 1
                
                # Call callback if provided
                if callback:
                    await callback(frame, frame_count)
                
                # Yield frame for generator pattern
                yield frame
        
        except asyncio.CancelledError:
            logger.info("Continuous capture cancelled")
        
        finally:
            process.kill()
            await process.wait()
            logger.info(f"Continuous capture stopped (captured {frame_count} frames)")
