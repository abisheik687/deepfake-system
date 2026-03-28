
import cv2
import yt_dlp
import time
import threading
import queue
import numpy as np
from collections import deque

class StreamLoader:
    """
    KAVACH-AI Day 4: Stream Ingestion
    Handles connecting to YouTube Live or RTSP streams and buffering frames.
    """
    def __init__(self, source_url, buffer_size=128, sample_rate=1):
        self.source_url = source_url
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate # Process 1 frame every N frames
        
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self.stream_url = None
        self.fps = 0
        self.width = 0
        self.height = 0
        
        # Thread for reading
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        """Resolve content URL and start reading thread"""
        print(f"Resolving URL: {self.source_url}...")
        try:
            if "youtube.com" in self.source_url or "youtu.be" in self.source_url:
                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'quiet': True,
                    'no_warnings': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.source_url, download=False)
                    self.stream_url = info['url']
                    print("YouTube URL resolved.")
            else:
                self.stream_url = self.source_url
                
            # Open capture
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                raise ValueError("Could not open stream.")
                
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Stream started: {self.width}x{self.height} @ {self.fps} FPS")
            self.thread.start()
            return self
            
        except Exception as e:
            print(f"Error starting stream: {e}")
            self.stopped = True
            return self

    def _update(self):
        """Background thread to read frames"""
        count = 0
        while not self.stopped:
            if not self.cap.isOpened():
                self.stop()
                break
                
            if self.frame_queue.full():
                time.sleep(0.01)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                print("Stream ended or failed to read.")
                self.stop()
                break
                
            # Adaptive sampling logic could go here
            # For now, simple decimation
            count += 1
            if count % self.sample_rate == 0:
                self.frame_queue.put(frame)
                count = 0

    def read(self):
        """Get next frame from buffer"""
        if self.stopped and self.frame_queue.empty():
            return None
            
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
    def running(self):
        return not self.stopped or not self.frame_queue.empty()
