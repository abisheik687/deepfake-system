
import os
import glob
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path

# KAVACH-AI Day 3: Audio Extraction
# Extracts .wav files from video datasets for training

def extract_audio(video_path, output_root):
    try:
        filename = Path(video_path).stem
        # Preserve directory structure relative to input root? 
        # For now, just dump into output_root/filename.wav or flat structure?
        # Let's use a flat structure with prefixes or subfolders.
        
        # Decide output path
        # Assuming input is like data/raw/manipulated_sequences/.../video.mp4
        # We want data/processed/audio/manipulated_sequences/.../video.wav
        
        # Simplified: Output root + filename.wav (potential collisions, better to use subfolders)
        save_path = os.path.join(output_root, f"{filename}.wav")
        
        if os.path.exists(save_path):
            return

        # FFmpeg command: Extract audio, convert to 16kHz mono wav
        cmd = [
            'ffmpeg',
            '-y',                 # Overwrite output
            '-i', video_path,     # Input
            '-vn',                # No video
            '-acodec', 'pcm_s16le', # 16-bit PCM
            '-ar', '16000',       # 16kHz sampling rate
            '-ac', '1',           # Mono
            '-loglevel', 'error', # Quiet
            save_path
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error processing {video_path}: {e}")
    except Exception as e:
        print(f"Unexpected error on {video_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract audio from videos")
    parser.add_argument("--input", required=True, help="Input directory containing MP4s")
    parser.add_argument("--output", required=True, help="Output directory for WAV files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # Recursive search for videos
    videos = glob.glob(os.path.join(args.input, "**/*.mp4"), recursive=True)
    print(f"Found {len(videos)} videos in {args.input}")
    
    if len(videos) == 0:
        print("No videos found! Please check input path.")
        return

    # Process in parallel? FFmpeg is fast, sequential is fine for now.
    for video in tqdm(videos):
        extract_audio(video, args.output)

if __name__ == "__main__":
    main()
