
import os
import cv2
import glob
import argparse
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from pathlib import Path

# KAVACH-AI Day 2: Face Extraction
# Extracts face crops from video datasets for training

def get_face_detector():
    mp_face_detection = mp.solutions.face_detection
    return mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def extract_faces_from_video(video_path, output_root, detector, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    filename = Path(video_path).stem
    
    # Organize output by original video name
    save_dir = os.path.join(output_root, filename)
    os.makedirs(save_dir, exist_ok=True)
    
    count = 0
    frame_idx = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames) if total_frames > 0 else 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % step == 0 and count < max_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)
            
            if results.detections:
                for idx, detection in enumerate(results.detections):
                    h, w, c = frame.shape
                    bboxC = detection.location_data.relative_bounding_box
                    
                    # Expand bbox slightly for context
                    margin = 0.2
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    box_w = int(bboxC.width * w)
                    box_h = int(bboxC.height * h)
                    
                    # Apply margin
                    x1 = max(0, x - int(box_w * margin))
                    y1 = max(0, y - int(box_h * margin))
                    x2 = min(w, x + box_w + int(box_w * margin))
                    y2 = min(h, y + box_h + int(box_h * margin))
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        save_path = os.path.join(save_dir, f"frame_{frame_idx}_face_{idx}.jpg")
                        cv2.imwrite(save_path, face_crop)
                        count += 1
                        
        frame_idx += 1
    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Extract faces from videos")
    parser.add_argument("--input", required=True, help="Input directory containing MP4s")
    parser.add_argument("--output", required=True, help="Output directory for face crops")
    parser.add_argument("--count", type=int, default=30, help="Frames per video to extract")
    args = parser.parse_args()

    detector = get_face_detector()
    
    # Recursive search for videos
    videos = glob.glob(os.path.join(args.input, "**/*.mp4"), recursive=True)
    print(f"Found {len(videos)} videos in {args.input}")
    
    if len(videos) == 0:
        print("No videos found! Please check input path.")
        return

    for video in tqdm(videos):
        extract_faces_from_video(video, args.output, detector, max_frames=args.count)

if __name__ == "__main__":
    main()
