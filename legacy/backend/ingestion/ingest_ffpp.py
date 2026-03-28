import os
import argparse
from pathlib import Path
import cv2
from loguru import logger
from backend.ml.mtcnn_extractor import FaceExtractor
from backend.ml.manifest_builder import ManifestBuilder

def ingest_ffpp(raw_dir: str, processed_dir: str, limit: int = None):
    """
    Ingests FaceForensics++ data by extracting faces and building a manifest.
    """
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    
    extractor = FaceExtractor()
    
    categories = ["real", "fake"]
    
    for cat in categories:
        cat_raw = raw_path / cat
        cat_proc = proc_path / cat
        cat_proc.mkdir(parents=True, exist_ok=True)
        
        if not cat_raw.exists():
            logger.warning(f"Category directory {cat_raw} does not exist. Skipping.")
            continue
            
        videos = list(cat_raw.glob("*.mp4"))
        if limit:
            videos = videos[:limit]
            
        logger.info(f"Processing {len(videos)} videos in {cat}...")
        
        for vid in videos:
            cap = cv2.VideoCapture(str(vid))
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract faces from frame
                faces = extractor.mtcnn(frame) # Simplified call for demo
                
                if faces is not None:
                    for i, face in enumerate(faces):
                        face_name = f"{vid.stem}_f{count}_{i}.jpg"
                        extractor.save_face(face, cat_proc / face_name)
                
                count += 1
                # Sample every 10 frames for dataset diversity
                cap.set(cv2.POS_FRAMES, count * 10)
                
            cap.release()
            
    # Build manifest after extraction
    builder = ManifestBuilder(proc_path)
    builder.build_from_directory()
    builder.split_manifest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAVACH-AI FF++ Ingestion")
    parser.add_argument("--raw", type=str, default="./data/raw", help="Path to raw videos")
    parser.add_argument("--proc", type=str, default="./data/processed", help="Path for processed faces")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos per category")
    args = parser.parse_args()
    
    ingest_ffpp(args.raw, args.proc, args.limit)
