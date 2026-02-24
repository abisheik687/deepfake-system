"""
DeepShield AI — Face Extraction Preprocessing

Extracts face crops from dataset videos/images, saves to disk.
Must be run BEFORE training.

Usage:
    python training/preprocessing/extract_faces.py \
        --dataset celeb_df \
        --input  path/to/raw_videos \
        --output data/datasets/celeb_df \
        --fps    4.0

Output structure:
    data/datasets/celeb_df/
    ├── real/
    │   ├── video1_frame0042.jpg
    │   └── ...
    └── fake/
        ├── manipulated1_frame0010.jpg
        └── ...
"""

import os
import sys
import argparse
import cv2
from pathlib import Path
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.detection.face_extractor import FaceExtractor

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_faces_from_video(
    video_path: Path,
    output_dir: Path,
    extractor: FaceExtractor,
    sample_fps: float = 4.0,
    min_face_size: int = 80,
    quality: int = 90,
) -> int:
    """Extract face crops from a video file. Returns count of saved crops."""
    cap     = cv2.VideoCapture(str(video_path))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step    = max(1, int(vid_fps / sample_fps))
    count   = 0
    frame_i = 0
    stem    = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i % step == 0:
            faces = extractor.detect(frame)
            h_f, w_f = frame.shape[:2]
            for (x, y, w, h) in faces:
                if w < min_face_size or h < min_face_size:
                    continue
                # 10% padding
                px = int(w * 0.1); py = int(h * 0.1)
                x1 = max(0, x-px); y1 = max(0, y-py)
                x2 = min(w_f, x+w+px); y2 = min(h_f, y+h+py)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                fname = output_dir / f"{stem}_f{frame_i:06d}_face{count%10}.jpg"
                cv2.imwrite(str(fname), crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
                count += 1
        frame_i += 1

    cap.release()
    return count


def extract_faces_from_image(
    image_path: Path,
    output_dir: Path,
    extractor: FaceExtractor,
    min_face_size: int = 80,
) -> int:
    """Extract face crops from a single image."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        return 0
    faces = extractor.detect(frame)
    h_f, w_f = frame.shape[:2]
    count = 0
    for (x, y, w, h) in faces:
        if w < min_face_size or h < min_face_size:
            continue
        crop = frame[max(0,y-10):min(h_f,y+h+10), max(0,x-10):min(w_f,x+w+10)]
        if crop.size == 0:
            continue
        fname = output_dir / f"{image_path.stem}_face{count}.jpg"
        cv2.imwrite(str(fname), crop)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Extract face crops from deepfake datasets")
    parser.add_argument("--dataset", required=True, help="Dataset name (for logging)")
    parser.add_argument("--input",   required=True, help="Input directory with real/ and fake/ subdirs (or flat structure)")
    parser.add_argument("--output",  required=True, help="Output directory for face crops")
    parser.add_argument("--fps",     type=float, default=4.0, help="Video sampling FPS (default 4)")
    parser.add_argument("--workers", type=int,   default=1,   help="Parallel workers (default 1)")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    extractor = FaceExtractor(min_confidence=0.6)
    logger.info(f"Processing dataset: {args.dataset}")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")

    total_crops = 0

    for label in ["real", "fake"]:
        src = input_dir / label
        dst = output_dir / label
        dst.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            logger.warning(f"Skipping missing: {src}")
            continue

        # Collect all files
        files = [f for f in src.rglob("*") if f.is_file()]
        logger.info(f"[{label}] Found {len(files)} files")

        for f in tqdm(files, desc=f"[{label}]"):
            if f.suffix.lower() in SUPPORTED_EXTS:
                n = extract_faces_from_video(f, dst, extractor, sample_fps=args.fps)
            elif f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                n = extract_faces_from_image(f, dst, extractor)
            else:
                continue
            total_crops += n

    logger.success(f"Done! Extracted {total_crops} face crops to {output_dir}")
    print(f"\n✅ Extracted {total_crops} face crops")
    print(f"   real/  → {len(list((output_dir/'real').glob('*.jpg')))} images")
    print(f"   fake/  → {len(list((output_dir/'fake').glob('*.jpg')))} images")
    print(f"\nNext: python training/train.py --dataset {args.dataset} --data_dir {output_dir.parent}")


if __name__ == "__main__":
    main()
