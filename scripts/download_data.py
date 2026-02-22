
import os
import requests
from tqdm import tqdm
import argparse

# KAVACH-AI Data Acquisition Script
# Downloads subsets of FaceForensics++ or similar datasets for training
# Usage: python download_data.py --dataset faceforensics --count 500

DATA_DIR = "data/raw"

def download_file(url, folder):
    local_filename = url.split('/')[-1]
    path = os.path.join(folder, local_filename)
    
    if os.path.exists(path):
        print(f"Skipping {local_filename}, already exists.")
        return path

    print(f"Downloading {local_filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    return path

def setup_faceforensics_structure():
    # FaceForensics++ structure
    paths = [
        os.path.join(DATA_DIR, "manipulated_sequences/Deepfakes/c23/videos"),
        os.path.join(DATA_DIR, "original_sequences/youtube/c23/videos")
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)
    print(f"Created data directories in {DATA_DIR}")

def main():
    parser = argparse.ArgumentParser(description="KAVACH-AI Data Downloader")
    parser.add_argument("--dataset", type=str, default="faceforensics", help="Dataset to download")
    args = parser.parse_args()

    setup_faceforensics_structure()
    
    print("NOTE: FaceForensics++ requires a script/access form. ")
    print("For this blitz, we will assume you have the `download_faceforensicspp.py` script or manually place videos.")
    print("Checking for local files...")
    
    # Check if files exist
    real_path = os.path.join(DATA_DIR, "original_sequences/youtube/c23/videos")
    fake_path = os.path.join(DATA_DIR, "manipulated_sequences/Deepfakes/c23/videos")
    
    real_count = len([f for f in os.listdir(real_path) if f.endswith('.mp4')]) if os.path.exists(real_path) else 0
    fake_count = len([f for f in os.listdir(fake_path) if f.endswith('.mp4')]) if os.path.exists(fake_path) else 0
    
    print(f"Status: Found {real_count} real videos, {fake_count} fake videos.")
    
    if real_count < 10 or fake_count < 10:
        print("WARNING: Insufficient training data. Please create a 'data' folder and drop MP4 files there.")

if __name__ == "__main__":
    main()
