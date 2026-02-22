
import os
import zipfile
import shutil
import requests
from tqdm import tqdm

FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
INSTALL_DIR = "tools"

def download_ffmpeg():
    if os.path.exists(os.path.join(INSTALL_DIR, "ffmpeg/bin/ffmpeg.exe")):
        print("FFmpeg already installed.")
        return

    os.makedirs(INSTALL_DIR, exist_ok=True)
    zip_path = os.path.join(INSTALL_DIR, "ffmpeg.zip")
    
    print(f"Downloading FFmpeg from {FFMPEG_URL}...")
    with requests.get(FFMPEG_URL, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print("Extracting FFmpeg...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(INSTALL_DIR)
    
    # Rename folder to just 'ffmpeg'
    extracted_folder = [d for d in os.listdir(INSTALL_DIR) if d.startswith("ffmpeg-master")][0]
    source = os.path.join(INSTALL_DIR, extracted_folder)
    destination = os.path.join(INSTALL_DIR, "ffmpeg")
    
    if os.path.exists(destination):
        shutil.rmtree(destination)
        
    shutil.move(source, destination)
    
    os.remove(zip_path)
    print("FFmpeg installed to tools/ffmpeg/bin/")
    print("Please add this to your System PATH or use absolute paths.")

if __name__ == "__main__":
    download_ffmpeg()
