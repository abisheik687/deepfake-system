#!/usr/bin/env python3
"""
KAVACH-AI Dataset Downloader
Automatically downloads FaceForensics++, Celeb-DF v2, WaveFake, and ASVspoof 2021
"""

import os
import sys
import argparse
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

# Dataset configurations
DATASETS = {
    'faceforensics': {
        'name': 'FaceForensics++ (c23)',
        'url': 'https://github.com/ondyari/FaceForensics',
        'size': '~500GB',
        'description': 'Video deepfake dataset with multiple manipulation methods',
        'script_url': 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/download-FaceForensics.py',
        'compression': 'c23',
        'types': ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    },
    'celebdf': {
        'name': 'Celeb-DF v2',
        'url': 'https://github.com/yuezunli/celeb-deepfakeforensics',
        'size': '~5.8GB',
        'description': 'High-quality celebrity deepfake videos',
        'download_url': 'https://www.dropbox.com/s/example/Celeb-DF-v2.zip',  # Placeholder
        'requires_agreement': True
    },
    'wavefake': {
        'name': 'WaveFake',
        'url': 'https://zenodo.org/record/5642694',
        'size': '~3GB',
        'description': 'Audio deepfake dataset with multiple vocoders',
        'zenodo_id': '5642694'
    },
    'asvspoof': {
        'name': 'ASVspoof 2021',
        'url': 'https://www.asvspoof.org/',
        'size': '~10GB',
        'description': 'Audio spoofing and deepfake detection challenge dataset',
        'requires_registration': True
    }
}


class DatasetDownloader:
    def __init__(self, output_dir: str = './data', datasets: Optional[List[str]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = datasets or ['faceforensics', 'celebdf', 'wavefake', 'asvspoof']
        
    def download_file(self, url: str, dest: Path, desc: Optional[str] = None) -> bool:
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest, 'wb') as f, tqdm(
                desc=desc or dest.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def verify_checksum(self, file_path: Path, expected_md5: str) -> bool:
        """Verify file integrity with MD5 checksum"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest() == expected_md5
    
    def download_faceforensics(self) -> bool:
        """Download FaceForensics++ dataset"""
        print("\n" + "="*70)
        print("📥 Downloading FaceForensics++ (c23)")
        print("="*70)
        
        dataset_dir = self.output_dir / 'faceforensics'
        dataset_dir.mkdir(exist_ok=True)
        
        # Download the official download script
        script_path = dataset_dir / 'download-FaceForensics.py'
        config = DATASETS['faceforensics']
        
        print(f"\n📄 Downloading official download script...")
        if not self.download_file(config['script_url'], script_path, "Download script"):
            print("⚠️  Failed to download script. Please download manually from:")
            print(f"   {config['url']}")
            return False
        
        print(f"\n✅ Download script saved to: {script_path}")
        print(f"\n⚠️  IMPORTANT: FaceForensics++ requires registration and server credentials")
        print(f"   1. Register at: {config['url']}")
        print(f"   2. Obtain server credentials")
        print(f"   3. Run the download script:")
        print(f"      python {script_path} --output_path {dataset_dir} --compression {config['compression']}")
        
        # Create a sample command file
        cmd_file = dataset_dir / 'download_command.sh'
        with open(cmd_file, 'w') as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"# FaceForensics++ Download Command\n")
            f.write(f"# Replace USERNAME and PASSWORD with your credentials\n\n")
            f.write(f"python {script_path} \\\n")
            f.write(f"  --output_path {dataset_dir} \\\n")
            f.write(f"  --compression {config['compression']} \\\n")
            f.write(f"  --type all \\\n")
            f.write(f"  --server_username USERNAME \\\n")
            f.write(f"  --server_password PASSWORD\n")
        
        os.chmod(cmd_file, 0o755)
        print(f"\n📝 Sample command saved to: {cmd_file}")
        
        return True
    
    def download_celebdf(self) -> bool:
        """Download Celeb-DF v2 dataset"""
        print("\n" + "="*70)
        print("📥 Downloading Celeb-DF v2")
        print("="*70)
        
        dataset_dir = self.output_dir / 'celebdf'
        dataset_dir.mkdir(exist_ok=True)
        
        config = DATASETS['celebdf']
        
        print(f"\n⚠️  Celeb-DF v2 requires agreement to terms of use")
        print(f"   1. Visit: {config['url']}")
        print(f"   2. Read and agree to the terms")
        print(f"   3. Download the dataset manually")
        print(f"   4. Extract to: {dataset_dir}")
        
        # Create README
        readme = dataset_dir / 'README.txt'
        with open(readme, 'w') as f:
            f.write(f"Celeb-DF v2 Dataset\n")
            f.write(f"===================\n\n")
            f.write(f"Official URL: {config['url']}\n")
            f.write(f"Size: {config['size']}\n")
            f.write(f"Description: {config['description']}\n\n")
            f.write(f"Download Instructions:\n")
            f.write(f"1. Visit the official repository\n")
            f.write(f"2. Fill out the agreement form\n")
            f.write(f"3. Download the dataset\n")
            f.write(f"4. Extract all files to this directory\n")
        
        print(f"\n📝 Instructions saved to: {readme}")
        return True
    
    def download_wavefake(self) -> bool:
        """Download WaveFake dataset from Zenodo"""
        print("\n" + "="*70)
        print("📥 Downloading WaveFake")
        print("="*70)
        
        dataset_dir = self.output_dir / 'wavefake'
        dataset_dir.mkdir(exist_ok=True)
        
        config = DATASETS['wavefake']
        zenodo_api = f"https://zenodo.org/api/records/{config['zenodo_id']}"
        
        print(f"\n🔍 Fetching dataset metadata from Zenodo...")
        
        try:
            response = requests.get(zenodo_api)
            response.raise_for_status()
            data = response.json()
            
            files = data.get('files', [])
            if not files:
                print("❌ No files found in Zenodo record")
                return False
            
            print(f"\n📦 Found {len(files)} files:")
            for i, file_info in enumerate(files, 1):
                filename = file_info['key']
                size_mb = file_info['size'] / (1024 * 1024)
                print(f"   {i}. {filename} ({size_mb:.1f} MB)")
            
            # Download all files
            for file_info in files:
                filename = file_info['key']
                download_url = file_info['links']['self']
                dest_path = dataset_dir / filename
                
                if dest_path.exists():
                    print(f"\n⏭️  Skipping {filename} (already exists)")
                    continue
                
                print(f"\n📥 Downloading {filename}...")
                if not self.download_file(download_url, dest_path, filename):
                    print(f"❌ Failed to download {filename}")
                    continue
                
                # Verify checksum if available
                if 'checksum' in file_info:
                    expected_md5 = file_info['checksum'].split(':')[-1]
                    print(f"🔐 Verifying checksum...")
                    if self.verify_checksum(dest_path, expected_md5):
                        print(f"✅ Checksum verified")
                    else:
                        print(f"❌ Checksum mismatch!")
                        return False
            
            print(f"\n✅ WaveFake dataset downloaded to: {dataset_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading WaveFake: {e}")
            print(f"   Please download manually from: {config['url']}")
            return False
    
    def download_asvspoof(self) -> bool:
        """Download ASVspoof 2021 dataset"""
        print("\n" + "="*70)
        print("📥 Downloading ASVspoof 2021")
        print("="*70)
        
        dataset_dir = self.output_dir / 'asvspoof'
        dataset_dir.mkdir(exist_ok=True)
        
        config = DATASETS['asvspoof']
        
        print(f"\n⚠️  ASVspoof 2021 requires registration")
        print(f"   1. Visit: {config['url']}")
        print(f"   2. Register for the challenge")
        print(f"   3. Download the dataset from the participant portal")
        print(f"   4. Extract to: {dataset_dir}")
        
        # Create README
        readme = dataset_dir / 'README.txt'
        with open(readme, 'w') as f:
            f.write(f"ASVspoof 2021 Dataset\n")
            f.write(f"=====================\n\n")
            f.write(f"Official URL: {config['url']}\n")
            f.write(f"Size: {config['size']}\n")
            f.write(f"Description: {config['description']}\n\n")
            f.write(f"Download Instructions:\n")
            f.write(f"1. Register at the official website\n")
            f.write(f"2. Access the participant portal\n")
            f.write(f"3. Download LA (Logical Access) and DF (Deepfake) tracks\n")
            f.write(f"4. Extract all files to this directory\n\n")
            f.write(f"Expected structure:\n")
            f.write(f"  asvspoof/\n")
            f.write(f"    LA/\n")
            f.write(f"      ASVspoof2021_LA_train/\n")
            f.write(f"      ASVspoof2021_LA_dev/\n")
            f.write(f"      ASVspoof2021_LA_eval/\n")
            f.write(f"    DF/\n")
            f.write(f"      ASVspoof2021_DF_train/\n")
            f.write(f"      ASVspoof2021_DF_dev/\n")
            f.write(f"      ASVspoof2021_DF_eval/\n")
        
        print(f"\n📝 Instructions saved to: {readme}")
        return True
    
    def run(self) -> Dict[str, bool]:
        """Run the download process for all selected datasets"""
        print("\n" + "="*70)
        print("🚀 KAVACH-AI Dataset Downloader")
        print("="*70)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"Datasets to download: {', '.join(self.datasets)}")
        
        results = {}
        
        for dataset in self.datasets:
            if dataset == 'faceforensics':
                results[dataset] = self.download_faceforensics()
            elif dataset == 'celebdf':
                results[dataset] = self.download_celebdf()
            elif dataset == 'wavefake':
                results[dataset] = self.download_wavefake()
            elif dataset == 'asvspoof':
                results[dataset] = self.download_asvspoof()
            else:
                print(f"\n⚠️  Unknown dataset: {dataset}")
                results[dataset] = False
        
        # Summary
        print("\n" + "="*70)
        print("📊 Download Summary")
        print("="*70)
        
        for dataset, success in results.items():
            status = "✅ Complete" if success else "⚠️  Manual action required"
            print(f"  {DATASETS[dataset]['name']}: {status}")
        
        print("\n" + "="*70)
        print("📁 Dataset Directory Structure:")
        print("="*70)
        print(f"\n{self.output_dir}/")
        for dataset in self.datasets:
            print(f"  ├── {dataset}/")
        
        print("\n💡 Next Steps:")
        print("   1. Complete any manual downloads as instructed above")
        print("   2. Run: python scripts/train/build_manifest.py")
        print("   3. Start training: python scripts/train/train_image.py")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for KAVACH-AI training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Datasets:
  faceforensics  - FaceForensics++ (c23) video dataset (~500GB)
  celebdf        - Celeb-DF v2 video dataset (~5.8GB)
  wavefake       - WaveFake audio dataset (~3GB)
  asvspoof       - ASVspoof 2021 audio dataset (~10GB)

Examples:
  # Download all datasets
  python download_datasets.py

  # Download specific datasets
  python download_datasets.py --datasets wavefake asvspoof

  # Specify output directory
  python download_datasets.py --output ./datasets
        """
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./data',
        help='Output directory for datasets (default: ./data)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['faceforensics', 'celebdf', 'wavefake', 'asvspoof'],
        help='Specific datasets to download (default: all)'
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(
        output_dir=args.output,
        datasets=args.datasets
    )
    
    results = downloader.run()
    
    # Exit with error if any download failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()

# Made with Bob
