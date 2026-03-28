#!/usr/bin/env python3
"""
KAVACH-AI Manifest Builder
Creates train/val/test CSV splits from downloaded datasets
Stratified by fake method for balanced training
"""

import os
import sys
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random
from tqdm import tqdm

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Supported file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


class ManifestBuilder:
    def __init__(self, data_dir: str, output_dir: str, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)
        
        self.samples = {
            'image': [],
            'video': [],
            'audio': []
        }
        
    def scan_faceforensics(self) -> List[Dict]:
        """Scan FaceForensics++ dataset"""
        print("\n📂 Scanning FaceForensics++...")
        
        ff_dir = self.data_dir / 'faceforensics'
        if not ff_dir.exists():
            print(f"⚠️  FaceForensics++ directory not found: {ff_dir}")
            return []
        
        samples = []
        manipulation_types = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        
        for manip_type in manipulation_types:
            manip_dir = ff_dir / manip_type
            if not manip_dir.exists():
                continue
            
            label = 0 if manip_type == 'original' else 1
            method = 'real' if manip_type == 'original' else manip_type.lower()
            
            # Scan for videos
            video_files = []
            for ext in VIDEO_EXTENSIONS:
                video_files.extend(manip_dir.rglob(f'*{ext}'))
            
            for video_path in video_files:
                samples.append({
                    'path': str(video_path.relative_to(self.data_dir)),
                    'label': label,
                    'method': method,
                    'dataset': 'faceforensics',
                    'type': 'video'
                })
        
        print(f"   Found {len(samples)} samples")
        return samples
    
    def scan_celebdf(self) -> List[Dict]:
        """Scan Celeb-DF v2 dataset"""
        print("\n📂 Scanning Celeb-DF v2...")
        
        celebdf_dir = self.data_dir / 'celebdf'
        if not celebdf_dir.exists():
            print(f"⚠️  Celeb-DF directory not found: {celebdf_dir}")
            return []
        
        samples = []
        
        # Celeb-DF structure: Celeb-real/ and Celeb-synthesis/
        real_dir = celebdf_dir / 'Celeb-real'
        fake_dir = celebdf_dir / 'Celeb-synthesis'
        
        # Scan real videos
        if real_dir.exists():
            for ext in VIDEO_EXTENSIONS:
                for video_path in real_dir.rglob(f'*{ext}'):
                    samples.append({
                        'path': str(video_path.relative_to(self.data_dir)),
                        'label': 0,
                        'method': 'real',
                        'dataset': 'celebdf',
                        'type': 'video'
                    })
        
        # Scan fake videos
        if fake_dir.exists():
            for ext in VIDEO_EXTENSIONS:
                for video_path in fake_dir.rglob(f'*{ext}'):
                    samples.append({
                        'path': str(video_path.relative_to(self.data_dir)),
                        'label': 1,
                        'method': 'deepfake',
                        'dataset': 'celebdf',
                        'type': 'video'
                    })
        
        print(f"   Found {len(samples)} samples")
        return samples
    
    def scan_wavefake(self) -> List[Dict]:
        """Scan WaveFake dataset"""
        print("\n📂 Scanning WaveFake...")
        
        wavefake_dir = self.data_dir / 'wavefake'
        if not wavefake_dir.exists():
            print(f"⚠️  WaveFake directory not found: {wavefake_dir}")
            return []
        
        samples = []
        
        # WaveFake structure: real/ and fake/ subdirectories
        for label, subdir in [(0, 'real'), (1, 'fake')]:
            audio_dir = wavefake_dir / subdir
            if not audio_dir.exists():
                continue
            
            for ext in AUDIO_EXTENSIONS:
                for audio_path in audio_dir.rglob(f'*{ext}'):
                    # Extract vocoder method from filename if available
                    method = 'real' if label == 0 else 'tts_vocoder'
                    if label == 1:
                        # WaveFake uses different vocoders (melgan, hifiGAN, etc.)
                        filename = audio_path.stem.lower()
                        if 'melgan' in filename:
                            method = 'melgan'
                        elif 'hifigan' in filename:
                            method = 'hifigan'
                        elif 'waveglow' in filename:
                            method = 'waveglow'
                    
                    samples.append({
                        'path': str(audio_path.relative_to(self.data_dir)),
                        'label': label,
                        'method': method,
                        'dataset': 'wavefake',
                        'type': 'audio'
                    })
        
        print(f"   Found {len(samples)} samples")
        return samples
    
    def scan_asvspoof(self) -> List[Dict]:
        """Scan ASVspoof 2021 dataset"""
        print("\n📂 Scanning ASVspoof 2021...")
        
        asvspoof_dir = self.data_dir / 'asvspoof'
        if not asvspoof_dir.exists():
            print(f"⚠️  ASVspoof directory not found: {asvspoof_dir}")
            return []
        
        samples = []
        
        # ASVspoof has LA (Logical Access) and DF (Deepfake) tracks
        for track in ['LA', 'DF']:
            track_dir = asvspoof_dir / track
            if not track_dir.exists():
                continue
            
            # Each track has train/dev/eval splits with protocol files
            for split in ['train', 'dev', 'eval']:
                split_dir = track_dir / f'ASVspoof2021_{track}_{split}'
                protocol_file = split_dir / f'ASVspoof2021.{track}.cm.{split}.trl.txt'
                
                if not protocol_file.exists():
                    continue
                
                # Parse protocol file
                with open(protocol_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        speaker_id, filename, _, attack_type, label_str = parts[:5]
                        label = 0 if label_str == 'bonafide' else 1
                        method = 'real' if label == 0 else attack_type
                        
                        audio_path = split_dir / 'flac' / f'{filename}.flac'
                        if audio_path.exists():
                            samples.append({
                                'path': str(audio_path.relative_to(self.data_dir)),
                                'label': label,
                                'method': method,
                                'dataset': f'asvspoof_{track.lower()}',
                                'type': 'audio',
                                'split': split  # Pre-defined split
                            })
        
        print(f"   Found {len(samples)} samples")
        return samples
    
    def stratified_split(self, samples: List[Dict], by_method: bool = True) -> Tuple[List, List, List]:
        """Split samples into train/val/test with stratification"""
        
        # Group by method for stratification
        if by_method:
            groups = defaultdict(list)
            for sample in samples:
                groups[sample['method']].append(sample)
        else:
            groups = {'all': samples}
        
        train, val, test = [], [], []
        
        for method, method_samples in groups.items():
            random.shuffle(method_samples)
            n = len(method_samples)
            
            train_end = int(n * TRAIN_RATIO)
            val_end = train_end + int(n * VAL_RATIO)
            
            train.extend(method_samples[:train_end])
            val.extend(method_samples[train_end:val_end])
            test.extend(method_samples[val_end:])
        
        # Shuffle final splits
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        
        return train, val, test
    
    def save_manifest(self, samples: List[Dict], split_name: str, data_type: str):
        """Save manifest to CSV file"""
        output_file = self.output_dir / f'{data_type}_{split_name}.csv'
        
        fieldnames = ['path', 'label', 'method', 'dataset', 'type']
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample in samples:
                writer.writerow({k: sample[k] for k in fieldnames if k in sample})
        
        print(f"   ✅ Saved {len(samples)} samples to {output_file}")
    
    def save_statistics(self, stats: Dict):
        """Save dataset statistics to JSON"""
        output_file = self.output_dir / 'dataset_stats.json'
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n📊 Statistics saved to {output_file}")
    
    def build(self):
        """Build complete manifest"""
        print("\n" + "="*70)
        print("🔨 Building Dataset Manifest")
        print("="*70)
        print(f"\nData directory: {self.data_dir.absolute()}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Random seed: {self.seed}")
        
        # Scan all datasets
        all_samples = []
        all_samples.extend(self.scan_faceforensics())
        all_samples.extend(self.scan_celebdf())
        all_samples.extend(self.scan_wavefake())
        all_samples.extend(self.scan_asvspoof())
        
        if not all_samples:
            print("\n❌ No samples found! Please download datasets first.")
            sys.exit(1)
        
        # Group by type
        by_type = defaultdict(list)
        for sample in all_samples:
            by_type[sample['type']].append(sample)
        
        print("\n" + "="*70)
        print("📊 Dataset Statistics")
        print("="*70)
        
        stats = {
            'total_samples': len(all_samples),
            'by_type': {},
            'by_dataset': defaultdict(int),
            'by_label': defaultdict(int),
            'by_method': defaultdict(int)
        }
        
        # Process each type
        for data_type, samples in by_type.items():
            print(f"\n{data_type.upper()} Samples: {len(samples)}")
            
            # Check if ASVspoof has pre-defined splits
            has_predefined_split = any('split' in s for s in samples)
            
            if has_predefined_split and data_type == 'audio':
                # Use pre-defined splits for ASVspoof
                train = [s for s in samples if s.get('split') == 'train']
                val = [s for s in samples if s.get('split') == 'dev']
                test = [s for s in samples if s.get('split') == 'eval']
                
                # Add remaining samples without pre-defined split
                remaining = [s for s in samples if 'split' not in s]
                if remaining:
                    r_train, r_val, r_test = self.stratified_split(remaining)
                    train.extend(r_train)
                    val.extend(r_val)
                    test.extend(r_test)
            else:
                # Stratified split
                train, val, test = self.stratified_split(samples)
            
            print(f"   Train: {len(train)} ({len(train)/len(samples)*100:.1f}%)")
            print(f"   Val:   {len(val)} ({len(val)/len(samples)*100:.1f}%)")
            print(f"   Test:  {len(test)} ({len(test)/len(samples)*100:.1f}%)")
            
            # Save manifests
            self.save_manifest(train, 'train', data_type)
            self.save_manifest(val, 'val', data_type)
            self.save_manifest(test, 'test', data_type)
            
            # Update statistics
            stats['by_type'][data_type] = {
                'total': len(samples),
                'train': len(train),
                'val': len(val),
                'test': len(test)
            }
            
            for sample in samples:
                stats['by_dataset'][sample['dataset']] += 1
                stats['by_label'][sample['label']] += 1
                stats['by_method'][sample['method']] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats['by_dataset'] = dict(stats['by_dataset'])
        stats['by_label'] = dict(stats['by_label'])
        stats['by_method'] = dict(stats['by_method'])
        
        self.save_statistics(stats)
        
        print("\n" + "="*70)
        print("✅ Manifest Build Complete")
        print("="*70)
        print(f"\nTotal samples: {stats['total_samples']}")
        print(f"Real samples: {stats['by_label'].get(0, 0)}")
        print(f"Fake samples: {stats['by_label'].get(1, 0)}")
        
        print("\n💡 Next Steps:")
        print("   1. Review the generated CSV files in the output directory")
        print("   2. Start training: python scripts/train/train_image.py")
        print("   3. Or train audio: python scripts/train/train_audio.py")


def main():
    parser = argparse.ArgumentParser(
        description='Build train/val/test manifest from downloaded datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory containing downloaded datasets (default: ./data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./manifests',
        help='Output directory for manifest files (default: ./manifests)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    
    args = parser.parse_args()
    
    builder = ManifestBuilder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    builder.build()


if __name__ == '__main__':
    main()

# Made with Bob
