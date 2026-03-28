#!/usr/bin/env python3
"""
KAVACH-AI Audio Model Trainer
Trains RawNet2 + LCNN for audio deepfake detection
"""

import os
import sys
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchaudio
import torchaudio.transforms as T

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Training configuration
CONFIG = {
    'sample_rate': 16000,
    'n_mels': 80,
    'n_fft': 512,
    'hop_length': 160,
    'batch_size': 64,
    'num_epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'gradient_accumulation': 2,
    'early_stopping_patience': 5,
    'num_workers': 4,
    'mixed_precision': True,
    'max_audio_length': 4.0  # seconds
}


class AudioDeepfakeDataset(Dataset):
    """Dataset for audio deepfake detection"""
    
    def __init__(self, manifest_path, data_dir: str, max_length: float = 4.0):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.sample_rate = CONFIG['sample_rate']
        
        # Load manifest
        self.samples = []
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['type'] == 'audio':
                    self.samples.append({
                        'path': self.data_dir / row['path'],
                        'label': int(row['label']),
                        'method': row['method']
                    })
        
        print(f"Loaded {len(self.samples)} audio samples")
        
        # Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels']
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        try:
            waveform, sr = torchaudio.load(sample['path'])
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Trim or pad to max_length
            max_samples = int(self.max_length * self.sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            elif waveform.shape[1] < max_samples:
                padding = max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return silence as fallback
            waveform = torch.zeros(1, int(self.max_length * self.sample_rate))
        
        # Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)  # Log scale
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return mel_spec, label


class RawNet2(nn.Module):
    """RawNet2 architecture for audio deepfake detection"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Sinc layer (first layer processes raw waveform)
        self.conv1 = nn.Conv1d(1, 128, kernel_size=251, stride=1, padding=125)
        self.bn1 = nn.BatchNorm1d(128)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(128, 128),
            self._make_res_block(128, 256),
            self._make_res_block(256, 256),
            self._make_res_block(256, 512),
        ])
        
        # GRU layer
        self.gru = nn.GRU(512, 256, num_layers=2, batch_first=True, bidirectional=True)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x shape: (batch, 1, time) for raw waveform
        # or (batch, n_mels, time) for mel-spectrogram
        
        if x.dim() == 4:  # (batch, 1, n_mels, time)
            x = x.squeeze(1)  # (batch, n_mels, time)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        for block in self.res_blocks:
            residual = x
            x = block(x)
            if x.shape[1] != residual.shape[1]:
                residual = nn.functional.adaptive_avg_pool1d(residual, x.shape[2])
                residual = nn.functional.pad(residual, (0, 0, 0, x.shape[1] - residual.shape[1]))
            x = x + residual
        
        # GRU
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Take last timestep
        
        # Classifier
        x = self.fc(x)
        
        return x


class LCNN(nn.Module):
    """Light CNN for audio deepfake detection"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time)
        x = self.features(x)
        x = self.classifier(x)
        return x


class AudioTrainer:
    """Audio model trainer"""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=CONFIG['num_epochs']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler() if CONFIG['mixed_precision'] else None
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]} [Train]')
        
        self.optimizer.zero_grad()
        
        for batch_idx, (spectrograms, labels) in enumerate(pbar):
            spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
            
            if self.scaler:
                with autocast():
                    outputs = self.model(spectrograms)
                    loss = self.criterion(outputs, labels)
                    loss = loss / CONFIG['gradient_accumulation']
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(spectrograms)
                loss = self.criterion(outputs, labels)
                loss = loss / CONFIG['gradient_accumulation']
                loss.backward()
                
                if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * CONFIG['gradient_accumulation']
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]} [Val]')
            
            for spectrograms, labels in pbar:
                spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
                
                outputs = self.model(spectrograms)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{total_loss/(pbar.n+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': CONFIG
        }
        
        checkpoint_path = self.output_dir / f'{self.model_name}_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, best_path)
            print(f'✅ Saved best model to {best_path}')
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*70}")
        print(f"Training {self.model_name}")
        print(f"{'='*70}\n")
        
        for epoch in range(CONFIG['num_epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            self.scheduler.step()
            
            print(f'\nEpoch {epoch+1}/{CONFIG["num_epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    f'{self.model_name}_train_loss': train_loss,
                    f'{self.model_name}_val_loss': val_loss,
                    f'{self.model_name}_val_acc': val_acc
                }, step=epoch)
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= CONFIG['early_stopping_patience']:
                print(f'\n⚠️  Early stopping at epoch {epoch+1}')
                break
        
        print(f'\n✅ Training complete!')
        print(f'   Best Val Loss: {self.best_val_loss:.4f}')
        print(f'   Best Val Acc: {max(self.val_accuracies):.2f}%')
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc': max(self.val_accuracies)
        }


def main():
    parser = argparse.ArgumentParser(description='Train audio deepfake detection models')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--manifest-dir', type=str, default='./manifests')
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--model', type=str, choices=['rawnet2', 'lcnn', 'both'], default='both')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    CONFIG['batch_size'] = args.batch_size
    CONFIG['num_epochs'] = args.epochs
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load datasets
    train_dataset = AudioDeepfakeDataset(
        manifest_path=Path(args.manifest_dir) / 'audio_train.csv',
        data_dir=args.data_dir
    )
    
    val_dataset = AudioDeepfakeDataset(
        manifest_path=Path(args.manifest_dir) / 'audio_val.csv',
        data_dir=args.data_dir
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment('kavach-ai-audio-training')
        mlflow.start_run()
        mlflow.log_params(CONFIG)
    
    results = {}
    
    if args.model in ['rawnet2', 'both']:
        model = RawNet2()
        trainer = AudioTrainer(model, 'rawnet2', train_loader, val_loader, device, args.output_dir)
        results['rawnet2'] = trainer.train()
    
    if args.model in ['lcnn', 'both']:
        model = LCNN()
        trainer = AudioTrainer(model, 'lcnn', train_loader, val_loader, device, args.output_dir)
        results['lcnn'] = trainer.train()
    
    results_file = Path(args.output_dir) / 'audio_training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if MLFLOW_AVAILABLE:
        mlflow.end_run()
    
    print(f"\n💡 Next: python scripts/train/export_onnx.py")


if __name__ == '__main__':
    main()

# Made with Bob
