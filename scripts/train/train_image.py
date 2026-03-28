#!/usr/bin/env python3
"""
KAVACH-AI Image Model Trainer
Trains EfficientNet-B4 + Xception ensemble for deepfake detection
"""

import os
import sys
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not available. Install with: pip install mlflow")

# Training configuration
CONFIG = {
    'image_size': 224,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'gradient_accumulation': 4,
    'early_stopping_patience': 5,
    'num_workers': 4,
    'mixed_precision': True,
    'save_best_only': True
}


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection"""
    
    def __init__(self, manifest_path, data_dir: str, transform=None, augment=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        
        # Load manifest
        self.samples = []
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['type'] == 'image':
                    self.samples.append({
                        'path': self.data_dir / row['path'],
                        'label': int(row['label']),
                        'method': row['method']
                    })
        
        print(f"Loaded {len(self.samples)} samples from {manifest_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size']))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return image, label


def get_transforms(augment=False):
    """Get image transforms"""
    
    if augment:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class EfficientNetB4(nn.Module):
    """EfficientNet-B4 for deepfake detection"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.efficientnet_b4(pretrained=pretrained)
        # Replace classifier
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, 2)
        )
    
    def forward(self, x):
        return self.model(x)


class XceptionNet(nn.Module):
    """Xception-like architecture for deepfake detection"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        # Use ResNet50 as base (Xception not in torchvision)
        self.model = models.resnet50(pretrained=pretrained)
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 2)
        )
    
    def forward(self, x):
        return self.model(x)


class Trainer:
    """Model trainer with early stopping and checkpointing"""
    
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
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=CONFIG['num_epochs']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision training
        self.scaler = GradScaler() if CONFIG['mixed_precision'] else None
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics
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
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Mixed precision training
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / CONFIG['gradient_accumulation']
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / CONFIG['gradient_accumulation']
                loss.backward()
                
                if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * CONFIG['gradient_accumulation']
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]} [Val]')
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{total_loss/(pbar.n+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': CONFIG
        }
        
        # Save latest
        checkpoint_path = self.output_dir / f'{self.model_name}_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
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
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate schedule
            self.scheduler.step()
            
            # Log metrics
            print(f'\nEpoch {epoch+1}/{CONFIG["num_epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    f'{self.model_name}_train_loss': train_loss,
                    f'{self.model_name}_train_acc': train_acc,
                    f'{self.model_name}_val_loss': val_loss,
                    f'{self.model_name}_val_acc': val_acc,
                    f'{self.model_name}_lr': self.optimizer.param_groups[0]['lr']
                }, step=epoch)
            
            # Early stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if CONFIG['save_best_only']:
                if is_best:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Check early stopping
            if self.patience_counter >= CONFIG['early_stopping_patience']:
                print(f'\n⚠️  Early stopping triggered after {epoch+1} epochs')
                break
        
        print(f'\n✅ Training complete!')
        print(f'   Best Val Loss: {self.best_val_loss:.4f}')
        print(f'   Best Val Acc: {max(self.val_accuracies):.2f}%')
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc': max(self.val_accuracies),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description='Train image deepfake detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing datasets')
    parser.add_argument('--manifest-dir', type=str, default='./manifests',
                        help='Directory containing manifest files')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--model', type=str, choices=['efficientnet', 'xception', 'both'],
                        default='both', help='Model to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['batch_size'] = args.batch_size
    CONFIG['num_epochs'] = args.epochs
    CONFIG['learning_rate'] = args.lr
    CONFIG['mixed_precision'] = not args.no_amp
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load datasets
    print(f"\n📂 Loading datasets...")
    train_dataset = DeepfakeDataset(
        manifest_path=Path(args.manifest_dir) / 'image_train.csv',
        data_dir=args.data_dir,
        transform=get_transforms(augment=True)
    )
    
    val_dataset = DeepfakeDataset(
        manifest_path=Path(args.manifest_dir) / 'image_val.csv',
        data_dir=args.data_dir,
        transform=get_transforms(augment=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    # MLflow setup
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment('kavach-ai-image-training')
        mlflow.start_run()
        mlflow.log_params(CONFIG)
    
    # Train models
    results = {}
    
    if args.model in ['efficientnet', 'both']:
        print(f"\n{'='*70}")
        print("Training EfficientNet-B4")
        print(f"{'='*70}")
        
        model = EfficientNetB4(pretrained=True)
        trainer = Trainer(
            model=model,
            model_name='efficientnet_b4',
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=args.output_dir
        )
        results['efficientnet_b4'] = trainer.train()
    
    if args.model in ['xception', 'both']:
        print(f"\n{'='*70}")
        print("Training Xception")
        print(f"{'='*70}")
        
        model = XceptionNet(pretrained=True)
        trainer = Trainer(
            model=model,
            model_name='xception',
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=args.output_dir
        )
        results['xception'] = trainer.train()
    
    # Save results
    results_file = Path(args.output_dir) / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to {results_file}")
    
    if MLFLOW_AVAILABLE:
        mlflow.end_run()
    
    print(f"\n💡 Next steps:")
    print(f"   1. Run: python scripts/train/export_onnx.py")
    print(f"   2. Run: python scripts/train/benchmark.py")


if __name__ == '__main__':
    main()

# Made with Bob
