#!/usr/bin/env python3
"""
KAVACH-AI Model Benchmarking
Evaluates trained models on test set with comprehensive metrics
"""

import os
import sys
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import model architectures and datasets
sys.path.append(str(Path(__file__).parent))
from train_image import EfficientNetB4, XceptionNet, DeepfakeDataset, get_transforms
from train_audio import RawNet2, LCNN, AudioDeepfakeDataset


class ModelBenchmark:
    """Comprehensive model benchmarking"""
    
    def __init__(self, checkpoint_dir: str, data_dir: str, manifest_dir: str, output_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.data_dir = Path(data_dir)
        self.manifest_dir = Path(manifest_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")
    
    def load_model(self, model: nn.Module, checkpoint_path: Path) -> nn.Module:
        """Load model from checkpoint"""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        return model
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        model_name: str
    ) -> Dict:
        """Evaluate model on test set"""
        print(f"\n{'='*70}")
        print(f"Evaluating {model_name}")
        print(f"{'='*70}")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs)
        
        # Equal Error Rate (EER)
        fpr, tpr = self.calculate_roc(all_labels, all_probs)
        eer = self.calculate_eer(fpr, tpr)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'eer': float(eer),
            'confusion_matrix': cm.tolist(),
            'num_samples': len(all_labels),
            'num_real': int((all_labels == 0).sum()),
            'num_fake': int((all_labels == 1).sum())
        }
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall:    {recall*100:.2f}%")
        print(f"   F1 Score:  {f1*100:.2f}%")
        print(f"   AUC:       {auc:.4f}")
        print(f"   EER:       {eer*100:.2f}%")
        print(f"\n   Samples: {len(all_labels)} (Real: {results['num_real']}, Fake: {results['num_fake']})")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, model_name)
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
        
        return results
    
    def calculate_roc(self, labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate ROC curve"""
        thresholds = np.linspace(0, 1, 1000)
        fpr_list = []
        tpr_list = []
        
        for threshold in thresholds:
            preds = (scores >= threshold).astype(int)
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        
        return np.array(fpr_list), np.array(tpr_list)
    
    def calculate_eer(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        """Calculate Equal Error Rate"""
        fnr = 1 - tpr
        eer_threshold = np.nanargmin(np.absolute(fnr - fpr))
        eer = (fpr[eer_threshold] + fnr[eer_threshold]) / 2
        return eer
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        output_path = self.output_dir / f'{model_name}_confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved confusion matrix to {output_path}")
    
    def benchmark_image_models(self) -> Dict:
        """Benchmark all image models"""
        print(f"\n{'='*70}")
        print("IMAGE MODEL BENCHMARKING")
        print(f"{'='*70}")
        
        # Load test dataset
        test_dataset = DeepfakeDataset(
            manifest_path=self.manifest_dir / 'image_test.csv',
            data_dir=self.data_dir,
            transform=get_transforms(augment=False)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        results = {}
        
        # EfficientNet-B4
        try:
            model = EfficientNetB4(pretrained=False)
            checkpoint_path = self.checkpoint_dir / 'efficientnet_b4_best.pth'
            model = self.load_model(model, checkpoint_path)
            results['efficientnet_b4'] = self.evaluate_model(model, test_loader, 'EfficientNet-B4')
        except Exception as e:
            print(f"⚠️  Failed to benchmark EfficientNet-B4: {e}")
            results['efficientnet_b4'] = None
        
        # Xception
        try:
            model = XceptionNet(pretrained=False)
            checkpoint_path = self.checkpoint_dir / 'xception_best.pth'
            model = self.load_model(model, checkpoint_path)
            results['xception'] = self.evaluate_model(model, test_loader, 'Xception')
        except Exception as e:
            print(f"⚠️  Failed to benchmark Xception: {e}")
            results['xception'] = None
        
        return results
    
    def benchmark_audio_models(self) -> Dict:
        """Benchmark all audio models"""
        print(f"\n{'='*70}")
        print("AUDIO MODEL BENCHMARKING")
        print(f"{'='*70}")
        
        # Load test dataset
        test_dataset = AudioDeepfakeDataset(
            manifest_path=self.manifest_dir / 'audio_test.csv',
            data_dir=self.data_dir
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4
        )
        
        results = {}
        
        # RawNet2
        try:
            model = RawNet2()
            checkpoint_path = self.checkpoint_dir / 'rawnet2_best.pth'
            model = self.load_model(model, checkpoint_path)
            results['rawnet2'] = self.evaluate_model(model, test_loader, 'RawNet2')
        except Exception as e:
            print(f"⚠️  Failed to benchmark RawNet2: {e}")
            results['rawnet2'] = None
        
        # LCNN
        try:
            model = LCNN()
            checkpoint_path = self.checkpoint_dir / 'lcnn_best.pth'
            model = self.load_model(model, checkpoint_path)
            results['lcnn'] = self.evaluate_model(model, test_loader, 'LCNN')
        except Exception as e:
            print(f"⚠️  Failed to benchmark LCNN: {e}")
            results['lcnn'] = None
        
        return results
    
    def generate_report(self, image_results: Dict, audio_results: Dict):
        """Generate comprehensive benchmark report"""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        
        all_results = {
            'image_models': image_results,
            'audio_models': audio_results,
            'timestamp': str(Path(__file__).stat().st_mtime)
        }
        
        # Save JSON results
        results_file = self.output_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📊 Results saved to {results_file}")
        
        # Print summary table
        print(f"\n{'='*70}")
        print("MODEL PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'Accuracy':<12} {'AUC':<10} {'F1':<10} {'EER':<10}")
        print(f"{'-'*70}")
        
        for model_name, results in {**image_results, **audio_results}.items():
            if results:
                print(f"{model_name:<20} {results['accuracy']*100:>10.2f}% {results['auc']:>9.4f} {results['f1_score']:>9.4f} {results['eer']*100:>8.2f}%")
        
        print(f"{'='*70}")
        
        # Best model
        all_valid_results = {k: v for k, v in {**image_results, **audio_results}.items() if v is not None}
        if all_valid_results:
            best_model = max(all_valid_results.items(), key=lambda x: x[1]['accuracy'])
            print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']*100:.2f}%)")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Review confusion matrices in {self.output_dir}")
        print(f"   2. Deploy best models to production")
        print(f"   3. Update README with benchmark results")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark trained deepfake detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing datasets')
    parser.add_argument('--manifest-dir', type=str, default='./manifests',
                        help='Directory containing manifest files')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--models', type=str, choices=['image', 'audio', 'all'],
                        default='all', help='Which models to benchmark')
    
    args = parser.parse_args()
    
    benchmark = ModelBenchmark(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir
    )
    
    image_results = {}
    audio_results = {}
    
    if args.models in ['image', 'all']:
        image_results = benchmark.benchmark_image_models()
    
    if args.models in ['audio', 'all']:
        audio_results = benchmark.benchmark_audio_models()
    
    benchmark.generate_report(image_results, audio_results)
    
    print(f"\n✅ Benchmarking complete!")


if __name__ == '__main__':
    main()

# Made with Bob
