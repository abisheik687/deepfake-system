#!/usr/bin/env python3
"""
KAVACH-AI ONNX Exporter
Exports trained PyTorch models to ONNX format with verification
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

# Import model architectures
sys.path.append(str(Path(__file__).parent))
from train_image import EfficientNetB4, XceptionNet
from train_audio import RawNet2, LCNN


class ONNXExporter:
    """ONNX model exporter with verification"""
    
    def __init__(self, checkpoint_dir: str, output_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Load PyTorch checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        return checkpoint
    
    def export_image_model(self, model_name: str, model: nn.Module, input_shape: tuple):
        """Export image model to ONNX"""
        print(f"\n{'='*70}")
        print(f"Exporting {model_name} to ONNX")
        print(f"{'='*70}")
        
        # Load checkpoint
        checkpoint_path = self.checkpoint_dir / f'{model_name}_best.pth'
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return False
        
        checkpoint = self.load_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Export to ONNX
        onnx_path = self.output_dir / f'{model_name}.onnx'
        
        print(f"📦 Exporting to {onnx_path}...")
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Exported to {onnx_path}")
        
        # Verify ONNX model
        print(f"🔍 Verifying ONNX model...")
        
        try:
            # Check ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX model is valid")
            
            # Test inference
            ort_session = ort.InferenceSession(str(onnx_path))
            
            # PyTorch inference
            with torch.no_grad():
                torch_output = model(dummy_input).cpu().numpy()
            
            # ONNX inference
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            ort_output = ort_session.run(None, ort_inputs)[0]
            
            # Compare outputs
            max_diff = np.abs(torch_output - ort_output).max()
            print(f"📊 Max difference between PyTorch and ONNX: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                print(f"✅ Parity check passed!")
            else:
                print(f"⚠️  Parity check warning: difference is {max_diff:.6f}")
            
            # Get model info
            file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"📁 Model size: {file_size_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def export_audio_model(self, model_name: str, model: nn.Module, input_shape: tuple):
        """Export audio model to ONNX"""
        print(f"\n{'='*70}")
        print(f"Exporting {model_name} to ONNX")
        print(f"{'='*70}")
        
        checkpoint_path = self.checkpoint_dir / f'{model_name}_best.pth'
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return False
        
        checkpoint = self.load_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        dummy_input = torch.randn(input_shape).to(self.device)
        onnx_path = self.output_dir / f'{model_name}.onnx'
        
        print(f"📦 Exporting to {onnx_path}...")
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Exported to {onnx_path}")
        
        # Verify
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX model is valid")
            
            ort_session = ort.InferenceSession(str(onnx_path))
            
            with torch.no_grad():
                torch_output = model(dummy_input).cpu().numpy()
            
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            ort_output = ort_session.run(None, ort_inputs)[0]
            
            max_diff = np.abs(torch_output - ort_output).max()
            print(f"📊 Max difference: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                print(f"✅ Parity check passed!")
            
            file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"📁 Model size: {file_size_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def export_all(self):
        """Export all trained models"""
        print(f"\n{'='*70}")
        print("KAVACH-AI ONNX Model Exporter")
        print(f"{'='*70}")
        print(f"\nCheckpoint directory: {self.checkpoint_dir.absolute()}")
        print(f"Output directory: {self.output_dir.absolute()}")
        
        results = {}
        
        # Image models
        print(f"\n{'='*70}")
        print("IMAGE MODELS")
        print(f"{'='*70}")
        
        # EfficientNet-B4
        model = EfficientNetB4(pretrained=False)
        results['efficientnet_b4'] = self.export_image_model(
            'efficientnet_b4',
            model,
            input_shape=(1, 3, 224, 224)
        )
        
        # Xception
        model = XceptionNet(pretrained=False)
        results['xception'] = self.export_image_model(
            'xception',
            model,
            input_shape=(1, 3, 224, 224)
        )
        
        # Audio models
        print(f"\n{'='*70}")
        print("AUDIO MODELS")
        print(f"{'='*70}")
        
        # RawNet2
        model = RawNet2()
        results['rawnet2'] = self.export_audio_model(
            'rawnet2',
            model,
            input_shape=(1, 80, 400)  # (batch, n_mels, time)
        )
        
        # LCNN
        model = LCNN()
        results['lcnn'] = self.export_audio_model(
            'lcnn',
            model,
            input_shape=(1, 1, 80, 400)  # (batch, channels, n_mels, time)
        )
        
        # Summary
        print(f"\n{'='*70}")
        print("EXPORT SUMMARY")
        print(f"{'='*70}")
        
        for model_name, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"  {model_name}: {status}")
        
        # Save results
        results_file = self.output_dir / 'export_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Results saved to {results_file}")
        
        # List exported models
        onnx_files = list(self.output_dir.glob('*.onnx'))
        if onnx_files:
            print(f"\n📁 Exported ONNX models:")
            for onnx_file in onnx_files:
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                print(f"   {onnx_file.name} ({size_mb:.2f} MB)")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Copy ONNX models to: models/")
        print(f"   2. Run: python scripts/train/benchmark.py")
        print(f"   3. Update backend to use ONNX models")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Export trained models to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory containing PyTorch checkpoints'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Output directory for ONNX models'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['efficientnet_b4', 'xception', 'rawnet2', 'lcnn', 'all'],
        default='all',
        help='Specific model to export'
    )
    
    args = parser.parse_args()
    
    exporter = ONNXExporter(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir
    )
    
    if args.model == 'all':
        exporter.export_all()
    else:
        # Export specific model
        if args.model == 'efficientnet_b4':
            model = EfficientNetB4(pretrained=False)
            exporter.export_image_model(args.model, model, (1, 3, 224, 224))
        elif args.model == 'xception':
            model = XceptionNet(pretrained=False)
            exporter.export_image_model(args.model, model, (1, 3, 224, 224))
        elif args.model == 'rawnet2':
            model = RawNet2()
            exporter.export_audio_model(args.model, model, (1, 80, 400))
        elif args.model == 'lcnn':
            model = LCNN()
            exporter.export_audio_model(args.model, model, (1, 1, 80, 400))


if __name__ == '__main__':
    main()

# Made with Bob
