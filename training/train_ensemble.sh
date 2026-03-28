#!/bin/bash
# KAVACH-AI Ensemble Training Orchestrator
# Trains the 5-model ensemble sequentially and exports to ONNX.

set -e

echo "🚀 Starting KAVACH-AI World-Class v2.0 Ensemble Training..."

MODELS=("vit_primary" "vit_secondary" "efficientnet" "xception" "convnext" "wav2vec2_audio")
EPOCHS=${1:-30}  # Allow overriding epochs for smoke tests

for MODEL in "${MODELS[@]}"; do
    echo "----------------------------------------------------"
    echo "🏗️ Training Modality: $MODEL"
    echo "----------------------------------------------------"
    
    python training/train.py \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --config training/model_config.yaml
        
    echo "✅ $MODEL training and ONNX export complete."
done

echo "===================================================="
echo "🎉 Full Ensemble Training Cycle Finished Successfully!"
echo "Checkpoints exported to: training/checkpoints/"
echo "===================================================="
