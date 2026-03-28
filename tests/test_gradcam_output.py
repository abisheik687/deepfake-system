import pytest
import torch
import numpy as np
from PIL import Image
from backend.detection.gradcam import GradCAM
import timm

def test_gradcam_logic():
    # Create a small dummy model
    model = timm.create_model('resnet18', pretrained=False, num_classes=2).eval()
    
    # Target the last convolution of the last block
    last_conv = model.layer4[-1].conv2
    
    cam = GradCAM(model, last_conv)
    
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)
    
    logits, heatmap = cam.generate(dummy_input, class_idx=1)
    
    assert logits.shape == (1, 2)
    assert heatmap.shape == (7, 7) # Standard ResNet18 last feature map size
    assert heatmap.max() <= 1.0
    assert heatmap.min() >= 0.0
    
    cam.remove_hooks()
