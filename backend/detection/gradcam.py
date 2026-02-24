"""
DeepShield AI — Grad-CAM (Gradient-weighted Class Activation Mapping)

Visualises WHICH regions of a face the CNN uses to decide FAKE vs REAL.
Works with any PyTorch model — EfficientNet-B4, XceptionNet, ViT, etc.

Usage:
    cam = GradCAM(model, target_layer)
    logits, heatmap = cam.generate(tensor_input, class_idx=1)
    # heatmap → float32 numpy array [0,1], same spatial layout as input

Reference: Selvaraju et al. 2017 — "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization"
"""

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger


class GradCAM:
    """
    Hook-based Grad-CAM that works with any CNN architecture in timm.

    Parameters
    ----------
    model       : nn.Module  — the deepfake classifier
    target_layer: nn.Module  — last conv layer to hook (use _find_last_conv)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self._activations = output.detach().clone()

        def backward_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach().clone()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int = 1,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Run a forward + backward pass and compute Grad-CAM.

        Parameters
        ----------
        input_tensor : shape (1, 3, H, W), already on device
        class_idx    : 0 = REAL, 1 = FAKE

        Returns
        -------
        logits   : model output tensor  (1, num_classes)
        heatmap  : float32 numpy array  (H, W) values in [0, 1]
        """
        self.model.zero_grad()

        # --- Forward ---
        output = self.model(input_tensor)

        # Handle binary (1 output) vs 2-class models
        if output.shape[-1] == 1:
            score = output[0, 0]
        else:
            score = output[0, class_idx]

        # --- Backward ---
        score.backward()

        # --- Compute CAM ---
        gradients   = self._gradients   # (1, C, H, W)
        activations = self._activations # (1, C, H, W)

        if gradients is None or activations is None:
            logger.warning("GradCAM: no gradients captured — returning empty heatmap")
            return output.detach(), np.zeros((7, 7), dtype=np.float32)

        # Global-Average-Pool gradients over spatial dims
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        heatmap = cam.squeeze().cpu().numpy()  # (H, W)
        return output.detach(), heatmap


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ — improved localisation for multi-instance cases.
    Same interface as GradCAM.
    Reference: Chattopadhyay et al. 2018
    """

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int = 1,
    ) -> tuple[torch.Tensor, np.ndarray]:
        self.model.zero_grad()
        output = self.model(input_tensor)

        if output.shape[-1] == 1:
            score = output[0, 0]
        else:
            score = output[0, class_idx]

        score.backward()

        grads = self._gradients       # (1, C, H, W)
        acts  = self._activations     # (1, C, H, W)

        if grads is None or acts is None:
            return output.detach(), np.zeros((7, 7), dtype=np.float32)

        # Grad-CAM++ weights
        grads_sq   = grads ** 2
        grads_cube = grads ** 3
        alpha_num  = grads_sq
        alpha_den  = 2 * grads_sq + (grads_cube * acts).sum(dim=[2, 3], keepdim=True)
        alpha_den  = torch.where(alpha_den != 0, alpha_den, torch.ones_like(alpha_den))
        alpha      = alpha_num / alpha_den

        relu_grads = F.relu(score.exp() * grads)
        weights    = (alpha * relu_grads).sum(dim=[2, 3], keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        heatmap = cam.squeeze().cpu().numpy()
        return output.detach(), heatmap
