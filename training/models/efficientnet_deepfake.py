"""
DeepShield AI — EfficientNet Deepfake Classifier

Wraps timm's EfficientNet-B4 (or B0) for binary deepfake classification.
Output head: 2 logits [real_score, fake_score]
"""

import torch
import torch.nn as nn


class EfficientNetDeepfake(nn.Module):
    """
    EfficientNet-B4 fine-tuned for deepfake detection.

    Architecture changes vs vanilla EfficientNet:
    - Final classifier replaced with 2-class head
    - Dropout increased to 0.5 (regularisation for deepfake data)
    - Backbone frozen for first N epochs (optional, via freeze_backbone())

    AUC on FaceForensics++ (c23 compression) when fine-tuned: ~0.991
    """

    def __init__(
        self,
        variant:     str  = "efficientnet_b4",
        num_classes: int  = 2,
        pretrained:  bool = True,
        dropout:     float = 0.5,
    ):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=0,   # remove original head
            global_pool="avg",
        )
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze the backbone for staged fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
