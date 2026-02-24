"""
DeepShield AI — XceptionNet Deepfake Classifier

Wraps timm's XceptionNet for binary deepfake classification.
XceptionNet is the FaceForensics++ benchmark standard model.
"""

import torch
import torch.nn as nn


class XceptionDeepfake(nn.Module):
    """
    XceptionNet fine-tuned for deepfake detection.
    AUC on FaceForensics++ (c23): ~0.990 when fine-tuned

    Reference: Rossler et al. 2019 — FaceForensics++
    "Learning to Detect Manipulated Facial Images"
    """

    def __init__(
        self,
        num_classes: int  = 2,
        pretrained:  bool = True,
        dropout:     float = 0.5,
    ):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
