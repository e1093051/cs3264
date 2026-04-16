"""
DeiT-Small baseline for 3-class face authenticity classification.

Architecture choice:
  - Pretrained on ImageNet for strong visual representations.
  - Uses self-attention to capture global relationships across image patches.
  - More data-efficient than vanilla ViT, making it suitable for smaller datasets.
  - Final embedding is used as the penultimate feature vector for downstream models.

Training strategy (implemented in train_cnn.py):
  1. Freeze backbone; train head for UNFREEZE_AFTER epochs with high LR.
  2. Unfreeze backbone; fine-tune all weights with low LR.
"""

import timm
import torch
import torch.nn as nn

import config


class DeiTSmallClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        # num_classes=0 removes the original classification head
        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )

        in_features = self.backbone.num_features  # typically 384

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate DeiT-Small features (no classification head)."""
        return self.backbone(x)