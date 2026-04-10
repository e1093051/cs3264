"""
EfficientNet-B0 baseline for 3-class face authenticity classification.

Architecture choice:
  - Pretrained on ImageNet (strong general visual features).
  - Lightweight (5.3 M params) — fast to fine-tune, good for constrained GPU budgets.
  - Global average pooling → 1280-d feature vector used by ensemble models.

Training strategy (implemented in train_cnn.py):
  1. Freeze backbone; train head for UNFREEZE_AFTER epochs with high LR.
  2. Unfreeze backbone; fine-tune all weights with low LR.
"""

import timm
import torch
import torch.nn as nn

import config


class EfficientNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained:  bool = True,
        dropout:     float = 0.3,
    ):
        super().__init__()
        # num_classes=0 removes the original classification head
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
        )
        in_features = self.backbone.num_features  # 1280
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1280-d penultimate features (no classification head)."""
        return self.backbone(x)
