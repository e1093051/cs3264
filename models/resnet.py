"""
ResNet-50 fine-tuned for 3-class face authenticity classification.

Architecture choice:
  - Standard residual blocks — fast on GPU, works well at 224x224.
  - 25 M params, 2048-d feature vector — higher capacity than EfficientNet-B0.
  - Strong baseline in image forensics and manipulation detection literature.

Training strategy is identical to EfficientNet (see train_cnn.py).
"""

import timm
import torch
import torch.nn as nn

import config


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained:  bool = True,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=0,
        )
        in_features = self.backbone.num_features  # 2048
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 2048-d penultimate features (no classification head)."""
        return self.backbone(x)
