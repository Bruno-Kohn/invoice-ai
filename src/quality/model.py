"""Quality classification model using MobileNetV2 transfer learning."""

import torch
import torch.nn as nn
from torchvision import models


class QualityClassifier(nn.Module):
    """MobileNetV2-based classifier for document quality assessment.

    Classifies images into 3 classes: ready, marginal, not_ready.
    Uses pretrained ImageNet weights with a custom classification head.

    Args:
        num_classes: Number of output classes (default: 3).
        pretrained: Whether to use pretrained ImageNet weights.
        dropout: Dropout rate before final classification layer.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all backbone layers (only train classifier head)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
