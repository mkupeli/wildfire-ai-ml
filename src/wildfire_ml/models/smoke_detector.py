"""SmokeDetector: MobileNetV3-Small backbone + binary sigmoid head."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class SmokeDetector(nn.Module):
    """Binary smoke classifier.

    Backbone: MobileNetV3-Small (ImageNet pretrained when pretrained=True)
    Head: Linear(576, 128) -> Hardswish -> Dropout(0.2) -> Linear(128, 1)
    Output: (B, 1) raw logit (no sigmoid; export'ta wrapped)
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                f"SmokeDetector only supports num_classes=1 (binary); got {num_classes}"
            )
        self.num_classes = num_classes
        self.pretrained = pretrained

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        # Replace classifier with binary head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 1) logit

    def freeze_backbone(self) -> None:
        """Freeze backbone (features), only train classifier head."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("classifier")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for full fine-tune."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def backbone_params(self):
        return [p for n, p in self.backbone.named_parameters() if not n.startswith("classifier")]

    def head_params(self):
        return [p for n, p in self.backbone.named_parameters() if n.startswith("classifier")]
