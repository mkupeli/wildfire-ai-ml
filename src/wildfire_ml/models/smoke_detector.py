"""Smoke detection model wrapper.

Wraps PyroNear pyro-vision backbone for fine-tuning on Turkey-specific data.
Implementation: Phase 1 Sprint 2.
"""

import torch
import torch.nn as nn


class SmokeDetector(nn.Module):
    """Smoke vs no-smoke binary classifier.

    Phase 0.5: skeleton. Real architecture in Phase 1 Sprint 2.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self._backbone: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in Phase 1 Sprint 2")
