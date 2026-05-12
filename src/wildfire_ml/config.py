"""Hyperparameter and path configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_root: Path = Path("data/raw")
    models_root: Path = Path("models")

    # Training
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 1e-4
    num_workers: int = 4
    device: str = "cpu"
    seed: int = 42

    # Fine-tune schedule
    freeze_epochs: int = 5
    weight_decay: float = 1e-4
    patience: int = 5

    # Model
    model_name: str = "mobilenetv3-small"
    num_classes: int = 1  # binary: sigmoid head

    # Export
    onnx_opset: int = 17
    int8_quantize: bool = True
