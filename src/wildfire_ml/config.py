"""Hyperparameter and path configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_root: Path = Path("data")
    models_root: Path = Path("models")

    # Training
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 1e-4
    num_workers: int = 4
    device: str = "cpu"

    # Model
    model_name: str = "pyronear-base"
    num_classes: int = 2  # smoke vs no-smoke

    # Export
    onnx_opset: int = 17
    int8_quantize: bool = True

    seed: int = 42
