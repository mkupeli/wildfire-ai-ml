"""Phase 0.5 smoke tests: imports work, basic class instantiation."""


def test_torch_imports() -> None:
    import torch
    assert torch.__version__


def test_config_loads() -> None:
    from wildfire_ml.config import Config
    cfg = Config()
    assert cfg.batch_size > 0
    assert cfg.num_classes >= 2


def test_dataset_class_importable() -> None:
    from wildfire_ml.data.dataset import FireDataset
    assert FireDataset is not None


def test_smoke_detector_instantiates() -> None:
    from wildfire_ml.models.smoke_detector import SmokeDetector
    model = SmokeDetector(num_classes=2)
    assert model.num_classes == 2
