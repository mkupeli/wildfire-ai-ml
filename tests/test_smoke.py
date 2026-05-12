"""Phase 0.5 smoke tests + Sprint 2-C pipeline testleri."""
import io

import numpy as np
import torch
from PIL import Image

# --- Mevcut 4 test (Phase 0.5 — korunur) ---


def test_torch_imports() -> None:
    import torch

    assert torch.__version__


def test_config_loads() -> None:
    from wildfire_ml.config import Config

    cfg = Config()
    assert cfg.batch_size > 0
    assert cfg.num_classes >= 1  # binary: 1; multi-class: >1


def test_dataset_class_importable() -> None:
    from wildfire_ml.data.dataset import FireDataset

    assert FireDataset is not None


def test_smoke_detector_instantiates() -> None:
    from wildfire_ml.models.smoke_detector import SmokeDetector

    model = SmokeDetector(num_classes=1)
    assert model.num_classes == 1


# --- Sprint 2-C: 5 yeni test ---


def test_dataset_dummy_sample():
    """Dummy dataset shape + label kontrol (gercek dataset I/O yok)."""
    from torch.utils.data import Dataset

    class DummyDataset(Dataset):
        def __init__(self, n: int = 5) -> None:
            self.n = n

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int):
            return torch.rand(3, 224, 224), 0

    ds = DummyDataset(5)
    assert len(ds) == 5
    tensor, label = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert isinstance(label, int)


def test_model_forward():
    from wildfire_ml.models.smoke_detector import SmokeDetector

    model = SmokeDetector(num_classes=1, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1)


def test_transforms_preprocess_symmetric():
    """Backend inference_service.preprocess_image ile birebir sabitler (Karar #6)."""
    from wildfire_ml.data.transforms import preprocess_image_numpy

    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    arr = preprocess_image_numpy(buf.getvalue())
    assert arr.shape == (1, 3, 224, 224)
    assert arr.dtype == np.float32
    # ImageNet normalize sonrasi deger araligi
    assert arr.min() > -3.0
    assert arr.max() < 3.0


def test_train_dry_run():
    """Mock dataset + 1 epoch + tek batch -> loss > 0, optimizer step calisiyor."""
    from torch.utils.data import DataLoader, Dataset

    from wildfire_ml.models.smoke_detector import SmokeDetector

    class DummyDS(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return torch.rand(3, 224, 224), idx % 2

    loader = DataLoader(DummyDS(), batch_size=2)
    model = SmokeDetector(num_classes=1, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    before = next(model.parameters()).clone()
    model.train()
    for imgs, labels in loader:
        labels = labels.float().unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0
        break  # tek batch
    after = next(model.parameters())
    assert not torch.allclose(before, after)


def test_export_onnx_smoke(tmp_path):
    """Tiny model -> ONNX FP32 + INT8 export -> file exists."""
    from wildfire_ml.export import export_onnx
    from wildfire_ml.models.smoke_detector import SmokeDetector

    model = SmokeDetector(num_classes=1, pretrained=False)
    ckpt = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), ckpt)

    fp32 = tmp_path / "model_fp32.onnx"
    int8 = tmp_path / "model_int8.onnx"
    export_onnx(ckpt, fp32, int8, int8=True)
    assert fp32.exists()
    assert int8.exists()
    assert fp32.stat().st_size > 0
    assert int8.stat().st_size > 0
