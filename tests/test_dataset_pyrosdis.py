"""Pyro-SDIS dataset loader integration testleri.

Sprint 2-D: `_load_pyrosdis` Arrow inline bytes loader (Karar: in-memory bytes).
Cache `data/raw/pyro-sdis/.../*.arrow` yoksa testler skip edilir.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from PIL import Image

PYROSDIS_CACHE = Path(
    "data/raw/pyro-sdis/pyronear___pyro-sdis/default/0.0.0/"
    "a1e553ec4d806f71fc6db744cc22bc3469487382"
)


def _cache_present() -> bool:
    return PYROSDIS_CACHE.exists() and (PYROSDIS_CACHE / "pyro-sdis-val.arrow").exists()


pytestmark = pytest.mark.skipif(
    not _cache_present(),
    reason="Pyro-SDIS Arrow cache yok (data/raw/pyro-sdis/...). "
    "scripts/download_pyrosdis.py ile indir.",
)


def test_pyrosdis_val_loads_nonzero_samples():
    """Val split yuklenir, > 0 sample doner."""
    from wildfire_ml.data.dataset import FireDataset

    ds = FireDataset(Path("data/raw"), "val", transform=None)
    assert len(ds) > 0, "val split bos donmemeli"


def test_pyrosdis_sample_label_is_binary():
    """Ilk birkac orneğin labeli {0, 1} icinde."""
    from wildfire_ml.data.dataset import FireDataset

    ds = FireDataset(Path("data/raw"), "val", transform=None)
    for _src, label in ds.samples[:10]:
        assert label in {0, 1}, f"Label binary olmali, got: {label}"


def test_pyrosdis_image_bytes_openable():
    """Inline bytes PIL.Image.open ile RGB acilabilir."""
    from wildfire_ml.data.dataset import FireDataset

    ds = FireDataset(Path("data/raw"), "val", transform=None)
    source, _label = ds.samples[0]
    # Pyro-SDIS Arrow -> bytes; D-Fire disk -> Path
    assert isinstance(source, (bytes, bytearray, Path)), \
        f"source bytes veya Path olmali, got: {type(source)}"
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source)).convert("RGB")
    else:
        img = Image.open(source).convert("RGB")
    w, h = img.size
    assert w > 0 and h > 0


def test_pyrosdis_getitem_returns_tensor():
    """__getitem__ transformsiz raw tensor doner (3, H, W) float."""
    from wildfire_ml.data.dataset import FireDataset

    ds = FireDataset(Path("data/raw"), "val", transform=None)
    img_tensor, label = ds[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.ndim == 3
    assert img_tensor.shape[0] == 3
    assert img_tensor.dtype == torch.float32
    assert label in {0, 1}


def test_pyrosdis_label_distribution_has_both_classes():
    """Val split hem smoke hem no-smoke ornegi icermeli (binary smoke detector saglikli kalsin)."""
    from wildfire_ml.data.dataset import FireDataset

    ds = FireDataset(Path("data/raw"), "val", transform=None)
    labels = [lbl for _src, lbl in ds.samples]
    pos = sum(labels)
    neg = len(labels) - pos
    assert pos > 0, "smoke (label=1) ornegi yok"
    assert neg > 0, "no-smoke (label=0) ornegi yok"
