"""FireDataset: pyro-sdis + D-Fire union loader."""
from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from PIL import Image
from torch.utils.data import Dataset

# Pyro-SDIS HuggingFace cache (datasets v3 layout) altinda Arrow shard'larin
# bulundugu sabit dizin. download_pyrosdis.py bu yola yazar.
# Karar: in-memory bytes -> disk israfi yok, format donusumu yok (Sprint 2-D)
_PYROSDIS_CACHE_REL = Path(
    "pyro-sdis/pyronear___pyro-sdis/default/0.0.0/"
    "a1e553ec4d806f71fc6db744cc22bc3469487382"
)


class FireDataset(Dataset):
    """Smoke binary classifier dataset (label: 1=smoke/fire, 0=none).

    Combines:
    - pyro-sdis (HuggingFace cache at data/raw/pyro-sdis/)
    - D-Fire (Kaggle download at data/raw/dfire/{images,labels}/{split}/)

    Sample format: (source, label) where `source` is either:
    - pathlib.Path -> D-Fire image path on disk
    - bytes -> Pyro-SDIS image bytes (inline from Arrow shard)

    `__getitem__` handles both via PIL.Image.open (path veya BytesIO).
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform=None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.samples: list[tuple[object, int]] = []
        self.samples.extend(self._load_pyrosdis(split))
        self.samples.extend(self._load_dfire(split))

        if split == "train":
            random.Random(seed).shuffle(self.samples)

    def _load_pyrosdis(self, split: str) -> list[tuple[bytes, int]]:
        """Load pyro-sdis from HuggingFace Arrow cache.

        Label kurali (PyroNear pyro-sdis card teyitli):
        - annotations YOLO formatli string (class_id cx cy w h)
        - Tum bbox'lar tek class = smoke
        - annotations bos -> label=0 (no smoke)
        - annotations dolu -> label=1 (smoke)
        Kaynak: https://huggingface.co/datasets/pyronear/pyro-sdis
        """
        cache_dir = self.root / _PYROSDIS_CACHE_REL
        if not cache_dir.exists():
            return []

        if split == "train":
            shards = sorted(cache_dir.glob("pyro-sdis-train-*.arrow"))
        elif split == "val":
            shards = [cache_dir / "pyro-sdis-val.arrow"]
        else:
            return []

        pairs: list[tuple[bytes, int]] = []
        for shard_path in shards:
            if not shard_path.exists():
                continue
            with pa.memory_map(str(shard_path), "r") as src:
                reader = ipc.open_stream(src)
                table = reader.read_all()
            img_col = table["image"]
            ann_col = table["annotations"]
            for i in range(table.num_rows):
                ann = ann_col[i].as_py()
                label = 0 if (ann is None or ann.strip() == "") else 1
                # image struct: {"bytes": <binary>, "path": <str>}
                img_bytes = img_col[i].as_py()["bytes"]
                pairs.append((img_bytes, label))
        return pairs

    def _load_dfire(self, split: str) -> list[tuple[Path, int]]:
        """Load D-Fire YOLO format. Returns (image_path, label) pairs.

        YOLO class_id: 0=fire, 1=smoke, 2=none
        Label: 1 if any line has class_id in {0,1}, else 0
        """
        dfire_root = self.root / "dfire"
        images_dir = dfire_root / "images" / split
        labels_dir = dfire_root / "labels" / split
        if not images_dir.exists():
            return []

        pairs: list[tuple[Path, int]] = []
        image_paths = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        )
        for img_path in image_paths:
            label_path = labels_dir / f"{img_path.stem}.txt"
            label = 0
            if label_path.exists():
                content = label_path.read_text().strip()
                if content:
                    for line in content.splitlines():
                        parts = line.split()
                        if parts and parts[0] in {"0", "1"}:
                            label = 1
                            break
            pairs.append((img_path, label))
        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        source, label = self.samples[idx]
        # source: Path (D-Fire disk) veya bytes (Pyro-SDIS Arrow inline)
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source)).convert("RGB")
        else:
            img = Image.open(source).convert("RGB")
        img_np = np.array(img)
        if self.transform is not None:
            img_tensor = self.transform(image=img_np)["image"]
        else:
            # Fallback raw tensor (CHW float [0,1])
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        return img_tensor, label
