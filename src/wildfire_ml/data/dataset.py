"""FireDataset: pyro-sdis + D-Fire union loader."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FireDataset(Dataset):
    """Smoke binary classifier dataset (label: 1=smoke/fire, 0=none).

    Combines:
    - pyro-sdis (HuggingFace cache at data/raw/pyro-sdis/)
    - D-Fire (Kaggle download at data/raw/dfire/{images,labels}/{split}/)
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

        self.samples: list[tuple[Path, int]] = []
        self.samples.extend(self._load_pyrosdis(split))
        self.samples.extend(self._load_dfire(split))

        if split == "train":
            random.Random(seed).shuffle(self.samples)

    def _load_pyrosdis(self, split: str) -> list[tuple[Path, int]]:
        """Load pyro-sdis from HuggingFace cache. Returns (path, label) pairs."""
        # NOT: HuggingFace datasets format flexibility — bu bir ornek yukleyici.
        # Gercek kullanim kullanicinin `datasets.load_dataset(...)` ile cache'i indirip
        # disk path'lerini parse etmesini gerektirir.
        # TODO Sprint 2-D real loader (HuggingFace dataset -> disk path mapping)
        return []

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
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        if self.transform is not None:
            img_tensor = self.transform(image=img_np)["image"]
        else:
            # Fallback raw tensor (CHW float [0,1])
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        return img_tensor, label
