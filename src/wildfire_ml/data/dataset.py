"""PyTorch dataset for fire/smoke images.

Implementation: Phase 1 Sprint 2.
"""

from pathlib import Path

from torch.utils.data import Dataset


class FireDataset(Dataset):
    """Loads pyro-sdis + D-Fire + custom Beynam images.

    Phase 0.5: skeleton only. Real loading logic in Phase 1 Sprint 2.
    """

    def __init__(self, root: Path, split: str = "train", transform=None) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError("Implement in Phase 1 Sprint 2")
