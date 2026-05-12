"""Evaluation: F1, precision, recall."""
from __future__ import annotations

import argparse

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from wildfire_ml.config import Config
from wildfire_ml.data.dataset import FireDataset
from wildfire_ml.data.transforms import build_transforms
from wildfire_ml.models.smoke_detector import SmokeDetector


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).cpu()
            probs = torch.sigmoid(logits).squeeze(1)
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
    if not all_labels:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    return {
        "f1": f1_score(all_labels, preds, zero_division=0),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(args.device)
    model = SmokeDetector(num_classes=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    val_ds = FireDataset(args.data_root, "val", build_transforms("val"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    metrics = evaluate(model, val_loader, device)
    print(
        f"F1: {metrics['f1']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
