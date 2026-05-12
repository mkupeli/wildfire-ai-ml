"""Training loop CLI."""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wildfire_ml.config import Config
from wildfire_ml.data.dataset import FireDataset
from wildfire_ml.data.transforms import build_transforms
from wildfire_ml.evaluate import evaluate
from wildfire_ml.export import export_onnx, write_model_card
from wildfire_ml.models.smoke_detector import SmokeDetector


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_ds = FireDataset(cfg.data_root, "train", build_transforms("train"), seed=cfg.seed)
    val_ds = FireDataset(cfg.data_root, "val", build_transforms("val"))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    if len(val_ds) == 0:
        logging.warning(
            "val_ds is empty — evaluation will return zero metrics. "
            "Check data/raw/dfire/images/val/ and pyro-sdis cache."
        )

    model = SmokeDetector(num_classes=1, pretrained=True).to(device)
    model.freeze_backbone()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    best_f1 = 0.0
    patience_counter = 0
    best_metrics: dict = {}
    cfg.models_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.models_root / "best_checkpoint.pt"

    for epoch in range(cfg.num_epochs):
        # freeze_epochs sonrasi backbone unfreeze + differential LR
        if epoch == cfg.freeze_epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone_params(), "lr": cfg.lr * 0.1},
                    {"params": model.head_params(), "lr": cfg.lr},
                ],
                weight_decay=cfg.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.num_epochs - cfg.freeze_epochs
            )

        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{cfg.num_epochs} | F1: {metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stop @ epoch {epoch + 1}")
                break

    # ONNX export
    fp32 = cfg.models_root / "smoke_detector_fp32.onnx"
    int8 = cfg.models_root / "smoke_detector_int8.onnx"
    export_onnx(checkpoint_path, fp32, int8, int8=True)

    # Model card
    write_model_card(best_metrics, cfg, cfg.models_root / "model_card.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    train(cfg)


if __name__ == "__main__":
    main()
