"""Smoke test training: pipeline'in uctan uca calistigini dogrular.

Tam egitim DEGIL — sadece:
- 1 epoch
- 50 batch (max_steps)
- 500 train + 100 val sample (Subset)
- batch_size=16 + mixed precision (AMP)
- freeze_backbone tum epoch boyunca (sadece head ogrenir)

Cikti:
- artifacts/smoke_test/checkpoint.pt
- models/smoke_detector_smoketest_v0.onnx (export.py ile)

Kullanim:
    python scripts/smoke_test_train.py --device cuda
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from wildfire_ml.data.dataset import FireDataset
from wildfire_ml.data.transforms import build_transforms
from wildfire_ml.evaluate import evaluate
from wildfire_ml.models.smoke_detector import SmokeDetector


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_smoke_test(
    data_root: Path,
    out_dir: Path,
    device_str: str = "cuda",
    train_subset: int = 500,
    val_subset: int = 100,
    max_steps: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
) -> Path:
    set_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    print(f"[smoke] device={device}")

    full_train = FireDataset(data_root, "train", build_transforms("train"), seed=seed)
    full_val = FireDataset(data_root, "val", build_transforms("val"))
    print(f"[smoke] full train={len(full_train)}, val={len(full_val)}")

    # Subset (deterministic — full_train icinden ilk N)
    train_ds = Subset(full_train, list(range(min(train_subset, len(full_train)))))
    val_ds = Subset(full_val, list(range(min(val_subset, len(full_val)))))
    print(f"[smoke] subset train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows + Arrow + bytes -> fork sorunlu, 0 guvenli
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = SmokeDetector(num_classes=1, pretrained=True).to(device)
    model.freeze_backbone()  # tum epoch boyunca; sadece head ogrenir
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[smoke] trainable params: {trainable:,} / {total:,} (head only)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    use_amp = device.type == "cuda"
    # torch 2.x: GradScaler yeni API
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    losses: list[float] = []
    step = 0
    t0 = time.time()
    for imgs, labels in train_loader:
        if step >= max_steps:
            break
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        if step % 10 == 0:
            print(f"[smoke] step {step:3d}/{max_steps} loss={loss.item():.4f}")
        step += 1

    train_time = time.time() - t0
    final_train_loss = losses[-1] if losses else float("nan")
    first_train_loss = losses[0] if losses else float("nan")
    avg_last5 = float(np.mean(losses[-5:])) if losses else float("nan")
    print(
        f"[smoke] train done in {train_time:.1f}s | "
        f"first_loss={first_train_loss:.4f} last_loss={final_train_loss:.4f} "
        f"avg_last5={avg_last5:.4f}"
    )

    # Val pass — F1 onemsiz, pipeline calisiyor mu test
    val_metrics = evaluate(model, val_loader, device)
    # Val loss ayrica hesapla (evaluate sadece f1/p/r)
    model.eval()
    val_losses = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            logits = model(imgs)
            val_losses.append(criterion(logits, labels).item())
    val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
    print(
        f"[smoke] val_loss={val_loss:.4f} | "
        f"F1={val_metrics['f1']:.4f} P={val_metrics['precision']:.4f} "
        f"R={val_metrics['recall']:.4f}"
    )

    # Checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[smoke] checkpoint -> {ckpt_path}")
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="artifacts/smoke_test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-subset", type=int, default=500)
    parser.add_argument("--val-subset", type=int, default=100)
    args = parser.parse_args()

    run_smoke_test(
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        device_str=args.device,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
