"""Full baseline training: Pyro-SDIS + D-Fire union -> smoke detector.

Sprint 2 Parça (b) — gercek baseline egitim (tur 2 revizyonu).

Plan (tur 2 — architect kararlari):
- Epochs: 20 (freeze_epochs=5 -> sadece head; sonraki 15 epoch full unfreeze + diff LR)
- Batch size: 16 lokal / 64-128 Kaggle P100
- Precision: AMP (torch.cuda.amp) — fp16 mixed
- Loss: BCEWithLogitsLoss(pos_weight=n_neg/n_pos) — dinamik class imbalance dengelemesi
- Optimizer: AdamW + CosineAnnealingLR
- Metrik: ROC-AUC, F1, precision, recall, accuracy, confusion matrix, val loss
- BEST checkpoint kriteri: val ROC-AUC (F1 degil — tur 1'de F1 yanilticiydi)
- Intermediate checkpoint: her 5 epoch'ta checkpoint_epoch{N}.pt (Kaggle 12 sa session
  limiti reservi)

Kullanim:
    python scripts/train_smoke_detector.py --device cuda --batch-size 16 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

# Dogrudan `python scripts/train_smoke_detector.py` calistirildiginda paket src/ altinda
# kaldigi icin sys.path'e eklenmez (sadece pytest pythonpath=["src"] biliyordu) -> bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wildfire_ml.data.dataset import FireDataset  # noqa: E402
from wildfire_ml.data.transforms import build_transforms  # noqa: E402
from wildfire_ml.models.smoke_detector import SmokeDetector  # noqa: E402


@dataclass
class TrainConfig:
    data_root: str = "data/raw"
    out_dir: str = "artifacts/full_train"
    device: str = "cuda"
    epochs: int = 20
    freeze_epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-3
    head_lr_after_unfreeze: float = 1e-3
    backbone_lr_after_unfreeze: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0  # Windows + Arrow inline bytes -> 0 guvenli; Kaggle Linux'ta 2
    seed: int = 42
    amp: bool = True
    checkpoint_every: int = 5  # Her N epoch'ta intermediate checkpoint
    pos_weight: float | None = None  # Train set'inden hesaplanir; final summary'ye gider


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(samples: list[tuple[object, int]]) -> tuple[float, int, int]:
    """train_ds.samples (FireDataset.samples) -> (pos_weight, n_pos, n_neg).

    BCEWithLogitsLoss(pos_weight=n_neg/n_pos) class imbalance dengelemesi icin.
    Train set bir kez sayilir (etiketler bellekte zaten — I/O yok).
    """
    n_pos = sum(1 for _, label in samples if label == 1)
    n_neg = len(samples) - n_pos
    if n_pos == 0:
        raise RuntimeError("train set'inde pozitif ornek yok — pos_weight hesaplanamaz")
    if n_neg == 0:
        # Az olasi ama guvenli: 1.0 dondur
        return 1.0, n_pos, n_neg
    return n_neg / n_pos, n_pos, n_neg


def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict:
    """Tam val pass: ROC-AUC + F1 + P + R + accuracy + CM + val_loss."""
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels_t = labels.float().unsqueeze(1).to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels_t)
            losses.append(loss.item())
            probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    if not all_labels:
        return {
            "val_loss": float("nan"),
            "roc_auc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "n_samples": 0,
        }

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    try:
        roc_auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        # Tek class varsa
        roc_auc = float("nan")
    cm = confusion_matrix(all_labels, preds, labels=[0, 1]).tolist()
    return {
        "val_loss": float(np.mean(losses)),
        "roc_auc": roc_auc,
        "f1": float(f1_score(all_labels, preds, zero_division=0)),
        "precision": float(precision_score(all_labels, preds, zero_division=0)),
        "recall": float(recall_score(all_labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(all_labels, preds)),
        "confusion_matrix": cm,
        "n_samples": len(all_labels),
    }


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "cfg.device=cuda but torch.cuda.is_available()=False. "
            "Aborting — CPU fallback yapma talimati."
        )
    print(f"[train] device={device}")
    if device.type == "cuda":
        print(f"[train] gpu={torch.cuda.get_device_name(0)}")
        print(
            f"[train] vram total={torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    print("[train] loading datasets...")
    train_ds = FireDataset(
        Path(cfg.data_root), "train", build_transforms("train"), seed=cfg.seed
    )
    val_ds = FireDataset(Path(cfg.data_root), "val", build_transforms("val"))
    print(f"[train] train={len(train_ds)} val={len(val_ds)}")

    # --- Dinamik pos_weight (architect karari — tur 1'de eksik, F1 yanilticiydi) ---
    pos_weight_val, n_pos, n_neg = compute_pos_weight(train_ds.samples)
    cfg.pos_weight = float(pos_weight_val)
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    print(
        f"[train] pos_weight={pos_weight_val:.4f} (n_pos={n_pos}, n_neg={n_neg})"
    )

    # persistent_workers True yalnizca num_workers>0 oldugunda anlamli
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=(4 if cfg.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=(4 if cfg.num_workers > 0 else None),
    )

    model = SmokeDetector(num_classes=1, pretrained=True).to(device)
    model.freeze_backbone()  # Ilk freeze_epochs kadar sadece head
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] params trainable={trainable:,} / total={total:,} (head only)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    # Phase 1 scheduler (head-only) — T_max = freeze_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.freeze_epochs, 1)
    )

    use_amp = cfg.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict] = []
    best_roc_auc = -1.0
    best_metrics: dict = {}
    checkpoint_path = out_dir / "best_checkpoint.pt"

    for epoch in range(cfg.epochs):
        # Unfreeze phase
        if epoch == cfg.freeze_epochs:
            print(f"[train] === Epoch {epoch + 1}: UNFREEZE backbone ===")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone_params(), "lr": cfg.backbone_lr_after_unfreeze},
                    {"params": model.head_params(), "lr": cfg.head_lr_after_unfreeze},
                ],
                weight_decay=cfg.weight_decay,
            )
            # Phase 2 scheduler — T_max = kalan epoch sayisi (epochs - freeze_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(cfg.epochs - cfg.freeze_epochs, 1)
            )

        model.train()
        epoch_losses: list[float] = []
        t0 = time.time()
        n_batches = len(train_loader)
        for step, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels_t = labels.float().unsqueeze(1).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss = criterion(logits, labels_t)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels_t)
                loss.backward()
                optimizer.step()
            epoch_losses.append(loss.item())

            if step % 100 == 0:
                elapsed = time.time() - t0
                vram = (
                    torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0.0
                )
                print(
                    f"[train] epoch {epoch + 1}/{cfg.epochs} "
                    f"step {step:4d}/{n_batches} "
                    f"loss={loss.item():.4f} "
                    f"elapsed={elapsed:.1f}s "
                    f"vram={vram:.2f}GB"
                )

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        train_time = time.time() - t0
        print(f"[train] epoch {epoch + 1} train_loss={train_loss:.4f} time={train_time:.1f}s")

        val_metrics = evaluate_full(model, val_loader, device, criterion)
        print(
            f"[train] epoch {epoch + 1} val_loss={val_metrics['val_loss']:.4f} "
            f"ROC-AUC={val_metrics['roc_auc']:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"P={val_metrics['precision']:.4f} "
            f"R={val_metrics['recall']:.4f} "
            f"acc={val_metrics['accuracy']:.4f}"
        )
        print(f"[train] epoch {epoch + 1} CM={val_metrics['confusion_matrix']}")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_time_sec": train_time,
                **val_metrics,
            }
        )

        # BEST kriteri: ROC-AUC (tur 1'de F1 idi; class imbalance + recall=1.0
        # capkurusu yuzunden yanilticiydi)
        if val_metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = val_metrics["roc_auc"]
            best_metrics = val_metrics
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[train] new best ROC-AUC {best_roc_auc:.4f} -> {checkpoint_path}")

        # Intermediate checkpoint (Kaggle 12 sa session reservi)
        if cfg.checkpoint_every > 0 and (epoch + 1) % cfg.checkpoint_every == 0:
            inter_path = out_dir / f"checkpoint_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), inter_path)
            print(f"[train] intermediate checkpoint -> {inter_path}")

    # Final dump
    summary = {
        "config": asdict(cfg),
        "history": history,
        "best_metrics": best_metrics,
        "best_checkpoint": str(checkpoint_path),
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[train] done. summary -> {out_dir / 'training_summary.json'}")
    print(
        f"[train] BEST val: ROC-AUC={best_metrics.get('roc_auc', 0):.4f} "
        f"F1={best_metrics.get('f1', 0):.4f} "
        f"acc={best_metrics.get('accuracy', 0):.4f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="artifacts/full_train")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        device=args.device,
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        checkpoint_every=args.checkpoint_every,
    )
    train(cfg)


if __name__ == "__main__":
    main()
