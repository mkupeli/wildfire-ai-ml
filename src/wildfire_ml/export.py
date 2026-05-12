"""ONNX export with sigmoid wrapper + INT8 quantization."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import torch
import torch.nn as nn
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantType, quantize_dynamic

from wildfire_ml import __version__
from wildfire_ml.config import Config
from wildfire_ml.models.smoke_detector import SmokeDetector

MODEL_CARD_TEMPLATE = """# Model Card — Smoke Detector

## Model Details
- **Architecture**: MobileNetV3-Small + binary sigmoid head
- **Backbone**: torchvision.models.mobilenet_v3_small (ImageNet pretrained)
- **Version**: {version}
- **Date**: {date_str}
- **License**: Apache 2.0

## Training Configuration
- Epochs: {epochs} (freeze_epochs={freeze_epochs})
- Batch size: {batch_size}
- Learning rate: {lr}
- Device: {device}
- Seed: {seed}

## Datasets
- pyro-sdis (Apache 2.0) — https://huggingface.co/datasets/pyronear/pyro-sdis
- D-Fire (akademik kullanim) — https://github.com/gaiasd/DFireDataset

## Metrics (validation)
- F1: {f1:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}

## Attribution
Bu model pyro-sdis ve D-Fire acik veri setleri uzerine fine-tune edilmistir.
Backbone: torchvision MobileNetV3-Small (BSD-3-Clause).
PyroNear ekibine atif: https://github.com/pyronear (Apache 2.0).
Tam atif listesi: CREDITS.md

## Limitations
- Gece / termal goruntu kapsam disi
- Beynam PoC verisiyle henuz fine-tune yapilmadi (Faz 1 ilerisi)
- Confidence threshold kalibrasyon: 50+ pilot gonullu sonrasi
"""


def write_model_card(metrics: dict, cfg: Config, output_path: Path) -> None:
    content = MODEL_CARD_TEMPLATE.format(
        version=__version__,
        date_str=date.today().isoformat(),
        epochs=cfg.num_epochs,
        freeze_epochs=getattr(cfg, "freeze_epochs", 5),
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=cfg.device,
        seed=cfg.seed,
        f1=metrics.get("f1", 0.0),
        precision=metrics.get("precision", 0.0),
        recall=metrics.get("recall", 0.0),
    )
    output_path.write_text(content, encoding="utf-8")


def export_onnx(
    checkpoint_path: Path,
    fp32_path: Path,
    int8_path: Path | None = None,
    int8: bool = True,
) -> None:
    """Export PyTorch checkpoint -> ONNX (FP32) + optional INT8 quantize."""
    model = SmokeDetector(num_classes=1, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Sigmoid wrap: backend inference_service [0,1] output bekliyor
    model_with_sigmoid = nn.Sequential(model, nn.Sigmoid())
    model_with_sigmoid.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    fp32_path = Path(fp32_path)
    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False: eski tracing exporter kullan (opset 17 + quantize_dynamic uyumlu)
    torch.onnx.export(
        model_with_sigmoid,
        dummy,
        str(fp32_path),
        opset_version=17,
        input_names=["input"],
        output_names=["smoke_prob"],
        dynamic_axes=None,
        dynamo=False,
    )

    # Dogrulama: ORT session + shape + aralik
    sess = InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})
    assert out[0].shape == (1, 1), f"Unexpected ONNX output shape: {out[0].shape}"
    assert 0.0 <= float(out[0][0][0]) <= 1.0, "Sigmoid output out of [0,1] range"

    if int8 and int8_path is not None:
        int8_path = Path(int8_path)
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fp32-out", type=str, default="models/smoke_detector_fp32.onnx")
    parser.add_argument("--int8-out", type=str, default="models/smoke_detector_int8.onnx")
    parser.add_argument("--no-int8", action="store_true")
    args = parser.parse_args()

    export_onnx(
        Path(args.checkpoint),
        Path(args.fp32_out),
        Path(args.int8_out) if not args.no_int8 else None,
        int8=not args.no_int8,
    )
    print(f"FP32: {args.fp32_out}")
    if not args.no_int8:
        print(f"INT8: {args.int8_out}")


if __name__ == "__main__":
    main()
