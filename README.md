# wildfire-ai-ml

> Wildfire-AI projesi için ML eğitim ve ONNX export pipeline'ı.

> ML training and ONNX export pipeline for the Wildfire-AI project.

## TR

### Genel bakış
Detaylı proje bağlamı için [wildfire-ai-docs](https://github.com/wildfire-ai-tr/wildfire-ai-docs).

Bu repo Faz 0.5 iskeletindedir; gerçek eğitim kodu (PyroNear fine-tune, ONNX export, INT8 quantization) Faz 1 Sprint 2'de uygulanacak.

### Kurulum
- Python 3.12+
- `pip install -r requirements.txt` (CPU PyTorch için CI dosyasındaki ek index URL'i bkz.)

### Eğitim (placeholder)
- `python -m wildfire_ml.train --config ...` — Faz 1 Sprint 2'de implementasyon

### Test
- `pytest -q`

### Veri
- `data/raw/` — `.gitignore`'da. `scripts/download_pyrosdis.py` ve `scripts/download_dfire.py` ile indirilir.
- `models/` — eğitilmiş ağırlıklar `.gitignore`'da; `model_card.md` versiyonlanır.

## EN

### Overview
See [wildfire-ai-docs](https://github.com/wildfire-ai-tr/wildfire-ai-docs) for full project context.

This repo is in Phase 0.5 skeleton state; actual training code (PyroNear fine-tune, ONNX export, INT8 quantization) will be implemented in Phase 1 Sprint 2.

### Setup
- Python 3.12+
- `pip install -r requirements.txt`

### Train (placeholder)
- `python -m wildfire_ml.train --config ...` — Phase 1 Sprint 2

### Test
- `pytest -q`

## Lisans / License

Apache 2.0 — see `LICENSE`.

## Atıflar / Credits

Detailed list: `CREDITS.md`.
