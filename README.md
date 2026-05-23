# wildfire-ai-ml

> Wildfire-AI projesi için ML eğitim ve ONNX export pipeline'ı.

> ML training and ONNX export pipeline for the Wildfire-AI project.

## TR

### Genel bakış
Detaylı proje bağlamı için [wildfire-ai-docs](https://github.com/wildfire-ai-tr/wildfire-ai-docs).

İki ML çıktısı: **risk modeli** (Sprint 4/5/6 — XGBoost ONNX `risk_model_b1.onnx` üretildi ve backend'e deploy'lu) ve **smoke detector** (Sprint 2 — kod hazır, model henüz eğitilmedi).

**Mevcut durum (2026-05-23):**
- ✅ `src/wildfire_ml/models/smoke_detector.py` — `SmokeDetector` (MobileNetV3-Small + binary sigmoid head)
- ✅ `src/wildfire_ml/train.py` — eğitim döngüsü
- ✅ `src/wildfire_ml/export.py` — ONNX export + INT8 dynamic quantization
- ✅ `scripts/download_pyrosdis.py` — pyro-sdis dataset (33,636 örnek, 3.1 GB) indirilebiliyor, HF cache'te
- ⚠️ Eğitim **hiç çalıştırılmadı** → checkpoint yok → `smoke_detector_int8.onnx` üretilmedi → ne backend'de ne mobile asset'te smoke inference fiilen çalışır (graceful fallback `local_confidence=0` / `confidence_score=None`)
- ⚠️ Dataset YOLO bbox formatında (`class x_c y_c w h`), SmokeDetector ise binary classifier → eğitim öncesi annotations → binary label adaptasyonu gerekir (class==1 bbox varsa label=1)

### Kurulum
- Python 3.12+ (test edildi: 3.13)
- `pip install -r requirements.txt datasets`
- `datasets` paketi `requirements.txt`'de değil — `scripts/download_pyrosdis.py` için ayrıca gerekir
- PyTorch varsayılan **CPU**; GPU için ek index URL (CI dosyasına bak)

### Eğitim
- `python -m wildfire_ml.train --config ...` — script var; CPU'da 30k örnek MobileNetV3 fine-tune saatler/günler. GPU önerilir. Annotations adaptör'ü uygulanmalı (bkz. dataset notu).
- `python -m wildfire_ml.export` — eğitilmiş checkpoint → ONNX INT8 export
- Çıktı: `models/smoke_detector_int8.onnx` → kopyala: backend `models/` + mobile `assets/models/`

### Risk modeli (ayrı pipeline)
- `src/wildfire_ml/risk/` altında: `train_risk.py`, `export_risk.py`, feature schema vs.
- Çıktı: `models/risk_model_b1.onnx` (HAZIR, versiyonlu değil — model dosyası `.gitignore`'da)
- Karar #6: ML+backend preprocess senkron (`PREPROCESS_SYMMETRIC`)

### Test
- `pytest -q`

### Veri
- `data/raw/` — `.gitignore`'da. `scripts/download_pyrosdis.py` ile indirilir; HF datasets `cache_dir` argümanı bu sürümde override etmiyor → veri `~/.cache/huggingface/hub/datasets--pyronear--pyro-sdis/`'a düşer (3.1 GB).
- `data/processed/beynam_real_dataset.csv` (46 MB) — risk modeli (b1) training input
- `models/` — eğitilmiş ağırlıklar `.gitignore`'da; `model_card.md`, `risk_feature_schema.json`, `risk_model_v2_card.md` versiyonlanır.

## EN

### Overview
See [wildfire-ai-docs](https://github.com/wildfire-ai-tr/wildfire-ai-docs) for full project context.

Two ML outputs: the **risk model** (Sprint 4/5/6 — XGBoost ONNX `risk_model_b1.onnx` produced and deployed to backend) and the **smoke detector** (Sprint 2 — code ready, model not yet trained).

**Current state (2026-05-23):**
- ✅ `src/wildfire_ml/models/smoke_detector.py` — `SmokeDetector` (MobileNetV3-Small + binary sigmoid head)
- ✅ `src/wildfire_ml/train.py` — training loop
- ✅ `src/wildfire_ml/export.py` — ONNX export + INT8 dynamic quantization
- ✅ `scripts/download_pyrosdis.py` — pyro-sdis dataset (33,636 samples, 3.1 GB) downloadable, stored in HF cache
- ⚠️ Training **never run** → no checkpoint → `smoke_detector_int8.onnx` not produced → smoke inference does not actually run in either backend or mobile asset (graceful fallback `local_confidence=0` / `confidence_score=None`)
- ⚠️ Dataset is YOLO bbox format (`class x_c y_c w h`) but SmokeDetector is a binary classifier → an adapter (class==1 bbox present → label=1) is needed before training

### Setup
- Python 3.12+ (tested with 3.13)
- `pip install -r requirements.txt datasets`
- `datasets` is not in `requirements.txt` — needed separately for `scripts/download_pyrosdis.py`
- PyTorch default **CPU**; GPU needs the extra index URL (see CI)

### Train
- `python -m wildfire_ml.train --config ...` — script exists; CPU fine-tune of MobileNetV3 on 30k samples takes hours/days, GPU recommended. Annotation adapter must be applied (see dataset note).
- `python -m wildfire_ml.export` — trained checkpoint → ONNX INT8 export
- Output: `models/smoke_detector_int8.onnx` → copy to: backend `models/` + mobile `assets/models/`

### Risk model (separate pipeline)
- Under `src/wildfire_ml/risk/`: `train_risk.py`, `export_risk.py`, feature schema, etc.
- Output: `models/risk_model_b1.onnx` (READY, weight file is `.gitignore`'d)
- Decision #6: ML and backend preprocess kept in sync (`PREPROCESS_SYMMETRIC`)

### Test
- `pytest -q`

## Lisans / License

Apache 2.0 — see `LICENSE`.

## Atıflar / Credits

Detailed list: `CREDITS.md`.
