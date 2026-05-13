# Model Kartı — Wildfire Risk Classifier v1

## TR

### Amaç
30 günlük yangın riski ikili sınıflandırması. Pilot bölge: Beynam Ormanı (Bala/Ankara), 250m grid.

### Uyarı — Sentetik Veri
Sprint 4-B çıktısı **sentetik veri** ile eğitildi. PoC pipeline doğrulaması içindir; gerçek WorldCover/DEM/FIRMS/Open-Meteo verisi Sprint 5'te entegre edilecek. **Üretimde kullanılamaz.**

### Model Detayları
- Mimari: XGBoost binary classifier (sklearn API), `XGBClassifier`
- Versiyon: v1 (Sprint 4-B, 0.1.0a4)
- Feature sayısı: 24 (FEATURE_COLUMNS, config.py)
- Hedef: `fire_occurred_within_30d` binary

### Eğitim Verisi
- Kaynak: `SyntheticRiskDataGenerator` (synthetic-v1)
- 5000 sample, 60/20/20 stratified split

### Hiperparametreler (default XGBoostConfig)
max_depth=6, learning_rate=0.05, n_estimators=500, scale_pos_weight=auto, min_child_weight=5, eval_metric=aucpr, early_stopping_rounds=30

### Kısıtlamalar
- Sentetik veri: gerçek dağılımları yansıtmaz
- Pilot bölge dışında geçerlilik bilinmiyor
- Push trigger yok (Karar #7 uyumu)
- 112 bağlantısı yok (Karar #2 uyumu)

### KVKK
Anonim grid hücre bazlı; kişisel veri içermez.

### Lisans
Apache 2.0

### Atıflar
- XGBoost (Apache 2.0): Chen & Guestrin (2016)
- SHAP (MIT): Lundberg & Lee (2017)
- scikit-learn (BSD-3): Pedregosa et al. (2011)
- skl2onnx (MIT), onnxmltools (MIT)
- Kavzoglu, T., Sahin, E. K., & Sener, I. (2019). Modelling forest fire susceptibility in Mediterranean Türkiye via machine learning.

## EN

### Purpose
30-day wildfire risk binary classification. Pilot region: Beynam Forest (Bala/Ankara), 250m grid.

### Warning — Synthetic Data
Sprint 4-B output trained on **synthetic data**. For PoC pipeline validation only; real WorldCover/DEM/FIRMS/Open-Meteo integration scheduled for Sprint 5. **Not for production use.**

### Model Details
- Architecture: XGBoost binary classifier (sklearn API), `XGBClassifier`
- Version: v1 (Sprint 4-B, 0.1.0a4)
- Feature count: 24 (FEATURE_COLUMNS, config.py)
- Target: `fire_occurred_within_30d` binary

### Training Data
- Source: `SyntheticRiskDataGenerator` (synthetic-v1)
- 5000 samples, 60/20/20 stratified split

### Hyperparameters (default XGBoostConfig)
Same as TR section.

### Limitations
- Synthetic data: does not reflect real distributions
- Validity outside pilot region unknown
- No push trigger (Decision #7 compliance)
- No 112 connection (Decision #2 compliance)

### KVKK
Anonymous grid-cell based; no personal data.

### License
Apache 2.0

### Citations
Same as TR section.
