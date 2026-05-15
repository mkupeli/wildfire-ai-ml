# Model Kartı — Wildfire Risk Classifier v2

## TR

### Amaç
30 günlük yangın riski ikili sınıflandırması. Pilot bölge: Beynam Ormanı (Bala/Ankara), 250m grid.

### Sprint 5 Notu (2026-05-14)
cos(lat) PREPROCESS_SYMMETRIC fix (Karar #6) uygulandı; sentetik veri ile retrain edildi. `compute_slope_aspect` artık `mid_lat` parametresiyle longitude derecesi enlemle birlikte daralan örnekleme adımını (`res_m / cos(mid_lat)`) kullanır — backend `risk_feature_service._compute_slope_aspect_sampled` ile birebir simetrik. **Önemli**: `SyntheticRiskDataGenerator.generate()` slope'u doğrudan `rng.beta(2.0, 5.0, n) * 45.0` ile üretiyor ve `compute_slope_aspect` çağırmıyor; bu nedenle Sprint 5 retrain'inde model ağırlıkları v1'e çok yakın çıkar. Sprint 5'in değeri preprocess kontratı tutarlılığıdır. Gerçek DEM ile anlamlı retrain Sprint 6'da yapılacaktır.

### Uyarı — Sentetik Veri
Sprint 5 çıktısı **sentetik veri** ile eğitildi. PoC pipeline doğrulaması içindir; gerçek WorldCover/DEM/FIRMS/Open-Meteo verisi Sprint 6'da entegre edilecek. **Üretimde kullanılamaz.**

### Model Detayları
- Mimari: XGBoost binary classifier (sklearn API), `XGBClassifier`
- Versiyon: v2 (Sprint 5, 0.1.0a5)
- Feature sayısı: 24 (FEATURE_COLUMNS, config.py)
- Hedef: `fire_occurred_within_30d` binary

### Eğitim Verisi
- Kaynak: `SyntheticRiskDataGenerator` (synthetic-v2)
- 5000 sample, 60/20/20 stratified split

### Hiperparametreler (default XGBoostConfig)
max_depth=6, learning_rate=0.05, n_estimators=500, scale_pos_weight=auto, min_child_weight=5, eval_metric=aucpr, early_stopping_rounds=30

### Kısıtlamalar
- Sentetik veri: gerçek dağılımları yansıtmaz
- Sentetik slope üretimi gradient pipeline'ı bypass eder → cos(lat) fix sayısal ağırlıkları neredeyse etkilemez (Sprint 5 minimum scope)
- Pilot bölge dışında geçerlilik bilinmiyor
- Push trigger yok (Karar #7 uyumu)
- 112 bağlantısı yok (Karar #2 uyumu)

### Kararlar
- Karar #6: PREPROCESS_SYMMETRIC — backend ile ML aynı preprocess formüllerini kullanır (cos(lat) düzeltmesi dahil)
- Karar #7: filter v0 — push trigger yok

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

### Sprint 5 Note (2026-05-14)
cos(lat) PREPROCESS_SYMMETRIC fix (Decision #6) applied; retrained on synthetic data. `compute_slope_aspect` now accepts a `mid_lat` parameter and uses a sampling step (`res_m / cos(mid_lat)`) that mirrors how longitude degrees narrow with latitude — exactly symmetric with backend `risk_feature_service._compute_slope_aspect_sampled`. **Important**: `SyntheticRiskDataGenerator.generate()` produces slope directly via `rng.beta(2.0, 5.0, n) * 45.0` and bypasses `compute_slope_aspect`; therefore Sprint 5 retraining yields weights very close to v1. Sprint 5's value is preprocess-contract consistency. Meaningful retraining with real DEM is scheduled for Sprint 6.

### Warning — Synthetic Data
Sprint 5 output trained on **synthetic data**. For PoC pipeline validation only; real WorldCover/DEM/FIRMS/Open-Meteo integration scheduled for Sprint 6. **Not for production use.**

### Model Details
- Architecture: XGBoost binary classifier (sklearn API), `XGBClassifier`
- Version: v2 (Sprint 5, 0.1.0a5)
- Feature count: 24 (FEATURE_COLUMNS, config.py)
- Target: `fire_occurred_within_30d` binary

### Training Data
- Source: `SyntheticRiskDataGenerator` (synthetic-v2)
- 5000 samples, 60/20/20 stratified split

### Hyperparameters (default XGBoostConfig)
Same as TR section.

### Limitations
- Synthetic data: does not reflect real distributions
- Synthetic slope generator bypasses the gradient pipeline → the cos(lat) fix has near-zero effect on numerical weights (Sprint 5 minimum scope)
- Validity outside pilot region unknown
- No push trigger (Decision #7 compliance)
- No 112 connection (Decision #2 compliance)

### Decisions
- Decision #6: PREPROCESS_SYMMETRIC — backend and ML share identical preprocess formulas (including the cos(lat) correction)
- Decision #7: filter v0 — no push trigger

### KVKK
Anonymous grid-cell based; no personal data.

### License
Apache 2.0

### Citations
Same as TR section.
