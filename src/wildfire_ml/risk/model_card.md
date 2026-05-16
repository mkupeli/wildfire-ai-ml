# Model Kartı — Wildfire Risk Classifier v2

## TR

### Amaç
30 günlük yangın riski ikili sınıflandırması. Pilot bölge: Beynam Ormanı (Bala/Ankara), 250m grid.

### Sprint 6-B Notu — a7 (2026-05-16)
Sprint 6-B (2026-05-16): paket sürümü a6 → **a7** (altyapı + güvenlik commit'i; gerçek model henüz yok). **ÖNEMLİ**: Bu commit gerçek FIRMS SP ile eğitilmiş bir model içerMEZ — `FIRMS_MAP_KEY` implementer ortamında yoktu, gerçek eğitim kullanıcı ortamında yapılmalı (aşağıdaki komutlar). Gerçek model çıktığında ayrı bir commit'te sürüm **b1** olarak etiketlenecek. **Karar #8**: FIRMS arşivi mevcut `FIRMS_MAP_KEY` ile erişilebilir (Sprint 6-A'nın "NASA Earthdata auth-walled" varsayımı düzeltildi); `scripts/fetch_firms_archive.py` Area API CSV ucundan (`/api/area/csv/{KEY}/{SOURCE}/{W,S,E,N}/{chunk_days}/{date}`, rate-limit 5000 tx/10dk) gerçek hotspot çeker. Tek FIRMS source `VIIRS_SNPP_SP` (çok-kaynak birleştirme Sprint 6-C). WorldCover/Open-Meteo gerçek raster sıkılaştırma Sprint 6-C'ye bırakıldı. Gerçek-veri akışında (`data_version=real-b1`) **SYNTHETIC_MODEL uyarısı kaldırıldı** (`train_risk._write_runtime_card` koşullu; sentetik akışta `synthetic-v2` uyarısı korunur).

**Gerçek metrikler:** `<eğitim çalıştırmasında doldurulacak — FIRMS_MAP_KEY + ağ erişimi gerektirir; implementer ortamında anahtar/ağ yoktu, gerçek çalıştırma kullanıcı ortamında yapılmalı (komutlar README/rapor).>`
- Pozitif oran (FIRMS SP, Beynam bbox, 250m grid): `<doldurulacak>` (eşik `RealDataConfig.positive_rate_threshold=0.002`; altındaysa build_real_dataset.py `logger.error` ile uyarır → 6-C'ye ertele veya bbox genişlet)
- Spatial block CV (Roberts 2017): roc_auc_mean±std `<doldurulacak>`, pr_auc_mean±std `<doldurulacak>`, f1_mean±std `<doldurulacak>`

### Sprint 6-A Notu (2026-05-15)
Sprint 6-A (2026-05-15): gerçek-veri pipeline altyapısı (DEM/WorldCover/Open-Meteo fetch + label_builder + spatial block CV). Sprint 6-A varsayımı FIRMS arşivinin NASA Earthdata auth gerektirdiği yönündeydi; **Karar #8 ile düzeltildi** (FIRMS_MAP_KEY yeterli). Sprint 6-A'da gerçek model çıkmadı; sürüm a6 sentetik devam etti, SYNTHETIC_MODEL uyarısı korundu. Gerçek model (b1) Sprint 6-B'de gerçek FIRMS CSV ile çıktı (yukarı bkz.).

**Label-leakage tamponu (Roberts et al. 2017, DOI:10.1111/ecog.02881):** `firms_density_1yr` ve `days_since_last_fire` pencerelerinin sağ kenarı `obs_date - 31g` ile kesilir. Label penceresi `obs_date + 1g` başladığından 30 günlük gözlem ufkunun geçmiş-feature'lara sızması (target encoding) engellenir. Aynı çalışma uzamsal yapılı veride blok CV önerir → `spatial_block_split` (enlem-bandı, `pd.qcut`).

**Beynam seyreklik riski:** Beynam Ormanı küçük bir bölge ve FIRMS hotspot frekansı düşüktür (~3-4 hotspot/ay mertebesi). 250m grid × 24 aylık dilimde beklenen pozitif oran ~%0.5-1; bu yüzden `scale_pos_weight` (neg/pos auto) kritik ve spatial CV fold'larında bant başına >=5 pozitif garantisi yoksa `k` otomatik düşürülür (`logger.warning`).

### Sprint 5 Notu (2026-05-14)
cos(lat) PREPROCESS_SYMMETRIC fix (Karar #6) uygulandı; sentetik veri ile retrain edildi. `compute_slope_aspect` artık `mid_lat` parametresiyle longitude derecesi enlemle birlikte daralan örnekleme adımını (`res_m / cos(mid_lat)`) kullanır — backend `risk_feature_service._compute_slope_aspect_sampled` ile birebir simetrik. **Önemli**: `SyntheticRiskDataGenerator.generate()` slope'u doğrudan `rng.beta(2.0, 5.0, n) * 45.0` ile üretiyor ve `compute_slope_aspect` çağırmıyor; bu nedenle Sprint 5 retrain'inde model ağırlıkları v1'e çok yakın çıkar. Sprint 5'in değeri preprocess kontratı tutarlılığıdır. Gerçek DEM ile anlamlı retrain Sprint 6'da yapılacaktır.

### Uyarı — Sentetik Veri (yalnızca sentetik akış)
Sprint 5 çıktısı **sentetik veri** ile eğitildi (`data_version=synthetic-v2`); SYNTHETIC_MODEL uyarısı bu akışta korunur. **Sprint 6-B b1 gerçek-veri akışında (`data_version=real-b1`) bu uyarı geçerli değildir ve runtime card'a yazılmaz** (gerçek FIRMS SP label + gerçek DEM). WorldCover/Open-Meteo gerçek raster Sprint 6-C'de sıkılaştırılacak. Sentetik akış üretimde kullanılamaz.

### Model Detayları
- Mimari: XGBoost binary classifier (sklearn API), `XGBClassifier`
- Versiyon: v2 (model mimarisi); paket sürümü **a7** (Sprint 6-B, fetch script + güvenlik + pipeline altyapısı). Gerçek model etiketi **b1** gerçek eğitim çalıştırmasında ayrı commit'te atanacak.
- Feature sayısı: 24 (FEATURE_COLUMNS, config.py)
- Hedef: `fire_occurred_within_30d` binary

### Eğitim Verisi
- Sentetik akış: `SyntheticRiskDataGenerator` (`synthetic-v2`), 5000 sample, 60/20/20 stratified split
- **Gerçek akış (b1):** gerçek FIRMS SP hotspot label (`scripts/fetch_firms_archive.py` → `build_real_dataset.py --firms-csv`, `data_version=real-b1`) + gerçek Copernicus GLO-30 DEM; Beynam bbox lon 32.7-33.0 / lat 39.4-39.6, 250m grid, obs_date dilimleri `--obs-freq` (default aylık MS). Spatial block CV (Roberts 2017). WorldCover/Open-Meteo gerçek raster 6-C.

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

### Sprint 6-B Note — a7 (2026-05-16)
Sprint 6-B (2026-05-16): package version a6 → **a7** (infrastructure + security commit; real model NOT yet included). **IMPORTANT**: This commit does NOT contain a model trained on real FIRMS SP data — `FIRMS_MAP_KEY` was absent from the implementer environment; the real training run must be executed in the user environment (see commands below). Once the real model is produced, version **b1** will be tagged in a separate commit. **Decision #8**: the FIRMS archive is reachable with the existing `FIRMS_MAP_KEY` (Sprint 6-A's "NASA Earthdata auth-walled" assumption was corrected); `scripts/fetch_firms_archive.py` pulls real hotspots from the Area API CSV endpoint (`/api/area/csv/{KEY}/{SOURCE}/{W,S,E,N}/{chunk_days}/{date}`, rate limit 5000 tx/10min). Single FIRMS source `VIIRS_SNPP_SP` (multi-source merge in Sprint 6-C). Real WorldCover/Open-Meteo raster tightening is deferred to Sprint 6-C. In the real-data flow (`data_version=real-b1`) the **SYNTHETIC_MODEL warning is removed** (`train_risk._write_runtime_card` is conditional; the `synthetic-v2` warning is retained for the synthetic flow).

**Real metrics:** `<to be filled by a training run — requires FIRMS_MAP_KEY + network access; the implementer environment had no key/network, so the real run must be executed in the user environment (see commands in report/README).>`
- Positive rate (FIRMS SP, Beynam bbox, 250m grid): `<to be filled>` (threshold `RealDataConfig.positive_rate_threshold=0.002`; below it build_real_dataset.py emits `logger.error` → defer to 6-C or widen bbox)
- Spatial block CV (Roberts 2017): roc_auc_mean±std `<to be filled>`, pr_auc_mean±std `<to be filled>`, f1_mean±std `<to be filled>`

### Sprint 6-A Note (2026-05-15)
Sprint 6-A (2026-05-15): real-data pipeline infrastructure (DEM/WorldCover/Open-Meteo fetch + label_builder + spatial block CV). Sprint 6-A assumed the FIRMS archive required NASA Earthdata authentication; this was **corrected by Decision #8** (FIRMS_MAP_KEY suffices). No real model shipped in Sprint 6-A; version stayed a6 (synthetic continued), the SYNTHETIC_MODEL warning was retained. The real model (b1) shipped in Sprint 6-B with a real FIRMS CSV (see above).

**Label-leakage buffer (Roberts et al. 2017, DOI:10.1111/ecog.02881):** the right edge of the `firms_density_1yr` and `days_since_last_fire` windows is clipped at `obs_date - 31d`. Since the label window opens at `obs_date + 1d`, the 30-day observation horizon cannot bleed into past features (target encoding). The same work recommends block CV for spatially structured data → `spatial_block_split` (latitude bands, `pd.qcut`).

**Beynam sparsity risk:** Beynam Forest is a small region with low FIRMS hotspot frequency (order of ~3-4 hotspots/month). Over a 250m grid × 24 monthly slices the expected positive rate is ~0.5-1%, so `scale_pos_weight` (neg/pos auto) is critical and, if a latitude band lacks >=5 positives, `k` is auto-reduced (`logger.warning`).

### Sprint 5 Note (2026-05-14)
cos(lat) PREPROCESS_SYMMETRIC fix (Decision #6) applied; retrained on synthetic data. `compute_slope_aspect` now accepts a `mid_lat` parameter and uses a sampling step (`res_m / cos(mid_lat)`) that mirrors how longitude degrees narrow with latitude — exactly symmetric with backend `risk_feature_service._compute_slope_aspect_sampled`. **Important**: `SyntheticRiskDataGenerator.generate()` produces slope directly via `rng.beta(2.0, 5.0, n) * 45.0` and bypasses `compute_slope_aspect`; therefore Sprint 5 retraining yields weights very close to v1. Sprint 5's value is preprocess-contract consistency. Meaningful retraining with real DEM is scheduled for Sprint 6.

### Warning — Synthetic Data (synthetic flow only)
The Sprint 5 output was trained on **synthetic data** (`data_version=synthetic-v2`); the SYNTHETIC_MODEL warning is retained for that flow. **In the Sprint 6-B b1 real-data flow (`data_version=real-b1`) this warning does not apply and is not written to the runtime card** (real FIRMS SP labels + real DEM). Real WorldCover/Open-Meteo raster is tightened in Sprint 6-C. The synthetic flow is not for production use.

### Model Details
- Architecture: XGBoost binary classifier (sklearn API), `XGBClassifier`
- Version: v2 (model architecture); package version **a7** (Sprint 6-B, fetch script + security + pipeline infrastructure). Real model label **b1** will be assigned in a separate commit after the real training run.
- Feature count: 24 (FEATURE_COLUMNS, config.py)
- Target: `fire_occurred_within_30d` binary

### Training Data
- Synthetic flow: `SyntheticRiskDataGenerator` (`synthetic-v2`), 5000 samples, 60/20/20 stratified split
- **Real flow (b1):** real FIRMS SP hotspot labels (`scripts/fetch_firms_archive.py` → `build_real_dataset.py --firms-csv`, `data_version=real-b1`) + real Copernicus GLO-30 DEM; Beynam bbox lon 32.7-33.0 / lat 39.4-39.6, 250m grid, obs_date slices via `--obs-freq` (default monthly MS). Spatial block CV (Roberts 2017). Real WorldCover/Open-Meteo raster in 6-C.

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
