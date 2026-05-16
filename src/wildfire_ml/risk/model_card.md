# Model Kartı — Wildfire Risk Classifier v2

## TR

### Amaç
30 günlük yangın riski ikili sınıflandırması. Pilot bölge: Beynam Ormanı (Bala/Ankara), 250m grid.

### Sprint 6-B Notu — b1 (2026-05-16)
Sprint 6-B (2026-05-16): **gerçek model b1 eğitildi** (paket sürümü → **b1**). Gerçek `FIRMS_MAP_KEY` ile `scripts/fetch_firms_archive.py` Beynam bbox 2024-01..2025-12 arşivinden VIIRS_SNPP_SP hotspot çekti (147 chunk, chunk_days=5). **Karar #8**: FIRMS arşivi mevcut `FIRMS_MAP_KEY` ile erişilebilir (Sprint 6-A'nın "NASA Earthdata auth-walled" varsayımı düzeltildi). Gerçek-çalıştırmada 5 bug bulundu+düzeltildi: (1) httpx logger `FIRMS_MAP_KEY` sızıntısı, (2) `DEFAULT_CHUNK_DAYS` 7→5 (FIRMS Area API day_range ≤5), (3) `label_builder` VIIRS tek-harf confidence (l/n/h) normalizasyonu, (4) `export_risk` onnxmltools feature_names temizleme, (5) `export_risk` `ai.onnx.ml` opset uyumu. Tek FIRMS source `VIIRS_SNPP_SP` (çok-kaynak 6-C). Gerçek DEM + gerçek Open-Meteo + gerçek FIRMS label kullanıldı; WorldCover one-hot sentetik kaldı (gerçek raster sınıflama Sprint 6-C). Gerçek-veri akışında (`data_version=real-b1`) **SYNTHETIC_MODEL uyarısı yazılmaz** (`train_risk._write_runtime_card` koşullu; sentetik akışta `synthetic-v2` uyarısı korunur).

**Gerçek metrikler (2026-05-16; gerçek FIRMS SP + Copernicus GLO-30 DEM + Open-Meteo ERA5):**
- FIRMS: 24 ayda 7 ham hotspot → confidence filtresi sonrası **5** (nominal; 2 low/sun-glint elendi)
- Pozitif oran: **%8.25** (0.0825); 220008 satır (9167 grid hücre × 24 aylık obs_date). Öngörülen ~%0.5-1'den yüksek — 10km FIRMS haversine yarıçapı küçük Beynam bbox'ında geniş kapsıyor (eşik 0.002 fazlasıyla aşıldı; seyreklik fallback gerekmedi).
- Spatial block CV (Roberts 2017, 5 fold, leakage-safe): ROC-AUC **0.819 ±0.037**, PR-AUC 0.471 ±0.128, F1 0.279 ±0.120 (toplam pozitif 18143)
- Test (holdout, 60/20/20 stratified, n_test=44002): ROC-AUC **0.858**, PR-AUC 0.566, F1 0.314, Precision 0.198, Recall **0.761**, scale_pos_weight 11.13
- ONNX: `risk_model_b1.onnx`, smoke test passed (XGBoost↔ONNX rtol 1e-3, atol 1e-4)
- Yorum: yüksek recall (0.76) "riskli bölge kaçırmama" önceliğine uygun; düşük precision (0.20) seyrek-pozitif + risk-haritası tarama bağlamında kabul edilebilir. WorldCover one-hot sentetik olduğundan land-cover sinyali eksik — gerçek raster 6-C'de eklenince iyileşmesi beklenir.

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
- Versiyon: v2 (model mimarisi); paket sürümü **b1** (Sprint 6-B, gerçek FIRMS SP + DEM + Open-Meteo ile eğitildi 2026-05-16). Artefaktlar: `models/risk_model_v2.ubj` (XGBoost), `models/risk_model_b1.onnx` (backend), `models/risk_feature_schema.json`.
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

### Sprint 6-B Note — b1 (2026-05-16)
Sprint 6-B (2026-05-16): **real model b1 trained** (package version → **b1**). With a real `FIRMS_MAP_KEY`, `scripts/fetch_firms_archive.py` pulled VIIRS_SNPP_SP hotspots from the Beynam bbox 2024-01..2025-12 archive (147 chunks, chunk_days=5). **Decision #8**: the FIRMS archive is reachable with the existing `FIRMS_MAP_KEY` (Sprint 6-A's "NASA Earthdata auth-walled" assumption corrected). The real run surfaced+fixed 5 bugs: (1) httpx logger `FIRMS_MAP_KEY` leak, (2) `DEFAULT_CHUNK_DAYS` 7→5 (FIRMS Area API day_range ≤5), (3) `label_builder` VIIRS single-letter confidence (l/n/h) normalization, (4) `export_risk` onnxmltools feature_names clearing, (5) `export_risk` `ai.onnx.ml` opset compatibility. Single FIRMS source `VIIRS_SNPP_SP` (multi-source in 6-C). Real DEM + real Open-Meteo + real FIRMS labels used; WorldCover one-hot stayed synthetic (real raster classification in Sprint 6-C). In the real-data flow (`data_version=real-b1`) the **SYNTHETIC_MODEL warning is not written** (`train_risk._write_runtime_card` conditional; `synthetic-v2` warning retained for the synthetic flow).

**Real metrics (2026-05-16; real FIRMS SP + Copernicus GLO-30 DEM + Open-Meteo ERA5):**
- FIRMS: 7 raw hotspots over 24 months → 5 after confidence filter (nominal; 2 low/sun-glint dropped)
- Positive rate: **8.25%** (0.0825); 220008 rows (9167 grid cells × 24 monthly obs_date). Higher than the predicted ~0.5-1% — the 10km FIRMS haversine radius covers a large share of the small Beynam bbox (threshold 0.002 far exceeded; sparsity fallback not needed).
- Spatial block CV (Roberts 2017, 5 folds, leakage-safe): ROC-AUC **0.819 ±0.037**, PR-AUC 0.471 ±0.128, F1 0.279 ±0.120 (total positives 18143)
- Test (holdout, 60/20/20 stratified, n_test=44002): ROC-AUC **0.858**, PR-AUC 0.566, F1 0.314, Precision 0.198, Recall **0.761**, scale_pos_weight 11.13
- ONNX: `risk_model_b1.onnx`, smoke test passed (XGBoost↔ONNX rtol 1e-3, atol 1e-4)
- Note: high recall (0.76) fits the "don't miss risky areas" priority; low precision (0.20) is acceptable for a sparse-positive risk-map screening context. Land-cover signal is missing because WorldCover one-hot is synthetic — expected to improve once real raster lands in 6-C.

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
- Version: v2 (model architecture); package version **b1** (Sprint 6-B, trained on real FIRMS SP + DEM + Open-Meteo 2026-05-16). Artifacts: `models/risk_model_v2.ubj` (XGBoost), `models/risk_model_b1.onnx` (backend), `models/risk_feature_schema.json`.
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
