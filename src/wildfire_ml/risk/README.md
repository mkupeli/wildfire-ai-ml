# Risk Haritası ML Modülü (Sprint 4-A)

## TR

Türkiye yangın risk haritası için XGBoost modelinin veri katmanı.

### Kapsam
- 24 feature + 1 target (binary, 30 gün)
- Topografya (DEM), Land cover (WorldCover 8 sınıf one-hot), Meteoroloji (Open-Meteo + VPD/FFMC), Yangın tarihçesi (FIRMS), İnsan baskısı (OSM)
- Beynam bbox (~39.4-39.6 lat, 32.7-33.0 lng), 250m grid

### Uyarı
Sprint 4-A çıktısı **sentetik veri**dir. Gerçek WorldCover/DEM/FIRMS/Open-Meteo entegrasyonu Sprint 5'te (saha denemesi öncesi) yapılacak.

### Kullanım
```python
from wildfire_ml.risk import SyntheticRiskDataGenerator, RiskConfig, load_risk_dataset, train_val_test_split

cfg = RiskConfig(seed=42, n_samples=5000)
df = load_risk_dataset(cfg=cfg)  # path=None → synthetic
train, val, test = train_val_test_split(df, cfg)
```

### PREPROCESS SYMMETRIC
`config.py::FEATURE_COLUMNS` + `schema.json` backend Sprint 4-C `risk_service.py` ile simetrik olmak zorundadır. JSON Schema export değişiklik kontrolü için tek kaynak.

---

## EN

Data layer for the wildfire risk map XGBoost model in Türkiye.

### Scope
- 24 features + 1 binary target (30-day horizon)
- Topography (DEM), Land cover (WorldCover 8-class one-hot), Meteorology (Open-Meteo + VPD/FFMC), Fire history (FIRMS), Human pressure (OSM)
- Beynam bbox (~39.4-39.6 lat, 32.7-33.0 lng), 250m grid

### Warning
Sprint 4-A output is **synthetic data**. Real WorldCover/DEM/FIRMS/Open-Meteo integration scheduled for Sprint 5 (before field trial).

### Usage
```python
from wildfire_ml.risk import SyntheticRiskDataGenerator, RiskConfig, load_risk_dataset, train_val_test_split

cfg = RiskConfig(seed=42, n_samples=5000)
df = load_risk_dataset(cfg=cfg)  # path=None → synthetic
train, val, test = train_val_test_split(df, cfg)
```

### PREPROCESS SYMMETRIC
`config.py::FEATURE_COLUMNS` + `schema.json` must remain symmetric with backend Sprint 4-C `risk_service.py`. JSON Schema export is the single source of truth for change tracking.
