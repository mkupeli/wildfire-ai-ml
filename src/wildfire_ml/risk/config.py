# SPDX-License-Identifier: Apache-2.0
"""Risk haritası feature schema + RiskConfig dataclass.

PREPROCESS SYMMETRIC: Bu dosyadaki FEATURE_COLUMNS ve FEATURE_SCHEMA
backend `wildfire-ai-backend/app/services/risk_service.py` (Sprint 4-C)
ile simetrik olmak zorundadır. JSON Schema export (schema.json) backend'in
single source of truth'u olur.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    seed: int = 42
    n_samples: int = 5000
    grid_resolution_m: int = 250
    beynam_lat_min: float = 39.4
    beynam_lat_max: float = 39.6
    beynam_lon_min: float = 32.7
    beynam_lon_max: float = 33.0
    target_horizon_days: int = 30
    target_positive_rate: float = 0.15
    test_size: float = 0.20
    val_size: float = 0.20  # train'in bu yüzdesi val olur (60/20/20 sonuç)
    data_version: str = "synthetic-v1"


SYNTHETIC_DATA_NOTE = (
    "WARNING: synthetic placeholder data, NOT real observations. "
    "Use only for pipeline validation. Sprint 5 will replace with real "
    "WorldCover + DEM + FIRMS + Open-Meteo data."
)

TOPO_FEATURES: list[str] = [
    "elevation_m",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
]

WORLDCOVER_CLASSES: list[str] = [
    "lc_tree_cover",
    "lc_shrubland",
    "lc_grassland",
    "lc_cropland",
    "lc_built_up",
    "lc_bare_sparse",
    "lc_herbaceous_wetland",
    "lc_moss_lichen",
]

METEO_FEATURES: list[str] = [
    "temp_c",
    "rh_pct",
    "wind_speed_ms",
    "wind_dir_sin",
    "wind_dir_cos",
    "vpd_kpa",
    "ffmc_approx",
]

FIRE_HIST_FEATURES: list[str] = [
    "firms_density_1yr",
    "days_since_last_fire",
]

HUMAN_FEATURES: list[str] = [
    "road_dist_m",
    "settlement_dist_m",
    "picnic_area",
]

FEATURE_COLUMNS: list[str] = (
    TOPO_FEATURES
    + WORLDCOVER_CLASSES
    + METEO_FEATURES
    + FIRE_HIST_FEATURES
    + HUMAN_FEATURES
)  # 4 + 8 + 7 + 2 + 3 = 24

assert len(FEATURE_COLUMNS) == 24, f"Beklenen 24 feature, got {len(FEATURE_COLUMNS)}"

TARGET_COLUMN: str = "fire_occurred_within_30d"

FEATURE_SCHEMA: dict = {
    # Topografya
    "elevation_m": {"dtype": "float32", "min": 900.0, "max": 1600.0, "unit": "m", "description": "Deniz seviyesinden yükseklik"},
    "slope_deg": {"dtype": "float32", "min": 0.0, "max": 45.0, "unit": "deg", "description": "Yüzey eğimi"},
    "aspect_sin": {"dtype": "float32", "min": -1.0, "max": 1.0, "unit": "unitless", "description": "sin(aspect) yön sürekliliği"},
    "aspect_cos": {"dtype": "float32", "min": -1.0, "max": 1.0, "unit": "unitless", "description": "cos(aspect) yön sürekliliği"},
    # Land cover one-hot (8)
    **{cls: {"dtype": "int8", "min": 0, "max": 1, "unit": "binary", "description": f"WorldCover one-hot {cls}"} for cls in WORLDCOVER_CLASSES},
    # Meteoroloji
    "temp_c": {"dtype": "float32", "min": -10.0, "max": 45.0, "unit": "C", "description": "2m sıcaklık"},
    "rh_pct": {"dtype": "float32", "min": 5.0, "max": 100.0, "unit": "pct", "description": "Bağıl nem"},
    "wind_speed_ms": {"dtype": "float32", "min": 0.0, "max": 25.0, "unit": "m/s", "description": "10m rüzgar hızı"},
    "wind_dir_sin": {"dtype": "float32", "min": -1.0, "max": 1.0, "unit": "unitless", "description": "sin(wind_dir)"},
    "wind_dir_cos": {"dtype": "float32", "min": -1.0, "max": 1.0, "unit": "unitless", "description": "cos(wind_dir)"},
    "vpd_kpa": {"dtype": "float32", "min": 0.0, "max": 10.0, "unit": "kPa", "description": "Vapor pressure deficit (VPD)"},
    "ffmc_approx": {"dtype": "float32", "min": 0.0, "max": 101.0, "unit": "unitless", "description": "FFMC yaklaşımı (Van Wagner 1987 single-step)"},
    # Fire history
    "firms_density_1yr": {
        "dtype": "float32", "min": 0.0, "max": 100.0, "unit": "count",
        "description": (
            "FIRMS hotspot density (10km, 1yr). "
            "PoC: sentetik veri Poisson(0.3) ile üretildi, gerçek max ~4-5. "
            "Schema max=100 gerçek veri için geniş band; Sprint 5'te gözden geçirilecek."
        )
    },
    "days_since_last_fire": {"dtype": "float32", "min": 0.0, "max": 3650.0, "unit": "days", "description": "Son FIRMS yangınından beri"},
    # Human pressure
    "road_dist_m": {"dtype": "float32", "min": 0.0, "max": 5000.0, "unit": "m", "description": "En yakın yol mesafesi"},
    "settlement_dist_m": {"dtype": "float32", "min": 0.0, "max": 10000.0, "unit": "m", "description": "En yakın yerleşim mesafesi"},
    "picnic_area": {"dtype": "int8", "min": 0, "max": 1, "unit": "binary", "description": "Piknik/kamp alanı"},
    # Target
    TARGET_COLUMN: {"dtype": "int8", "min": 0, "max": 1, "unit": "binary", "description": "30 gün içinde yangın olayı"},
}

# Gelecek (Sprint 5 gerçek MGM verisi):
# FUTURE_FEATURES = ["dmc", "dc", "isi", "bui", "fwi_index", "dsr"]
