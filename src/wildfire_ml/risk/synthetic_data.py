# SPDX-License-Identifier: Apache-2.0
"""Synthetic risk dataset generator.

WARNING: Bu modül PoC için sahte ama plausible feature dağılımları üretir.
GERÇEK VERİ DEĞİL — Sprint 5'te gerçek WorldCover/DEM/FIRMS/Open-Meteo ile değiştirilecek.

Reproducibility: RiskConfig.seed sabit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FEATURE_COLUMNS,
    HUMAN_FEATURES,  # noqa: F401 — explicit import for clarity
    SYNTHETIC_DATA_NOTE,
    TARGET_COLUMN,
    TOPO_FEATURES,  # noqa: F401
    WORLDCOVER_CLASSES,
    METEO_FEATURES,  # noqa: F401
    FIRE_HIST_FEATURES,  # noqa: F401
    RiskConfig,
)
from .features import compute_ffmc_approx, compute_vpd


class SyntheticRiskDataGenerator:
    """Beynam bbox için sentetik risk feature DataFrame'i üretir."""

    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self.cfg = cfg or RiskConfig()

    def generate(self, n: int | None = None) -> pd.DataFrame:
        n = self.cfg.n_samples if n is None else n
        rng = np.random.default_rng(self.cfg.seed)

        data: dict[str, np.ndarray] = {}
        # Koordinatlar — bilgi amaçlı, FEATURE_COLUMNS'ta değil ama metadata olarak
        lat = rng.uniform(self.cfg.beynam_lat_min, self.cfg.beynam_lat_max, n).astype(np.float64)
        lon = rng.uniform(self.cfg.beynam_lon_min, self.cfg.beynam_lon_max, n).astype(np.float64)

        # Topografya
        data["elevation_m"] = rng.uniform(900.0, 1600.0, n).astype(np.float32)
        data["slope_deg"] = (rng.beta(2.0, 5.0, n) * 45.0).astype(np.float32)
        aspect_deg = rng.uniform(0.0, 360.0, n)
        data["aspect_sin"] = np.sin(np.radians(aspect_deg)).astype(np.float32)
        data["aspect_cos"] = np.cos(np.radians(aspect_deg)).astype(np.float32)

        # Land cover — multinomial (Beynam'da forest+shrub+grass ağırlıklı)
        lc_probs = [0.40, 0.20, 0.15, 0.10, 0.05, 0.05, 0.03, 0.02]  # 8 sınıf, toplam=1
        lc_idx = rng.choice(len(WORLDCOVER_CLASSES), size=n, p=lc_probs)
        for i, cls in enumerate(WORLDCOVER_CLASSES):
            data[cls] = (lc_idx == i).astype(np.int8)

        # Meteoroloji
        data["temp_c"] = rng.normal(25.0, 8.0, n).clip(-10.0, 45.0).astype(np.float32)
        data["rh_pct"] = (rng.beta(2.0, 5.0, n) * 100.0).clip(5.0, 100.0).astype(np.float32)
        data["wind_speed_ms"] = rng.exponential(3.0, n).clip(0.0, 25.0).astype(np.float32)
        wind_dir_deg = rng.uniform(0.0, 360.0, n)
        data["wind_dir_sin"] = np.sin(np.radians(wind_dir_deg)).astype(np.float32)
        data["wind_dir_cos"] = np.cos(np.radians(wind_dir_deg)).astype(np.float32)
        # Türev: VPD + FFMC
        data["vpd_kpa"] = compute_vpd(data["temp_c"], data["rh_pct"])
        data["ffmc_approx"] = compute_ffmc_approx(data["temp_c"], data["rh_pct"], data["wind_speed_ms"])

        # Fire history
        data["firms_density_1yr"] = rng.poisson(0.3, n).astype(np.float32)
        data["days_since_last_fire"] = rng.exponential(500.0, n).clip(0.0, 3650.0).astype(np.float32)

        # Human pressure
        data["road_dist_m"] = rng.exponential(800.0, n).clip(0.0, 5000.0).astype(np.float32)
        data["settlement_dist_m"] = rng.exponential(2000.0, n).clip(0.0, 10000.0).astype(np.float32)
        data["picnic_area"] = (rng.random(n) < 0.05).astype(np.int8)

        df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
        df[TARGET_COLUMN] = self._generate_target(df, rng)

        # Metadata
        df.attrs["source"] = SYNTHETIC_DATA_NOTE
        df.attrs["data_version"] = self.cfg.data_version
        df.attrs["lat"] = lat  # info-only, NOT in FEATURE_COLUMNS
        df.attrs["lon"] = lon

        return df

    def _generate_target(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """Feature-correlated logistic scoring → binary target."""
        if len(df) == 0:
            return np.array([], dtype=np.int8)
        # Yüksek risk = yüksek VPD + düşük RH + tree_cover/shrubland + yakın picnic + düşük settlement_dist
        score = (
            0.15 * (df["vpd_kpa"] / 5.0)
            + 0.10 * (1.0 - df["rh_pct"] / 100.0)
            + 0.15 * df["lc_tree_cover"]
            + 0.10 * df["lc_shrubland"]
            + 0.10 * df["picnic_area"]
            + 0.05 * (df["ffmc_approx"] / 100.0)
            + 0.05 * (1.0 - df["settlement_dist_m"] / 10000.0)
            + 0.05 * (df["firms_density_1yr"] / 5.0)
            - 0.5  # bias to hit ~%15 positive
        )
        prob = 1.0 / (1.0 + np.exp(-score * 2.0))
        # Adjust bias to hit target rate roughly
        threshold = np.quantile(prob, 1.0 - self.cfg.target_positive_rate)
        return (prob >= threshold).astype(np.int8)
