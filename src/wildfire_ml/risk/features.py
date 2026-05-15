# SPDX-License-Identifier: Apache-2.0
"""Feature engineering helpers — VPD, slope/aspect, FIRMS density, FFMC approx.

PREPROCESS SYMMETRIC: Backend Sprint 4-C inference de bu fonksiyonları
(veya birebir aynı formülleri) kullanmalı.
"""
from __future__ import annotations

from math import radians as _radians, cos as _cos

import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd


def compute_vpd(temp_c: np.ndarray | float, rh_pct: np.ndarray | float) -> np.ndarray:
    """Tetens formülü ile vapor pressure deficit (kPa).

    e_sat = 0.6112 * exp(17.67 * T / (T + 243.5))
    VPD = e_sat * (1 - RH/100)
    Clip [0, 10].
    """
    t = np.asarray(temp_c, dtype=np.float64)
    rh = np.asarray(rh_pct, dtype=np.float64)
    e_sat = 0.6112 * np.exp(17.67 * t / (t + 243.5))
    vpd = e_sat * (1.0 - rh / 100.0)
    return np.clip(vpd, 0.0, 10.0).astype(np.float32)


def compute_slope_aspect(
    elevation_grid: np.ndarray,
    res_m: float,
    mid_lat: float = 39.5,
) -> tuple[np.ndarray, np.ndarray]:
    """DEM grid'inden slope (deg) ve aspect (deg) hesapla — gradient based.

    PREPROCESS SYMMETRIC (Karar #6): bbox orta enleminde cos(lat) faktörü
    uygulanır; longitude derecesi enlemle birlikte daralır, bu yüzden
    doğu-batı yönündeki örnekleme adımı `res_m / cos(mid_lat)` olur.
    Kuzey-güney adımı (`res_m`) sabittir. Beynam (~39.5°N) default;
    Kızılcahamam (~40.4°N) için çağrı `mid_lat=40.4` geçer. Backend
    `wildfire-ai-backend/app/services/risk_feature_service.py:_compute_slope_aspect_sampled`
    aynı formülü kullanır (Karar #6 PREPROCESS_SYMMETRIC).

    Sprint 5'te gerçek DEM raster için kullanılır. NOTE: Dönülen slope/aspect
    değerlerinde clip uygulanmamıştır. slope_deg teorik olarak 90 dereceye
    ulaşabilir; downstream kullanım [0, 60] clip varsayıyorsa çağıran taraf
    clip etmeli. Sprint 6 gerçek DEM entegrasyonunda gözden geçirilecek.
    """
    dy_spacing = res_m  # kuzey-güney sabit
    dx_spacing = res_m / _cos(_radians(mid_lat))  # doğu-batı cos(lat) düzeltmeli
    dy, dx = np.gradient(elevation_grid, dy_spacing, dx_spacing)
    slope_rad = np.arctan(np.hypot(dx, dy))
    aspect_rad = np.arctan2(-dx, dy)  # north=0, clockwise
    slope_deg = np.rad2deg(slope_rad)
    aspect_deg = (np.rad2deg(aspect_rad) + 360.0) % 360.0
    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def compute_firms_density(
    lat: np.ndarray, lon: np.ndarray, firms_df: "pd.DataFrame", radius_km: float = 10.0
) -> np.ndarray:
    """Her (lat, lon) merkez için radius_km içindeki FIRMS nokta sayısı.

    firms_df: pandas DataFrame, 'latitude' + 'longitude' sütunları.
    Haversine formülü.
    """
    import pandas as pd  # local — import only when called

    if not isinstance(firms_df, pd.DataFrame) or len(firms_df) == 0:
        return np.zeros_like(lat, dtype=np.float32)

    counts = np.zeros(len(lat), dtype=np.float32)
    firms_lat = firms_df["latitude"].to_numpy()
    firms_lon = firms_df["longitude"].to_numpy()
    R = 6371.0  # km
    for i in range(len(lat)):
        dlat = np.radians(firms_lat - lat[i])
        dlon = np.radians(firms_lon - lon[i])
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat[i])) * np.cos(np.radians(firms_lat)) * np.sin(dlon / 2) ** 2
        )
        dist = 2 * R * np.arcsin(np.sqrt(a))
        counts[i] = int((dist <= radius_km).sum())
    return counts


def compute_ffmc_approx(
    temp_c: np.ndarray | float,
    rh_pct: np.ndarray | float,
    wind_ms: np.ndarray | float,
    prev_ffmc: float = 85.0,
) -> np.ndarray:
    """Van Wagner (1987) FFMC single-step yaklaşımı (PoC için basitleştirilmiş).

    Tam iteratif FFMC zinciri Sprint 5'te (gerçek MGM verisi ile) replace edilir.
    prev_ffmc=85.0: orta-kuru başlangıç koşulu.
    wind_ms: Rüzgar hızı m/s. Fonksiyon içinde km/h'a dönüştürülür
    (Van Wagner 1987 Eq. formülü km/h biriminde tanımlıdır).
    """
    t = np.asarray(temp_c, dtype=np.float64)
    h = np.clip(np.asarray(rh_pct, dtype=np.float64), 5.0, 100.0)
    w = np.asarray(wind_ms, dtype=np.float64)
    w_kmh = w * 3.6  # Van Wagner (1987): katsayı 0.0365 km/h biriminde

    # Equilibrium moisture content (Van Wagner Eq.4)
    Ed = 0.942 * (h ** 0.679) + 11.0 * np.exp((h - 100.0) / 10.0) + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h))

    # Basitleştirilmiş drying: m = Ed + (prev_m - Ed) * exp(-1)
    prev_m = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)
    m = Ed + (prev_m - Ed) * np.exp(-1.0 - 0.0365 * w_kmh)

    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return np.clip(ffmc, 0.0, 101.0).astype(np.float32)
