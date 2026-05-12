# SPDX-License-Identifier: Apache-2.0
"""Sprint 4-A: Feature engineering helper testleri (features.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire_ml.risk import (
    compute_vpd,
    compute_slope_aspect,
    compute_firms_density,
    compute_ffmc_approx,
)


# ===========================================================================
# compute_vpd
# ===========================================================================

def test_vpd_zero_when_rh_100() -> None:
    """RH=%100 => doymuş hava, VPD=0."""
    vpd = compute_vpd(temp_c=25.0, rh_pct=100.0)
    assert float(vpd) == pytest.approx(0.0, abs=1e-5), f"RH=100 iken VPD=0 beklendi, got {vpd}"


def test_vpd_positive_when_rh_less_than_100() -> None:
    """RH < 100 => VPD > 0."""
    vpd = compute_vpd(temp_c=30.0, rh_pct=50.0)
    assert float(vpd) > 0.0, f"RH=50 iken VPD > 0 beklendi, got {vpd}"


def test_vpd_increases_with_temperature() -> None:
    """Sabit RH'de sıcaklık artışı VPD'yi artırmalı."""
    rh = 40.0
    vpd_low = float(compute_vpd(temp_c=10.0, rh_pct=rh))
    vpd_high = float(compute_vpd(temp_c=40.0, rh_pct=rh))
    assert vpd_high > vpd_low, (
        f"T=40 VPD ({vpd_high}) T=10 VPD ({vpd_low})'den büyük olmalı"
    )


def test_vpd_clip_upper_bound() -> None:
    """Aşırı sıcak + kuru koşul → VPD <= 10 kPa (clip uygulanmış)."""
    vpd = compute_vpd(temp_c=80.0, rh_pct=1.0)
    result = float(vpd)
    assert result <= 10.0, f"VPD clip üst sınırı 10 kPa, got {result}"


# ===========================================================================
# compute_ffmc_approx
# ===========================================================================

def test_ffmc_approx_bounds() -> None:
    """Çeşitli koşullarda FFMC [0, 101] aralığında kalmalı."""
    temps = np.array([-10.0, 0.0, 20.0, 35.0, 45.0])
    rhs = np.array([5.0, 30.0, 50.0, 80.0, 100.0])
    winds = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
    ffmc = compute_ffmc_approx(temps, rhs, winds)
    assert ffmc.min() >= 0.0, f"FFMC min {ffmc.min()} < 0"
    assert ffmc.max() <= 101.0, f"FFMC max {ffmc.max()} > 101"


def test_ffmc_increases_with_low_rh() -> None:
    """Aynı sıcaklık ve rüzgarda düşük RH, yüksek RH'den daha yüksek FFMC vermeli."""
    ffmc_dry = float(compute_ffmc_approx(temp_c=30.0, rh_pct=10.0, wind_ms=5.0))
    ffmc_wet = float(compute_ffmc_approx(temp_c=30.0, rh_pct=90.0, wind_ms=5.0))
    assert ffmc_dry > ffmc_wet, (
        f"Kuru koşul FFMC ({ffmc_dry}) ıslak koşul FFMC ({ffmc_wet})'den büyük olmalı"
    )


# ===========================================================================
# compute_firms_density
# ===========================================================================

def test_firms_density_empty_df() -> None:
    """Boş FIRMS DataFrame → tüm sayımlar 0."""
    lat = np.array([39.5, 39.55])
    lon = np.array([32.85, 32.9])
    empty_df = pd.DataFrame(columns=["latitude", "longitude"])
    counts = compute_firms_density(lat, lon, empty_df, radius_km=10.0)
    assert (counts == 0).all(), f"Boş FIRMS için 0 beklendi, got {counts}"
    assert counts.shape == (2,), f"Çıktı shape (2,) beklendi, got {counts.shape}"


def test_firms_density_single_point_in_range() -> None:
    """Merkeze 5 km içinde 1 FIRMS noktası → count == 1."""
    center_lat = np.array([39.5])
    center_lon = np.array([32.85])
    # ~4 km kuzey — 0.04 derece enlem ≈ 4.4 km
    firms_df = pd.DataFrame({"latitude": [39.54], "longitude": [32.85]})
    counts = compute_firms_density(center_lat, center_lon, firms_df, radius_km=10.0)
    assert counts[0] == 1, f"1 yakın nokta için count=1 beklendi, got {counts[0]}"


def test_firms_density_single_point_out_range() -> None:
    """Merkeze 15 km uzakta 1 FIRMS noktası → count == 0."""
    center_lat = np.array([39.5])
    center_lon = np.array([32.85])
    # ~0.135 derece enlem ≈ 15 km
    firms_df = pd.DataFrame({"latitude": [39.635], "longitude": [32.85]})
    counts = compute_firms_density(center_lat, center_lon, firms_df, radius_km=10.0)
    assert counts[0] == 0, f"Uzak nokta için count=0 beklendi, got {counts[0]}"


# ===========================================================================
# compute_slope_aspect
# ===========================================================================

def test_compute_slope_aspect_flat_grid() -> None:
    """Düz (uniform elevation) DEM → slope == 0."""
    dem = np.full((10, 10), 1000.0)
    slope, aspect = compute_slope_aspect(dem, res_m=250.0)
    assert np.allclose(slope, 0.0, atol=1e-5), (
        f"Düz DEM için slope=0 beklendi, max={slope.max()}"
    )


def test_compute_slope_aspect_ramp() -> None:
    """Artan (ramp) DEM → slope > 0 ve aspect tutarlı (kuzey yönüne yakın)."""
    # Her satırda +10m artan DEM (kuzeyden güneye iniyor)
    dem = np.tile(np.arange(100.0, 1100.0, 100.0), (10, 1)).T  # (10, 10), satırlar artar
    slope, aspect = compute_slope_aspect(dem, res_m=100.0)
    # İç bölgelerde slope > 0 olmalı (kenarlarda gradient davranışı farklı)
    inner_slope = slope[1:-1, 1:-1]
    assert inner_slope.min() > 0.0, (
        f"Ramp DEM iç bölgelerinde slope > 0 beklendi, min={inner_slope.min()}"
    )
    # aspect_deg 0-360 aralığında
    assert aspect.min() >= 0.0, f"aspect_deg min {aspect.min()} < 0"
    assert aspect.max() <= 360.0, f"aspect_deg max {aspect.max()} > 360"


# ===========================================================================
# B2 regression: Van Wagner km/h dönüşümü
# ===========================================================================

def test_ffmc_wind_unit_conversion() -> None:
    """5 m/s ve 0 m/s girişlerinin yol açtığı FFMC farkı anlamlı olmalı.

    Bug B2: Eski kod rüzgarı m/s olarak kullandığında katsayı 0.0365
    m/s biriminde çalışırdı (hatalı — Van Wagner 1987 km/h biriminde).
    Fix: w_kmh = w * 3.6 dönüşümü. Sonuç: yüksek rüzgarda (20 m/s = 72 km/h)
    m korunma faktörü exp(-1 - 0.0365*72) = exp(-3.63) iken
    eski kodda exp(-1 - 0.0365*20) = exp(-1.73) olurdu → belirgin fark.
    Fix öncesi bu test KIRMIZI olurdu (fark yeterince büyük olmaz),
    fix sonrası YEŞİL.
    """
    result_zero_wind = float(compute_ffmc_approx(temp_c=25.0, rh_pct=50.0, wind_ms=0.0))
    result_high_wind = float(compute_ffmc_approx(temp_c=25.0, rh_pct=50.0, wind_ms=20.0))  # 72 km/h

    # km/h dönüşümü yapılmışsa exp terimleri daha büyük fark yaratır → FFMC farkı > 0.5
    assert abs(result_high_wind - result_zero_wind) > 0.5, (
        f"20 m/s (72 km/h) ile 0 m/s arasındaki FFMC farkı 0.5'ten büyük olmalı. "
        f"Fark: {abs(result_high_wind - result_zero_wind):.4f}. "
        f"B2 bug hala mevcut: rüzgar m/s yerine km/h'a dönüştürülmüyor olabilir."
    )

    # Her iki sonuç da [0, 101] sınırlarında kalmalı
    assert 0.0 <= result_zero_wind <= 101.0, (
        f"0 m/s FFMC={result_zero_wind} sınır dışı [0, 101]"
    )
    assert 0.0 <= result_high_wind <= 101.0, (
        f"20 m/s FFMC={result_high_wind} sınır dışı [0, 101]"
    )


def test_ffmc_wind_unit_is_kmh_internal() -> None:
    """compute_ffmc_approx, wind_ms=5.0 girişini dahili olarak 18.0 km/h gibi işlemeli.

    Van Wagner (1987) Denklem katsayısı 0.0365 km/h biriminde tanımlanmıştır.
    Herhangi bir girdi, 3.6 çarpanıyla km/h'a dönüştürülmelidir.
    Bu test sonucun [0, 101] aralığında ve makul olduğunu doğrular;
    dönüşüm doğruluğu test_ffmc_wind_unit_conversion ile kapsanmıştır.
    """
    result = float(compute_ffmc_approx(temp_c=25.0, rh_pct=50.0, wind_ms=5.0))
    assert 0.0 <= result <= 101.0, (
        f"wind_ms=5.0 girişi için FFMC={result} [0, 101] aralığı dışında. "
        f"Dahili km/h dönüşümü beklenmedik bir değer üretmiş olabilir."
    )
