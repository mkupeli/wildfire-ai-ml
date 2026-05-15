# SPDX-License-Identifier: Apache-2.0
"""Sprint 6-A: label_builder.build_labels() unit testleri.

Kapsam:
  - Label penceresi (obs+1g .. obs+30g) pozitif / negatif
  - Leakage tampon: firms_density_1yr sağ kenarı = obs_date - 31g
  - Confidence filtresi: low satırlar elenir
  - days_since_last_fire default (no history)
  - BallTree haversine 10km sınır testi
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn", reason="scikit-learn kurulu değil — label_builder testleri skip")

from wildfire_ml.risk.label_builder import build_labels, EARTH_RADIUS_M
from wildfire_ml.risk.config import RealDataConfig


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _make_grid(lat: float, lon: float, obs_date: str) -> pd.DataFrame:
    """Tek hücreli grid_df oluştur."""
    return pd.DataFrame({
        "lat": [lat],
        "lon": [lon],
        "obs_date": [obs_date],
    })


def _make_firms(lat: float, lon: float, acq_date: str, confidence: str | None = "nominal") -> pd.DataFrame:
    """Tek hotspot'lu firms_df oluştur."""
    row = {"latitude": lat, "longitude": lon, "acq_date": acq_date}
    if confidence is not None:
        row["confidence"] = confidence
    return pd.DataFrame([row])


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """İki nokta arası haversine mesafe (metre)."""
    R = EARTH_RADIUS_M
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# test_label_window_positive
# ---------------------------------------------------------------------------

def test_label_window_positive() -> None:
    """obs_date=2025-06-01, hotspot @ 2025-06-15 (pencere içi) → label=1."""
    grid = _make_grid(39.5, 32.85, "2025-06-01")
    # Aynı konumda hotspot, obs+14g → label penceresi (obs+1g..obs+30g) içinde
    firms = _make_firms(39.5, 32.85, "2025-06-15", confidence="nominal")

    result = build_labels(grid, firms)

    assert result["fire_occurred_within_30d"].iloc[0] == 1, (
        "obs_date+14g içinde 10km'deki hotspot → fire_occurred_within_30d=1 beklendi."
    )


# ---------------------------------------------------------------------------
# test_label_negative_no_hotspot_in_window
# ---------------------------------------------------------------------------

def test_label_negative_no_hotspot_in_window() -> None:
    """obs_date=2025-06-01, pencerede hotspot yok → label=0."""
    grid = _make_grid(39.5, 32.85, "2025-06-01")
    # Sadece geçmişte hotspot var; label penceresinde (06-02..07-01) hiçbiri yok
    firms = _make_firms(39.5, 32.85, "2025-05-15", confidence="nominal")

    result = build_labels(grid, firms)

    assert result["fire_occurred_within_30d"].iloc[0] == 0, (
        "Label penceresi dışındaki hotspot → fire_occurred_within_30d=0 beklendi."
    )


# ---------------------------------------------------------------------------
# test_label_leakage_buffer
# ---------------------------------------------------------------------------

def test_label_leakage_buffer() -> None:
    """Leakage tampon (31g): density window = [obs-365g, obs-31g].

    Senaryo (obs_date=2025-06-01):
      A) 2025-05-20 → obs-12g; tampon içinde (obs-31g = 2025-05-01; 05-20 > 05-01)
         → density_window dışında! (05-20 > cutoff=05-01 → dahil değil)

    Dikkat: cutoff = obs_date - 31g = 2025-05-01.
    Density window: [obs-365g=2024-06-01, cutoff=2025-05-01].

    Hotspot tarihleri:
      - 2025-06-15  → label penceresi içi → density'e GİRMEMELİ
      - 2025-05-20  → cutoff (2025-05-01) sonrası → density'e GİRMEMELİ
      - 2025-04-01  → density window içi ([2024-06-01..2025-05-01]) → SAYILMALI
      - 2024-07-15  → density window içi → SAYILMALI
    """
    obs_date = "2025-06-01"
    grid = _make_grid(39.5, 32.85, obs_date)

    # Tüm hotspot'lar aynı konumda (10km içi)
    rows = [
        {"latitude": 39.5, "longitude": 32.85, "acq_date": "2025-06-15", "confidence": "nominal"},  # label penceresi
        {"latitude": 39.5, "longitude": 32.85, "acq_date": "2025-05-20", "confidence": "nominal"},  # tampon içi (cutoff sonrası)
        {"latitude": 39.5, "longitude": 32.85, "acq_date": "2025-04-01", "confidence": "nominal"},  # density window içi
        {"latitude": 39.5, "longitude": 32.85, "acq_date": "2024-07-15", "confidence": "nominal"},  # density window içi
    ]
    firms = pd.DataFrame(rows)

    result = build_labels(grid, firms)

    # Label: 2025-06-15 label penceresinde → 1
    assert result["fire_occurred_within_30d"].iloc[0] == 1, (
        "2025-06-15 label penceresi içi → label=1 beklendi."
    )

    # Density: sadece 2025-04-01 ve 2024-07-15 sayılmalı → 2
    density = result["firms_density_1yr"].iloc[0]
    assert density == 2.0, (
        f"firms_density_1yr=2 beklendi (cutoff=2025-05-01; 05-20 tampon içi, 06-15 label penceresi → dışarda). "
        f"Gerçek: {density}. "
        "Bu test label leakage tamponunun (obs_date-31g cutoff) doğru çalıştığını kanıtlar."
    )


# ---------------------------------------------------------------------------
# test_confidence_filter_low_excluded
# ---------------------------------------------------------------------------

def test_confidence_filter_low_excluded() -> None:
    """Low confidence hotspot'lar label hesabına dahil edilmez."""
    grid = _make_grid(39.5, 32.85, "2025-06-01")

    # Yalnızca low confidence hotspot label penceresinde
    firms_low = pd.DataFrame([{
        "latitude": 39.5, "longitude": 32.85,
        "acq_date": "2025-06-10",
        "confidence": "low",
    }])
    result_low = build_labels(grid, firms_low)
    assert result_low["fire_occurred_within_30d"].iloc[0] == 0, (
        "Sadece 'low' confidence hotspot → filtre sonrası label=0 beklendi."
    )

    # Nominal + high karışık: label penceresinde low var, ama nominal yok
    firms_mixed = pd.DataFrame([
        {"latitude": 39.5, "longitude": 32.85, "acq_date": "2025-06-10", "confidence": "low"},
        {"latitude": 39.5, "longitude": 32.85, "acq_date": "2025-05-01", "confidence": "nominal"},  # geçmişte
    ])
    result_mixed = build_labels(grid, firms_mixed)
    # Label penceresinde yalnızca low var → label=0 (low elindi), nominal geçmişte → label=0
    assert result_mixed["fire_occurred_within_30d"].iloc[0] == 0, (
        "Low penceresinde, nominal geçmişte → label=0 beklendi."
    )

    # Nominal label penceresinde → label=1
    firms_nominal = pd.DataFrame([
        {"latitude": 39.5, "longitude": 32.85, "acq_date": "2025-06-10", "confidence": "low"},
        {"latitude": 39.5, "longitude": 32.85, "acq_date": "2025-06-12", "confidence": "nominal"},
    ])
    result_nominal = build_labels(grid, firms_nominal)
    assert result_nominal["fire_occurred_within_30d"].iloc[0] == 1, (
        "Nominal label penceresinde → label=1 beklendi (low elindi, nominal sayıldı)."
    )


# ---------------------------------------------------------------------------
# test_days_since_last_fire_no_history
# ---------------------------------------------------------------------------

def test_days_since_last_fire_no_history() -> None:
    """Geçmişte hiç hotspot yok → days_since_last_fire == 3650 (default)."""
    grid = _make_grid(39.5, 32.85, "2025-06-01")
    # Sadece label penceresinde hotspot var, geçmişte yok
    firms = _make_firms(39.5, 32.85, "2025-06-15", confidence="nominal")

    result = build_labels(grid, firms)

    # Geçmiş (cutoff=2025-05-01 öncesi) hotspot yok → default 3650
    days = result["days_since_last_fire"].iloc[0]
    assert days == 3650.0, (
        f"Geçmişte hotspot yok → days_since_last_fire=3650.0 beklendi, got {days}."
    )


def test_days_since_last_fire_empty_firms() -> None:
    """firms_df tamamen boş → days_since_last_fire == 3650."""
    grid = _make_grid(39.5, 32.85, "2025-06-01")
    firms_empty = pd.DataFrame(columns=["latitude", "longitude", "acq_date", "confidence"])

    result = build_labels(grid, firms_empty)

    days = result["days_since_last_fire"].iloc[0]
    assert days == 3650.0, (
        f"firms_df boş → days_since_last_fire=3650 beklendi, got {days}."
    )
    assert result["fire_occurred_within_30d"].iloc[0] == 0
    assert result["firms_density_1yr"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# test_haversine_10km_boundary
# ---------------------------------------------------------------------------

def test_haversine_10km_boundary() -> None:
    """BallTree haversine: 9.9km içi dahil, 10.1km dışı hariç.

    Referans noktası: (39.5, 32.85). Kuzey yönünde ~10km ≈ 0.09° enlem.
    _haversine_m ile tam mesafeyi hesaplayıp margin ile test eder.
    """
    cfg = RealDataConfig(firms_radius_m=10000.0)
    obs_date = "2025-06-01"
    center_lat = 39.5
    center_lon = 32.85
    grid = _make_grid(center_lat, center_lon, obs_date)

    # 1 derece enlem ≈ 111_111m → 0.09° ≈ 9999m
    deg_per_m = 1.0 / 111_111.0
    offset_inside = 9900 * deg_per_m    # ~9.9km
    offset_outside = 10100 * deg_per_m  # ~10.1km

    lat_inside = center_lat + offset_inside
    lat_outside = center_lat + offset_outside

    # Gerçek mesafeleri doğrula
    dist_inside = _haversine_m(center_lat, center_lon, lat_inside, center_lon)
    dist_outside = _haversine_m(center_lat, center_lon, lat_outside, center_lon)
    assert dist_inside < 10000.0, f"Test kurulumu hatalı: dist_inside={dist_inside:.1f}m >= 10000m"
    assert dist_outside > 10000.0, f"Test kurulumu hatalı: dist_outside={dist_outside:.1f}m <= 10000m"

    # 9.9km içi hotspot → label=1
    firms_inside = _make_firms(lat_inside, center_lon, "2025-06-15", confidence="nominal")
    result_inside = build_labels(grid, firms_inside, cfg=cfg)
    assert result_inside["fire_occurred_within_30d"].iloc[0] == 1, (
        f"~{dist_inside:.0f}m içi hotspot (10km sınırı içinde) → label=1 beklendi."
    )

    # 10.1km dışı hotspot → label=0
    firms_outside = _make_firms(lat_outside, center_lon, "2025-06-15", confidence="nominal")
    result_outside = build_labels(grid, firms_outside, cfg=cfg)
    assert result_outside["fire_occurred_within_30d"].iloc[0] == 0, (
        f"~{dist_outside:.0f}m dışı hotspot (10km sınırı dışında) → label=0 beklendi."
    )


# ---------------------------------------------------------------------------
# test_label_exactly_at_boundary_days
# ---------------------------------------------------------------------------

def test_label_exactly_at_window_boundary() -> None:
    """obs+30g (window son gün) dahil, obs+31g dahil değil."""
    grid = _make_grid(39.5, 32.85, "2025-06-01")

    # obs+30g = 2025-07-01 → dahil
    firms_last_day = _make_firms(39.5, 32.85, "2025-07-01", confidence="nominal")
    result = build_labels(grid, firms_last_day)
    assert result["fire_occurred_within_30d"].iloc[0] == 1, (
        "obs+30g (2025-07-01) label penceresinin son günü → label=1 beklendi."
    )

    # obs+31g = 2025-07-02 → dahil değil
    firms_over = _make_firms(39.5, 32.85, "2025-07-02", confidence="nominal")
    result_over = build_labels(grid, firms_over)
    assert result_over["fire_occurred_within_30d"].iloc[0] == 0, (
        "obs+31g (2025-07-02) label penceresi dışı → label=0 beklendi."
    )

    # obs+1g = 2025-06-02 → dahil (ilk gün)
    firms_first_day = _make_firms(39.5, 32.85, "2025-06-02", confidence="nominal")
    result_first = build_labels(grid, firms_first_day)
    assert result_first["fire_occurred_within_30d"].iloc[0] == 1, (
        "obs+1g (2025-06-02) label penceresinin ilk günü → label=1 beklendi."
    )

    # obs+0g = 2025-06-01 → obs_date kendisi dahil DEĞİL
    firms_same_day = _make_firms(39.5, 32.85, "2025-06-01", confidence="nominal")
    result_same = build_labels(grid, firms_same_day)
    assert result_same["fire_occurred_within_30d"].iloc[0] == 0, (
        "obs_date kendisi (2025-06-01) label penceresine dahil değil → label=0 beklendi."
    )


# ---------------------------------------------------------------------------
# test_no_confidence_column_uses_all_rows
# ---------------------------------------------------------------------------

def test_no_confidence_column_uses_all_rows() -> None:
    """firms_df'te 'confidence' sütunu yoksa tüm satırlar kullanılır."""
    grid = _make_grid(39.5, 32.85, "2025-06-01")
    # confidence sütunu yok
    firms = pd.DataFrame([{
        "latitude": 39.5, "longitude": 32.85, "acq_date": "2025-06-10",
    }])

    result = build_labels(grid, firms)
    assert result["fire_occurred_within_30d"].iloc[0] == 1, (
        "confidence sütunu olmayan firms_df → tüm satırlar kullanılır → label=1 beklendi."
    )


# ---------------------------------------------------------------------------
# test_multiple_grid_rows
# ---------------------------------------------------------------------------

def test_multiple_grid_rows() -> None:
    """Birden fazla grid satırı bağımsız olarak etiketlenir."""
    grid = pd.DataFrame({
        "lat": [39.5, 40.0],   # İkinci nokta çok uzakta (>10km)
        "lon": [32.85, 32.85],
        "obs_date": ["2025-06-01", "2025-06-01"],
    })
    # Sadece birinci noktanın yakınında hotspot
    firms = _make_firms(39.5, 32.85, "2025-06-15", confidence="nominal")

    result = build_labels(grid, firms)

    assert result["fire_occurred_within_30d"].iloc[0] == 1, "1. hücre (yakın) → label=1"
    assert result["fire_occurred_within_30d"].iloc[1] == 0, "2. hücre (uzak >50km) → label=0"
