# SPDX-License-Identifier: Apache-2.0
"""Sprint 4-A: SyntheticRiskDataGenerator unit testleri."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire_ml.risk import (
    RiskConfig,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    WORLDCOVER_CLASSES,
    SYNTHETIC_DATA_NOTE,
    SyntheticRiskDataGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gen(seed: int = 42, n_samples: int = 100) -> SyntheticRiskDataGenerator:
    cfg = RiskConfig(seed=seed, n_samples=n_samples)
    return SyntheticRiskDataGenerator(cfg)


def _generate(seed: int = 42, n: int = 100) -> pd.DataFrame:
    return _make_gen(seed=seed, n_samples=n).generate(n)


# ---------------------------------------------------------------------------
# Shape & schema
# ---------------------------------------------------------------------------

def test_generate_returns_expected_shape() -> None:
    """n=100 ile generate() → (100, 25): 24 feature + 1 target sütunu."""
    df = _generate(n=100)
    assert len(df) == 100, f"Beklenen satır sayısı 100, got {len(df)}"
    assert len(df.columns) == 25, f"Beklenen 25 sütun (24 feature + target), got {len(df.columns)}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_generate_is_reproducible() -> None:
    """Aynı seed ile iki çağrı özdeş DataFrame döndürmeli."""
    df1 = _generate(seed=42, n=200)
    df2 = _generate(seed=42, n=200)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_different_seeds_differ() -> None:
    """Farklı seed'ler farklı DataFrame üretmeli."""
    df1 = _generate(seed=1, n=100)
    df2 = _generate(seed=2, n=100)
    # En az bir sütunda değer farklı olmalı
    assert not df1.equals(df2), "seed=1 ve seed=2 özdeş DataFrame döndürdü — reproducibility hatası"


# ---------------------------------------------------------------------------
# One-hot integrity
# ---------------------------------------------------------------------------

def test_one_hot_sums_to_one() -> None:
    """Her satırda WorldCover one-hot sütunları toplamı tam olarak 1 olmalı."""
    df = _generate()
    row_sums = df[WORLDCOVER_CLASSES].sum(axis=1)
    assert (row_sums == 1).all(), (
        f"WorldCover one-hot satır toplamı 1 değil. "
        f"Hatalı satır indexleri: {row_sums[row_sums != 1].index.tolist()}"
    )


# ---------------------------------------------------------------------------
# Null check
# ---------------------------------------------------------------------------

def test_no_nulls() -> None:
    """DataFrame hiç NaN/None içermemeli."""
    df = _generate()
    assert not df.isnull().any().any(), (
        f"Null değer bulunan sütunlar: {df.columns[df.isnull().any()].tolist()}"
    )


# ---------------------------------------------------------------------------
# Target rate
# ---------------------------------------------------------------------------

def test_target_positive_rate_in_range() -> None:
    """Pozitif etiket oranı 0.05–0.40 aralığında olmalı (n=1000, geniş tolerans)."""
    cfg = RiskConfig(seed=42, n_samples=1000, target_positive_rate=0.15)
    df = SyntheticRiskDataGenerator(cfg).generate()
    rate = df["fire_occurred_within_30d"].mean()
    assert 0.05 < rate < 0.40, f"Pozitif oran beklenen 0.05-0.40 aralığı dışında: {rate:.4f}"


# ---------------------------------------------------------------------------
# Feature bounds
# ---------------------------------------------------------------------------

def test_elevation_bounds() -> None:
    """elevation_m: 900 <= x <= 1600."""
    df = _generate(n=500)
    assert df["elevation_m"].min() >= 900.0, f"elevation_m min çok düşük: {df['elevation_m'].min()}"
    assert df["elevation_m"].max() <= 1600.0, f"elevation_m max çok yüksek: {df['elevation_m'].max()}"


def test_rh_bounds() -> None:
    """rh_pct: 5 <= x <= 100."""
    df = _generate(n=500)
    assert df["rh_pct"].min() >= 5.0, f"rh_pct min {df['rh_pct'].min()} < 5"
    assert df["rh_pct"].max() <= 100.0, f"rh_pct max {df['rh_pct'].max()} > 100"


def test_vpd_non_negative() -> None:
    """vpd_kpa >= 0 (clip edilmiş olmalı)."""
    df = _generate(n=500)
    neg_mask = df["vpd_kpa"] < 0
    assert not neg_mask.any(), f"Negatif vpd_kpa değerleri var: {df.loc[neg_mask, 'vpd_kpa'].tolist()}"


def test_ffmc_bounds() -> None:
    """ffmc_approx: 0 <= x <= 101."""
    df = _generate(n=500)
    assert df["ffmc_approx"].min() >= 0.0, f"ffmc_approx min {df['ffmc_approx'].min()} < 0"
    assert df["ffmc_approx"].max() <= 101.0, f"ffmc_approx max {df['ffmc_approx'].max()} > 101"


# ---------------------------------------------------------------------------
# DataFrame attrs (metadata)
# ---------------------------------------------------------------------------

def test_synthetic_note_in_attrs() -> None:
    """df.attrs['source'] == SYNTHETIC_DATA_NOTE."""
    df = _generate()
    assert "source" in df.attrs, "df.attrs 'source' anahtarı eksik"
    assert df.attrs["source"] == SYNTHETIC_DATA_NOTE, (
        f"Beklenen: {SYNTHETIC_DATA_NOTE!r}\nGelen: {df.attrs['source']!r}"
    )


def test_beynam_lat_bounds() -> None:
    """df.attrs['lat'] değerleri Beynam bbox içinde (39.4 – 39.6)."""
    cfg = RiskConfig(seed=42)
    df = SyntheticRiskDataGenerator(cfg).generate(n=200)
    lat = df.attrs["lat"]
    assert lat.min() >= cfg.beynam_lat_min, f"lat min {lat.min()} < {cfg.beynam_lat_min}"
    assert lat.max() <= cfg.beynam_lat_max, f"lat max {lat.max()} > {cfg.beynam_lat_max}"


def test_beynam_lon_bounds() -> None:
    """df.attrs['lon'] değerleri Beynam bbox içinde (32.7 – 33.0)."""
    cfg = RiskConfig(seed=42)
    df = SyntheticRiskDataGenerator(cfg).generate(n=200)
    lon = df.attrs["lon"]
    assert lon.min() >= cfg.beynam_lon_min, f"lon min {lon.min()} < {cfg.beynam_lon_min}"
    assert lon.max() <= cfg.beynam_lon_max, f"lon max {lon.max()} > {cfg.beynam_lon_max}"


# ---------------------------------------------------------------------------
# B1 regression: n=0 falsy bug fix
# ---------------------------------------------------------------------------

def test_generate_n_zero_returns_empty() -> None:
    """n=0 ile generate() → boş ama schema-uyumlu DataFrame.

    Bug B1: Eski kod `n = n or self.cfg.n_samples` kullandığında n=0 falsy
    değer olarak değerlendirilir ve n_samples (varsayılan 5000) satır dönerdi.
    Fix: `n = self.cfg.n_samples if n is None else n` — n=0 artık 0 satır demek.
    Fix öncesi bu test KIRMIZI (len=5000), fix sonrası YEŞİL (len=0).
    """
    gen = SyntheticRiskDataGenerator(RiskConfig(seed=42))
    df = gen.generate(0)
    assert len(df) == 0, (
        f"n=0 ile generate() boş DataFrame beklendi (0 satır), got {len(df)} satır. "
        f"B1 bug hala mevcut: n=0 falsy olarak n_samples'a fallback ediyor olabilir."
    )
    assert set(df.columns) == set(FEATURE_COLUMNS + [TARGET_COLUMN]), (
        f"Boş DataFrame'de sütun şeması korunmalı. "
        f"Eksik: {set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)}"
    )
