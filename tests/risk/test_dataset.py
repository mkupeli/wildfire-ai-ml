# SPDX-License-Identifier: Apache-2.0
"""Sprint 4-A: load_risk_dataset + train_val_test_split testleri."""
from __future__ import annotations

import pandas as pd
import pytest

from wildfire_ml.risk import (
    RiskConfig,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_risk_dataset,
    train_val_test_split,
)


# ---------------------------------------------------------------------------
# load_risk_dataset
# ---------------------------------------------------------------------------

def test_load_synthetic_returns_dataframe() -> None:
    """path=None → sentetik DataFrame döner."""
    cfg = RiskConfig(seed=42, n_samples=100)
    df = load_risk_dataset(path=None, cfg=cfg)
    assert isinstance(df, pd.DataFrame), f"pd.DataFrame beklendi, got {type(df)}"
    assert len(df) == 100


def test_column_completeness() -> None:
    """FEATURE_COLUMNS + TARGET_COLUMN'un tamamı DataFrame'de mevcut olmalı."""
    cfg = RiskConfig(seed=42, n_samples=100)
    df = load_risk_dataset(path=None, cfg=cfg)
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    actual = set(df.columns)
    missing = expected - actual
    assert not missing, f"Eksik sütunlar: {missing}"


# ---------------------------------------------------------------------------
# train_val_test_split
# ---------------------------------------------------------------------------

def _get_base_df(n: int = 300) -> tuple[pd.DataFrame, RiskConfig]:
    cfg = RiskConfig(seed=42, n_samples=n)
    df = load_risk_dataset(path=None, cfg=cfg)
    return df, cfg


def test_split_sizes() -> None:
    """60/20/20 split — her bölmenin boyutu ±5% tolerans içinde olmalı."""
    df, cfg = _get_base_df(n=300)
    train, val, test = train_val_test_split(df, cfg)
    n = len(df)
    train_ratio = len(train) / n
    val_ratio = len(val) / n
    test_ratio = len(test) / n

    assert abs(train_ratio - 0.60) <= 0.05, f"Train ratio {train_ratio:.3f}, beklenen ~0.60"
    assert abs(val_ratio - 0.20) <= 0.05, f"Val ratio {val_ratio:.3f}, beklenen ~0.20"
    assert abs(test_ratio - 0.20) <= 0.05, f"Test ratio {test_ratio:.3f}, beklenen ~0.20"
    # Toplam satır korunmalı
    assert len(train) + len(val) + len(test) == n


def test_split_reproducibility() -> None:
    """Aynı config ile iki split çağrısı özdeş bölümler üretmeli."""
    df, cfg = _get_base_df(n=300)
    train1, val1, test1 = train_val_test_split(df, cfg)
    train2, val2, test2 = train_val_test_split(df, cfg)
    pd.testing.assert_frame_equal(train1, train2, check_like=False)
    pd.testing.assert_frame_equal(val1, val2, check_like=False)
    pd.testing.assert_frame_equal(test1, test2, check_like=False)


def test_stratification_target_ratio() -> None:
    """Train/val/test'te pozitif sınıf oranı birbirine ±5% yakın olmalı (stratified)."""
    cfg = RiskConfig(seed=42, n_samples=1000)
    df = load_risk_dataset(path=None, cfg=cfg)
    train, val, test = train_val_test_split(df, cfg)
    overall_rate = df[TARGET_COLUMN].mean()
    for name, split in [("train", train), ("val", val), ("test", test)]:
        rate = split[TARGET_COLUMN].mean()
        assert abs(rate - overall_rate) <= 0.05, (
            f"{name} pozitif oranı {rate:.4f}, genel oran {overall_rate:.4f} — fark >{0.05}"
        )


# ---------------------------------------------------------------------------
# CSV roundtrip
# ---------------------------------------------------------------------------

def test_load_from_csv_roundtrip(tmp_path) -> None:
    """generate → to_csv → load_risk_dataset(path=...) → özdeş DataFrame (sütun değerleri)."""
    cfg = RiskConfig(seed=42, n_samples=100)
    df_orig = load_risk_dataset(path=None, cfg=cfg)

    csv_path = tmp_path / "risk_test.csv"
    df_orig.to_csv(csv_path, index=False)

    df_loaded = load_risk_dataset(path=csv_path, cfg=cfg)

    # Sütun listesi
    assert set(df_orig.columns) == set(df_loaded.columns), "Sütun listeleri farklı"

    # Sayısal değerler — CSV round-trip float precision toleransıyla
    numeric_cols = [c for c in FEATURE_COLUMNS if df_orig[c].dtype.kind == "f"]
    for col in numeric_cols:
        pd.testing.assert_series_equal(
            df_orig[col].reset_index(drop=True),
            df_loaded[col].astype(df_orig[col].dtype).reset_index(drop=True),
            check_names=True,
            rtol=1e-4,
            check_exact=False,
        )


# ---------------------------------------------------------------------------
# Validation: missing column raises ValueError
# ---------------------------------------------------------------------------

def test_column_missing_raises(tmp_path) -> None:
    """Eksik sütunlu CSV yüklenmeye çalışılırsa ValueError fırlatılmalı."""
    cfg = RiskConfig(seed=42, n_samples=50)
    df = load_risk_dataset(path=None, cfg=cfg)

    # Bir feature sütununu kaldır
    df_bad = df.drop(columns=[FEATURE_COLUMNS[0]])
    bad_csv = tmp_path / "bad.csv"
    df_bad.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="Eksik sütunlar"):
        load_risk_dataset(path=bad_csv, cfg=cfg)


# ---------------------------------------------------------------------------
# NH4 fix: CSV yüklemesinde df.attrs["source"] ve ["data_version"] set edilmeli
# ---------------------------------------------------------------------------

def test_load_from_csv_sets_attrs_source(tmp_path) -> None:
    """CSV yüklendikten sonra df.attrs['source'] ve ['data_version'] set edilmeli.

    NH4 fix: load_risk_dataset(path=...) artık CSV path'ini df.attrs["source"]
    olarak ve cfg.data_version değerini df.attrs["data_version"] olarak saklar.
    Bu, aşağı akış pipeline'larının (model training, logging) veri kökenini
    takip edebilmesi için gereklidir.
    """
    from wildfire_ml.risk import SyntheticRiskDataGenerator

    gen = SyntheticRiskDataGenerator(RiskConfig(seed=42))
    df_orig = gen.generate(50)

    csv_path = tmp_path / "risk_test.csv"
    df_orig.to_csv(csv_path, index=False)

    cfg = RiskConfig(seed=42, data_version="synthetic-test-v1")
    df_loaded = load_risk_dataset(path=csv_path, cfg=cfg)

    assert "source" in df_loaded.attrs, (
        "df.attrs['source'] anahtarı eksik. "
        "NH4 fix: load_risk_dataset(path=...) df.attrs['source'] = str(path) set etmeli."
    )
    assert df_loaded.attrs["source"] == str(csv_path), (
        f"df.attrs['source'] beklenen {str(csv_path)!r}, "
        f"got {df_loaded.attrs.get('source')!r}"
    )
    assert "data_version" in df_loaded.attrs, (
        "df.attrs['data_version'] anahtarı eksik. "
        "NH4 fix: load_risk_dataset(path=...) df.attrs['data_version'] = cfg.data_version set etmeli."
    )
    assert df_loaded.attrs["data_version"] == "synthetic-test-v1", (
        f"df.attrs['data_version'] beklenen 'synthetic-test-v1', "
        f"got {df_loaded.attrs.get('data_version')!r}"
    )


# ---------------------------------------------------------------------------
# Sprint 6-A: spatial_block_split testleri
# ---------------------------------------------------------------------------

import logging
import numpy as np

from wildfire_ml.risk.dataset import spatial_block_split


def _make_spatial_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """lat/lon + TARGET_COLUMN içeren minimal gerçek-veri benzeri DataFrame."""
    rng = np.random.default_rng(seed)
    lats = rng.uniform(39.4, 39.6, size=n)
    lons = rng.uniform(32.7, 33.0, size=n)
    # %15 pozitif oran — düzenli dağılım
    target = (rng.random(n) < 0.15).astype(np.int8)
    return pd.DataFrame({"lat": lats, "lon": lons, TARGET_COLUMN: target})


def test_spatial_block_split_no_leakage() -> None:
    """Her val_df enlem aralığı train_df enlem aralığı ile örtüşmez (band sızdırmaz).

    spatial_block_split qcut ile k enlem bandı kurar; her fold'da 1 bant val,
    kalan k-1 bant train'e gider. Aynı enlem bandı hem train hem val'da OLMAZ.
    """
    df = _make_spatial_df(n=300)
    folds = spatial_block_split(df, k=3, lat_col="lat")

    assert len(folds) >= 2, "En az 2 fold beklendi."

    for fold_i, (train_df, val_df) in enumerate(folds):
        val_lats = set(val_df["lat"].tolist())
        train_lats = set(train_df["lat"].tolist())
        overlap = val_lats & train_lats
        assert len(overlap) == 0, (
            f"Fold {fold_i}: val_df ve train_df aynı enlem değerlerini paylaşıyor "
            f"({len(overlap)} nokta). Uzamsal sızıntı var!"
        )


def test_spatial_block_split_covers_all() -> None:
    """k fold val_df'leri birleşimi tüm satırları tam bir kez kapsar."""
    df = _make_spatial_df(n=300)
    folds = spatial_block_split(df, k=5, lat_col="lat")

    # Tüm val index'lerini topla
    val_indices_all: list[int] = []
    for _, val_df in folds:
        # reset_index(drop=True) sonrası orijinal index bilgisi kaybolabilir;
        # lat değerleri eşsizse onları kullan; değilse satır içeriklerini.
        val_indices_all.extend(val_df["lat"].tolist())

    # Orijinal df'teki tüm lat değerleri val'da tam bir kez yer almalı
    expected_lats = sorted(df["lat"].tolist())
    actual_lats = sorted(val_indices_all)

    assert len(actual_lats) == len(expected_lats), (
        f"val_df birleşimi {len(actual_lats)} satır, orijinal df {len(expected_lats)} satır. "
        "Bazı satırlar eksik veya çift sayılmış."
    )
    assert actual_lats == expected_lats, (
        "val_df'lerin lat değerleri birleşimi orijinal df lat değerleriyle özdeş değil."
    )


def test_spatial_block_split_low_positive_reduces_k() -> None:
    """Toplam pozitif <5*k iken k otomatik düşürülür ve warning loglanır.

    k=5 ile total_pos=3 → koşul: total_pos < 5*k = 25 → k sürekli düşer
    → k=2'de 5*2=10 > 3 hala sağlanır ama k=2 min sınırı → döngü durur.
    Sonuç: folds sayısı <5 (max k=2).
    """
    rng = np.random.default_rng(0)
    n = 20
    lats = rng.uniform(39.4, 39.6, size=n)
    target = np.zeros(n, dtype=np.int8)
    target[:3] = 1  # Sadece 3 pozitif
    df_low = pd.DataFrame({
        "lat": lats,
        "lon": rng.uniform(32.7, 33.0, size=n),
        TARGET_COLUMN: target,
    })

    # caplog yerine el ile handler: caplog fixture burada yok
    import logging as _log
    handler_called: list[str] = []

    class _H(_log.Handler):
        def emit(self, record: _log.LogRecord) -> None:
            if record.levelno >= _log.WARNING:
                handler_called.append(record.getMessage())

    root = _log.getLogger("wildfire_ml.risk.dataset")
    h = _H()
    root.addHandler(h)
    root.setLevel(_log.WARNING)
    try:
        folds = spatial_block_split(df_low, k=5, lat_col="lat")
    finally:
        root.removeHandler(h)

    # k düşürüldü → fold sayısı < 5 olmalı
    assert len(folds) < 5, (
        f"Düşük pozitif örnek ile k azaltılması beklendi; fold sayısı {len(folds)} (beklenen <5)."
    )
    # Warning mesajı loglandı
    assert any("düşürülüyor" in m or "yetersiz" in m for m in handler_called), (
        f"k azaltılırken warning loglanmadı. Yakalanan mesajlar: {handler_called}"
    )


def test_spatial_block_split_low_positive_reduces_k_caplog(caplog) -> None:
    """caplog ile: düşük pozitif → k azaltma warning yakalanır."""
    rng = np.random.default_rng(7)
    n = 20
    lats = rng.uniform(39.4, 39.6, size=n)
    target = np.zeros(n, dtype=np.int8)
    target[:3] = 1  # 3 pozitif, k=5 için 5*5=25 > 3

    df_low = pd.DataFrame({
        "lat": lats,
        "lon": rng.uniform(32.7, 33.0, size=n),
        TARGET_COLUMN: target,
    })

    with caplog.at_level(logging.WARNING, logger="wildfire_ml.risk.dataset"):
        folds = spatial_block_split(df_low, k=5, lat_col="lat")

    assert len(folds) < 5, (
        f"k azaltılması beklendi → fold sayısı {len(folds)} < 5."
    )
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "düşürülüyor" in str(m) or "yetersiz" in str(m)
        for m in warning_messages
    ), (
        f"'düşürülüyor' veya 'yetersiz' içeren WARNING loglanmadı. "
        f"Yakalanan: {warning_messages}"
    )


def test_train_val_test_split_unchanged() -> None:
    """Regression: train_val_test_split mevcut davranışı değişmedi.

    spatial_block_split eklenmesi train_val_test_split'i etkilememeli.
    60/20/20 oranlar ve stratifikasyon korunur.
    """
    cfg = RiskConfig(seed=42, n_samples=500)
    df = load_risk_dataset(path=None, cfg=cfg)
    train, val, test = train_val_test_split(df, cfg)
    n = len(df)

    # Oran kontrolü
    assert abs(len(train) / n - 0.60) <= 0.05, f"Train ratio {len(train)/n:.3f}"
    assert abs(len(val) / n - 0.20) <= 0.05, f"Val ratio {len(val)/n:.3f}"
    assert abs(len(test) / n - 0.20) <= 0.05, f"Test ratio {len(test)/n:.3f}"
    assert len(train) + len(val) + len(test) == n, "Toplam satır korunmalı"

    # Stratifikasyon: TARGET_COLUMN oranı ±5%
    overall_rate = df[TARGET_COLUMN].mean()
    for name, split in [("train", train), ("val", val), ("test", test)]:
        rate = split[TARGET_COLUMN].mean()
        assert abs(rate - overall_rate) <= 0.05, (
            f"{name} pozitif oran {rate:.4f} vs genel {overall_rate:.4f}"
        )
