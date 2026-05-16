# SPDX-License-Identifier: Apache-2.0
"""Sprint 6-B: build_real_dataset.py T3 + T4 testleri.

T3) build_real_dataset + mock gerçek FIRMS CSV:
    5 hotspot 2024-06-01 Beynam bbox içi → data_version="real-b1",
    positive_rate > 0, meta label_source="firms_archive".

T4) Seyreklik eşiği:
    positive_rate < positive_rate_threshold → logger.error/warning
    "pozitif oran < eşik" mesajı (bbox dışı hotspot CSV; network YOK).

Gerçek ağ çağrısı YAPILMAZ — FIRMS CSV mock olarak tmp_path'e yazılır.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# build_real_dataset modülünü scripts/ altından import et
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def _load_build_module():
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    spec = importlib.util.spec_from_file_location(
        "build_real_dataset",
        _SCRIPTS_DIR / "build_real_dataset.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_brd = _load_build_module()


# ---------------------------------------------------------------------------
# T3: FIRMS CSV (5 Beynam-içi hotspot) → data_version="real-b1",
#     positive_rate > 0, label_source="firms_archive"
# ---------------------------------------------------------------------------

def _beynam_firms_csv(path: Path, n_hotspots: int = 5) -> Path:
    """Beynam bbox (lat 39.4..39.6, lon 32.7..33.0) içinde n hotspot üret.

    Tarih: 2024-06-01. obs_date MS frekansı ile 2024-06-01 olacak → label
    penceresi 2024-06-02..2024-07-01. Hotspot tarihini 2024-06-10 (pencere içi)
    olarak seçiyoruz → label=1 beklenir.
    """
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(n_hotspots):
        rows.append({
            "latitude": float(rng.uniform(39.45, 39.55)),
            "longitude": float(rng.uniform(32.75, 32.95)),
            "acq_date": "2024-06-10",  # obs_date=2024-06-01 için pencere içi
            "confidence": "nominal",
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def test_t3_real_firms_csv_produces_real_b1(tmp_path: Path) -> None:
    """T3: 5 Beynam-içi hotspot CSV → data_version='real-b1', label_source='firms_archive'.

    positive_rate > 0 beklenir: 2024-06-10 hotspot'lar, obs_date 2024-06-01
    (MS frekansı ilk dilim) → label penceresi [2024-06-02, 2024-07-01] → içeride.
    Grid 5000m → az hücre → en az 1 pozitif beklenir.
    """
    firms_csv = tmp_path / "firms_beynam.csv"
    _beynam_firms_csv(firms_csv, n_hotspots=5)

    out_csv = tmp_path / "real_b1_dataset.csv"
    argv = [
        "--out", str(out_csv),
        "--firms-csv", str(firms_csv),
        "--grid-resolution-m", "5000",
    ]
    ret = _brd.main(argv)
    assert ret == 0, f"main() 0 döndürmeli, got {ret!r}"

    assert out_csv.exists(), f"Çıktı CSV oluşturulmadı: {out_csv}"

    meta_path = out_csv.with_suffix(".meta.json")
    assert meta_path.exists(), f".meta.json bulunamadı: {meta_path}"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # data_version == "real-b1" (Sprint 6-B, Karar #8)
    assert meta.get("data_version") == "real-b1", (
        f"data_version='real-b1' beklendi, got {meta.get('data_version')!r}. "
        "build_real_dataset.py'nin FIRMS CSV dalında data_version='real-b1' set etmesi gerekir."
    )

    # label_source == "firms_archive"
    assert meta.get("label_source") == "firms_archive", (
        f"label_source='firms_archive' beklendi, got {meta.get('label_source')!r}"
    )

    # positive_rate > 0 (en az 1 Beynam hotspot label oluşturmalı)
    pos_rate = meta.get("positive_rate", 0.0)
    assert pos_rate > 0.0, (
        f"positive_rate > 0 beklendi (5 Beynam-içi hotspot label penceresi içinde), "
        f"got {pos_rate}. label_builder.build_labels() haversine 10km sorgusu "
        "yanlış çalışıyor olabilir veya obs_date penceresi uyuşmuyor."
    )


# ---------------------------------------------------------------------------
# T4: Seyreklik eşiği — positive_rate < threshold → logger.error
# ---------------------------------------------------------------------------

def _outside_bbox_firms_csv(path: Path) -> Path:
    """Beynam bbox DIŞINDA hotspot — spatial komşu yok → tüm label=0."""
    df = pd.DataFrame([{
        "latitude": 38.0,   # Beynam bbox: 39.4..39.6
        "longitude": 32.85,
        "acq_date": "2024-06-10",
        "confidence": "nominal",
    }])
    df.to_csv(path, index=False)
    return path


def test_t4_sparse_positive_rate_logs_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """T4: bbox dışı hotspot → positive_rate=0 < threshold → logger.error/warning 'pozitif oran < eşik'.

    Network çağrısı YOK — sadece yerel CSV mock.
    """
    firms_csv = tmp_path / "firms_outside.csv"
    _outside_bbox_firms_csv(firms_csv)

    out_csv = tmp_path / "sparse_dataset.csv"
    argv = [
        "--out", str(out_csv),
        "--firms-csv", str(firms_csv),
        "--grid-resolution-m", "5000",
    ]

    with caplog.at_level(logging.ERROR, logger="build_real_dataset"):
        ret = _brd.main(argv)

    assert ret == 0, f"main() hata vermemeli (seyreklik uyarısı hata DEĞİL): {ret!r}"

    # logger.error ile "pozitif oran < eşik" (veya benzer) mesajı beklendi
    error_messages = [
        r.message for r in caplog.records
        if r.levelno >= logging.ERROR
    ]
    has_sparse_error = any(
        ("pozitif" in m and ("eşik" in m or "threshold" in m or "<" in m))
        or "pozitif oran" in m
        for m in error_messages
    )
    assert has_sparse_error, (
        "logger.error 'pozitif oran < eşik' (veya benzer) beklendi. "
        f"Yakalanan ERROR mesajları: {error_messages}\n"
        f"Tüm mesajlar: {[r.message for r in caplog.records]}"
    )


def test_t4_sparse_no_error_when_synthetic_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """T4 negatif: sentetik fallback dalında (--firms-csv yok) seyreklik error LOGLANMAZ.

    Seyreklik kontrolü yalnızca 'firms_archive' label_source'u için anlamlıdır.
    Sentetik ~%15 pozitif oran → eşiğin üstünde; ama kontrol yapılmamalı zaten.
    """
    out_csv = tmp_path / "synth_dataset.csv"
    argv = [
        "--out", str(out_csv),
        "--grid-resolution-m", "5000",
    ]

    with caplog.at_level(logging.ERROR, logger="build_real_dataset"):
        ret = _brd.main(argv)

    assert ret == 0

    # Sentetik fallback dalında "pozitif oran < eşik" ERROR gelmemeli
    error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    has_sparse_error = any("pozitif oran" in m for m in error_messages)
    assert not has_sparse_error, (
        "Sentetik fallback dalında seyreklik ERROR gelmemeli. "
        f"Beklenmedik ERROR mesajları: {error_messages}"
    )


# ---------------------------------------------------------------------------
# obs_date_slices: obs_freq parametresi testi
# ---------------------------------------------------------------------------

def test_obs_date_slices_ms_freq() -> None:
    """obs_date_slices: MS frekansıyla 2024-01..2025-12 → 24 dilim."""
    from wildfire_ml.risk.config import RealDataConfig

    real_cfg = RealDataConfig()
    slices = _brd.obs_date_slices(real_cfg, obs_freq="MS")
    assert len(slices) == 24, (
        f"MS frekansıyla 24 aylık dilim beklendi, got {len(slices)}. "
        f"İlk: {slices[0]}, Son: {slices[-1]}"
    )


def test_obs_date_slices_2w_freq() -> None:
    """obs_date_slices: 2W frekansıyla 2024..2025 → MS'den daha fazla dilim."""
    from wildfire_ml.risk.config import RealDataConfig

    real_cfg = RealDataConfig()
    slices_ms = _brd.obs_date_slices(real_cfg, obs_freq="MS")
    slices_2w = _brd.obs_date_slices(real_cfg, obs_freq="2W")
    assert len(slices_2w) > len(slices_ms), (
        f"2W frekansı MS'den daha fazla dilim üretmeli. "
        f"MS={len(slices_ms)}, 2W={len(slices_2w)}"
    )
