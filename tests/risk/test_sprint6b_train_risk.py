# SPDX-License-Identifier: Apache-2.0
"""Sprint 6-B: train_risk.py T5 + T6 testleri.

T5) Spatial CV fold atlama: fold başına <5 pozitif mock dataset →
    "fold atlanıyor" log, cv_n_folds < k, RuntimeError YOK.

T6) SYNTHETIC uyarısı koşulu:
    data_version="real-b1" → runtime card'da "Sentetik veri ile eğitildi" YOK.
    data_version="synthetic-v2" → VAR.

Gerçek network çağrısı YAPILMAZ — sentetik/mock veriler kullanılır.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildfire_ml.risk.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RealDataConfig,
    RiskConfig,
    XGBoostConfig,
)
from wildfire_ml.risk.train_risk import _write_runtime_card, spatial_cv_evaluate


# ---------------------------------------------------------------------------
# Yardımcı: az pozitifli mock dataset
# ---------------------------------------------------------------------------

def _make_low_positive_df(
    n: int = 60,
    n_positive: int = 4,
    seed: int = 99,
) -> pd.DataFrame:
    """k=5 fold için yetersiz pozitif örnek içeren minimal DataFrame.

    n_positive=4 → k=5 ile her fold'da ortalama <1 pozitif → fold atlanmalı.
    spatial_block_split: lat sütunu gerektirir.
    """
    rng = np.random.default_rng(seed)
    from wildfire_ml.risk.synthetic_data import SyntheticRiskDataGenerator

    gen = SyntheticRiskDataGenerator(RiskConfig(seed=seed, n_samples=n))
    df = gen.generate(n=n)

    # TARGET_COLUMN: yalnızca n_positive satır 1, geri kalan 0
    target = np.zeros(n, dtype=np.int8)
    target[:n_positive] = 1
    df[TARGET_COLUMN] = target

    # lat/lon sütunları ekle (spatial_block_split gerektirir)
    df["lat"] = rng.uniform(39.4, 39.6, size=n)
    df["lon"] = rng.uniform(32.7, 33.0, size=n)

    return df


# ---------------------------------------------------------------------------
# T5: Fold atlama — <5 pozitif/fold → "fold atlanıyor" log, cv_n_folds < k
# ---------------------------------------------------------------------------

def test_t5_spatial_cv_fold_skip_low_positive_caplog(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T5: fold başına <5 pozitif → uyarı loglanır, cv_n_folds<k, RuntimeError YOK.

    n_positive=4, k=5 → spatial_block_split k'yi 2'ye düşürür (her bant için
    >=5 pozitif garantisi). 'düşürülüyor' WARNING loglanır.
    cv_n_folds = üretilen fold sayısı (≤ başlangıç k=5'ten az).
    RuntimeError fırlatılmamalı.
    """
    df = _make_low_positive_df(n=60, n_positive=4)

    xgb_cfg = XGBoostConfig(n_estimators=10, verbosity=0)

    # dataset.py logger'ını da yakala (spatial_block_split mesajları oradan gelir)
    with caplog.at_level(logging.WARNING):
        # RuntimeError fırlatılmamalı
        result = spatial_cv_evaluate(df, xgb_cfg, seed=42, k=5)

    # cv_n_folds < başlangıç k=5 (k azaltması veya fold atlama nedeniyle)
    cv_n_folds = result.get("cv_n_folds", None)
    assert cv_n_folds is not None, "spatial_cv_evaluate 'cv_n_folds' döndürmeli"
    assert isinstance(cv_n_folds, int), (
        f"cv_n_folds int olmalı, got {type(cv_n_folds)}"
    )
    assert cv_n_folds < 5, (
        f"Düşük pozitif oranla cv_n_folds < 5 beklendi, got {cv_n_folds}. "
        "spatial_block_split k azaltması etkili olmalı."
    )

    # k azaltması veya seyreklik uyarısı içeren WARNING loglanmış olmalı
    all_messages = [r.message for r in caplog.records]
    has_warning_log = any(
        "düşürülüyor" in m
        or "yetersiz" in m
        or "atlanıyor" in m
        or "tek sınıf" in m
        or "seyrekli" in m
        for m in all_messages
    )
    assert has_warning_log, (
        f"k azaltması veya seyreklik WARNING beklendi. "
        f"Yakalanan mesajlar: {all_messages}"
    )


def test_t5_spatial_cv_no_exception_with_all_negative_folds() -> None:
    """T5: tüm foldlar atlanınca bile RuntimeError YOK, cv_n_folds=0 döner."""
    # n_positive=0: tüm foldlar tek sınıf → hepsi atlanır
    df = _make_low_positive_df(n=30, n_positive=0)
    xgb_cfg = XGBoostConfig(n_estimators=5, verbosity=0)

    # RuntimeError fırlatılmamalı
    result = spatial_cv_evaluate(df, xgb_cfg, seed=42, k=5)

    assert "cv_n_folds" in result
    assert result["cv_n_folds"] == 0, (
        f"Tüm foldlar atlanınca cv_n_folds=0 beklendi, got {result['cv_n_folds']}"
    )
    # Metrikler 0.0 olmalı (hiç fold yok)
    assert result["cv_roc_auc_mean"] == 0.0
    assert result["cv_pr_auc_mean"] == 0.0


# ---------------------------------------------------------------------------
# T6: _write_runtime_card SYNTHETIC uyarısı koşulu
# ---------------------------------------------------------------------------

def test_t6_synthetic_v2_card_has_synthetic_warning(tmp_path: Path) -> None:
    """T6: data_version='synthetic-v2' → runtime card 'Sentetik veri ile eğitildi' içerir."""
    card_path = tmp_path / "card_synthetic.md"
    metrics = {
        "trained_at": "2024-01-01T00:00:00+00:00",
        "roc_auc": 0.75,
        "pr_auc": 0.60,
        "f1": 0.55,
        "precision": 0.60,
        "recall": 0.50,
        "n_train": 300,
        "n_val": 100,
        "n_test": 100,
        "scale_pos_weight": 5.0,
    }
    xgb_cfg = XGBoostConfig()

    _write_runtime_card(
        card_path,
        metrics,
        xgb_cfg,
        data_version="synthetic-v2",
    )

    content = card_path.read_text(encoding="utf-8")
    assert "Sentetik veri ile eğitildi" in content, (
        f"data_version='synthetic-v2' → 'Sentetik veri ile eğitildi' card'da olmalı. "
        f"Card içeriği:\n{content}"
    )


def test_t6_real_b1_card_no_synthetic_warning(tmp_path: Path) -> None:
    """T6: data_version='real-b1' → runtime card 'Sentetik veri ile eğitildi' İÇERMEZ."""
    card_path = tmp_path / "card_real_b1.md"
    metrics = {
        "trained_at": "2024-06-01T00:00:00+00:00",
        "roc_auc": 0.80,
        "pr_auc": 0.65,
        "f1": 0.60,
        "precision": 0.65,
        "recall": 0.55,
        "n_train": 400,
        "n_val": 130,
        "n_test": 130,
        "scale_pos_weight": 4.0,
    }
    xgb_cfg = XGBoostConfig()

    _write_runtime_card(
        card_path,
        metrics,
        xgb_cfg,
        data_version="real-b1",
    )

    content = card_path.read_text(encoding="utf-8")
    assert "Sentetik veri ile eğitildi" not in content, (
        f"data_version='real-b1' → 'Sentetik veri ile eğitildi' card'da OLMAMALI. "
        f"Card içeriği:\n{content}"
    )


def test_t6_real_b1_card_has_real_data_note(tmp_path: Path) -> None:
    """T6 ek: data_version='real-b1' → card'da 'real-b1' ve gerçek-veri notu bulunur."""
    card_path = tmp_path / "card_real_note.md"
    metrics = {
        "trained_at": "2024-06-01T00:00:00+00:00",
        "roc_auc": 0.80,
        "pr_auc": 0.65,
        "f1": 0.60,
        "precision": 0.65,
        "recall": 0.55,
        "n_train": 400,
        "n_val": 130,
        "n_test": 130,
        "scale_pos_weight": 4.0,
    }
    xgb_cfg = XGBoostConfig()

    _write_runtime_card(
        card_path,
        metrics,
        xgb_cfg,
        data_version="real-b1",
    )

    content = card_path.read_text(encoding="utf-8")
    # "real-b1" data_version card'da bulunmalı
    assert "real-b1" in content, (
        f"data_version='real-b1' card'da geçmeli. Card:\n{content}"
    )
    # Gerçek-veri akışı notu (SYNTHETIC uyarısının yerine)
    assert "Gerçek-veri akışı" in content or "gerçek" in content.lower(), (
        f"real-b1 card'da gerçek-veri notu beklendi. Card:\n{content}"
    )


def test_t6_default_data_version_is_synthetic() -> None:
    """T6: _write_runtime_card default data_version='synthetic-v2' → SYNTHETIC uyarısı var."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        card_path = Path(f.name)

    metrics = {
        "trained_at": "2024-01-01T00:00:00+00:00",
        "roc_auc": 0.70,
        "pr_auc": 0.55,
        "f1": 0.50,
        "precision": 0.55,
        "recall": 0.45,
        "n_train": 200,
        "n_val": 67,
        "n_test": 67,
        "scale_pos_weight": 6.0,
    }

    # data_version default değeri (synthetic-v2)
    _write_runtime_card(card_path, metrics, XGBoostConfig())

    content = card_path.read_text(encoding="utf-8")
    card_path.unlink(missing_ok=True)

    assert "Sentetik veri ile eğitildi" in content, (
        "Default data_version (synthetic-v2) → 'Sentetik veri ile eğitildi' olmalı. "
        f"Card:\n{content}"
    )
