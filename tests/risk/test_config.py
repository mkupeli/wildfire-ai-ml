# SPDX-License-Identifier: Apache-2.0
"""Sprint 6-A: RealDataConfig dataclass default değer testleri."""
from __future__ import annotations

from wildfire_ml.risk.config import RealDataConfig


def test_real_data_config_defaults() -> None:
    """RealDataConfig() → beklenen default değerler."""
    cfg = RealDataConfig()

    assert cfg.label_window_days == 30, (
        f"label_window_days beklenen 30, got {cfg.label_window_days}"
    )
    assert cfg.leakage_buffer_days == 31, (
        f"leakage_buffer_days beklenen 31, got {cfg.leakage_buffer_days}"
    )
    assert cfg.firms_radius_m == 10_000.0, (
        f"firms_radius_m beklenen 10000.0, got {cfg.firms_radius_m}"
    )
    assert set(cfg.firms_confidence_levels) == {"nominal", "high"}, (
        f"firms_confidence_levels beklenen ('nominal','high'), got {cfg.firms_confidence_levels}"
    )
    assert cfg.spatial_cv_k == 5, (
        f"spatial_cv_k beklenen 5, got {cfg.spatial_cv_k}"
    )


def test_real_data_config_frozen_like_immutable() -> None:
    """RealDataConfig değerleri override edilebilir (frozen=False), ancak defaults sabit."""
    cfg_custom = RealDataConfig(label_window_days=14, spatial_cv_k=3)
    assert cfg_custom.label_window_days == 14
    assert cfg_custom.spatial_cv_k == 3
    # Diğer default'lar korunmalı
    assert cfg_custom.leakage_buffer_days == 31
    assert cfg_custom.firms_radius_m == 10_000.0


def test_real_data_config_confidence_levels_tuple() -> None:
    """firms_confidence_levels tuple[str, ...] tipinde olmalı."""
    cfg = RealDataConfig()
    assert isinstance(cfg.firms_confidence_levels, tuple), (
        f"firms_confidence_levels tuple beklendi, got {type(cfg.firms_confidence_levels)}"
    )
    for level in cfg.firms_confidence_levels:
        assert isinstance(level, str), f"Her confidence level str beklendi, got {type(level)}"
