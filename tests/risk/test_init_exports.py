# SPDX-License-Identifier: Apache-2.0
"""Sprint 6-A/Tur 2: wildfire_ml.risk __init__.py export testleri (#1 NICE).

Kapsam:
  - `from wildfire_ml.risk import RealDataConfig, spatial_block_split, build_labels`
    başarıyla import edilmeli (ImportError olmamalı).
  - `wildfire_ml.risk.__all__` şu üçünü de içermeli:
    {'RealDataConfig', 'spatial_block_split', 'build_labels'}.
  - Bu üç isim gerçekten callable / kullanılabilir nesne olmalı.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Import testleri
# ---------------------------------------------------------------------------

def test_import_real_data_config() -> None:
    """RealDataConfig __init__.py'den import edilebilmeli."""
    try:
        from wildfire_ml.risk import RealDataConfig  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"'RealDataConfig' wildfire_ml.risk'ten import edilemedi: {exc}")


def test_import_spatial_block_split() -> None:
    """spatial_block_split __init__.py'den import edilebilmeli."""
    try:
        from wildfire_ml.risk import spatial_block_split  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"'spatial_block_split' wildfire_ml.risk'ten import edilemedi: {exc}")


def test_import_build_labels() -> None:
    """build_labels __init__.py'den import edilebilmeli."""
    try:
        from wildfire_ml.risk import build_labels  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"'build_labels' wildfire_ml.risk'ten import edilemedi: {exc}")


def test_all_contains_required_exports() -> None:
    """__all__ üç zorunlu export'u içermeli: RealDataConfig, spatial_block_split, build_labels."""
    import wildfire_ml.risk as r

    required = {"RealDataConfig", "spatial_block_split", "build_labels"}
    missing = required - set(getattr(r, "__all__", []))
    assert not missing, (
        f"wildfire_ml.risk.__all__ içinde eksik export'lar: {missing}. "
        f"Mevcut __all__: {getattr(r, '__all__', 'YOK')}"
    )


def test_exports_are_callable() -> None:
    """RealDataConfig, spatial_block_split, build_labels callable (class/function) olmalı."""
    from wildfire_ml.risk import RealDataConfig, build_labels, spatial_block_split

    assert callable(RealDataConfig), "RealDataConfig callable değil"
    assert callable(spatial_block_split), "spatial_block_split callable değil"
    assert callable(build_labels), "build_labels callable değil"


def test_real_data_config_instantiable() -> None:
    """RealDataConfig() default argümanlarla instantiate edilebilmeli."""
    from wildfire_ml.risk import RealDataConfig

    cfg = RealDataConfig()
    assert cfg is not None, "RealDataConfig() None döndürdü"
