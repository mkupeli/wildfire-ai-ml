# SPDX-License-Identifier: Apache-2.0
"""Sprint 4-A BONUS: schema.json ile config.py::FEATURE_SCHEMA simetri testi (Karar #6).

Bu test, export_risk_schema.py çalıştırıldıktan sonra schema.json'un
config.py ile birebir senkronize olduğunu garanti eder.
"""
from __future__ import annotations

import json
from pathlib import Path

from wildfire_ml.risk import FEATURE_SCHEMA


# schema.json konumu (src/wildfire_ml/risk/schema.json)
_SCHEMA_JSON_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "wildfire_ml" / "risk" / "schema.json"
)


def test_schema_json_exists() -> None:
    """schema.json dosyası beklenen konumda mevcut olmalı."""
    assert _SCHEMA_JSON_PATH.exists(), f"schema.json bulunamadı: {_SCHEMA_JSON_PATH}"


def test_schema_json_matches_feature_schema() -> None:
    """schema.json içeriği config.py::FEATURE_SCHEMA ile birebir eşleşmeli.

    Karar #6 simetri garantisi: backend sprint 4-C bu dosyayı single source of
    truth olarak kullanır; config.py değiştiğinde export unutulursa bu test kırmızıya döner.
    """
    with open(_SCHEMA_JSON_PATH, encoding="utf-8") as f:
        json_schema = json.load(f)

    # Anahtar seti eşleşmeli
    json_keys = set(json_schema.keys())
    config_keys = set(FEATURE_SCHEMA.keys())
    assert json_keys == config_keys, (
        f"Anahtar uyumsuzluğu.\n"
        f"Sadece JSON'da: {json_keys - config_keys}\n"
        f"Sadece config'de: {config_keys - json_keys}"
    )

    # Her anahtarın tüm alanları eşleşmeli
    mismatches: list[str] = []
    for key in config_keys:
        for field in ("dtype", "min", "max", "unit", "description"):
            json_val = json_schema[key].get(field)
            cfg_val = FEATURE_SCHEMA[key].get(field)
            if json_val != cfg_val:
                mismatches.append(
                    f"  [{key}][{field}]: JSON={json_val!r}, config={cfg_val!r}"
                )

    assert not mismatches, (
        f"schema.json ile FEATURE_SCHEMA arasında {len(mismatches)} uyumsuzluk:\n"
        + "\n".join(mismatches)
    )
