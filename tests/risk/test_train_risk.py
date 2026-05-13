# SPDX-License-Identifier: Apache-2.0
"""Sprint 4-A: train_risk() unit + integration testleri (T-TR-1..6)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import xgboost as xgb

from wildfire_ml.risk import FEATURE_COLUMNS, FEATURE_SCHEMA, RiskConfig
from wildfire_ml.risk.train_risk import train_risk


# ---------------------------------------------------------------------------
# Module-scoped fixture — modeli tek kez eğit, tüm testler yeniden kullanır.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_result(tmp_path_factory):
    """n_samples=200 ile train_risk çalıştır, (metrics, output_dir) döndür."""
    out = tmp_path_factory.mktemp("risk_train")
    metrics = train_risk(
        risk_cfg=RiskConfig(n_samples=200, seed=42),
        output_dir=out,
        run_shap=False,
    )
    return metrics, out


# ---------------------------------------------------------------------------
# T-TR-1: metrics dict anahtar + tip kontrolü
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "roc_auc", "pr_auc", "f1", "precision", "recall",
    "confusion_matrix", "n_train", "n_val", "n_test",
}


def test_train_risk_returns_metrics(trained_result):
    """T-TR-1: train_risk() REQUIRED_KEYS içeren dict döndürmeli."""
    metrics, _ = trained_result
    missing = REQUIRED_KEYS - set(metrics.keys())
    assert not missing, f"Eksik metrik anahtarları: {missing}"

    # Float / int / list tip kontrolü
    float_keys = {"roc_auc", "pr_auc", "f1", "precision", "recall"}
    for key in float_keys:
        assert isinstance(metrics[key], float), (
            f"metrics['{key}'] float beklendi, got {type(metrics[key])}"
        )

    int_keys = {"n_train", "n_val", "n_test"}
    for key in int_keys:
        assert isinstance(metrics[key], int), (
            f"metrics['{key}'] int beklendi, got {type(metrics[key])}"
        )

    assert isinstance(metrics["confusion_matrix"], list), (
        f"metrics['confusion_matrix'] list beklendi, got {type(metrics['confusion_matrix'])}"
    )


# ---------------------------------------------------------------------------
# T-TR-2: model dosyası oluştu mu?
# ---------------------------------------------------------------------------

def test_train_risk_saves_model_file(trained_result):
    """T-TR-2: output_dir/risk_model_v1.ubj oluşmalı."""
    _, out = trained_result
    model_path = out / "risk_model_v1.ubj"
    assert model_path.exists(), f"risk_model_v1.ubj bulunamadı: {model_path}"
    assert model_path.stat().st_size > 0, "risk_model_v1.ubj boş"


# ---------------------------------------------------------------------------
# T-TR-3: schema.json kopyalandı mı?
# ---------------------------------------------------------------------------

def test_train_risk_saves_schema_json(trained_result):
    """T-TR-3: risk_feature_schema.json oluşmalı ve içerik FEATURE_SCHEMA ile JSON-eşit olmalı."""
    _, out = trained_result
    schema_path = out / "risk_feature_schema.json"
    assert schema_path.exists(), f"risk_feature_schema.json bulunamadı: {schema_path}"

    # İçerik karşılaştırması: kaynak schema.json → FEATURE_SCHEMA ile eşleşmeli
    loaded = json.loads(schema_path.read_text(encoding="utf-8"))

    # FEATURE_SCHEMA'nın JSON-serializable kopyasıyla karşılaştır
    expected = json.loads(json.dumps(FEATURE_SCHEMA))
    assert loaded == expected, (
        "risk_feature_schema.json içeriği FEATURE_SCHEMA ile uyuşmuyor."
    )


# ---------------------------------------------------------------------------
# T-TR-4: runtime model card oluştu mu?
# ---------------------------------------------------------------------------

def test_train_risk_saves_runtime_card(trained_result):
    """T-TR-4: risk_model_v1_card.md oluşmalı ve 'ROC-AUC' substring içermeli."""
    _, out = trained_result
    card_path = out / "risk_model_v1_card.md"
    assert card_path.exists(), f"risk_model_v1_card.md bulunamadı: {card_path}"
    content = card_path.read_text(encoding="utf-8")
    assert "ROC-AUC" in content.upper(), (
        f"'ROC-AUC' metni card dosyasında bulunamadı.\nİçerik başı: {content[:200]}"
    )


# ---------------------------------------------------------------------------
# T-TR-5: ROC-AUC >= 0.65
# ---------------------------------------------------------------------------

def test_train_risk_roc_auc_above_chance(trained_result):
    """T-TR-5: Sentetik feature-correlated veri ile roc_auc >= 0.65 beklenir."""
    metrics, _ = trained_result
    roc = metrics["roc_auc"]
    assert roc >= 0.65, (
        f"roc_auc beklenen >= 0.65, got {roc:.4f}. "
        "Sentetik veri feature-target korelasyonu sağlamıyor olabilir."
    )


# ---------------------------------------------------------------------------
# T-TR-6: feature_names_in_ == FEATURE_COLUMNS
# ---------------------------------------------------------------------------

def test_feature_names_preserved_after_fit(trained_result):
    """T-TR-6: Kaydedilen model yüklenince feature_names_in_ == FEATURE_COLUMNS."""
    _, out = trained_result
    model_path = out / "risk_model_v1.ubj"
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(str(model_path))

    assert hasattr(loaded_model, "feature_names_in_"), (
        "Yüklenen modelde 'feature_names_in_' attribute eksik. "
        "DataFrame pass ile eğitim yapılmadı olabilir."
    )
    actual = list(loaded_model.feature_names_in_)
    assert actual == FEATURE_COLUMNS, (
        f"feature_names_in_ uyuşmuyor.\n"
        f"Beklenen: {FEATURE_COLUMNS}\n"
        f"Gerçek:   {actual}"
    )
