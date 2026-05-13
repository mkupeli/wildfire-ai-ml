# SPDX-License-Identifier: Apache-2.0
"""Sprint 4-A: run_shap_analysis() testleri (T-SH-1..5)."""
from __future__ import annotations

import json

import numpy as np
import pytest

shap = pytest.importorskip("shap", reason="shap kurulu değil — T-SH testleri skip")

from wildfire_ml.risk import FEATURE_COLUMNS, RiskConfig
from wildfire_ml.risk.dataset import load_risk_dataset, train_val_test_split
from wildfire_ml.risk.train_risk import train_risk
from wildfire_ml.risk.shap_analysis import run_shap_analysis


# ---------------------------------------------------------------------------
# Module-scoped fixture — modeli eğit + SHAP verileri hazırla
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shap_inputs(tmp_path_factory):
    """Modeli eğit; X_train, X_val (numpy), shap output_dir döndür."""
    out = tmp_path_factory.mktemp("risk_shap")

    # Modeli eğit ve .ubj kaydet
    train_risk(
        risk_cfg=RiskConfig(n_samples=200, seed=42),
        output_dir=out,
        run_shap=False,
    )

    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(str(out / "risk_model_v1.ubj"))

    # Aynı seed ile dataset yeniden üret → split
    cfg = RiskConfig(n_samples=200, seed=42)
    df = load_risk_dataset(cfg=cfg)
    train_df, val_df, _ = train_val_test_split(df, cfg)

    # shap_analysis.py np.vstack kullanıyor → numpy array gerek
    X_train = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    X_val = val_df[FEATURE_COLUMNS].values.astype(np.float32)

    shap_out = tmp_path_factory.mktemp("shap_out")
    result = run_shap_analysis(model, X_train, X_val, FEATURE_COLUMNS, shap_out)
    return result, shap_out


# ---------------------------------------------------------------------------
# T-SH-1: top 10 dict döndürmeli
# ---------------------------------------------------------------------------

def test_shap_returns_dict_top10(shap_inputs):
    """T-SH-1: run_shap_analysis() 10 elemanlı dict döndürmeli."""
    result, _ = shap_inputs
    assert isinstance(result, dict), f"dict beklendi, got {type(result)}"
    assert len(result) == 10, (
        f"10 feature beklendi, got {len(result)}. Keys: {list(result.keys())}"
    )


# ---------------------------------------------------------------------------
# T-SH-2: shap_importance.json oluştu mu?
# ---------------------------------------------------------------------------

def test_shap_saves_json(shap_inputs):
    """T-SH-2: shap_importance.json oluşmalı; 'feature_order' ve 'features' anahtarları içermeli."""
    _, shap_out = shap_inputs
    json_path = shap_out / "shap_importance.json"
    assert json_path.exists(), f"shap_importance.json bulunamadı: {json_path}"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "feature_order" in data, "'feature_order' anahtarı JSON'da eksik"
    assert "features" in data, "'features' anahtarı JSON'da eksik"


# ---------------------------------------------------------------------------
# T-SH-3: PNG dosyaları oluştu mu?
# ---------------------------------------------------------------------------

def test_shap_saves_pngs(shap_inputs):
    """T-SH-3: shap_summary_bar.png ve shap_summary_beeswarm.png oluşmalı."""
    _, shap_out = shap_inputs
    bar_png = shap_out / "shap_summary_bar.png"
    bee_png = shap_out / "shap_summary_beeswarm.png"
    assert bar_png.exists(), f"shap_summary_bar.png bulunamadı: {bar_png}"
    assert bar_png.stat().st_size > 0, "shap_summary_bar.png boş"
    assert bee_png.exists(), f"shap_summary_beeswarm.png bulunamadı: {bee_png}"
    assert bee_png.stat().st_size > 0, "shap_summary_beeswarm.png boş"


# ---------------------------------------------------------------------------
# T-SH-4: feature_order == FEATURE_COLUMNS
# ---------------------------------------------------------------------------

def test_shap_feature_order_matches_config(shap_inputs):
    """T-SH-4: JSON 'feature_order' listesi FEATURE_COLUMNS ile özdeş olmalı."""
    _, shap_out = shap_inputs
    json_path = shap_out / "shap_importance.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["feature_order"] == FEATURE_COLUMNS, (
        f"JSON feature_order FEATURE_COLUMNS ile uyuşmuyor.\n"
        f"Beklenen: {FEATURE_COLUMNS}\n"
        f"Gerçek:   {data['feature_order']}"
    )


# ---------------------------------------------------------------------------
# T-SH-5: SHAP_MAX_SAMPLES == 2000
# ---------------------------------------------------------------------------

def test_shap_max_samples_constant():
    """T-SH-5: config.SHAP_MAX_SAMPLES == 2000."""
    from wildfire_ml.risk.config import SHAP_MAX_SAMPLES
    assert SHAP_MAX_SAMPLES == 2000, (
        f"SHAP_MAX_SAMPLES beklenen 2000, got {SHAP_MAX_SAMPLES}"
    )
