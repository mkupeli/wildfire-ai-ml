# SPDX-License-Identifier: Apache-2.0
"""Sprint 4-A: export_risk_onnx() testleri (T-EX-1..5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wildfire_ml.risk import FEATURE_COLUMNS, RiskConfig
from wildfire_ml.risk.train_risk import train_risk
from wildfire_ml.risk.export_risk import export_risk_onnx


# ---------------------------------------------------------------------------
# Module-scoped fixture — modeli tek kez eğit + ONNX export et
# skl2onnx / onnxmltools / onnxruntime eksikse tüm export testleri skip olur.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def export_result(tmp_path_factory):
    """Model eğit → ONNX export et. (model_path, onnx_path, out_dir) döndür."""
    pytest.importorskip("skl2onnx", reason="skl2onnx kurulu değil — T-EX-1..4 skip")
    pytest.importorskip("onnxmltools", reason="onnxmltools kurulu değil — T-EX-1..4 skip")
    pytest.importorskip("onnxruntime", reason="onnxruntime kurulu değil — T-EX-1..4 skip")

    out = tmp_path_factory.mktemp("risk_export")
    train_risk(
        risk_cfg=RiskConfig(n_samples=200, seed=42),
        output_dir=out,
        run_shap=False,
    )
    model_path = out / "risk_model_v1.ubj"
    onnx_path = out / "risk_model_v1.onnx"
    export_risk_onnx(model_path, onnx_path)
    return model_path, onnx_path, out


# ---------------------------------------------------------------------------
# T-EX-1: .onnx dosyası oluştu mu?
# ---------------------------------------------------------------------------

def test_export_produces_onnx_file(export_result):
    """T-EX-1: export_risk_onnx() sonrası .onnx dosyası oluşmalı."""
    _, onnx_path, _ = export_result
    assert onnx_path.exists(), f".onnx dosyası oluşmadı: {onnx_path}"
    assert onnx_path.stat().st_size > 0, ".onnx dosyası boş"


# ---------------------------------------------------------------------------
# T-EX-2: export_risk_onnx exception fırlatmamalı (smoke test dahil)
# ---------------------------------------------------------------------------

def test_export_smoke_test_passes(export_result):
    """T-EX-2: export_risk_onnx çağrısı (iç smoke test dahil) exception fırlatmamalı."""
    # Fixture içinde zaten başarıyla çalıştı; exception fırlatılmış olsaydı
    # fixture setup aşamasında hata alırdık. Burada ek ikinci çağrı yapıyoruz.
    model_path, _, out = export_result
    second_onnx = out / "risk_model_v1_smoke2.onnx"
    # Exception => test fail
    export_risk_onnx(model_path, second_onnx)


# ---------------------------------------------------------------------------
# T-EX-3: ONNX input shape ve isim kontrolü
# ---------------------------------------------------------------------------

def test_onnx_input_shape(export_result):
    """T-EX-3: Input adı 'features', shape [None, len(FEATURE_COLUMNS)]."""
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime kurulu değil")
    _, onnx_path, _ = export_result
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    assert len(inputs) >= 1, "ONNX modelinde hiç input yok"

    inp = inputs[0]
    assert inp.name == "features", (
        f"Input adı 'features' beklendi, got '{inp.name}'"
    )
    shape = inp.shape
    assert len(shape) == 2, f"Input shape 2-boyutlu beklendi, got {len(shape)}D: {shape}"

    # shape[0] == None veya dynamic (str/None)
    assert shape[0] is None or not isinstance(shape[0], int), (
        f"Input shape[0] dinamik (None) beklendi, got {shape[0]}"
    )
    assert shape[1] == len(FEATURE_COLUMNS), (
        f"Input shape[1] beklenen {len(FEATURE_COLUMNS)}, got {shape[1]}"
    )


# ---------------------------------------------------------------------------
# T-EX-4: ONNX çıktı olasılıkları [0, 1] aralığında
# ---------------------------------------------------------------------------

def test_onnx_output_range(export_result):
    """T-EX-4: Rastgele input ile ONNX 'probabilities' çıktısı 0 <= p <= 1."""
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime kurulu değil")
    _, onnx_path, _ = export_result
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(99)
    X = rng.random((5, len(FEATURE_COLUMNS))).astype(np.float32)
    outputs = sess.run(None, {"features": X})

    # outputs[0] = labels, outputs[1] = probabilities [N, 2]
    assert len(outputs) >= 2, (
        f"ONNX model en az 2 output beklendi (label + proba), got {len(outputs)}"
    )
    probabilities = outputs[1]
    assert probabilities.shape == (5, 2), (
        f"Olasılık çıktısı shape (5, 2) beklendi, got {probabilities.shape}"
    )
    assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0), (
        f"Olasılık değerleri [0, 1] dışına çıktı. "
        f"min={probabilities.min():.6f}, max={probabilities.max():.6f}"
    )


# ---------------------------------------------------------------------------
# T-EX-5: export_risk.py kaynak kodu — initial_type hardcoded sayı içermemeli
# Bu test skl2onnx/onnxruntime gerektirmez — sadece kaynak kodu okur.
# ---------------------------------------------------------------------------

def test_feature_columns_in_initial_type():
    """T-EX-5: export_risk.py 'initial_type' satırında hardcoded '24' yerine
    len(FEATURE_COLUMNS) kullanılmalı."""
    src_path = Path(__file__).parents[2] / "src" / "wildfire_ml" / "risk" / "export_risk.py"
    assert src_path.exists(), f"export_risk.py bulunamadı: {src_path}"
    source = src_path.read_text(encoding="utf-8")

    # initial_type satırını bul
    initial_type_lines = [
        line for line in source.splitlines()
        if "initial_type" in line and "FloatTensorType" in line
    ]
    assert initial_type_lines, (
        "export_risk.py içinde 'initial_type' + 'FloatTensorType' içeren satır bulunamadı."
    )

    # Satırda "len(FEATURE_COLUMNS)" geçmeli — hardcoded rakam değil
    for line in initial_type_lines:
        assert "len(FEATURE_COLUMNS)" in line, (
            f"initial_type satırında 'len(FEATURE_COLUMNS)' beklendi, "
            f"hardcoded rakam kullanılmış olabilir.\nSatır: {line.strip()}"
        )
