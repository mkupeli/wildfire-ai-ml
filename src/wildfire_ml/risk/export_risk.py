# SPDX-License-Identifier: Apache-2.0
"""ONNX export for wildfire risk XGBoost model (risk_model_v2).

Output kontratı (Sprint 4-C backend için):
- Input: 'features' float32 [N, len(FEATURE_COLUMNS)]  # = 24
- Output: 'probabilities' float32 [N, 2]; risk_score = probabilities[:, 1]

Sprint 5: model versiyon v2 — cos(lat) PREPROCESS_SYMMETRIC fix uygulandı
(Karar #6). Default model/onnx adları `risk_model_v2.*`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from .config import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def export_risk_onnx(
    model_path: Path | str,
    onnx_path: Path | str,
    opset: int = 15,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    """XGBoost .ubj → ONNX FP32. Smoke test inference match.

    Input: 'features' float32 [N, len(FEATURE_COLUMNS)]  # = 24
    opset=15 default — XGBoost TreeEnsemble operatörü opset 15'te stabil;
    opset 17 ONNX Runtime'da development/limited support.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from onnxruntime import InferenceSession
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
    from skl2onnx import update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

    model_path = Path(model_path)
    onnx_path = Path(onnx_path)

    # XGBClassifier'ı yükle
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    # skl2onnx XGBClassifier converter'ını kaydet
    update_registered_converter(
        xgb.XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )

    initial_type = [("features", FloatTensorType([None, len(FEATURE_COLUMNS)]))]
    onx = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=opset,
        options={id(model): {"zipmap": False}},
    )
    onnx_path.write_bytes(onx.SerializeToString())
    logger.info(f"ONNX exported: {onnx_path}")

    # Smoke test
    sess = InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    X = rng.random((10, len(FEATURE_COLUMNS))).astype(np.float32)
    xgb_proba = model.predict_proba(X)[:, 1]
    ort_outputs = sess.run(None, {"features": X})
    # ort_outputs: [label, probabilities] (zipmap=False)
    ort_proba = ort_outputs[1][:, 1]
    np.testing.assert_allclose(ort_proba, xgb_proba, rtol=rtol, atol=atol)
    logger.info(f"Smoke test passed (rtol={rtol}, atol={atol})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export XGBoost risk model to ONNX")
    parser.add_argument("--model-path", default="models/risk_model_v2.ubj")
    parser.add_argument("--onnx-path", default="models/risk_model_v2.onnx")
    parser.add_argument("--opset", type=int, default=15)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export_risk_onnx(args.model_path, args.onnx_path, args.opset, atol=args.atol)


if __name__ == "__main__":
    main()
