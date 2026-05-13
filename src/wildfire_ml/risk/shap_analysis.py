# SPDX-License-Identifier: Apache-2.0
"""SHAP feature importance analysis for wildfire risk model.

Headless plotting (matplotlib Agg backend) — WSL/CI uyumlu.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def run_shap_analysis(
    model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = 10,
) -> dict[str, float]:
    """SHAP TreeExplainer + PNG + JSON rapor.

    Returns: {feature_name: mean_abs_shap, ...} azalan sıra, top_n öğesi.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from .config import FEATURE_COLUMNS, SHAP_MAX_SAMPLES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    X_combined = np.vstack([X_train, X_val])

    # S1: SHAP performance guard — büyük dataset'lerde rastgele alt-örnekle.
    if len(X_combined) > SHAP_MAX_SAMPLES:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_combined), size=SHAP_MAX_SAMPLES, replace=False)
        X_combined = X_combined[idx]
        logger.info(
            f"SHAP: X_combined {len(idx)} sample'a indirgendi (max={SHAP_MAX_SAMPLES})"
        )

    shap_values = explainer.shap_values(X_combined)

    # XGBoost binary classifier: shap_values shape (n, 24) (sigmoid output için)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance_pairs = sorted(
        zip(feature_names, mean_abs.tolist()), key=lambda kv: kv[1], reverse=True
    )

    # JSON output — PREPROCESS SYMMETRIC: feature_order korunur
    assert list(feature_names) == FEATURE_COLUMNS, "Feature order mismatch (PREPROCESS SYMMETRIC)"
    importance_json = {
        "feature_order": FEATURE_COLUMNS,
        "features": [{"name": name, "mean_abs_shap": float(score)} for name, score in importance_pairs],
    }
    (output_dir / "shap_importance.json").write_text(
        json.dumps(importance_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # PNG plots (headless)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_combined, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_combined, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    top = dict(importance_pairs[:top_n])
    logger.info(f"Top {top_n} features: {list(top.keys())}")
    return top
