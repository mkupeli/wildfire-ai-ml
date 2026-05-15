# SPDX-License-Identifier: Apache-2.0
"""XGBoost binary classifier training for wildfire risk.

PREPROCESS SYMMETRIC: X_train = train_df[FEATURE_COLUMNS].values — feature
sırası config.py FEATURE_COLUMNS ile aynı. Backend Sprint 4-C aynı sırayı kullanacak.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost.callback import EarlyStopping

from .config import FEATURE_COLUMNS, TARGET_COLUMN, RiskConfig, XGBoostConfig
from .dataset import load_risk_dataset, train_val_test_split

logger = logging.getLogger(__name__)


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    return neg / max(pos, 1)


def train_risk(
    risk_cfg: RiskConfig | None = None,
    xgb_cfg: XGBoostConfig | None = None,
    csv_path: Path | str | None = None,
    output_dir: Path | str = Path("models"),
    run_shap: bool = True,
) -> dict[str, float]:
    """XGBoost classifier eğit, kaydet, metrikleri döndür."""
    risk_cfg = risk_cfg or RiskConfig()
    xgb_cfg = xgb_cfg or XGBoostConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_risk_dataset(path=csv_path, cfg=risk_cfg)
    train_df, val_df, test_df = train_val_test_split(df, risk_cfg)

    # T1: DataFrame olarak bırak — XGBoost 2.x feature_names_in_ otomatik set eder.
    # skl2onnx dönüşümünde feature isim bilgisi DataFrame üzerinden taşınır.
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].values.astype(np.int32)
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN].values.astype(np.int32)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values.astype(np.int32)

    scale_pos = xgb_cfg.scale_pos_weight or compute_scale_pos_weight(y_train)

    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg.n_estimators,
        max_depth=xgb_cfg.max_depth,
        learning_rate=xgb_cfg.learning_rate,
        scale_pos_weight=scale_pos,
        min_child_weight=xgb_cfg.min_child_weight,
        subsample=xgb_cfg.subsample,
        colsample_bytree=xgb_cfg.colsample_bytree,
        gamma=xgb_cfg.gamma,
        reg_alpha=xgb_cfg.reg_alpha,
        reg_lambda=xgb_cfg.reg_lambda,
        eval_metric=xgb_cfg.eval_metric,
        tree_method=xgb_cfg.tree_method,
        device=xgb_cfg.device,
        verbosity=xgb_cfg.verbosity,
        random_state=risk_cfg.seed,
        callbacks=[EarlyStopping(rounds=xgb_cfg.early_stopping_rounds, save_best=True)],
    )

    # XGBoost 2.x: DataFrame pass → feature_names_in_ otomatik set edilir.
    # Post-hoc atama (model.feature_names_in_ = ...) skl2onnx'e taşınmıyor; bu yüzden kaldırıldı.
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    model_path = output_dir / "risk_model_v2.ubj"
    model.save_model(str(model_path))
    logger.info(f"Model saved: {model_path}")

    # Test metrikleri
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(np.int32)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "scale_pos_weight": float(scale_pos),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # Runtime model card
    _write_runtime_card(output_dir / "risk_model_v2_card.md", metrics, xgb_cfg)

    # T2: schema.json kopyala — Sprint 4-C backend `models/risk_feature_schema.json` okuyacak.
    import shutil
    src_schema = Path(__file__).parent / "schema.json"
    if src_schema.exists():
        shutil.copy2(src_schema, output_dir / "risk_feature_schema.json")
        logger.info(f"Schema kopyalandı: {output_dir / 'risk_feature_schema.json'}")

    # SHAP
    if run_shap:
        from .shap_analysis import run_shap_analysis
        shap_importance = run_shap_analysis(
            model, X_train, X_val, FEATURE_COLUMNS, output_dir
        )
        metrics["shap_top5"] = list(shap_importance.items())[:5]

    return metrics


def _write_runtime_card(path: Path, metrics: dict, xgb_cfg: XGBoostConfig) -> None:
    lines = [
        "# Runtime Model Card — risk_model_v2",
        f"Trained: {metrics['trained_at']}",
        f"Test metrics: ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, "
        f"F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}",
        f"n_train={metrics['n_train']}, n_val={metrics['n_val']}, n_test={metrics['n_test']}",
        f"scale_pos_weight={metrics['scale_pos_weight']:.4f}",
        f"Hyperparams: max_depth={xgb_cfg.max_depth}, lr={xgb_cfg.learning_rate}, "
        f"n_estimators={xgb_cfg.n_estimators}, eval_metric={xgb_cfg.eval_metric}",
        "",
        "Sentetik veri ile eğitildi — production değil. Bkz. src/wildfire_ml/risk/model_card.md",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train wildfire risk XGBoost model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    risk_cfg = RiskConfig(seed=args.seed, n_samples=args.n_samples)
    xgb_cfg = XGBoostConfig(
        n_estimators=args.n_estimators or XGBoostConfig().n_estimators,
        max_depth=args.max_depth or XGBoostConfig().max_depth,
        learning_rate=args.learning_rate or XGBoostConfig().learning_rate,
    )

    metrics = train_risk(
        risk_cfg=risk_cfg, xgb_cfg=xgb_cfg, csv_path=args.csv_path,
        output_dir=args.output_dir, run_shap=not args.no_shap,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "shap_top5"}, indent=2))


if __name__ == "__main__":
    main()
