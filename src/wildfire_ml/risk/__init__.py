# SPDX-License-Identifier: Apache-2.0
"""Risk haritası ML sub-package — sentetik veri + feature pipeline (Sprint 4-A/B)."""
from .config import (
    RiskConfig,
    RealDataConfig,
    XGBoostConfig,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    WORLDCOVER_CLASSES,
    TOPO_FEATURES,
    METEO_FEATURES,
    FIRE_HIST_FEATURES,
    HUMAN_FEATURES,
    FEATURE_SCHEMA,
    SYNTHETIC_DATA_NOTE,
    SHAP_MAX_SAMPLES,
)
from .synthetic_data import SyntheticRiskDataGenerator
from .features import compute_vpd, compute_slope_aspect, compute_firms_density, compute_ffmc_approx
from .dataset import load_risk_dataset, train_val_test_split, spatial_block_split
from .label_builder import build_labels
from .train_risk import train_risk
from .shap_analysis import run_shap_analysis
from .export_risk import export_risk_onnx

__all__ = [
    "RiskConfig", "RealDataConfig", "XGBoostConfig",
    "FEATURE_COLUMNS", "TARGET_COLUMN", "WORLDCOVER_CLASSES",
    "TOPO_FEATURES", "METEO_FEATURES", "FIRE_HIST_FEATURES", "HUMAN_FEATURES",
    "FEATURE_SCHEMA", "SYNTHETIC_DATA_NOTE",
    "SHAP_MAX_SAMPLES",
    "SyntheticRiskDataGenerator",
    "compute_vpd", "compute_slope_aspect", "compute_firms_density", "compute_ffmc_approx",
    "load_risk_dataset", "train_val_test_split", "spatial_block_split",
    "build_labels",
    "train_risk",
    "run_shap_analysis",
    "export_risk_onnx",
]
