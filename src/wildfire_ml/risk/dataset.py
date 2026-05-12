# SPDX-License-Identifier: Apache-2.0
"""Dataset loader + train/val/test splitter for risk model."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, TARGET_COLUMN, RiskConfig
from .synthetic_data import SyntheticRiskDataGenerator


def load_risk_dataset(
    path: Path | str | None = None,
    cfg: RiskConfig | None = None,
) -> pd.DataFrame:
    """Risk dataset yükle. path=None ise sentetik üretir.

    Yüklenen DataFrame: FEATURE_COLUMNS + [TARGET_COLUMN] sütunları.
    """
    cfg = cfg or RiskConfig()
    if path is None:
        return SyntheticRiskDataGenerator(cfg).generate()
    df = pd.read_csv(Path(path))
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}")
    df.attrs["source"] = str(path)
    df.attrs["data_version"] = cfg.data_version
    return df


def train_val_test_split(
    df: pd.DataFrame, cfg: RiskConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split → (train, val, test). Default 60/20/20."""
    cfg = cfg or RiskConfig()
    train_val, test = train_test_split(
        df,
        test_size=cfg.test_size,
        stratify=df[TARGET_COLUMN],
        random_state=cfg.seed,
    )
    val_ratio = cfg.val_size / (1.0 - cfg.test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val[TARGET_COLUMN],
        random_state=cfg.seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
