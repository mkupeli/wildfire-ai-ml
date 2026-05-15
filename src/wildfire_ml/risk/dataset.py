# SPDX-License-Identifier: Apache-2.0
"""Dataset loader + train/val/test splitter for risk model."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, TARGET_COLUMN, RiskConfig
from .synthetic_data import SyntheticRiskDataGenerator

logger = logging.getLogger(__name__)


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


def spatial_block_split(
    df: pd.DataFrame,
    k: int = 5,
    lat_col: str = "lat",
    target_col: str = TARGET_COLUMN,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Enlem-bandı bazlı spatial block cross-validation split.

    Random stratified split komşu grid hücrelerini hem train hem val'a
    dağıtarak uzamsal otokorelasyondan kaynaklı iyimser metrik üretir.
    Roberts et al. 2017 (DOI:10.1111/ecog.02881) — uzamsal yapılı veride
    blok CV önerir. Burada `pd.qcut(lat, q=k)` ile ~eşit örnekli k enlem
    bandı kurulur; her fold'da 1 bant val, kalan k-1 bant train.

    Dönüş: k adet (train_df, val_df). Bantta yeterli pozitif örnek yoksa
    (her bant ≥5 pozitif) k otomatik düşürülür ve logger.warning verilir.
    Mevcut `train_val_test_split` (stratified random) bu fonksiyondan
    bağımsızdır ve değişmemiştir.
    """
    if lat_col not in df.columns:
        raise ValueError(
            f"spatial_block_split: '{lat_col}' sütunu yok. Gerçek-veri "
            f"dataset'i koordinat sütunları içermelidir."
        )
    if target_col not in df.columns:
        raise ValueError(f"spatial_block_split: '{target_col}' sütunu yok.")

    n = len(df)
    k = max(2, int(k))

    # Her bantta >=5 pozitif garantisi için k'yi gerekirse düşür.
    total_pos = int((df[target_col] == 1).sum())
    while k > 2 and total_pos < 5 * k:
        logger.warning(
            "spatial_block_split: toplam %d pozitif örnek %d bant için "
            "yetersiz (band başına >=5 hedef); k %d → %d düşürülüyor.",
            total_pos, k, k, k - 1,
        )
        k -= 1

    work = df.reset_index(drop=True)
    try:
        bands = pd.qcut(work[lat_col], q=k, labels=False, duplicates="drop")
    except ValueError:
        bands = pd.qcut(
            work[lat_col].rank(method="first"), q=k, labels=False, duplicates="drop"
        )
    work = work.assign(_band=bands)

    unique_bands = sorted(b for b in work["_band"].dropna().unique())
    if len(unique_bands) < k:
        logger.warning(
            "spatial_block_split: qcut yalnızca %d ayrık bant üretti "
            "(istenen k=%d); fiili k=%d.",
            len(unique_bands), k, len(unique_bands),
        )

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for b in unique_bands:
        val_mask = work["_band"] == b
        band_pos = int((work.loc[val_mask, target_col] == 1).sum())
        if band_pos < 5:
            logger.warning(
                "spatial_block_split: enlem bandı %s yalnızca %d pozitif "
                "örnek içeriyor (<5) — Beynam seyrekliği; fold yine de "
                "üretiliyor, metrik yorumunda dikkat.",
                b, band_pos,
            )
        val_df = work.loc[val_mask].drop(columns="_band").reset_index(drop=True)
        train_df = work.loc[~val_mask].drop(columns="_band").reset_index(drop=True)
        folds.append((train_df, val_df))

    logger.info(
        "spatial_block_split: %d fold, n=%d, toplam pozitif=%d",
        len(folds), n, total_pos,
    )
    return folds
