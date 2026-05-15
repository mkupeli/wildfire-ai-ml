# SPDX-License-Identifier: Apache-2.0
"""FIRMS hotspot arşivinden gerçek label + fire-history feature üretimi.

Karar #6 PREPROCESS_SYMMETRIC: Bu modül 10km haversine yarıçapını backend
`wildfire-ai-backend/app/services/risk_feature_service.py` ile *birebir aynı*
formülle uygular — `sklearn.neighbors.BallTree(metric='haversine')`,
EARTH_RADIUS_M = 6_371_000.0, radius_rad = 10_000.0 / EARTH_RADIUS_M,
koordinat sırası [lat, lon] RADYAN. Inference (backend) ve label üretimi
(burası) aynı uzamsal komşuluk tanımını paylaşır.

Label-leakage tamponu (Roberts et al. 2017, DOI:10.1111/ecog.02881,
"Cross-validation strategies for data with temporal, spatial, hierarchical,
or phylogenetic structure"): firms_density_1yr / days_since_last_fire
pencerelerinin sağ kenarı obs_date - leakage_buffer_days (31g) ile kesilir.
Label penceresi obs_date + 1g'den başladığından, 30g'lik gözlem ufkunun
geçmiş-feature'lara sızması (target encoding) engellenir.

WARNING: firms_df boş/None ise (NASA Earthdata auth-walled FIRMS arşivi
verilmediğinde) bu modül çağrılmaz; çağıran taraf sentetik fallback'e geçer
(bkz. scripts/build_real_dataset.py).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .config import RealDataConfig

logger = logging.getLogger(__name__)

# Karar #6 PREPROCESS_SYMMETRIC — backend risk_feature_service.py ile aynı sabit.
EARTH_RADIUS_M = 6_371_000.0


def build_labels(
    grid_df: pd.DataFrame,
    firms_df: pd.DataFrame,
    obs_date_col: str = "obs_date",
    cfg: RealDataConfig | None = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """grid_df'ye FIRMS arşivinden 3 sütun ekler ve DataFrame döndürür.

    Eklenen sütunlar:
      * fire_occurred_within_30d (int8) — TARGET. obs_date+1g .. obs_date+30g
        penceresinde 10km yarıçapta en az 1 hotspot varsa 1.
      * firms_density_1yr (float32) — obs_date-365g .. obs_date-31g
        penceresinde 10km içi hotspot sayısı (31g leakage tamponu).
      * days_since_last_fire (float32) — [tarih başı .. obs_date-31g]
        aralığında 10km içi en son hotspot'a gün; yoksa 3650.

    grid_df: en az [lat_col, lon_col, obs_date_col] sütunları.
    firms_df: en az ['latitude', 'longitude', 'acq_date'] ve opsiyonel
        'confidence' sütunları. confidence düşük ('low' — sun glint) satırlar
        elenir; confidence yoksa tüm satırlar tutulur.

    Vektörizasyon: tek BallTree(haversine) + query_radius. Her grid satırı
    için 10km komşu hotspot index'leri bir kez çekilir, zaman pencereleri
    NumPy maskeleriyle uygulanır. Backend BallTree pattern'i (Karar #6) ile
    aynı: query_radius çıktısı list-of-arrays, [lat, lon] radyan sırası.

    Referans: Roberts et al. 2017 (DOI:10.1111/ecog.02881) — spatial/temporal
    leakage tamponu; Karar #6 PREPROCESS_SYMMETRIC.
    """
    cfg = cfg or RealDataConfig()
    out = grid_df.copy()
    n = len(out)

    radius_rad = cfg.firms_radius_m / EARTH_RADIUS_M
    label_w = np.timedelta64(cfg.label_window_days, "D")
    buffer_d = np.timedelta64(cfg.leakage_buffer_days, "D")
    one_day = np.timedelta64(1, "D")
    year_d = np.timedelta64(365, "D")
    default_days_since = 3650.0

    obs_dates = pd.to_datetime(out[obs_date_col]).to_numpy(dtype="datetime64[ns]")

    # Boş / hiç hotspot kalmayan FIRMS → tüm satırlar negatif/default.
    if firms_df is None or len(firms_df) == 0:
        logger.warning(
            "build_labels: firms_df boş — tüm label=0, density=0, "
            "days_since_last_fire=%.0f", default_days_since,
        )
        out["fire_occurred_within_30d"] = np.zeros(n, dtype=np.int8)
        out["firms_density_1yr"] = np.zeros(n, dtype=np.float32)
        out["days_since_last_fire"] = np.full(n, default_days_since, dtype=np.float32)
        return out

    fdf = firms_df.copy()

    # Confidence filtre: 'low' (sun glint) hariç; cfg.firms_confidence_levels
    # ('nominal','high'). Sütun yoksa filtre atlanır.
    if "confidence" in fdf.columns:
        conf = fdf["confidence"].astype(str).str.strip().str.lower()
        before = len(fdf)
        fdf = fdf[conf.isin([c.lower() for c in cfg.firms_confidence_levels])]
        logger.info(
            "build_labels: confidence filtre %d → %d satır (levels=%s)",
            before, len(fdf), cfg.firms_confidence_levels,
        )
    else:
        logger.warning(
            "build_labels: firms_df 'confidence' sütunu yok — confidence "
            "filtresi atlandı (tüm hotspot'lar kullanılıyor)."
        )

    if len(fdf) == 0:
        logger.warning(
            "build_labels: confidence filtresi sonrası 0 hotspot — tüm label=0."
        )
        out["fire_occurred_within_30d"] = np.zeros(n, dtype=np.int8)
        out["firms_density_1yr"] = np.zeros(n, dtype=np.float32)
        out["days_since_last_fire"] = np.full(n, default_days_since, dtype=np.float32)
        return out

    fire_dates = pd.to_datetime(fdf["acq_date"]).to_numpy(dtype="datetime64[ns]")

    # BallTree haversine — [lat, lon] RADYAN sırası (Karar #6, backend ile aynı).
    hotspot_coords_rad = np.radians(
        np.column_stack([
            fdf["latitude"].to_numpy(dtype=np.float64),
            fdf["longitude"].to_numpy(dtype=np.float64),
        ])
    )
    tree = BallTree(hotspot_coords_rad, metric="haversine")

    query_coords_rad = np.radians(
        np.column_stack([
            out[lat_col].to_numpy(dtype=np.float64),
            out[lon_col].to_numpy(dtype=np.float64),
        ])
    )
    neighbor_idx = tree.query_radius(query_coords_rad, radius_rad)

    label = np.zeros(n, dtype=np.int8)
    density = np.zeros(n, dtype=np.float32)
    days_since = np.full(n, default_days_since, dtype=np.float32)

    for i in range(n):
        idx = neighbor_idx[i]
        if idx.size == 0:
            continue
        obs = obs_dates[i]
        local_fire = fire_dates[idx]

        # Label penceresi: (obs_date, obs_date+30g] → obs+1g .. obs+30g.
        label_lo = obs + one_day
        label_hi = obs + label_w
        if np.any((local_fire >= label_lo) & (local_fire <= label_hi)):
            label[i] = 1

        # Geçmiş feature kesim noktası: obs_date - leakage_buffer_days.
        cutoff = obs - buffer_d

        # firms_density_1yr: [obs-365g, obs-31g] içi hotspot sayısı.
        dens_lo = obs - year_d
        in_density = (local_fire >= dens_lo) & (local_fire <= cutoff)
        density[i] = float(np.count_nonzero(in_density))

        # days_since_last_fire: [-inf, obs-31g] içi en son hotspot.
        past = local_fire[local_fire <= cutoff]
        if past.size > 0:
            gap = (cutoff - past.max()) / one_day
            # cutoff baz alındı (obs değil) → leakage tamponu korunur; en taze
            # geçmiş hotspot bile en az buffer_days kadar eski sayılır.
            days_since[i] = float(gap) + float(cfg.leakage_buffer_days)

    out["fire_occurred_within_30d"] = label
    out["firms_density_1yr"] = density
    out["days_since_last_fire"] = np.clip(days_since, 0.0, 3650.0).astype(np.float32)

    pos_rate = float(label.mean()) if n else 0.0
    logger.info(
        "build_labels: n=%d, pozitif oran=%.4f, ort. density=%.2f",
        n, pos_rate, float(density.mean()) if n else 0.0,
    )
    return out
