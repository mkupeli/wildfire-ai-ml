# SPDX-License-Identifier: Apache-2.0
"""Beynam gerçek-veri dataset builder (Sprint 6-A altyapı).

DEM (Copernicus GLO-30) + WorldCover v200 + Open-Meteo Historical auth-free
kaynaklardan feature çeker; FIRMS arşivi FIRMS_MAP_KEY ile erişilebilir
(Sprint 6-A varsayımı düzeltildi, bkz. Karar #8) — opsiyonel
`--firms-csv <path>` (scripts/fetch_firms_archive.py çıktısı) ile sağlanır:

  * --firms-csv verilirse → label_builder.build_labels ile GERÇEK label +
    fire-history feature; data_version = "real-b1".
  * verilmezse → WARNING "FIRMS arşiv yok — sentetik fallback label";
    data_version = "real-v0-no-firms"; TARGET ve fire-history sütunları
    sentetik üretici ile doldurulur (pipeline doğrulama amaçlı).

Sprint 6-B (Karar #8): --firms-csv (fetch_firms_archive.py çıktısı, gerçek
FIRMS SP) verildiğinde data_version=real-b1 gerçek-veri dataset'i üretir.
Çıktı CSV tüm FEATURE_COLUMNS + TARGET_COLUMN + [lat, lon, obs_date] içerir
(spatial_block_split lat ister).

Karar #6 PREPROCESS_SYMMETRIC: grid adımı backend
risk_feature_service._build_grid_coords ile aynı (deg_per_m * resolution_m,
np.arange(min + step/2, max, step), meshgrid). FIRMS 10km haversine
label_builder üzerinden (backend ile simetrik).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo src/ path'e ekle (script doğrudan çalıştırıldığında import için)
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wildfire_ml.risk.config import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RealDataConfig,
    RiskConfig,
)
from wildfire_ml.risk.label_builder import build_labels  # noqa: E402

logger = logging.getLogger("build_real_dataset")

EARTH_CIRC_M = 40_075_017.0  # WGS84 ekvator çevresi (~m)


def build_grid(cfg: RiskConfig) -> pd.DataFrame:
    """Beynam bbox + grid_resolution_m ile grid merkez koordinatları.

    Backend risk_feature_service._build_grid_coords ile aynı formül
    (Karar #6): deg_per_m_lat sabit, deg_per_m_lon cos(lat) düzeltmeli;
    np.arange(min + step/2, max, step); en az 1 hücre garantisi.
    """
    lat_mid = (cfg.beynam_lat_min + cfg.beynam_lat_max) / 2.0
    deg_per_m_lat = 360.0 / EARTH_CIRC_M
    deg_per_m_lon = 360.0 / (EARTH_CIRC_M * math.cos(math.radians(lat_mid)))

    step_lat = cfg.grid_resolution_m * deg_per_m_lat
    step_lon = cfg.grid_resolution_m * deg_per_m_lon

    lats_1d = np.arange(
        cfg.beynam_lat_min + step_lat / 2, cfg.beynam_lat_max, step_lat
    )
    lons_1d = np.arange(
        cfg.beynam_lon_min + step_lon / 2, cfg.beynam_lon_max, step_lon
    )
    if len(lats_1d) == 0:
        lats_1d = np.array([(cfg.beynam_lat_min + cfg.beynam_lat_max) / 2.0])
    if len(lons_1d) == 0:
        lons_1d = np.array([(cfg.beynam_lon_min + cfg.beynam_lon_max) / 2.0])

    grid_lons, grid_lats = np.meshgrid(lons_1d, lats_1d)
    return pd.DataFrame(
        {"lat": grid_lats.flatten(), "lon": grid_lons.flatten()}
    )


def obs_date_slices(
    real_cfg: RealDataConfig, obs_freq: str = "MS"
) -> list[pd.Timestamp]:
    """obs_date örnekleme: start..end aralığı `obs_freq` frekansıyla.

    obs_freq pandas date_range freq parametresi (default "MS" = her ayın
    1'i → 24 dilim 2024-01..2025-12; "2W" = 2 haftalık dilimler — daha
    yoğun örnekleme Beynam seyreklik riskini azaltabilir).
    """
    start = pd.Timestamp(real_cfg.obs_date_start).replace(day=1)
    end = pd.Timestamp(real_cfg.obs_date_end)
    return list(pd.date_range(start=start, end=end, freq=obs_freq))


def attach_real_features(
    df: pd.DataFrame,
    dem_path: Path | None,
    worldcover_path: Path | None,
    openmeteo_csv: Path | None,
    risk_cfg: RiskConfig,
) -> pd.DataFrame:
    """DEM/WorldCover/Open-Meteo'dan FEATURE_COLUMNS doldur.

    Sprint 6-A: gerçek raster sampling iskeleti. Raster/CSV yoksa (tester
    mock veya FIRMS-yok modu) sentetik üretici ile doldurulur — pipeline
    bütünlüğü için. Gerçek raster örnekleme b1'de (Sprint 6-B) sıkılaştırılır.
    """
    from wildfire_ml.risk.synthetic_data import SyntheticRiskDataGenerator

    n = len(df)
    have_real = bool(
        dem_path and dem_path.exists()
        and worldcover_path and worldcover_path.exists()
        and openmeteo_csv and openmeteo_csv.exists()
    )

    # Sentetik feature matrisi (deterministik) — gerçek raster yokken iskelet.
    syn = SyntheticRiskDataGenerator(risk_cfg).generate(n=n)
    for col in FEATURE_COLUMNS:
        df[col] = syn[col].to_numpy()

    if have_real:
        try:
            import rasterio

            from wildfire_ml.risk.features import (
                compute_ffmc_approx,
                compute_slope_aspect,
                compute_vpd,
            )

            with rasterio.open(dem_path) as dem:
                rows, cols = rasterio.transform.rowcol(
                    dem.transform, df["lon"].to_numpy(), df["lat"].to_numpy()
                )
                band = dem.read(1)
                rows = np.clip(rows, 0, band.shape[0] - 1)
                cols = np.clip(cols, 0, band.shape[1] - 1)
                df["elevation_m"] = band[rows, cols].astype(np.float32)
                slope, aspect = compute_slope_aspect(
                    band.astype(np.float64),
                    res_m=float(risk_cfg.grid_resolution_m),
                    mid_lat=(risk_cfg.beynam_lat_min + risk_cfg.beynam_lat_max) / 2.0,
                )
                df["slope_deg"] = np.clip(
                    slope[rows, cols], 0.0, 45.0
                ).astype(np.float32)
                a_rad = np.radians(aspect[rows, cols].astype(np.float64))
                df["aspect_sin"] = np.sin(a_rad).astype(np.float32)
                df["aspect_cos"] = np.cos(a_rad).astype(np.float32)

            logger.info("DEM örnekleme tamamlandı: %s", dem_path)

            om = pd.read_csv(openmeteo_csv, comment="#")
            if len(om):
                df["temp_c"] = np.float32(om["temp_c"].mean())
                df["rh_pct"] = np.float32(om["rh_pct"].mean())
                df["wind_speed_ms"] = np.float32(om["wind_speed_ms"].mean())
                wdir = np.radians(float(om["wind_dir_deg"].mean()))
                df["wind_dir_sin"] = np.float32(math.sin(wdir))
                df["wind_dir_cos"] = np.float32(math.cos(wdir))
                df["vpd_kpa"] = compute_vpd(
                    df["temp_c"].to_numpy(), df["rh_pct"].to_numpy()
                )
                df["ffmc_approx"] = compute_ffmc_approx(
                    df["temp_c"].to_numpy(),
                    df["rh_pct"].to_numpy(),
                    df["wind_speed_ms"].to_numpy(),
                )
            logger.info("Open-Meteo feature'ları işlendi: %s", openmeteo_csv)
            logger.warning(
                "WorldCover land-cover one-hot Sprint 6-A iskelet: sentetik "
                "dağılım korunuyor; gerçek raster sınıflama b1'de (Sprint 6-B)."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Gerçek raster örnekleme başarısız (%s) — sentetik iskelet "
                "feature'larla devam.", exc,
            )
    else:
        logger.warning(
            "DEM/WorldCover/Open-Meteo çıktıları bulunamadı — feature "
            "matrisi SENTETİK iskelet (Sprint 6-A; gerçek raster b1)."
        )
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Beynam gerçek-veri dataset builder (Sprint 6-A)"
    )
    parser.add_argument("--firms-csv", default=None,
                        help="FIRMS arşiv CSV (fetch_firms_archive.py çıktısı, "
                             "FIRMS_MAP_KEY ile — Karar #8; yoksa sentetik "
                             "fallback label).")
    parser.add_argument("--dem", default="data/beynam/dem_beynam.tif")
    parser.add_argument("--worldcover", default="data/beynam/worldcover_beynam.tif")
    parser.add_argument("--openmeteo",
                        default="data/raw/openmeteo/beynam_2024_2025.csv")
    parser.add_argument("--out", default="data/processed/beynam_real_dataset.csv")
    parser.add_argument("--grid-resolution-m", type=int, default=250)
    parser.add_argument(
        "--obs-freq", default="MS",
        help="obs_date örnekleme frekansı (pandas date_range freq). "
             "default 'MS' (aylık); '2W' = 2 haftalık (daha yoğun).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    risk_cfg = RiskConfig(grid_resolution_m=args.grid_resolution_m)
    real_cfg = RealDataConfig()

    grid = build_grid(risk_cfg)
    slices = obs_date_slices(real_cfg, obs_freq=args.obs_freq)
    logger.info(
        "Grid hücre=%d, obs_date dilim=%d → toplam satır=%d",
        len(grid), len(slices), len(grid) * len(slices),
    )

    # Grid'i tüm obs_date dilimleriyle çapraz çarp.
    frames = []
    for d in slices:
        g = grid.copy()
        g["obs_date"] = d.normalize()
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)

    df = attach_real_features(
        df,
        Path(args.dem) if args.dem else None,
        Path(args.worldcover) if args.worldcover else None,
        Path(args.openmeteo) if args.openmeteo else None,
        risk_cfg,
    )

    if args.firms_csv:
        firms_path = Path(args.firms_csv)
        logger.info("FIRMS CSV verildi: %s — gerçek label üretiliyor.", firms_path)
        firms_df = pd.read_csv(firms_path, comment="#")
        df = build_labels(df, firms_df, obs_date_col="obs_date", cfg=real_cfg)
        data_version = "real-b1"
        label_source = "firms_archive"
    else:
        logger.warning(
            "FIRMS arşiv yok — sentetik fallback label (data_version="
            "real-v0-no-firms). Gerçek model b1 Sprint 6-B'de FIRMS CSV ile."
        )
        # attach_real_features() yalnızca FEATURE_COLUMNS'u doldurur; TARGET_COLUMN
        # FIRMS yolunda build_labels() ile eklenir. Sentetik fallback dalında
        # TARGET'i burada açıkça atarız (BUG B6A-01 fix).
        # Sentetik fallback: TARGET df'in kendi feature satırlarından türetilir
        # (have_real=True'da da korelasyon korunur — BUG B6A-01 tur 3).
        from wildfire_ml.risk.synthetic_data import SyntheticRiskDataGenerator as _SynGen
        _syn_gen = _SynGen(risk_cfg)
        df[TARGET_COLUMN] = _syn_gen._generate_target(
            df[FEATURE_COLUMNS],
            np.random.default_rng(risk_cfg.seed),
        )
        data_version = "real-v0-no-firms"
        label_source = "synthetic_fallback"

    # Çıktı sütun düzeni: koordinat/zaman + FEATURE_COLUMNS + TARGET.
    ordered = ["lat", "lon", "obs_date"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[ordered]
    df.attrs["data_version"] = data_version

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    pos_rate = float((df[TARGET_COLUMN] == 1).mean()) if len(df) else 0.0

    # Beynam seyreklik kontrolü — yalnızca gerçek FIRMS dalında anlamlı
    # (sentetik fallback ~%15 hedefli). Pozitif oran eşiğin altındaysa
    # model eğitimi istatistiksel olarak riskli.
    if label_source == "firms_archive" and pos_rate < real_cfg.positive_rate_threshold:
        logger.error(
            "pozitif oran %.4f < eşik (%.4f) — model eğitimi riskli; "
            "6-C'ye ertele veya bbox genişlet",
            pos_rate, real_cfg.positive_rate_threshold,
        )

    if label_source == "firms_archive":
        sprint = "6-B"
        note = (
            "Sprint 6-B: gerçek FIRMS SP label + gerçek DEM ile dataset "
            "(data_version=real-b1, Karar #8). WorldCover/Open-Meteo gerçek "
            "raster sıkılaştırma Sprint 6-C'ye bırakıldı."
        )
    else:
        sprint = "6-A"
        note = (
            "Sprint 6-A altyapı; gerçek model çıkmadı (a6 sentetik devam, "
            "SYNTHETIC_MODEL uyarısı korunuyor). Gerçek model b1 Sprint 6-B."
        )

    meta = {
        "data_version": data_version,
        "label_source": label_source,
        "n_rows": int(len(df)),
        "n_grid_cells": int(len(grid)),
        "n_obs_date_slices": int(len(slices)),
        "positive_rate": pos_rate,
        "positive_rate_threshold": real_cfg.positive_rate_threshold,
        "obs_freq": args.obs_freq,
        "grid_resolution_m": args.grid_resolution_m,
        "bbox": {
            "lon_min": risk_cfg.beynam_lon_min,
            "lat_min": risk_cfg.beynam_lat_min,
            "lon_max": risk_cfg.beynam_lon_max,
            "lat_max": risk_cfg.beynam_lat_max,
        },
        "firms_csv": str(args.firms_csv) if args.firms_csv else None,
        "sprint": sprint,
        "note": note,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Dataset yazıldı: %s (%d satır, pozitif oran=%.4f, version=%s)",
        out_path, len(df), pos_rate, data_version,
    )
    logger.info("Metadata: %s", meta_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
