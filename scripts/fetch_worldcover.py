# SPDX-License-Identifier: Apache-2.0
"""ESA WorldCover v200 indirici (auth-free AWS S3 public bucket).

Kaynak: ESA WorldCover 2021 v200 — AWS Open Data (registry.opendata.aws)
  s3://esa-worldcover/  →  https://esa-worldcover.s3.amazonaws.com/
  10m çözünürlük, 3°×3° COG tile, EPSG:4326.
Lisans: CC BY 4.0 © ESA WorldCover project / Contains modified Copernicus
        Sentinel data. Atıf zorunlu (CREDITS.md).

Auth-free anonim HTTPS GET. Aynı retry pattern (3x exp backoff). Beynam
bbox'a rasterio ile kırpılır. Sprint 6-A: script yazılır, GERÇEK AĞDAN
ÇALIŞTIRILMAZ (tester network mock'layacak).
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import httpx

logger = logging.getLogger("fetch_worldcover")

DEFAULT_BBOX = "32.7,39.4,33.0,39.6"  # Beynam: lon_min,lat_min,lon_max,lat_max
WORLDCOVER_S3_BASE = "https://esa-worldcover.s3.amazonaws.com"
WORLDCOVER_VERSION = "v200"  # 2021 v200
WORLDCOVER_YEAR = "2021"
MAX_RETRIES = 3
HTTP_TIMEOUT = 180.0


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--bbox lon_min,lat_min,lon_max,lat_max bekler, got: {s}")
    lon_min, lat_min, lon_max, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(f"Geçersiz bbox sırası: {s}")
    return lon_min, lat_min, lon_max, lat_max


def _retry_get(client: httpx.Client, url: str, **kw: object) -> httpx.Response:
    """3 deneme, exponential backoff (2^n sn). 403 ham döner (çağıran karar verir)."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(url, **kw)  # type: ignore[arg-type]
            if resp.status_code == 403:
                logger.warning("403 Forbidden: %s", url)
                return resp
            resp.raise_for_status()
            return resp
        except (httpx.HTTPError, httpx.TransportError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "GET hata (deneme %d/%d): %s — %ds backoff",
                attempt + 1, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"GET {MAX_RETRIES} denemede başarısız: {url}") from last_exc


def _worldcover_tile_names(bbox: tuple[float, float, float, float]) -> list[str]:
    """bbox'ı kapsayan 3°×3° WorldCover tile adları.

    Tile grid 3° adımlı; tile köşesi 3'ün katına floor edilir.
    Örn Beynam (~32.85E, 39.5N) → ESA_WorldCover_10m_2021_v200_N39E033_Map
    (tile köşe lat=floor(39/3)*3=39, lon=floor(32/3)*3=30 ... aslında
    33 değil; aşağıdaki formül 3°'lik ızgara köşesini hesaplar).
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    def _floor3(v: float) -> int:
        return int(math.floor(v / 3.0) * 3)

    names: list[str] = []
    lat = _floor3(lat_min)
    while lat <= _floor3(lat_max):
        lon = _floor3(lon_min)
        while lon <= _floor3(lon_max):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile = (
                f"ESA_WorldCover_10m_{WORLDCOVER_YEAR}_{WORLDCOVER_VERSION}_"
                f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}_Map"
            )
            names.append(tile)
            lon += 3
        lat += 3
    return names


def fetch_worldcover(
    bbox: tuple[float, float, float, float],
    out_path: Path,
) -> Path:
    """WorldCover COG indir → Beynam bbox'a kırp → GeoTIFF EPSG:4326 yaz."""
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
    from rasterio.windows import from_bounds

    lon_min, lat_min, lon_max, lat_max = bbox
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tiles = _worldcover_tile_names(bbox)
    logger.info("WorldCover tile sayısı: %d", len(tiles))

    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        datasets = []
        try:
            for tile in tiles:
                url = (
                    f"{WORLDCOVER_S3_BASE}/{WORLDCOVER_VERSION}/"
                    f"{WORLDCOVER_YEAR}/map/{tile}.tif"
                )
                logger.info("İndiriliyor: %s", url)
                resp = _retry_get(client, url)
                if resp.status_code == 403:
                    raise RuntimeError(
                        f"WorldCover S3 403 (auth-free beklenirdi): {url}"
                    )
                datasets.append(MemoryFile(resp.content).open())

            if len(datasets) == 1:
                src = datasets[0]
                window = from_bounds(
                    lon_min, lat_min, lon_max, lat_max, src.transform
                )
                data = src.read(1, window=window)
                win_transform = src.window_transform(window)
                profile = src.profile
            else:
                mosaic, win_transform = merge(
                    datasets, bounds=(lon_min, lat_min, lon_max, lat_max)
                )
                data = mosaic[0]
                profile = datasets[0].profile
        finally:
            for ds in datasets:
                ds.close()

    profile.update(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        transform=win_transform,
        crs="EPSG:4326",
        count=1,
        dtype=data.dtype,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.asarray(data), 1)

    logger.info("WorldCover yazıldı: %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ESA WorldCover v200 fetch (auth-free AWS S3)"
    )
    parser.add_argument("--bbox", default=DEFAULT_BBOX,
                        help="lon_min,lat_min,lon_max,lat_max (default Beynam)")
    parser.add_argument("--out", default="data/beynam/worldcover_beynam.tif")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bbox = _parse_bbox(args.bbox)
    fetch_worldcover(bbox, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
