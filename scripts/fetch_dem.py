# SPDX-License-Identifier: Apache-2.0
"""Copernicus DEM GLO-30 indirici (auth-free AWS S3 public bucket).

Kaynak: Copernicus DEM GLO-30 — AWS Open Data public bucket
  https://copernicus-dem-30m.s3.amazonaws.com/
Lisans: CC BY 4.0 © DLR e.V. 2010-2014 / © Airbus Defence and Space GmbH /
        © ESA. Atıf zorunlu (CREDITS.md).

NASA Earthdata gibi auth-wall YOK; anonim HTTPS GET ile çekilir. 403/erişim
sorununda --dem-source opentopo fallback (OpenTopography Global DEM API;
registration optional, anonim de çalışabilir ama rate-limit'lidir).

Sprint 6-A: script yazılır, GERÇEK AĞDAN ÇALIŞTIRILMAZ (tester network
mock'layacak). Çıktı GeoTIFF EPSG:4326.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import httpx

logger = logging.getLogger("fetch_dem")

DEFAULT_BBOX = "32.7,39.4,33.0,39.6"  # Beynam: lon_min,lat_min,lon_max,lat_max
COPERNICUS_S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
OPENTOPO_API = "https://portal.opentopography.org/API/globaldem"
MAX_RETRIES = 3
HTTP_TIMEOUT = 120.0


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--bbox lon_min,lat_min,lon_max,lat_max bekler, got: {s}")
    lon_min, lat_min, lon_max, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(f"Geçersiz bbox sırası: {s}")
    return lon_min, lat_min, lon_max, lat_max


def _retry_get(client: httpx.Client, url: str, **kw: object) -> httpx.Response:
    """3 deneme, exponential backoff (2^n sn). 403 → çağıran fallback'e bakar."""
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


def _copernicus_tile_names(bbox: tuple[float, float, float, float]) -> list[str]:
    """bbox'ı kapsayan 1°×1° Copernicus DEM GLO-30 tile dosya adları.

    Tile adlandırma: Copernicus_DSM_COG_10_N39_00_E032_00_DEM
      → S3 key: Copernicus_DSM_COG_10_N39_00_E032_00_DEM/<...>.tif
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    names: list[str] = []
    for lat in range(math.floor(lat_min), math.ceil(lat_max)):
        for lon in range(math.floor(lon_min), math.ceil(lon_max)):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile = (
                f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_"
                f"{ew}{abs(lon):03d}_00_DEM"
            )
            names.append(tile)
    return names


def fetch_dem(
    bbox: tuple[float, float, float, float],
    out_path: Path,
    dem_source: str = "aws_s3",
) -> Path:
    """DEM indir → bbox'a kırp → GeoTIFF EPSG:4326 yaz.

    aws_s3: Copernicus GLO-30 public bucket (anonim). 403 → opentopo fallback.
    opentopo: OpenTopography Global DEM API (COP30 dataset).
    """
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
    from rasterio.windows import from_bounds

    lon_min, lat_min, lon_max, lat_max = bbox
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        if dem_source == "aws_s3":
            tiles = _copernicus_tile_names(bbox)
            logger.info("Copernicus GLO-30 tile sayısı: %d", len(tiles))
            datasets = []
            try:
                for tile in tiles:
                    url = f"{COPERNICUS_S3_BASE}/{tile}/{tile}.tif"
                    logger.info("İndiriliyor: %s", url)
                    resp = _retry_get(client, url)
                    if resp.status_code == 403:
                        logger.warning(
                            "Copernicus S3 403 — --dem-source opentopo fallback'e geçiliyor."
                        )
                        return fetch_dem(bbox, out_path, dem_source="opentopo")
                    datasets.append(MemoryFile(resp.content).open())
                mosaic, transform = merge(
                    datasets, bounds=(lon_min, lat_min, lon_max, lat_max)
                )
                profile = datasets[0].profile
            finally:
                for ds in datasets:
                    ds.close()
            profile.update(
                driver="GTiff",
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                crs="EPSG:4326",
                count=1,
                dtype=mosaic.dtype,
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(mosaic[0], 1)

        elif dem_source == "opentopo":
            url = (
                f"{OPENTOPO_API}?demtype=COP30"
                f"&south={lat_min}&north={lat_max}"
                f"&west={lon_min}&east={lon_max}"
                f"&outputFormat=GTiff"
            )
            logger.info("OpenTopography fallback: %s", url)
            resp = _retry_get(client, url)
            with MemoryFile(resp.content) as mem, mem.open() as src:
                window = from_bounds(
                    lon_min, lat_min, lon_max, lat_max, src.transform
                )
                data = src.read(1, window=window)
                win_transform = src.window_transform(window)
                profile = src.profile
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
        else:
            raise ValueError(f"Bilinmeyen --dem-source: {dem_source}")

    logger.info("DEM yazıldı: %s (source=%s)", out_path, dem_source)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copernicus DEM GLO-30 fetch (auth-free)")
    parser.add_argument("--bbox", default=DEFAULT_BBOX,
                        help="lon_min,lat_min,lon_max,lat_max (default Beynam)")
    parser.add_argument("--out", default="data/beynam/dem_beynam.tif")
    parser.add_argument("--dem-source", choices=["aws_s3", "opentopo"],
                        default="aws_s3")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bbox = _parse_bbox(args.bbox)
    fetch_dem(bbox, Path(args.out), dem_source=args.dem_source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
