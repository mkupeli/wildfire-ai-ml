# SPDX-License-Identifier: Apache-2.0
"""Open-Meteo Historical (ERA5 reanalysis) saatlik meteoroloji indirici.

Kaynak: Open-Meteo Historical Weather API
  https://archive-api.open-meteo.com/v1/archive
Lisans: CC BY 4.0 — Open-Meteo (https://open-meteo.com). Veri ERA5
        reanalizine dayanır (Copernicus Climate Change Service). Atıf
        zorunlu (CREDITS.md). Auth-free; API anahtarı gerekmez.

Beynam centroid (lat=39.5, lon=32.85). Tek HTTPS istekte tüm tarih aralığı
+ tüm parametreler (saatlik). Retry 3x exp backoff; 429 rate-limit handle.
Sprint 6-A: script yazılır, GERÇEK AĞDAN ÇALIŞTIRILMAZ (tester mock'layacak).
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

import httpx

logger = logging.getLogger("fetch_openmeteo")

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
BEYNAM_LAT = 39.5
BEYNAM_LON = 32.85
HOURLY_PARAMS = (
    "temperature_2m,relative_humidity_2m,windspeed_10m,winddirection_10m"
)
MAX_RETRIES = 3
HTTP_TIMEOUT = 120.0


def _retry_get(client: httpx.Client, url: str, params: dict[str, str]) -> httpx.Response:
    """3 deneme, exp backoff. 429 (rate-limit) → Retry-After / exp backoff."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(url, params=params)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = int(retry_after) if (retry_after and retry_after.isdigit()) else 2 ** (attempt + 1)
                logger.warning(
                    "429 rate-limit (deneme %d/%d) — %ds bekleniyor",
                    attempt + 1, MAX_RETRIES, wait,
                )
                last_exc = RuntimeError(
                    f"Open-Meteo 429 rate-limit (deneme {attempt + 1}/{MAX_RETRIES}): {url}"
                )
                time.sleep(wait)
                continue
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
    raise RuntimeError(
        f"Open-Meteo {MAX_RETRIES} denemede başarısız: {url}"
    ) from last_exc


def fetch_openmeteo(
    start: str,
    end: str,
    out_path: Path,
    lat: float = BEYNAM_LAT,
    lon: float = BEYNAM_LON,
) -> Path:
    """Open-Meteo arşivinden saatlik meteoroloji çek → CSV yaz.

    Tek istek: start..end tüm aralık + 4 parametre (saatlik). CSV başına
    CC BY 4.0 lisans yorumu yazılır.
    """
    import pandas as pd

    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "start_date": start,
        "end_date": end,
        "hourly": HOURLY_PARAMS,
        "timezone": "UTC",
    }

    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        resp = _retry_get(client, ARCHIVE_API, params)
        payload = resp.json()

    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise RuntimeError(
            f"Open-Meteo yanıtında 'hourly' verisi yok: keys={list(payload)}"
        )

    df = pd.DataFrame(hourly)
    df = df.rename(
        columns={
            "time": "time_utc",
            "temperature_2m": "temp_c",
            "relative_humidity_2m": "rh_pct",
            "windspeed_10m": "wind_speed_ms",
            "winddirection_10m": "wind_dir_deg",
        }
    )

    license_header = (
        "# Data source: Open-Meteo Historical Weather API "
        "(https://open-meteo.com)\n"
        "# License: CC BY 4.0 Open-Meteo. Underlying: ERA5 reanalysis "
        "(Copernicus C3S). Attribution required (see CREDITS.md).\n"
        f"# Query: lat={lat}, lon={lon}, {start}..{end}, hourly={HOURLY_PARAMS}\n"
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    out_path.write_text(license_header + buf.getvalue(), encoding="utf-8")

    logger.info("Open-Meteo CSV yazıldı: %s (%d satır)", out_path, len(df))
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Open-Meteo Historical fetch (auth-free, CC BY 4.0)"
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--lat", type=float, default=BEYNAM_LAT)
    parser.add_argument("--lon", type=float, default=BEYNAM_LON)
    parser.add_argument(
        "--out", default="data/raw/openmeteo/beynam_2024_2025.csv"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_openmeteo(args.start, args.end, Path(args.out), lat=args.lat, lon=args.lon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
