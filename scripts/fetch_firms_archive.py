# SPDX-License-Identifier: Apache-2.0
"""NASA FIRMS arşiv (Standard Processing) hotspot indirici.

Kaynak: NASA FIRMS Area API — CSV arşiv ucu
  https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/
  {W,S,E,N}/{chunk_days}/{YYYY-MM-DD}
Lisans: NASA FIRMS Open Data Policy (Public Domain). Atıf zorunlu
        (CREDITS.md). NASA Earthdata auth GEREKMEZ — mevcut FIRMS_MAP_KEY
        ile erişilebilir (Sprint 6-A varsayımı düzeltildi, bkz. Karar #8).

Rate limit: 5000 transaction / 10 dk. Chunk'lar arası time.sleep(1.0).
Chunk döngüsü: start_date..end_date aralığı chunk_days parçalara bölünür,
her chunk için tek istek; sonuçlar birleştirilir ve
(latitude, longitude, acq_date) bazında drop_duplicates uygulanır.

Sprint 6-B: gerçek FIRMS SP label kaynağı. FIRMS_MAP_KEY env'den okunur
(scripts/ altında _common.py yok → fetch_dem.py env okuma deseni: stdlib
os.environ, ek bağımlılık yok). Anahtar yoksa anlamlı RuntimeError.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

logger = logging.getLogger("fetch_firms_archive")

FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DEFAULT_BBOX = "32.7,39.4,33.0,39.6"  # Beynam: lon_min,lat_min,lon_max,lat_max
DEFAULT_SOURCE = "VIIRS_SNPP_SP"  # Standard Processing (çok-kaynak birleştirme 6-C)
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_CHUNK_DAYS = 7
MAX_RETRIES = 3
HTTP_TIMEOUT = 120.0
CHUNK_SLEEP_S = 1.0  # rate-limit (5000 tx/10dk) güvenli aralık
REQUIRED_COLUMNS = ("latitude", "longitude", "acq_date", "confidence")

LICENSE_HEADER = (
    "# Data source: NASA FIRMS Area API (Standard Processing archive)\n"
    "# https://firms.modaps.eosdis.nasa.gov\n"
    "# License: NASA FIRMS Open Data Policy (Public Domain). "
    "Attribution required (see CREDITS.md).\n"
)


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--bbox lon_min,lat_min,lon_max,lat_max bekler, got: {s}")
    lon_min, lat_min, lon_max, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(f"Geçersiz bbox sırası: {s}")
    return lon_min, lat_min, lon_max, lat_max


def _get_firms_key() -> str:
    """FIRMS_MAP_KEY env'den oku (fetch_dem.py deseni: stdlib os.environ).

    scripts/ altında _common.py yok; ek bağımlılık (python-dotenv) eklemeden
    yalnızca process env okunur. Anahtar yoksa anlamlı RuntimeError.
    """
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key or key in ("<REPLACE_ME>", "REPLACE_ME"):
        raise RuntimeError(
            "FIRMS_MAP_KEY ortam değişkeni yok/placeholder. NASA FIRMS "
            "map_key gerekli (https://firms.modaps.eosdis.nasa.gov/api/area/). "
            "PowerShell: $env:FIRMS_MAP_KEY=\"<anahtar>\" ; "
            "bash: export FIRMS_MAP_KEY=<anahtar>"
        )
    return key


def _date_chunks(
    start: date, end: date, chunk_days: int
) -> list[tuple[date, int]]:
    """[start, end] aralığını chunk_days uzunluğunda parçalara böl.

    Döndürür: (chunk_start, span_days) listesi. Son chunk end'e taşmasın
    diye span kırpılır (FIRMS day_range parametresi gerçek gün sayısı).
    """
    if chunk_days < 1:
        raise ValueError(f"--chunk-days >= 1 olmalı, got {chunk_days}")
    chunks: list[tuple[date, int]] = []
    cur = start
    while cur <= end:
        remaining = (end - cur).days + 1
        span = min(chunk_days, remaining)
        chunks.append((cur, span))
        cur = cur + timedelta(days=span)
    return chunks


def _retry_get(
    client: httpx.Client, url: str, max_retries: int = MAX_RETRIES, *, map_key: str = ""
) -> httpx.Response:
    """`max_retries` deneme, exp backoff. 429 → Retry-After varsa onu, yoksa exp.

    403 → geçersiz map_key (anlamlı RuntimeError). fetch_dem.py /
    fetch_openmeteo.py retry deseni (SystemExit değil RuntimeError).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.get(url)
            if resp.status_code == 403:
                raise RuntimeError(
                    "FIRMS 403 Forbidden — geçersiz/eksik FIRMS_MAP_KEY "
                    f"veya yetkisiz erişim: {_mask_url(url, map_key)}"
                )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = (
                    int(retry_after)
                    if (retry_after and retry_after.isdigit())
                    else 2 ** (attempt + 1)
                )
                logger.warning(
                    "429 rate-limit (deneme %d/%d) — %ds bekleniyor",
                    attempt + 1, max_retries, wait,
                )
                last_exc = RuntimeError(
                    f"FIRMS 429 rate-limit (deneme {attempt + 1}/{max_retries}): "
                    f"{_mask_url(url, map_key)}"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except RuntimeError:
            # 403 anlamlı hatası retry'a girmez — doğrudan yukarı fırlat.
            raise
        except (httpx.HTTPError, httpx.TransportError) as exc:
            last_exc = RuntimeError(f"GET hata: {_mask_url(str(exc), map_key)}")
            wait = 2 ** attempt
            logger.warning(
                "GET hata (deneme %d/%d): %s — %ds backoff",
                attempt + 1, max_retries, _mask_url(str(exc), map_key), wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"FIRMS {max_retries} denemede başarısız: {_mask_url(url, map_key)}"
    ) from last_exc


def _build_url(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    chunk_days: int,
    chunk_start: date,
) -> str:
    """FIRMS Area API CSV URL.

    bbox parametre sırası lon_min,lat_min,lon_max,lat_max (W,S,E,N).
    FIRMS area koordinat sırası: west,south,east,north.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    coords = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    return (
        f"{FIRMS_API_BASE}/{map_key}/{source}/{coords}/"
        f"{chunk_days}/{chunk_start.isoformat()}"
    )


def _mask_url(url: str, map_key: str) -> str:
    """URL'deki map_key'i '***' ile maskele (log/exception güvenliği)."""
    if map_key:
        return url.replace(map_key, "***")
    return url


def fetch_firms_archive(
    bbox: tuple[float, float, float, float],
    out_path: Path,
    source: str = DEFAULT_SOURCE,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_retries: int = MAX_RETRIES,
) -> Path:
    """FIRMS arşivinden hotspot CSV çek → birleştir → dedup → CSV yaz.

    Çıktı CSV mutlaka latitude, longitude, acq_date, confidence sütunlarını
    içerir (ek FIRMS sütunları korunur). NASA FIRMS Open Data Policy lisans
    başlığı dosyaya yazılır.
    """
    import pandas as pd

    map_key = _get_firms_key()
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError(f"start_date > end_date: {start_date} > {end_date}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = _date_chunks(start, end, chunk_days)
    logger.info(
        "FIRMS %s %s — %d chunk (chunk_days=%d) çekilecek",
        source, f"{start_date}..{end_date}", len(chunks), chunk_days,
    )

    frames: list[pd.DataFrame] = []
    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        for i, (chunk_start, span) in enumerate(chunks):
            url = _build_url(map_key, source, bbox, span, chunk_start)
            logger.info(
                "Chunk %d/%d: %s (span=%dg)",
                i + 1, len(chunks), chunk_start.isoformat(), span,
            )
            resp = _retry_get(client, url, max_retries=max_retries, map_key=map_key)
            text = resp.text.strip()
            # FIRMS boş chunk: yalnızca header satırı veya tamamen boş.
            if not text or "\n" not in text:
                logger.info(
                    "Chunk %d/%d boş döndü (%s)", i + 1, len(chunks),
                    chunk_start.isoformat(),
                )
            else:
                chunk_df = pd.read_csv(io.StringIO(text))
                if len(chunk_df):
                    frames.append(chunk_df)
                    logger.info(
                        "Chunk %d/%d: %d hotspot", i + 1, len(chunks),
                        len(chunk_df),
                    )
            # Rate-limit güvenli aralık (son chunk dahil zarar yok).
            time.sleep(CHUNK_SLEEP_S)

    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        # Boş arşiv: yine de geçerli şema ile boş CSV üret (downstream
        # build_real_dataset.py read_csv kırılmasın).
        df = pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    # Zorunlu sütun kontrolü (boş arşiv hariç — boşta zaten şema atandı).
    if len(df):
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"FIRMS CSV beklenen sütunları eksik: {missing} "
                f"(mevcut: {list(df.columns)})"
            )
        before = len(df)
        df = df.drop_duplicates(
            subset=["latitude", "longitude", "acq_date"]
        ).reset_index(drop=True)
        logger.info(
            "drop_duplicates: %d → %d satır (lat,lon,acq_date)",
            before, len(df),
        )

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    query_note = (
        f"# Query: source={source}, bbox={bbox}, "
        f"{start_date}..{end_date}, chunk_days={chunk_days}\n"
    )
    out_path.write_text(LICENSE_HEADER + query_note + buf.getvalue(), encoding="utf-8")

    logger.info(
        "FIRMS arşiv CSV yazıldı: %s (%d hotspot, source=%s)",
        out_path, len(df), source,
    )
    if len(df) == 0:
        logger.warning(
            "FIRMS arşiv 0 hotspot döndü (%s, %s..%s). bbox/source/tarih "
            "aralığını kontrol et; downstream pozitif oran 0 olacaktır.",
            source, start_date, end_date,
        )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NASA FIRMS arşiv (SP) fetch — FIRMS_MAP_KEY ile (Karar #8)"
    )
    parser.add_argument(
        "--bbox", default=DEFAULT_BBOX,
        help="lon_min,lat_min,lon_max,lat_max (default Beynam)",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    parser.add_argument(
        "--out", default="data/raw/firms/firms_sp_beynam.csv"
    )
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    bbox = _parse_bbox(args.bbox)
    fetch_firms_archive(
        bbox,
        Path(args.out),
        source=args.source,
        start_date=args.start_date,
        end_date=args.end_date,
        chunk_days=args.chunk_days,
        max_retries=args.max_retries,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
