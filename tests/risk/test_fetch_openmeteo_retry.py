# SPDX-License-Identifier: Apache-2.0
"""Sprint 6-A/Tur 2: fetch_openmeteo.py 429 retry exhaustion testi (#4 NICE).

Kapsam:
  - _retry_get: 3 ardışık 429 yanıtı → MAX_RETRIES tüketilince RuntimeError raise.
  - RuntimeError.__cause__ not None (from last_exc zinciri) — fix #4 doğrulaması.
  - time.sleep mock'lanır (CI'da gerçek bekleme olmaz).

Mock stratejisi:
  - respx: httpx.Client.get çağrılarını intercept et, 3x 429 döndür.
  - unittest.mock.patch: time.sleep patched (test süresini kısaltır).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

# ---------------------------------------------------------------------------
# fetch_openmeteo modülünü scripts/ altından import et
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def _load_fetch_module():
    """fetch_openmeteo.py'yi importlib ile yükle."""
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    spec = importlib.util.spec_from_file_location(
        "fetch_openmeteo",
        _SCRIPTS_DIR / "fetch_openmeteo.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_fom = _load_fetch_module()

# ---------------------------------------------------------------------------
# 429 retry exhaustion testi
# ---------------------------------------------------------------------------

@respx.mock
def test_retry_exhaustion_429_raises_runtime_error() -> None:
    """3x 429 yanıtı → MAX_RETRIES tükenince RuntimeError raise edilmeli.

    Fix #4: last_exc ataması (satır 50) düzeltildi → __cause__ not None garantisi.
    """
    # MAX_RETRIES kez 429 döndür
    respx.get(_fom.ARCHIVE_API).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "1"}, text="rate limited")
    )

    params = {
        "latitude": str(_fom.BEYNAM_LAT),
        "longitude": str(_fom.BEYNAM_LON),
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "hourly": _fom.HOURLY_PARAMS,
        "timezone": "UTC",
    }

    with patch.object(time, "sleep", return_value=None):
        with httpx.Client(timeout=_fom.HTTP_TIMEOUT) as client:
            with pytest.raises(RuntimeError) as exc_info:
                _fom._retry_get(client, _fom.ARCHIVE_API, params)

    err = exc_info.value
    assert "başarısız" in str(err) or "denemede" in str(err), (
        f"RuntimeError mesajı beklenen 'başarısız'/'denemede' içermeli: {err!r}"
    )
    # Fix #4 doğrulaması: __cause__ not None (from last_exc zinciri)
    assert err.__cause__ is not None, (
        "RuntimeError.__cause__ None — fix #4 (last_exc ataması) doğru çalışmıyor! "
        "raise ... from last_exc zinciri eksik veya last_exc None kalıyor."
    )


@respx.mock
def test_retry_exhaustion_429_cause_is_runtime_error() -> None:
    """__cause__ RuntimeError tipinde olmalı (429 özel durumu için)."""
    respx.get(_fom.ARCHIVE_API).mock(
        return_value=httpx.Response(429, text="rate limited")
    )

    params = {
        "latitude": str(_fom.BEYNAM_LAT),
        "longitude": str(_fom.BEYNAM_LON),
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "hourly": _fom.HOURLY_PARAMS,
        "timezone": "UTC",
    }

    with patch.object(time, "sleep", return_value=None):
        with httpx.Client(timeout=_fom.HTTP_TIMEOUT) as client:
            with pytest.raises(RuntimeError) as exc_info:
                _fom._retry_get(client, _fom.ARCHIVE_API, params)

    cause = exc_info.value.__cause__
    assert isinstance(cause, RuntimeError), (
        f"__cause__ RuntimeError beklendi, got {type(cause)!r}: {cause}"
    )
    assert "429" in str(cause), (
        f"__cause__ mesajında '429' beklendi: {cause!r}"
    )


@respx.mock
def test_retry_success_after_transient_429() -> None:
    """İlk 2 deneme 429, 3. deneme 200 → başarılı dönüş (no exception)."""
    import json as _json

    ok_payload = {
        "hourly": {
            "time": ["2024-01-01T00:00"],
            "temperature_2m": [5.0],
            "relative_humidity_2m": [60.0],
            "windspeed_10m": [3.0],
            "winddirection_10m": [180.0],
        }
    }

    call_count = 0

    def side_effect(request, *_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(429, headers={"Retry-After": "1"}, text="rate limited")
        return httpx.Response(200, json=ok_payload)

    respx.get(_fom.ARCHIVE_API).mock(side_effect=side_effect)

    params = {
        "latitude": str(_fom.BEYNAM_LAT),
        "longitude": str(_fom.BEYNAM_LON),
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "hourly": _fom.HOURLY_PARAMS,
        "timezone": "UTC",
    }

    with patch.object(time, "sleep", return_value=None):
        with httpx.Client(timeout=_fom.HTTP_TIMEOUT) as client:
            resp = _fom._retry_get(client, _fom.ARCHIVE_API, params)

    assert resp.status_code == 200, (
        f"3. denemede 200 beklendi, got {resp.status_code}"
    )
    assert call_count == 3, f"Tam 3 deneme beklendi, got {call_count}"
