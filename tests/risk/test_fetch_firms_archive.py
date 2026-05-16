# SPDX-License-Identifier: Apache-2.0
"""Sprint 6-B: fetch_firms_archive.py birim + entegrasyon testleri.

Kapsam (architect T1, T2):
  T1a) 200 + geçerli CSV → çıktı CSV var, lat/lon/acq_date/confidence sütunları mevcut
  T1b) 200 + boş body → logger.warning, boş/header-only çıktı, RuntimeError YOK
  T1c) 429 x3 → RuntimeError (SystemExit değil)
  T1d) 403 → anlamlı RuntimeError (geçersiz map_key)
  T2)  Çakışan chunk sınırları → çıktıda (latitude,longitude,acq_date) duplicate YOK

Mock stratejisi:
  - respx ile httpx.Client.get intercept (fetch_openmeteo_retry test deseni)
  - time.sleep patch (CI'da gerçek bekleme olmaz)
  - FIRMS_MAP_KEY monkeypatch ile sahte değer
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd
import pytest
import respx

# ---------------------------------------------------------------------------
# fetch_firms_archive modülünü scripts/ altından import et
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def _load_firms_module():
    """fetch_firms_archive.py'yi importlib ile yükle."""
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    spec = importlib.util.spec_from_file_location(
        "fetch_firms_archive",
        _SCRIPTS_DIR / "fetch_firms_archive.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_ffa = _load_firms_module()

# Sahte FIRMS_MAP_KEY
FAKE_KEY = "TEST_MAP_KEY_FAKE"

# FIRMS API base pattern — respx URL match için
FIRMS_API_BASE = _ffa.FIRMS_API_BASE

# Minimal geçerli FIRMS CSV içeriği (sütunlar + 2 satır)
_VALID_CSV_BODY = (
    "latitude,longitude,acq_date,confidence,brightness,frp\n"
    "39.51,32.85,2024-06-01,nominal,320.0,5.1\n"
    "39.52,32.86,2024-06-02,high,330.0,6.2\n"
)

# Boş body (sadece header)
_EMPTY_BODY = "latitude,longitude,acq_date,confidence\n"

# Gerçek boş (no header, no newline)
_BLANK_BODY = ""


# ---------------------------------------------------------------------------
# Fixture: FIRMS_MAP_KEY env monkeypatch
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_fake_map_key(monkeypatch):
    """Tüm testlerde FIRMS_MAP_KEY sahte değer olarak ayarla."""
    monkeypatch.setenv("FIRMS_MAP_KEY", FAKE_KEY)


# ---------------------------------------------------------------------------
# T1a: 200 + geçerli CSV → çıktı CSV var, zorunlu sütunlar mevcut
# ---------------------------------------------------------------------------

@respx.mock
def test_t1a_valid_csv_output(tmp_path: Path) -> None:
    """T1a: 200 + geçerli CSV yanıtı → çıktı CSV oluşur, zorunlu sütunlar var."""
    # URL pattern: FIRMS_API_BASE/{key}/{source}/{coords}/{days}/{date}
    respx.get(url__startswith=FIRMS_API_BASE).mock(
        return_value=httpx.Response(200, text=_VALID_CSV_BODY)
    )

    out_path = tmp_path / "firms_out.csv"

    with patch.object(time, "sleep", return_value=None):
        result = _ffa.fetch_firms_archive(
            bbox=(32.7, 39.4, 33.0, 39.6),
            out_path=out_path,
            source="VIIRS_SNPP_SP",
            start_date="2024-06-01",
            end_date="2024-06-07",
            chunk_days=7,
            max_retries=3,
        )

    assert result == out_path, f"Dönüş değeri out_path olmalı: {result!r}"
    assert out_path.exists(), f"Çıktı CSV oluşturulmadı: {out_path}"

    # Lisans başlığı '#' satırları olduğu için comment="#" ile oku
    df = pd.read_csv(out_path, comment="#")
    assert len(df) > 0, "CSV'de en az 1 satır beklendi"

    for col in ("latitude", "longitude", "acq_date", "confidence"):
        assert col in df.columns, (
            f"Zorunlu sütun '{col}' çıktı CSV'de eksik. "
            f"Mevcut sütunlar: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# T1b: 200 + boş body → logger.warning, boş/header-only çıktı, hata YOK
# ---------------------------------------------------------------------------

@respx.mock
def test_t1b_empty_body_no_exception(tmp_path: Path, caplog) -> None:
    """T1b: 200 + boş/header-only yanıt → RuntimeError YOK, logger.warning/info yayılır."""
    import logging

    respx.get(url__startswith=FIRMS_API_BASE).mock(
        return_value=httpx.Response(200, text=_BLANK_BODY)
    )

    out_path = tmp_path / "firms_empty.csv"

    with caplog.at_level(logging.DEBUG, logger="fetch_firms_archive"):
        with patch.object(time, "sleep", return_value=None):
            # RuntimeError fırlatılmamalı
            result = _ffa.fetch_firms_archive(
                bbox=(32.7, 39.4, 33.0, 39.6),
                out_path=out_path,
                source="VIIRS_SNPP_SP",
                start_date="2024-06-01",
                end_date="2024-06-07",
                chunk_days=7,
                max_retries=3,
            )

    assert result == out_path
    assert out_path.exists(), "Boş arşivde bile çıktı CSV oluşturulmalı (şema korunur)"

    # Çıktı CSV okunabilmeli (comment="#" ile lisans başlığı atlanır)
    df = pd.read_csv(out_path, comment="#")
    # Boş arşiv: 0 satır, zorunlu sütunlar şema olarak mevcut
    assert len(df) == 0, f"Boş arşiv → 0 satır beklendi, got {len(df)}"
    for col in ("latitude", "longitude", "acq_date", "confidence"):
        assert col in df.columns, (
            f"Boş arşiv şema: '{col}' sütunu eksik. "
            f"Mevcut: {list(df.columns)}"
        )

    # 0 hotspot uyarısı loglanmış olmalı
    messages = [r.message for r in caplog.records]
    has_empty_warn = any(
        ("boş" in m or "0 hotspot" in m or "chunk" in m)
        for m in messages
    )
    assert has_empty_warn, (
        "Boş arşiv için bilgi/uyarı mesajı beklendi. "
        f"Log mesajları: {messages}"
    )


# ---------------------------------------------------------------------------
# T1c: 429 x3 → RuntimeError (SystemExit değil)
# ---------------------------------------------------------------------------

@respx.mock
def test_t1c_429_exhaustion_raises_runtime_error(tmp_path: Path) -> None:
    """T1c: max_retries kadar 429 → RuntimeError (SystemExit değil)."""
    respx.get(url__startswith=FIRMS_API_BASE).mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "1"},
            text="rate limited",
        )
    )

    out_path = tmp_path / "firms_429.csv"

    with patch.object(time, "sleep", return_value=None):
        with pytest.raises(RuntimeError) as exc_info:
            _ffa.fetch_firms_archive(
                bbox=(32.7, 39.4, 33.0, 39.6),
                out_path=out_path,
                source="VIIRS_SNPP_SP",
                start_date="2024-06-01",
                end_date="2024-06-07",
                chunk_days=7,
                max_retries=3,
            )

    err = exc_info.value
    # SystemExit olmamalı
    assert not isinstance(err, SystemExit), (
        f"RuntimeError beklendi ama SystemExit fırlatıldı: {err!r}"
    )
    # "429" veya "başarısız"/"denemede" içermeli
    msg = str(err)
    assert "429" in msg or "başarısız" in msg or "denemede" in msg, (
        f"RuntimeError mesajı '429'/'başarısız'/'denemede' içermeli: {msg!r}"
    )
    # Güvenlik (regression): FIRMS_MAP_KEY değeri ne final mesajda ne de
    # zincirlenmiş __cause__ mesajında görünmemeli (_mask_url fix).
    cause_msg = str(err.__cause__) if err.__cause__ is not None else ""
    assert FAKE_KEY not in msg, (
        f"Güvenlik: FIRMS_MAP_KEY 429 mesajına sızdı: {msg!r}"
    )
    assert FAKE_KEY not in cause_msg, (
        f"Güvenlik: FIRMS_MAP_KEY __cause__ 429 mesajına sızdı: {cause_msg!r}"
    )
    context_msg_429 = str(err.__context__) if err.__context__ is not None else ""
    assert FAKE_KEY not in context_msg_429, (
        f"Güvenlik: FIRMS_MAP_KEY __context__ 429 mesajına sızdı: {context_msg_429!r}"
    )


# ---------------------------------------------------------------------------
# T1d: 403 → anlamlı RuntimeError (geçersiz map_key)
# ---------------------------------------------------------------------------

@respx.mock
def test_t1d_403_raises_runtime_error_with_key_message(tmp_path: Path) -> None:
    """T1d: 403 → anlamlı RuntimeError; mesajda 'FIRMS_MAP_KEY'/'403'/'Forbidden' geçmeli."""
    respx.get(url__startswith=FIRMS_API_BASE).mock(
        return_value=httpx.Response(403, text="Forbidden")
    )

    out_path = tmp_path / "firms_403.csv"

    with patch.object(time, "sleep", return_value=None):
        with pytest.raises(RuntimeError) as exc_info:
            _ffa.fetch_firms_archive(
                bbox=(32.7, 39.4, 33.0, 39.6),
                out_path=out_path,
                source="VIIRS_SNPP_SP",
                start_date="2024-06-01",
                end_date="2024-06-07",
                chunk_days=7,
                max_retries=3,
            )

    msg = str(exc_info.value)
    # "403" veya "Forbidden" veya "map_key"/"FIRMS_MAP_KEY" içermeli
    assert any(kw in msg for kw in ("403", "Forbidden", "map_key", "FIRMS_MAP_KEY")), (
        f"403 RuntimeError anlamlı mesaj içermeli (403/Forbidden/map_key): {msg!r}"
    )
    # 403 → retry yapılmaz, hızlı fail (RuntimeError doğrudan fırlatılır)
    assert not isinstance(exc_info.value, SystemExit)
    # Güvenlik (regression): FIRMS_MAP_KEY değeri exception mesajına sızmamalı
    # (_mask_url fix doğrulaması).
    assert FAKE_KEY not in msg, (
        f"Güvenlik: FIRMS_MAP_KEY exception mesajına sızdı: {msg!r}"
    )


# ---------------------------------------------------------------------------
# T2: chunk birleştirme dedup — çakışan chunk sınırları → duplicate YOK
# ---------------------------------------------------------------------------

@respx.mock
def test_t2_chunk_dedup_no_duplicates(tmp_path: Path) -> None:
    """T2: aynı (lat,lon,acq_date) farklı chunk'larda dönse bile çıktıda duplicate YOK."""
    # Her chunk'ta aynı satırı döndür → birleşimde duplicate oluşur,
    # fetch_firms_archive drop_duplicates uygulamalı.
    _DUPLICATE_CSV = (
        "latitude,longitude,acq_date,confidence\n"
        "39.51,32.85,2024-06-01,nominal\n"
        "39.52,32.86,2024-06-03,high\n"
    )

    respx.get(url__startswith=FIRMS_API_BASE).mock(
        return_value=httpx.Response(200, text=_DUPLICATE_CSV)
    )

    out_path = tmp_path / "firms_dedup.csv"

    # 14 günlük aralık, 7 günlük chunk → 2 chunk, her biri aynı 2 satırı döndürür
    # → birleştirince 4 satır → drop_duplicates sonrası 2 satır
    with patch.object(time, "sleep", return_value=None):
        _ffa.fetch_firms_archive(
            bbox=(32.7, 39.4, 33.0, 39.6),
            out_path=out_path,
            source="VIIRS_SNPP_SP",
            start_date="2024-06-01",
            end_date="2024-06-14",
            chunk_days=7,
            max_retries=3,
        )

    df = pd.read_csv(out_path, comment="#")

    # (latitude, longitude, acq_date) tuple bazında duplicate kontrolü
    dup_mask = df.duplicated(subset=["latitude", "longitude", "acq_date"], keep=False)
    n_dups = int(dup_mask.sum())
    assert n_dups == 0, (
        f"Çıktı CSV'de (latitude,longitude,acq_date) duplicate var: {n_dups} satır. "
        f"drop_duplicates uygulanmadı olabilir.\n"
        f"DataFrame:\n{df[dup_mask]}"
    )

    # Toplam satır: 2 orijinal satır (dedup sonrası)
    assert len(df) == 2, (
        f"Dedup sonrası 2 satır beklendi (2 chunk × 2 satır → drop_dup → 2), "
        f"got {len(df)}"
    )


# ---------------------------------------------------------------------------
# _date_chunks birim testi — chunk döngüsü mantığı
# ---------------------------------------------------------------------------

def test_date_chunks_covers_full_range() -> None:
    """_date_chunks: start..end aralığındaki tüm günleri kapsar, chunk sınırları taşmaz."""
    from datetime import date

    start = date(2024, 6, 1)
    end = date(2024, 6, 21)  # 21 gün
    chunks = _ffa._date_chunks(start, end, chunk_days=7)

    # 21g / 7g = 3 chunk
    assert len(chunks) == 3, f"3 chunk beklendi, got {len(chunks)}: {chunks}"

    # Toplam span = 21 gün
    total_span = sum(span for _, span in chunks)
    assert total_span == 21, f"Toplam span 21g beklendi, got {total_span}"

    # Her chunk başlangıcı monoton artan
    dates = [d for d, _ in chunks]
    assert dates == sorted(dates), f"Chunk başlangıçları sıralı değil: {dates}"


def test_date_chunks_last_chunk_clipped() -> None:
    """Son chunk end'e taşmaz — artık günler kırpılır."""
    from datetime import date

    start = date(2024, 6, 1)
    end = date(2024, 6, 10)  # 10 gün
    chunks = _ffa._date_chunks(start, end, chunk_days=7)

    # İlk chunk 7g, ikinci chunk 3g (kırpılmış)
    assert len(chunks) == 2
    assert chunks[0][1] == 7
    assert chunks[1][1] == 3, (
        f"Son chunk 3g kırpılmış olmalı, got {chunks[1][1]}"
    )


# ---------------------------------------------------------------------------
# _get_firms_key: FIRMS_MAP_KEY yoksa RuntimeError
# ---------------------------------------------------------------------------

def test_get_firms_key_missing_raises(monkeypatch) -> None:
    """FIRMS_MAP_KEY yok → RuntimeError (anlamlı mesaj)."""
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FIRMS_MAP_KEY"):
        _ffa._get_firms_key()


def test_get_firms_key_placeholder_raises(monkeypatch) -> None:
    """FIRMS_MAP_KEY='<REPLACE_ME>' → RuntimeError (placeholder koruması)."""
    monkeypatch.setenv("FIRMS_MAP_KEY", "<REPLACE_ME>")
    with pytest.raises(RuntimeError, match="FIRMS_MAP_KEY"):
        _ffa._get_firms_key()


def test_get_firms_key_valid(monkeypatch) -> None:
    """Geçerli FIRMS_MAP_KEY → değer döner, hata yok."""
    monkeypatch.setenv("FIRMS_MAP_KEY", "VALID_TEST_KEY_123")
    key = _ffa._get_firms_key()
    assert key == "VALID_TEST_KEY_123"


# ---------------------------------------------------------------------------
# T1e (tur 3): httpx.ConnectError / TransportError → logger.warning maskeli
# ---------------------------------------------------------------------------

@respx.mock
def test_t1e_http_error_log_masking(tmp_path: Path, caplog) -> None:
    """T1e (tur 3): _retry_get httpx.TransportError dalı — log'a FAKE_KEY sızmamalı.

    Senaryo:
      - respx GET'i httpx.ConnectError ile side_effect ayarla;
        exception mesajı _build_url çıktısını (FAKE_KEY içeren URL) barındırır.
      - Tüm max_retries denemelerinde aynı hata fırlatılır.
      - Beklenti 1: caplog'daki hiçbir formatlanmış mesajda FAKE_KEY geçmemeli
        (_mask_url(str(exc), map_key) doğru çalışmalı).
      - Beklenti 2: retry tükenince fırlatılan RuntimeError'ın str()'inde
        ve __cause__'unun str()'inde FAKE_KEY geçmemeli (tur 2 deseni).
    """
    import logging

    # URL içinde FAKE_KEY geçen bir ConnectError — _retry_get'in
    # httpx.HTTPError/TransportError except bloğuna düşer (satır 140-145).
    # exception mesajına bilinçli olarak FAKE_KEY ekli tam URL koyuyoruz
    # ki maskeleme gerçekten tetiklendiği kanıtlanabilsin.
    fake_url_fragment = f"{FIRMS_API_BASE}/{FAKE_KEY}/VIIRS_SNPP_SP"
    conn_error = httpx.ConnectError(
        f"Connection refused: {fake_url_fragment}/32.7,39.4,33.0,39.6/7/2024-06-01"
    )

    # respx: tüm GET istekleri ConnectError fırlatsın
    respx.get(url__startswith=FIRMS_API_BASE).mock(side_effect=conn_error)

    out_path = tmp_path / "firms_conn_err.csv"

    with caplog.at_level(logging.WARNING, logger="fetch_firms_archive"):
        with patch.object(time, "sleep", return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                _ffa.fetch_firms_archive(
                    bbox=(32.7, 39.4, 33.0, 39.6),
                    out_path=out_path,
                    source="VIIRS_SNPP_SP",
                    start_date="2024-06-01",
                    end_date="2024-06-07",
                    chunk_days=7,
                    max_retries=3,
                )

    # --- Log maskeleme kontrolü ---
    # logger %s lazy-format kullanıyor; getMessage() formatlanmış mesajı verir.
    for record in caplog.records:
        formatted = record.getMessage()
        assert FAKE_KEY not in formatted, (
            f"Guvenlik: FIRMS_MAP_KEY log kaydina sizdi!\n"
            f"  logger: {record.name}\n"
            f"  level:  {record.levelname}\n"
            f"  msg:    {formatted!r}"
        )

    # caplog.text de kontrol et (ek güvence)
    assert FAKE_KEY not in caplog.text, (
        f"Guvenlik: FIRMS_MAP_KEY caplog.text icinde bulundu:\n{caplog.text!r}"
    )

    # En az 1 WARNING logu üretilmeli (retry sayısı kadar)
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warning_records) >= 1, (
        "httpx.ConnectError dalinda en az 1 WARNING log beklendi, "
        f"got {len(warning_records)}"
    )

    # --- RuntimeError maskeleme kontrolü (tur 2 deseniyle tutarlı) ---
    err = exc_info.value
    err_str = str(err)
    cause_str = str(err.__cause__) if err.__cause__ is not None else ""

    assert FAKE_KEY not in err_str, (
        f"Guvenlik: FIRMS_MAP_KEY RuntimeError mesajina sizdi: {err_str!r}"
    )
    assert FAKE_KEY not in cause_str, (
        f"Guvenlik: FIRMS_MAP_KEY __cause__ mesajina sizdi: {cause_str!r}"
    )
    context_str = str(err.__context__) if err.__context__ is not None else ""
    assert FAKE_KEY not in context_str, (
        f"Güvenlik: FIRMS_MAP_KEY __context__ mesajına sızdı: {context_str!r}"
    )


# ---------------------------------------------------------------------------
# T3a: httpx INFO logu caplog'a sızmamalı (regression — fix öncesi kırmızı)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def _clean_root_logger_filters():
    """Root logger'a test sırasında eklenen filter'ları temizle.

    _install_secret_filter() root logger'a _SecretRedactFilter ekliyor.
    Test izolasyonu için her testten önce/sonra root filter listesi orijinal
    haline döndürülür.
    """
    import logging as _logging

    before = list(_logging.root.filters)
    yield
    # Test sonrası eklenen filter'ları kaldır
    _logging.root.filters[:] = before


@respx.mock
def test_t3a_httpx_info_log_does_not_leak_secret(
    tmp_path: Path, caplog, _clean_root_logger_filters
) -> None:
    """T3a (regression): fetch_firms_archive() sonrası caplog'da FAKE_KEY OLMAMALI.

    Senaryo:
      - respx 200 + geçerli CSV body mock (_VALID_CSV_BODY sabitini kullan).
      - caplog TÜM logger'ları DEBUG seviyede yakalar (logger adı verilmez —
        httpx dahil root yakalanır).
      - httpx'in 'HTTP Request: GET <tam-url>' INFO logu FAKE_KEY içerir;
        _SecretRedactFilter fix öncesinde bu log caplog'a sızardı.
      - Fix sonrası: _install_secret_filter root logger'a eklenir →
        httpx INFO logu maskelenir → FAKE_KEY caplog'da GÖRÜNMEZ.

    Fix öncesi neden kırmızı olurdu:
      _install_secret_filter olmadan httpx, URL'yi olduğu gibi loglar:
      "HTTP Request: GET https://firms.../TEST_MAP_KEY_FAKE/... HTTP/1.1 200 OK"
      Bu kayıt caplog.records'a girer → FAKE_KEY in caplog.text → AssertionError.
    """
    import logging

    respx.get(url__startswith=FIRMS_API_BASE).mock(
        return_value=httpx.Response(200, text=_VALID_CSV_BODY)
    )

    out_path = tmp_path / "firms_t3a.csv"

    # logger adı VERİLMEZ — root yakalanır, httpx dahil tüm alt logger'lar
    with caplog.at_level(logging.DEBUG):
        with patch.object(time, "sleep", return_value=None):
            _ffa.fetch_firms_archive(
                bbox=(32.7, 39.4, 33.0, 39.6),
                out_path=out_path,
                source="VIIRS_SNPP_SP",
                start_date="2024-06-01",
                end_date="2024-06-07",
                chunk_days=7,
                max_retries=3,
            )

    # HER log kaydı için formatlanmış mesaj FAKE_KEY içermemeli
    leaking_records = [
        r for r in caplog.records if FAKE_KEY in r.getMessage()
    ]
    assert not leaking_records, (
        f"Güvenlik regression: {len(leaking_records)} log kaydı FAKE_KEY içeriyor!\n"
        + "\n".join(
            f"  [{r.name}] {r.levelname}: {r.getMessage()!r}"
            for r in leaking_records
        )
    )

    # caplog.text bütünsel kontrolü (ek güvence)
    assert FAKE_KEY not in caplog.text, (
        f"Güvenlik regression: FAKE_KEY caplog.text içinde bulundu.\n"
        f"caplog.text (ilk 500 karakter): {caplog.text[:500]!r}"
    )

    # Sanity: httpx gerçekten bir şey logladı mı? (test anlamlı mı?)
    httpx_records = [r for r in caplog.records if r.name.startswith("httpx")]
    assert len(httpx_records) >= 1, (
        "Sanity: httpx hiç log üretmedi — test anlamlılığı sorgulanabilir. "
        f"Tüm logger adları: {sorted({r.name for r in caplog.records})}"
    )


# ---------------------------------------------------------------------------
# T3b: _SecretRedactFilter lazy %-format (record.args) maskeleme birim testi
# ---------------------------------------------------------------------------

def test_t3b_secret_redact_filter_lazy_format() -> None:
    """T3b: _SecretRedactFilter doğrudan birim testi — lazy %-format maskeleme.

    Senaryo:
      - Elle LogRecord oluştur: msg='%s %s', args=(url_with_key, '200 OK').
      - record.getMessage() lazy evaluation ile FAKE_KEY içeren string döner.
      - _SecretRedactFilter(FAKE_KEY).filter(record) çağrılır.
      - Beklenti: filter True döner; record.getMessage() artık FAKE_KEY içermez;
        '***' içerir; record.args is None.

    Fix öncesi neden kırmızı olurdu:
      _SecretRedactFilter olmadan record.getMessage() ham %-format string döner
      → FAKE_KEY kaçınılmaz → assert FAKE_KEY not in record.getMessage() çöker.

    Ek senaryo — boş secret guard:
      _SecretRedactFilter("").filter(record) → True döner, record.msg değişmez.
    """
    import logging as _logging

    # --- Senaryo 1: normal maskeleme ---
    url_with_key = f"GET https://example.com/{FAKE_KEY}/path"
    record = _logging.LogRecord(
        name="httpx",
        level=_logging.INFO,
        pathname="",
        lineno=0,
        msg="HTTP Request: %s %s",
        args=(url_with_key, "200 OK"),
        exc_info=None,
    )

    # Filtreyi uygula
    filt = _ffa._SecretRedactFilter(FAKE_KEY)
    result = filt.filter(record)

    assert result is True, "filter() her zaman True dönmeli"
    formatted = record.getMessage()
    assert FAKE_KEY not in formatted, (
        f"FAKE_KEY maskelenmedi: {formatted!r}"
    )
    assert "***" in formatted, (
        f"'***' maskeleme izi formatlanmış mesajda olmalı: {formatted!r}"
    )
    assert record.args is None, (
        f"filter() sonrası record.args=None olmalı (lazy re-format engeli), "
        f"got: {record.args!r}"
    )

    # --- Senaryo 2: boş secret — record değişmemeli ---
    record2 = _logging.LogRecord(
        name="httpx",
        level=_logging.INFO,
        pathname="",
        lineno=0,
        msg="HTTP Request: %s %s",
        args=(url_with_key, "200 OK"),
        exc_info=None,
    )
    original_msg = record2.msg
    original_args = record2.args

    filt_empty = _ffa._SecretRedactFilter("")
    result2 = filt_empty.filter(record2)

    assert result2 is True, "Boş secret ile filter() True dönmeli"
    assert record2.msg == original_msg, (
        f"Boş secret: record.msg değişmemeli. got: {record2.msg!r}"
    )
    assert record2.args == original_args, (
        f"Boş secret: record.args değişmemeli. got: {record2.args!r}"
    )
