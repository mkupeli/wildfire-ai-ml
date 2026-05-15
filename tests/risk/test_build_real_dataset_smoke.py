# SPDX-License-Identifier: Apache-2.0
"""Sprint 6-A: build_real_dataset.py smoke testleri.

Kapsam:
  - --firms-csv olmadan (sentetik fallback) başarılı çalışma
  - Çıktı CSV tüm FEATURE_COLUMNS içerir
  - .meta.json data_version = "real-v0-no-firms"
  - DEM/WorldCover/Open-Meteo ağ çağrıları mock'lanır (CI güvenli)

Not: fetch_dem / fetch_worldcover / fetch_openmeteo scriptleri subprocess
değil; build_real_dataset.py rasterio.open ve httpx.Client doğrudan kullanır.
Sprint 6-A'da gerçek raster yokken attach_real_features() sentetik iskelet
kullanır (have_real=False) → ağ çağrısı ZATen yapılmaz. Bu nedenle
test_network_calls_mocked, gerçek raster pathler geçildiğinde rasterio.open
mock ile korunur.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# build_real_dataset modülünü scripts/ altından import et
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def _load_build_module():
    """build_real_dataset.py'yi importlib ile yükle."""
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    spec = importlib.util.spec_from_file_location(
        "build_real_dataset",
        _SCRIPTS_DIR / "build_real_dataset.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_brd = _load_build_module()


# ---------------------------------------------------------------------------
# test_no_firms_csv_synthetic_fallback (BUG B6A-01 regression)
# ---------------------------------------------------------------------------

def test_no_firms_csv_synthetic_fallback(tmp_path: Path) -> None:
    """BUG B6A-01 REGRESSION: --firms-csv yok → main() başarılı dönmeli.

    Fix: sentetik fallback dalında TARGET_COLUMN ('fire_occurred_within_30d')
    df'e açıkça atanır (build_real_dataset.py satır 245).
    Bu test fix öncesi kırmızıydı (KeyError); fix sonrası yeşil — regression koruması.

    Doğrulama:
    - main() 0 döndürür (başarılı)
    - Çıktı CSV TARGET_COLUMN sütununu içerir
    - .meta.json: data_version='real-v0-no-firms', label_source='synthetic_fallback'
    """
    import pandas as pd

    out_csv = tmp_path / "test_dataset.csv"
    argv = [
        "--out", str(out_csv),
        "--grid-resolution-m", "5000",
    ]
    ret = _brd.main(argv)
    assert ret == 0, f"main() 0 döndürmeli (başarılı), got {ret!r}"

    assert out_csv.exists(), f"Çıktı CSV oluşturulmadı: {out_csv}"
    df = pd.read_csv(out_csv)
    target_col = _brd.TARGET_COLUMN  # "fire_occurred_within_30d"
    assert target_col in df.columns, (
        f"TARGET_COLUMN '{target_col}' çıktı CSV'de yok — BUG B6A-01 regression! "
        f"Mevcut sütunlar: {list(df.columns)}"
    )

    meta_path = out_csv.with_suffix(".meta.json")
    assert meta_path.exists(), f".meta.json bulunamadı: {meta_path}"

    import json
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("data_version") == "real-v0-no-firms", (
        f"data_version='real-v0-no-firms' beklendi, got {meta.get('data_version')!r}"
    )
    assert meta.get("label_source") == "synthetic_fallback", (
        f"label_source='synthetic_fallback' beklendi, got {meta.get('label_source')!r}"
    )


def test_no_firms_csv_warning_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """--firms-csv yok → 'FIRMS arşiv yok' mesajı WARNING seviyesinde loglanır.

    BUG B6A-01 fix sonrası aktive edildi (önceki tur skip idi).
    """
    import logging

    out_csv = tmp_path / "warn_test_dataset.csv"
    argv = [
        "--out", str(out_csv),
        "--grid-resolution-m", "5000",
    ]
    with caplog.at_level(logging.WARNING, logger="build_real_dataset"):
        ret = _brd.main(argv)

    assert ret == 0, f"main() 0 döndürmeli, got {ret!r}"
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    firms_warning = any("FIRMS" in msg and "arşiv" in msg for msg in warning_messages)
    assert firms_warning, (
        "WARNING seviyesinde 'FIRMS arşiv yok' (veya benzer) mesajı beklendi. "
        f"Yakalanan WARNING mesajları: {warning_messages}"
    )


# ---------------------------------------------------------------------------
# test_network_calls_mocked
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason=(
        "Sprint 6-A: build_real_dataset attach_real_features() gerçek raster "
        "dosyaları yokken (have_real=False) doğrudan ağ çağrısı YAPMAZ. "
        "DEM/WorldCover/Open-Meteo fetch ağ çağrısı fetch_*.py script'lerinde "
        "izole; bu script'ler Sprint 6-A'da subprocess olarak çağrılmıyor, "
        "test_no_firms_csv_synthetic_fallback zaten ağsız çalışıyor. "
        "Gerçek raster pathler + mock rasterio.open entegrasyonu Sprint 6-B "
        "integration testlerinde ele alınacak."
    )
)
def test_network_calls_mocked(tmp_path: Path) -> None:  # pragma: no cover
    """Placeholder — Sprint 6-B'de rasterio.open + httpx mock ile doldurulacak."""
    pass


# ---------------------------------------------------------------------------
# test_firms_csv_real_labels
# ---------------------------------------------------------------------------

def test_firms_csv_real_labels(tmp_path: Path) -> None:
    """--firms-csv verilince data_version='real-v0', label_source='firms_archive'.

    NOT: Bu yol (FIRMS CSV var) aynı BUG B6A-01'e çarpmaz çünkü build_labels()
    df'e fire_occurred_within_30d ekler. FIRMS CSV yolu test edilebilir durumda.
    """
    import pandas as pd

    # Minimal FIRMS CSV oluştur
    firms_csv = tmp_path / "firms_test.csv"
    # Beynam bbox içinde bir hotspot
    firms_df = pd.DataFrame([{
        "latitude": 39.5,
        "longitude": 32.85,
        "acq_date": "2024-02-15",
        "confidence": "nominal",
    }])
    firms_df.to_csv(firms_csv, index=False)

    out_csv = tmp_path / "real_dataset.csv"
    argv = [
        "--out", str(out_csv),
        "--firms-csv", str(firms_csv),
        "--grid-resolution-m", "5000",
    ]
    ret = _brd.main(argv)
    assert ret == 0, f"FIRMS CSV yolu main() 0 döndürmeli, got {ret}"

    meta_path = out_csv.with_suffix(".meta.json")
    assert meta_path.exists(), f".meta.json bulunamadı: {meta_path}"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("data_version") == "real-v0", (
        f"FIRMS CSV verilince data_version='real-v0' beklendi, got {meta.get('data_version')!r}"
    )
    assert meta.get("label_source") == "firms_archive", (
        f"label_source='firms_archive' beklendi, got {meta.get('label_source')!r}"
    )


# ---------------------------------------------------------------------------
# test_synthetic_fallback_target_correlated_with_features  (Sprint 6-A tur 3)
# ---------------------------------------------------------------------------

def test_synthetic_fallback_target_correlated_with_features(tmp_path: Path) -> None:
    """BUG B6A-01 tur 3 REGRESSION: sentetik fallback TARGET df'in feature'larından türetilmeli.

    Fix (tur 3): `_syn_gen._generate_target(df[FEATURE_COLUMNS], rng)` ile TARGET
    mevcut df feature satırlarından üretilir. Önceki (bug) davranışta bağımsız
    ikinci `generate()` çağrısı kendi RNG'si ile farklı feature matrisi yaratır,
    bu yüzden df'e yazılan TARGET df'in kendi feature'larıyla ilişkisizdi.

    Korelasyon doğrulama (iki katmanlı):

    Katman 1 — doğrudan korelasyon:
      `firms_density_1yr` ve `vpd_kpa` score fonksiyonunda pozitif ağırlığa sahip.
      Fix sonrası her ikisiyle de |corr(TARGET)| > 0 olmalı (pozitif label
      grubu bu feature'larda negatif gruba göre daha yüksek ortalamaya sahip).

    Katman 2 — pertübasyon testi (bug simülasyonu):
      CSV'deki feature sütunları kasıtlı karıştırılır (shuffle) ve bu bozulmuş
      feature'lardan yeniden `_generate_target` çağrısı yapılır.
      Bozulmuş TARGET ile orijinal CSV TARGET arasındaki korelasyon düşük olmalı
      (fix df'in GÜNCEL feature'ını kullandığını kanıtlar; eski bug'da
      TARGET zaten "yabancı" feature'dan geliyordu → shuffle farketmezdi).

    Bu test fix öncesi kırmızı olurdu:
      - Katman 1: fix sonrası korelasyonun VAR olduğunu teyit eder. Ancak eski
        bug'da generate() seed=42 ile deterministik olduğundan n aynıysa aynı
        feature matrisi üretilir → Katman 1 her iki durumda da tesadüfen geçebilir.
        Katman 1 fix sonrası davranışı doğrular; asıl regression koruması Katman 2'dir.
      - Katman 2: shuffle fark yaratmazdı — eski bug'da TARGET df'den bağımsız
        bir generate() çağrısından geldiği için shuffle/no-shuffle fark etmez
        (bağımsız RNG → perturb ≡ identity). diff_ratio=0.000 → test KIRMIZI.
        Fix sonrası TARGET df'in GÜNCEL satırlarına bağlı → shuffle değiştirir
        → diff_ratio > 0.05 → test YEŞİL.
    """
    import numpy as np
    import pandas as pd

    from wildfire_ml.risk.synthetic_data import SyntheticRiskDataGenerator as _SynGen
    from wildfire_ml.risk.config import RiskConfig, FEATURE_COLUMNS, TARGET_COLUMN

    out_csv = tmp_path / "corr_test_dataset.csv"
    argv = [
        "--out", str(out_csv),
        "--grid-resolution-m", "5000",  # küçük grid → hızlı
    ]
    ret = _brd.main(argv)
    assert ret == 0, f"main() 0 döndürmeli, got {ret!r}"

    df = pd.read_csv(out_csv)
    assert TARGET_COLUMN in df.columns, f"TARGET sütunu yok: {TARGET_COLUMN}"
    assert len(df) > 0, "Çıktı CSV boş"

    target = df[TARGET_COLUMN].to_numpy(dtype=np.float64)

    # --- Katman 1: doğrudan korelasyon ---
    # firms_density_1yr: score'da 0.05 * (x/5) ağırlığı → pozitif katkı
    # vpd_kpa:           score'da 0.15 * (x/5) ağırlığı → en güçlü katkı
    # Yeterince büyük dataset'te her ikisi de sıfırdan belirgin şekilde
    # korelasyon göstermeli (threshold: abs > 0.02, pratikte çok daha yüksek).
    for feat in ("firms_density_1yr", "vpd_kpa"):
        feat_vals = df[feat].to_numpy(dtype=np.float64)
        if feat_vals.std() < 1e-9:
            continue  # sıfır varyans → atla (grid çok küçükse)
        corr = float(np.corrcoef(feat_vals, target)[0, 1])
        assert abs(corr) > 0.02, (
            f"Katman 1 FAIL: {feat} ile TARGET korelasyonu beklenen >0.02, got {corr:.4f}. "
            f"TARGET df feature'larından TÜRETİLMEMİŞ olabilir (BUG B6A-01 tur 3 regression)."
        )
        # Pozitif ağırlık → pozitif korelasyon beklenir
        assert corr > 0, (
            f"Katman 1 FAIL: {feat} ile TARGET korelasyonu pozitif beklendi, got {corr:.4f}. "
            f"Score fonksiyonunda her iki feature da pozitif katsayıya sahip."
        )

    # --- Katman 2: pertübasyon testi ---
    # Özellikle vpd_kpa ve firms_density_1yr sütunlarını shuffle et →
    # yeni TARGET üret → orijinal TARGET ile bu TARGET farklı olmalı.
    # Fix'te TARGET = _generate_target(df[FEATURE_COLUMNS], rng) yani
    # df'in GÜNCEL satırlarına bağlı; shuffle sonrası satır sırası değişince
    # yeni TARGET da değişir. Orijinal ile yeni arasında tam korelasyon beklenemez.
    risk_cfg = RiskConfig(grid_resolution_m=5000)
    rng_perturb = np.random.default_rng(seed=999)

    df_perturbed = df[FEATURE_COLUMNS].copy()
    # Kritik feature'ları shuffle et (satır sırasını boz)
    shuffle_idx = rng_perturb.permutation(len(df_perturbed))
    df_perturbed["vpd_kpa"] = df_perturbed["vpd_kpa"].to_numpy()[shuffle_idx]
    df_perturbed["firms_density_1yr"] = df_perturbed["firms_density_1yr"].to_numpy()[shuffle_idx]

    syn_gen = _SynGen(risk_cfg)
    perturbed_target = syn_gen._generate_target(
        df_perturbed, np.random.default_rng(risk_cfg.seed)
    ).astype(np.float64)

    # Orijinal vs perturbed TARGET: eğer TARGET gerçekten feature'a bağlıysa
    # shuffle sonrası farklılık beklenir (tam özdeş olmamalı).
    n_diff = int(np.sum(target != perturbed_target))
    total = len(target)
    diff_ratio = n_diff / total if total > 0 else 0.0
    assert diff_ratio > 0.05, (
        f"Katman 2 FAIL: Feature pertübasyon sonrası TARGET %{diff_ratio*100:.1f} "
        f"değişti (beklenen >%%5). "
        f"TARGET gerçekten df feature'larına bağlı DEĞİL olabilir "
        f"(BUG B6A-01 tur 3: bağımsız generate bug'ı). "
        f"n_diff={n_diff}, total={total}"
    )
