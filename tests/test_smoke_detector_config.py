"""Sprint 2 tur 2 — train script konfig regresyon emniyet kemeri.

Architect kararlari:
- epochs default 20 (tur 1: 5 -> yeterli ogrenme yok)
- freeze_epochs default 5 (tur 1: 3 -> head warm-up kisaydi)
- BCEWithLogitsLoss pos_weight = n_neg/n_pos (tur 1'de yoktu)
- Best checkpoint kriteri ROC-AUC (tur 1: F1 -> "her seye smoke" capkurusu)

Bu testler config drift'i erken yakalar; degisiklik kasitliysa testi de guncelle.
"""
from __future__ import annotations


def test_train_config_defaults_match_architect_decision():
    """TrainConfig dataclass default'lari architect kararina uymali."""
    from scripts.train_smoke_detector import TrainConfig

    cfg = TrainConfig()
    assert cfg.epochs == 20, f"epochs default 20 olmali (tur 2 karari), got {cfg.epochs}"
    assert cfg.freeze_epochs == 5, (
        f"freeze_epochs default 5 olmali (tur 2 karari), got {cfg.freeze_epochs}"
    )
    assert cfg.checkpoint_every == 5, (
        f"checkpoint_every default 5 olmali (Kaggle 12 sa session reservi), "
        f"got {cfg.checkpoint_every}"
    )
    # pos_weight None default -> runtime'da train set'inden hesaplanir
    assert cfg.pos_weight is None, (
        "pos_weight default None olmali (compute_pos_weight runtime'da set eder)"
    )


def test_compute_pos_weight_balanced():
    """Dengeli set: pos_weight ~ 1.0."""
    from scripts.train_smoke_detector import compute_pos_weight

    samples = [("x", 0)] * 50 + [("y", 1)] * 50
    pw, n_pos, n_neg = compute_pos_weight(samples)
    assert n_pos == 50 and n_neg == 50
    assert abs(pw - 1.0) < 1e-6


def test_compute_pos_weight_imbalanced():
    """Negatif agirlikli set (n_neg=90, n_pos=10) -> pw=9.0."""
    from scripts.train_smoke_detector import compute_pos_weight

    samples = [("x", 0)] * 90 + [("y", 1)] * 10
    pw, n_pos, n_neg = compute_pos_weight(samples)
    assert n_pos == 10 and n_neg == 90
    assert abs(pw - 9.0) < 1e-6


def test_compute_pos_weight_zero_positives_raises():
    """Pozitif yoksa hata firlatmali (sessizce 0'a bolmek tehlikeli)."""
    import pytest

    from scripts.train_smoke_detector import compute_pos_weight

    samples = [("x", 0)] * 10
    with pytest.raises(RuntimeError, match="pozitif"):
        compute_pos_weight(samples)


def test_transforms_random_resized_crop_scale():
    """RandomResizedCrop scale=(0.6, 1.0) olmali (tur 2 karari)."""
    from wildfire_ml.data.transforms import build_transforms

    pipeline = build_transforms("train")
    rrc = None
    for t in pipeline.transforms:
        if t.__class__.__name__ == "RandomResizedCrop":
            rrc = t
            break
    assert rrc is not None, "RandomResizedCrop train pipeline'inda olmali"
    scale = getattr(rrc, "scale", None)
    assert scale is not None
    assert abs(scale[0] - 0.6) < 1e-6, f"scale[0] 0.6 olmali, got {scale[0]}"
    assert abs(scale[1] - 1.0) < 1e-6, f"scale[1] 1.0 olmali, got {scale[1]}"
