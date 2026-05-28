"""Image transforms.

# PREPROCESS SYMMETRIC: wildfire-ai-backend/app/services/inference_service.py
# `preprocess_image` ile birebir sabitleri kullanir (Karar #6).
# Degisiklik gerekirse her iki dosya ayni anda guncellenmeli.
"""
from __future__ import annotations

import io

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image

# ImageNet normalize sabitleri (backend ile birebir)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def build_transforms(split: str = "train"):
    """Return albumentations Compose pipeline for given split."""
    if split == "train":
        # Yeni API dene, eski API'ye fallback
        try:
            fog = A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.15)
        except TypeError:
            fog = A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15)
        # RandomResizedCrop API: >=1.4.18 size=(h,w); eski versiyon height+width
        # scale=(0.6, 1.0) — tur 2 (architect karari): daha agresif crop -> daha cesitli
        # bbox kapsamlari, kucuk smoke patch'lerine direnc artar.
        try:
            rrc = A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.6, 1.0))
        except (TypeError, ValueError):
            rrc = A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(0.6, 1.0))
        # Resize API benzer: yeni surumde tek-arg veya size=
        return A.Compose([
            rrc,
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            fog,
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:  # val / test
        try:
            resize = A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        except (TypeError, ValueError):
            resize = A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        return A.Compose([
            resize,
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


def preprocess_image_numpy(image_bytes: bytes) -> np.ndarray:
    """Backend `inference_service.preprocess_image` ile birebir NumPy karsiligi.

    # PREPROCESS SYMMETRIC: ayni sabitler (224, BILINEAR, ImageNet mean/std)
    # Test amacli; production inference'da kullanilmaz.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = arr.transpose(2, 0, 1)  # CHW
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return np.expand_dims(arr, axis=0).astype(np.float32)  # (1, 3, 224, 224)
