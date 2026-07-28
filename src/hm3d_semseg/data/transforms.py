"""Explicit RGB/mask paired geometry and RGB-only photometry."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def horizontal_flip(rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.ascontiguousarray(rgb[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])


def resize_pair(
    rgb: np.ndarray, mask: np.ndarray, width: int, height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize aligned arrays; labels always use nearest-neighbor interpolation."""
    resampling = getattr(Image, "Resampling", Image)
    rgb_image = Image.fromarray(rgb, mode="RGB").resize(
        (width, height), resample=resampling.BILINEAR
    )
    mask_image = Image.fromarray(mask, mode="L").resize(
        (width, height), resample=resampling.NEAREST
    )
    return np.asarray(rgb_image).copy(), np.asarray(mask_image).copy()


def photometric_jitter(rgb: np.ndarray, amount: float, blur: bool, factor: float) -> np.ndarray:
    image = Image.fromarray(rgb, mode="RGB")
    image = ImageEnhance.Brightness(image).enhance(1.0 + amount * factor)
    image = ImageEnhance.Contrast(image).enhance(1.0 - amount * factor)
    image = ImageEnhance.Color(image).enhance(1.0 + amount * factor / 2.0)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    return np.asarray(image).copy()
