"""Explicit RGB/mask paired geometry and RGB-only photometry."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from hm3d_semseg.types import NumpyArray


def horizontal_flip(rgb: NumpyArray, mask: NumpyArray) -> Tuple[NumpyArray, NumpyArray]:
    return np.ascontiguousarray(rgb[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])


def resize_pair(
    rgb: NumpyArray, mask: NumpyArray, width: int, height: int
) -> Tuple[NumpyArray, NumpyArray]:
    """Resize aligned arrays; labels always use nearest-neighbor interpolation."""
    resampling = getattr(Image, "Resampling", Image)
    rgb_image = Image.fromarray(rgb, mode="RGB").resize(
        (width, height), resample=resampling.BILINEAR
    )
    mask_image = Image.fromarray(mask, mode="L").resize(
        (width, height), resample=resampling.NEAREST
    )
    return np.asarray(rgb_image).copy(), np.asarray(mask_image).copy()


def photometric_jitter(
    rgb: NumpyArray, amount: float, blur: bool, factor: float
) -> NumpyArray:
    image = Image.fromarray(rgb, mode="RGB")
    image = ImageEnhance.Brightness(image).enhance(1.0 + amount * factor)
    image = ImageEnhance.Contrast(image).enhance(1.0 - amount * factor)
    image = ImageEnhance.Color(image).enhance(1.0 + amount * factor / 2.0)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    return np.asarray(image).copy()
