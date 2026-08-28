"""Explicit RGB/mask paired geometry and RGB-only photometry."""

from __future__ import annotations

from typing import Sequence, Tuple

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


def random_resize_pair(
    rgb: NumpyArray,
    mask: NumpyArray,
    *,
    base_width: int,
    base_height: int,
    scale: float,
) -> Tuple[NumpyArray, NumpyArray]:
    """Resize while fitting inside a randomly scaled ``(width, height)`` box."""
    height, width = mask.shape
    fit_scale = min(base_width * scale / width, base_height * scale / height)
    target_width = max(1, round(width * fit_scale))
    target_height = max(1, round(height * fit_scale))
    return resize_pair(rgb, mask, target_width, target_height)


def random_crop_pair(
    rgb: NumpyArray,
    mask: NumpyArray,
    *,
    width: int,
    height: int,
    max_class_fraction: float,
    attempts: int,
    rng: np.random.Generator,
) -> Tuple[NumpyArray, NumpyArray]:
    """Take one paired crop, retrying to avoid a single-class-dominated target."""
    image_height, image_width = mask.shape
    crop_height = min(height, image_height)
    crop_width = min(width, image_width)
    candidate = (0, crop_height, 0, crop_width)
    for _ in range(attempts):
        top = int(rng.integers(0, image_height - crop_height + 1))
        left = int(rng.integers(0, image_width - crop_width + 1))
        candidate = (top, top + crop_height, left, left + crop_width)
        crop = mask[candidate[0] : candidate[1], candidate[2] : candidate[3]]
        valid_crop = crop[crop != 255]
        _, counts = np.unique(valid_crop, return_counts=True)
        if (
            max_class_fraction >= 1.0
            or (
                len(counts) > 1
                and counts.max() / counts.sum() < max_class_fraction
            )
        ):
            break
    top, bottom, left, right = candidate
    return (
        np.ascontiguousarray(rgb[top:bottom, left:right]),
        np.ascontiguousarray(mask[top:bottom, left:right]),
    )


def pad_pair(
    rgb: NumpyArray,
    mask: NumpyArray,
    *,
    width: int,
    height: int,
    rgb_fill: Sequence[int] = (124, 116, 104),
    mask_fill: int = 255,
) -> Tuple[NumpyArray, NumpyArray]:
    """Pad the bottom/right of a paired sample to a fixed training shape."""
    current_height, current_width = mask.shape
    if current_height > height or current_width > width:
        raise ValueError("pad target must not be smaller than the input")
    output_rgb = np.empty((height, width, 3), dtype=np.uint8)
    output_rgb[...] = np.asarray(rgb_fill, dtype=np.uint8)
    output_rgb[:current_height, :current_width] = rgb
    output_mask = np.full((height, width), mask_fill, dtype=mask.dtype)
    output_mask[:current_height, :current_width] = mask
    return output_rgb, output_mask


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


def photometric_distortion(
    rgb: NumpyArray, rng: np.random.Generator
) -> NumpyArray:
    """Apply the ADE20K recipe's randomized brightness/contrast/saturation/hue family."""

    def contrast(image: NumpyArray) -> NumpyArray:
        factor = float(rng.uniform(0.5, 1.5))
        adjusted = ImageEnhance.Contrast(Image.fromarray(image, mode="RGB")).enhance(
            factor
        )
        return np.asarray(adjusted, dtype=np.uint8).copy()

    result = rgb
    if rng.random() < 0.5:
        delta = float(rng.uniform(-32.0, 32.0))
        result = np.clip(result.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    contrast_first = bool(rng.integers(0, 2))
    if contrast_first and rng.random() < 0.5:
        result = contrast(result)
    if rng.random() < 0.5:
        factor = float(rng.uniform(0.5, 1.5))
        saturated = ImageEnhance.Color(Image.fromarray(result, mode="RGB")).enhance(
            factor
        )
        result = np.asarray(saturated, dtype=np.uint8).copy()
    if rng.random() < 0.5:
        hsv = np.asarray(Image.fromarray(result, mode="RGB").convert("HSV")).copy()
        hue_delta = round(float(rng.uniform(-18.0, 18.0)) * 255.0 / 180.0)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_delta) % 256
        result = np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB")).copy()
    if not contrast_first and rng.random() < 0.5:
        result = contrast(result)
    return result
