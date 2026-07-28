"""Mask visualization separate from lossless target storage."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from hm3d_semseg.taxonomy.constants import NUM_CLASSES


def palette() -> np.ndarray:
    rng = np.random.default_rng(12345)
    colors = rng.integers(32, 256, size=(NUM_CLASSES, 3), dtype=np.uint8)
    colors[0] = [80, 80, 80]
    return colors


def colorize_mask(
    mask: np.ndarray, ignore_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    colors = palette()
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < NUM_CLASSES)
    output[valid] = colors[mask[valid]]
    output[mask == 255] = np.asarray(ignore_color, dtype=np.uint8)
    return output


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = colorize_mask(mask)
    return np.clip(
        rgb.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
