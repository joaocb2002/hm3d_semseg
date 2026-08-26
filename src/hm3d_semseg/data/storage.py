"""Lossless masks and configurable RGB codecs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from hm3d_semseg.types import NumpyArray


def _atomic_image_save(
    image: Image.Image,
    path: Path,
    *,
    image_format: str,
    quality: Optional[int] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=str(path.parent)
    )
    os.close(fd)
    try:
        options: Dict[str, Any] = {"format": image_format}
        if quality is not None:
            options["quality"] = quality
        image.save(temporary, **options)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_rgb(rgb: NumpyArray, path: Path, codec: str, jpeg_quality: int = 95) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"RGB must be uint8 [H,W,3], got {rgb.shape} {rgb.dtype}")
    image = Image.fromarray(rgb, mode="RGB")
    if codec == "png":
        _atomic_image_save(image, path, image_format="PNG")
    elif codec == "jpeg":
        _atomic_image_save(image, path, image_format="JPEG", quality=jpeg_quality)
    elif codec == "webp":
        _atomic_image_save(image, path, image_format="WEBP", quality=100)
    else:
        raise ValueError(f"Unsupported RGB codec: {codec}")


def save_mask(mask: NumpyArray, path: Path) -> None:
    """Save class targets losslessly as one-channel uint8 PNG."""
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError(f"Mask must be uint8 [H,W], got {mask.shape} {mask.dtype}")
    _atomic_image_save(Image.fromarray(mask, mode="L"), path, image_format="PNG")


def load_mask(path: Path) -> NumpyArray:
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"Mask must be single-channel L mode, found {image.mode}: {path}")
        return np.asarray(image, dtype=np.uint8).copy()


def save_depth(depth: NumpyArray, path: Path) -> None:
    if depth.ndim != 2 or depth.dtype != np.float32:
        raise ValueError(f"Depth must be float32 [H,W], got {depth.shape} {depth.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, depth, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
