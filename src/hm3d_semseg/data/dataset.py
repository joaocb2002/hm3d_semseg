"""PyTorch-compatible offline dataset with native aspect ratio."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from hm3d_semseg.config.schema import AugmentationConfig
from hm3d_semseg.data.schema import load_manifest
from hm3d_semseg.data.storage import load_mask
from hm3d_semseg.data.transforms import horizontal_flip, photometric_jitter
from hm3d_semseg.exceptions import OptionalDependencyError

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class OfflineSegmentationDataset:
    """Load manifest records without implicit resizing or label reduction."""

    def __init__(
        self,
        root: Path,
        augment: bool = False,
        augmentation: Optional[AugmentationConfig] = None,
        seed: int = 2027,
    ) -> None:
        self.root = root.resolve()
        self.records = load_manifest(self.root / "manifest.jsonl")
        self.augment = augment
        self.augmentation = augmentation or AugmentationConfig()
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> Dict[str, Any]:
        try:
            import torch
        except ImportError as error:
            raise OptionalDependencyError(
                "PyTorch is required to load training tensors; install the train extra."
            ) from error
        record = self.records[index]
        with Image.open(self.root / record.rgb) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        mask = load_mask(self.root / record.mask)
        original_size = tuple(mask.shape)
        if rgb.shape[:2] != mask.shape:
            raise ValueError(f"RGB/mask mismatch for sample {record.sample_id}")
        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
            if rng.random() < self.augmentation.horizontal_flip_probability:
                rgb, mask = horizontal_flip(rgb, mask)
            rgb = photometric_jitter(
                rgb,
                self.augmentation.color_jitter,
                rng.random() < self.augmentation.blur_probability,
                float(rng.uniform(-1.0, 1.0)),
            )
            if self.augmentation.sensor_noise_std:
                noise = rng.normal(
                    0.0,
                    self.augmentation.sensor_noise_std * 255.0,
                    size=rgb.shape,
                )
                rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        normalized = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "pixel_values": torch.from_numpy(normalized.transpose(2, 0, 1).copy()),
            "labels": torch.from_numpy(mask.astype(np.int64, copy=True)),
            "sample_id": record.sample_id,
            "scene_id": record.scene_id,
            "original_size": original_size,
            "metadata_path": str(self.root / record.metadata),
        }
