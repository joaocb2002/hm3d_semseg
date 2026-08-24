"""PyTorch-compatible offline dataset with native aspect ratio."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from hm3d_semseg.config.schema import AugmentationConfig
from hm3d_semseg.data.schema import ManifestRecord, load_manifest
from hm3d_semseg.data.storage import load_mask
from hm3d_semseg.data.transforms import horizontal_flip, photometric_jitter
from hm3d_semseg.exceptions import OptionalDependencyError

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def select_manifest_records(
    records: Sequence[ManifestRecord],
    max_samples: Optional[int],
    *,
    strategy: str = "manifest_order",
    seed: int = 2027,
) -> List[ManifestRecord]:
    """Select a reproducible training subset without loading sample files."""
    available = list(records)
    if max_samples is None:
        return available
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    limit = min(max_samples, len(available))
    if strategy == "manifest_order":
        return available[:limit]
    if strategy != "scene_diverse":
        raise ValueError(f"Unknown sample-selection strategy: {strategy}")

    by_scene: Dict[str, List[ManifestRecord]] = defaultdict(list)
    for record in available:
        by_scene[record.scene_id].append(record)
    rng = np.random.default_rng(seed)
    scene_names = sorted(by_scene)
    scene_order = [scene_names[index] for index in rng.permutation(len(scene_names))]
    scene_records = {
        scene: [items[index] for index in rng.permutation(len(items))]
        for scene, items in by_scene.items()
    }
    selected: List[ManifestRecord] = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for scene in scene_order:
            items = scene_records[scene]
            if round_index < len(items):
                selected.append(items[round_index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        round_index += 1
    return selected


def records_for_sample_ids(
    records: Sequence[ManifestRecord], sample_ids: Sequence[str]
) -> List[ManifestRecord]:
    """Resolve sample IDs in the requested order and reject missing IDs."""
    requested = list(sample_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("sample_ids must not contain duplicates")
    by_id = {record.sample_id: record for record in records}
    missing = [sample_id for sample_id in requested if sample_id not in by_id]
    if missing:
        raise ValueError(f"Unknown sample IDs: {', '.join(missing[:10])}")
    return [by_id[sample_id] for sample_id in requested]


class OfflineSegmentationDataset:
    """Load manifest records without implicit resizing or label reduction."""

    def __init__(
        self,
        root: Path,
        augment: bool = False,
        augmentation: Optional[AugmentationConfig] = None,
        seed: int = 2027,
        max_samples: Optional[int] = None,
        sample_selection: str = "manifest_order",
        sample_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = root.resolve()
        self.records = load_manifest(self.root / "manifest.jsonl")
        if sample_ids is not None and max_samples is not None:
            raise ValueError("Set either sample_ids or max_samples, not both")
        if sample_ids is not None:
            self.records = records_for_sample_ids(self.records, sample_ids)
        else:
            self.records = select_manifest_records(
                self.records,
                max_samples,
                strategy=sample_selection,
                seed=seed,
            )
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
