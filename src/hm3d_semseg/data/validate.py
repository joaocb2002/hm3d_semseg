"""Complete offline dataset consistency validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import yaml
from PIL import Image

from hm3d_semseg.camera.profile import CameraProfile
from hm3d_semseg.data.schema import DATASET_SCHEMA_VERSION, load_manifest
from hm3d_semseg.data.storage import load_mask
from hm3d_semseg.exceptions import DatasetValidationError
from hm3d_semseg.utils.hashing import atomic_write_json, canonical_hash
from hm3d_semseg.visualization.masks import colorize_mask, overlay_mask


def validate_dataset(
    root: Path,
    raise_on_error: bool = True,
    artifact_output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate schema, files, shapes, targets, leakage, and class census."""
    root = root.resolve()
    errors: List[str] = []
    dataset_path = root / "dataset.yaml"
    manifest_path = root / "manifest.jsonl"
    if not dataset_path.is_file():
        errors.append("missing dataset.yaml")
        dataset = {}
    else:
        dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version {dataset.get('schema_version')!r}; "
            f"expected {DATASET_SCHEMA_VERSION!r}"
        )
    camera_path = root / "camera_profile.yaml"
    if not camera_path.is_file():
        errors.append("missing camera_profile.yaml")
    else:
        try:
            camera = CameraProfile.load(camera_path)
            if camera.profile_hash != dataset.get("camera_profile_hash"):
                errors.append("dataset.yaml camera_profile_hash does not match profile")
        except Exception as error:
            errors.append(f"invalid camera_profile.yaml: {error}")
    if not manifest_path.is_file():
        errors.append("missing manifest.jsonl")
        records = []
    else:
        try:
            records = load_manifest(manifest_path)
        except ValueError as error:
            errors.append(str(error))
            records = []
    ids = [record.sample_id for record in records]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate sample IDs: {', '.join(duplicates[:10])}")
    scene_splits: Dict[str, Set[str]] = defaultdict(set)
    class_counts = np.zeros(41, dtype=np.int64)
    ignored = 0
    total = 0
    for record in records:
        scene_splits[record.scene_id].add(record.split)
        paths = [root / record.rgb, root / record.mask, root / record.metadata]
        if record.depth:
            paths.append(root / record.depth)
        for path in paths:
            if not path.is_file():
                errors.append(f"{record.sample_id}: missing {path.relative_to(root)}")
        if any(not path.is_file() for path in paths[:3]):
            continue
        try:
            with Image.open(root / record.rgb) as image:
                image.load()
                rgb_size = image.size
            mask = load_mask(root / record.mask)
        except Exception as error:
            errors.append(f"{record.sample_id}: undecodable image: {error}")
            continue
        expected_shape = (record.height, record.width)
        if rgb_size != (record.width, record.height) or mask.shape != expected_shape:
            errors.append(
                f"{record.sample_id}: shape mismatch RGB={rgb_size}, mask={mask.shape}, "
                f"manifest={(record.width, record.height)}"
            )
        values = set(int(value) for value in np.unique(mask))
        invalid = sorted(values - set(range(41)) - {255})
        if invalid:
            errors.append(f"{record.sample_id}: invalid target IDs {invalid}")
        valid = mask != 255
        sample_counts = np.bincount(mask[valid], minlength=41)[:41]
        sample_ignored = int(np.sum(~valid))
        class_counts += sample_counts
        ignored += sample_ignored
        total += int(mask.size)
        if sample_counts.tolist() != record.class_histogram:
            errors.append(f"{record.sample_id}: class histogram differs from manifest")
        if sample_ignored != record.ignored_pixels:
            errors.append(f"{record.sample_id}: ignored-pixel count differs from manifest")
        if int(sample_counts[0]) != record.unknown_pixels:
            errors.append(f"{record.sample_id}: unknown-pixel count differs from manifest")
        if record.depth and (root / record.depth).is_file():
            try:
                depth = np.load(root / record.depth, allow_pickle=False)
                if depth.shape != expected_shape or depth.dtype != np.float32:
                    errors.append(
                        f"{record.sample_id}: depth shape/dtype mismatch "
                        f"{depth.shape} {depth.dtype}"
                    )
            except Exception as error:
                errors.append(f"{record.sample_id}: invalid depth array: {error}")
        if record.camera_profile_hash != dataset.get("camera_profile_hash"):
            errors.append(f"{record.sample_id}: camera-profile hash mismatch")
        if record.taxonomy_mapping_hash != dataset.get("taxonomy_mapping_hash"):
            errors.append(f"{record.sample_id}: taxonomy hash mismatch")
        try:
            metadata = json.loads((root / record.metadata).read_text(encoding="utf-8"))
            if metadata.get("sample_id") != record.sample_id:
                errors.append(f"{record.sample_id}: metadata sample ID mismatch")
        except Exception as error:
            errors.append(f"{record.sample_id}: invalid metadata JSON: {error}")
    leaked = sorted(scene for scene, splits in scene_splits.items() if len(splits) > 1)
    if leaked:
        errors.append(f"scene split leakage: {', '.join(leaked[:10])}")
    manifest_hash = canonical_hash([record.to_dict() for record in records])
    if dataset.get("manifest_hash") not in (None, manifest_hash):
        errors.append("dataset.yaml manifest_hash does not match the current manifest")
    report = {
        "dataset": str(root),
        "valid": not errors,
        "errors": errors,
        "samples": len(records),
        "scenes": len(scene_splits),
        "class_counts": class_counts.tolist(),
        "unknown_pixels": int(class_counts[0]),
        "ignored_pixels": ignored,
        "total_pixels": total,
        "unknown_fraction": float(class_counts[0] / max(1, total)),
        "ignored_fraction": float(ignored / max(1, total)),
        "manifest_hash": manifest_hash,
    }
    if artifact_output is not None:
        artifact_output.mkdir(parents=True, exist_ok=True)
        panels = []
        for record in sorted(records, key=lambda item: item.sample_id)[:8]:
            try:
                with Image.open(root / record.rgb) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                mask = load_mask(root / record.mask)
                panel = np.concatenate(
                    [rgb, colorize_mask(mask), overlay_mask(rgb, mask)], axis=1
                )
                panel_path = artifact_output / f"{record.sample_id}.png"
                Image.fromarray(panel, mode="RGB").save(panel_path)
                panels.append(str(panel_path))
            except Exception as error:
                errors.append(f"{record.sample_id}: could not create validation panel: {error}")
        report["manual_inspection_panels"] = panels
        report["valid"] = not errors
        report["errors"] = errors
        atomic_write_json(artifact_output / "validation_report.json", report)
    if errors and raise_on_error:
        raise DatasetValidationError(
            f"Dataset validation failed with {len(errors)} error(s):\n- "
            + "\n- ".join(errors[:25])
        )
    return report
