"""Resumable scene-grouped offline generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from hm3d_semseg.camera.profile import CameraProfile
from hm3d_semseg.camera.resolve import resolve_camera_profile
from hm3d_semseg.config.schema import ProjectConfig
from hm3d_semseg.data.schema import DATASET_SCHEMA_VERSION, ManifestRecord, load_manifest
from hm3d_semseg.data.storage import save_depth, save_mask, save_rgb
from hm3d_semseg.exceptions import ConfigurationError, DatasetValidationError
from hm3d_semseg.rendering.habitat import HabitatSceneRenderer
from hm3d_semseg.sampling.poses import PoseSampler, scene_seed
from hm3d_semseg.scenes.discovery import (
    discover_scenes,
    find_split_scene_dataset_config,
)
from hm3d_semseg.taxonomy.mapping import MatterportMapping, TaxonomyMapper
from hm3d_semseg.utils.hashing import atomic_write_json, atomic_write_text, canonical_hash
from hm3d_semseg.utils.provenance import collect_provenance


def _require_path(value: Optional[Path], name: str) -> Path:
    if value is None:
        raise ConfigurationError(f"Required path is unset: paths.{name}")
    return value


def _dataset_root(config: ProjectConfig) -> Path:
    if config.dataset.output_dir is not None:
        return config.dataset.output_dir
    generated = _require_path(config.paths.generated_data_root, "generated_data_root")
    return generated / config.dataset.name


def _load_camera(config: ProjectConfig, output_root: Path) -> CameraProfile:
    if config.camera.profile is not None:
        return CameraProfile.load(config.camera.profile)
    objectnav = _require_path(config.paths.objectnav_config, "objectnav_config")
    return resolve_camera_profile(objectnav, config.paths.habitat_lab_root)


def _read_scene_list(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _initialize_dataset(
    root: Path,
    config: ProjectConfig,
    camera: CameraProfile,
    mapping: MatterportMapping,
) -> None:
    contract = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "name": config.dataset.name,
        "split": config.dataset.split,
        "seed": config.sampling.seed,
        "camera_profile_hash": camera.profile_hash,
        "taxonomy_mapping_hash": mapping.sha256,
        "taxonomy_source": str(mapping.source),
        "taxonomy_source_url": mapping.source_url,
        "rgb_codec": config.dataset.rgb_codec,
        "store_depth": config.dataset.store_depth,
        "resolved_config_hash": canonical_hash(config.to_dict()),
    }
    dataset_file = root / "dataset.yaml"
    if dataset_file.exists():
        existing = yaml.safe_load(dataset_file.read_text(encoding="utf-8"))
        comparison_keys = (
            "schema_version",
            "split",
            "seed",
            "camera_profile_hash",
            "taxonomy_mapping_hash",
            "rgb_codec",
            "store_depth",
            "resolved_config_hash",
        )
        differences = [key for key in comparison_keys if existing.get(key) != contract.get(key)]
        if differences:
            raise DatasetValidationError(
                f"Refusing to resume incompatible dataset {root}; changed: "
                + ", ".join(differences)
            )
    else:
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(dataset_file, yaml.safe_dump(contract, sort_keys=False))
        camera.save(root / "camera_profile.yaml")
        atomic_write_text(
            root / "resolved_config.yaml",
            yaml.safe_dump(config.to_dict(), sort_keys=False),
        )
        atomic_write_json(
            root / "provenance.json",
            collect_provenance(config.paths.habitat_lab_root),
        )


def _record_paths(
    root: Path, split: str, scene_id: str, sample_id: str, codec: str, depth: bool
) -> Dict[str, Path]:
    base = root / split / scene_id
    extension = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[codec]
    paths = {
        "rgb": base / "rgb" / f"{sample_id}{extension}",
        "mask": base / "mask" / f"{sample_id}.png",
        "metadata": base / "metadata" / f"{sample_id}.json",
    }
    if depth:
        paths["depth"] = base / "depth" / f"{sample_id}.npy"
    return paths


def _record_failure(root: Path, value: Dict[str, Any]) -> None:
    path = root / "failures.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def generate_dataset(
    config: ProjectConfig,
    *,
    split_list: Optional[Path] = None,
    max_scenes: Optional[int] = None,
    max_samples_per_scene: Optional[int] = None,
    validation_only: bool = False,
) -> Dict[str, Any]:
    """Generate or safely resume a deterministic directory-backed dataset."""
    hm3d_root = _require_path(config.paths.hm3d_root, "hm3d_root")
    mapping_path = _require_path(config.paths.taxonomy_mapping, "taxonomy_mapping")
    root = _dataset_root(config)
    camera = _load_camera(config, root)
    mapping = MatterportMapping.from_file(mapping_path, config.taxonomy.expected_mapping_sha256)
    mapper = TaxonomyMapper(mapping, config.taxonomy)
    scene_ids = _read_scene_list(split_list)
    scenes = discover_scenes(
        hm3d_root, config.dataset.split, scene_ids=scene_ids, require_complete=True
    )
    scene_limit = max_scenes if max_scenes is not None else config.dataset.max_scenes
    if scene_limit is not None:
        scenes = scenes[:scene_limit]
    sample_limit = (
        max_samples_per_scene
        if max_samples_per_scene is not None
        else config.dataset.max_samples_per_scene
    )
    if sample_limit is None:
        sample_limit = config.sampling.positions_per_scene * config.sampling.yaws_per_position
    estimated_uncompressed = (
        len(scenes) * sample_limit * camera.rgb.width * camera.rgb.height * 4
    )
    summary: Dict[str, Any] = {
        "dataset": str(root),
        "scenes": len(scenes),
        "max_samples": len(scenes) * sample_limit,
        "estimated_uncompressed_bytes": estimated_uncompressed,
        "validation_only": bool(validation_only or config.dataset.validation_only),
    }
    if validation_only or config.dataset.validation_only:
        return summary
    if config.sampling.class_aware_fraction != 0.0:
        raise ConfigurationError(
            "The initial generator does not yet implement visibility-aware "
            "supplementary sampling; set sampling.class_aware_fraction to 0.0."
        )
    _initialize_dataset(root, config, camera, mapping)
    generation_request_hash = canonical_hash(
        {
            "scene_ids": [scene.scene_id for scene in scenes],
            "max_samples_per_scene": sample_limit,
        }
    )
    current_dataset_metadata = yaml.safe_load(
        (root / "dataset.yaml").read_text(encoding="utf-8")
    )
    if (
        current_dataset_metadata.get("generation_complete")
        and current_dataset_metadata.get("generation_request_hash") != generation_request_hash
    ):
        raise DatasetValidationError(
            "This dataset manifest is complete and immutable. Use a new dataset.name "
            "for a different scene/sample request."
        )
    atomic_write_text(
        root / "scene_lists" / f"requested_{config.dataset.split}.txt",
        "\n".join(scene.scene_id for scene in scenes) + "\n",
    )
    if config.camera.require_explicit_pitch and config.camera.pitch_degrees is None:
        raise ConfigurationError(
            "Full generation requires explicit camera.pitch_degrees because the local "
            "ObjectNav config exposes look actions but no pitch bounds."
        )
    pitches: Sequence[float] = config.camera.pitch_degrees or [0.0]
    scene_config = config.paths.scene_dataset_config or find_split_scene_dataset_config(
        hm3d_root, config.dataset.split
    )
    manifest_path = root / "manifest.jsonl"
    existing = load_manifest(manifest_path) if manifest_path.exists() else []
    existing_ids = {record.sample_id for record in existing}
    new_records: List[ManifestRecord] = []
    sampler = PoseSampler(config.sampling)
    for scene in scenes:
        assert scene.rgb_mesh is not None
        try:
            renderer = HabitatSceneRenderer(
                scene.rgb_mesh,
                scene_config,
                camera,
                config.dataset.store_depth,
            )
        except Exception as error:
            _record_failure(
                root,
                {
                    "scene_id": scene.scene_id,
                    "error": repr(error),
                    "continued": False,
                    "reason": "simulator initialization failed",
                },
            )
            raise
        with renderer:
            renderer.pathfinder.seed(scene_seed(config.sampling.seed, scene.scene_id))
            poses = sampler.sample(
                scene.scene_id,
                renderer.pathfinder.get_random_navigable_point,
                pitches,
                sample_limit,
            )
            accepted = 0
            for pose_index, pose in enumerate(poses):
                sample_id = f"{scene.scene_id}-{pose_index:06d}"
                if sample_id in existing_ids:
                    continue
                try:
                    frame = renderer.render(pose)
                    mask, decisions = mapper.map_semantic_observation(
                        frame.semantic_ids, frame.semantic_id_to_raw_name
                    )
                except Exception as error:
                    _record_failure(
                        root,
                        {
                            "scene_id": scene.scene_id,
                            "sample_id": sample_id,
                            "pose": pose.to_dict(),
                            "error": repr(error),
                            "continued": False,
                            "reason": "renderer or mapping state may be unsafe",
                        },
                    )
                    raise
                valid_fraction = float(np.mean(mask != config.taxonomy.ignore_index))
                unknown_fraction = float(np.mean(mask == 0))
                if valid_fraction < config.sampling.min_valid_pixel_fraction:
                    continue
                if unknown_fraction > config.sampling.max_unknown_fraction:
                    continue
                paths = _record_paths(
                    root,
                    config.dataset.split,
                    scene.scene_id,
                    sample_id,
                    config.dataset.rgb_codec,
                    config.dataset.store_depth,
                )
                save_rgb(
                    frame.rgb,
                    paths["rgb"],
                    config.dataset.rgb_codec,
                    config.dataset.jpeg_quality,
                )
                save_mask(mask, paths["mask"])
                if frame.depth is not None:
                    save_depth(frame.depth, paths["depth"])
                histogram = np.bincount(
                    mask[mask != config.taxonomy.ignore_index], minlength=41
                ).astype(int)
                metadata = {
                    "sample_id": sample_id,
                    "scene_id": scene.scene_id,
                    "split": config.dataset.split,
                    "pose": pose.to_dict(),
                    "camera_profile_hash": camera.profile_hash,
                    "taxonomy_mapping_hash": mapping.sha256,
                    "semantic_id_decisions": {
                        str(key): vars(value) for key, value in decisions.items()
                    },
                    "class_histogram": histogram.tolist(),
                    "ignored_pixels": int(np.sum(mask == config.taxonomy.ignore_index)),
                    "unknown_pixels": int(histogram[0]),
                }
                atomic_write_json(paths["metadata"], metadata)
                relative = {key: str(value.relative_to(root)) for key, value in paths.items()}
                record = ManifestRecord(
                    sample_id=sample_id,
                    split=config.dataset.split,
                    scene_id=scene.scene_id,
                    rgb=relative["rgb"],
                    mask=relative["mask"],
                    metadata=relative["metadata"],
                    depth=relative.get("depth"),
                    width=camera.rgb.width,
                    height=camera.rgb.height,
                    camera_profile_hash=camera.profile_hash,
                    taxonomy_mapping_hash=mapping.sha256,
                    class_histogram=histogram.tolist(),
                    ignored_pixels=metadata["ignored_pixels"],
                    unknown_pixels=metadata["unknown_pixels"],
                )
                new_records.append(record)
                existing_ids.add(sample_id)
                accepted += 1
                if accepted >= sample_limit:
                    break
            if new_records:
                with manifest_path.open("a", encoding="utf-8", newline="\n") as handle:
                    for record in new_records:
                        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                new_records.clear()
    records = load_manifest(manifest_path) if manifest_path.exists() else []
    summary["generated_samples"] = len(records)
    summary["manifest_hash"] = canonical_hash([record.to_dict() for record in records])
    atomic_write_text(
        root / "scene_lists" / f"manifest_{config.dataset.split}.txt",
        "\n".join(sorted({record.scene_id for record in records})) + "\n",
    )
    dataset_metadata = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    dataset_metadata["manifest_hash"] = summary["manifest_hash"]
    dataset_metadata["sample_count"] = len(records)
    dataset_metadata["generation_request_hash"] = generation_request_hash
    dataset_metadata["generation_complete"] = True
    atomic_write_text(root / "dataset.yaml", yaml.safe_dump(dataset_metadata, sort_keys=False))
    atomic_write_json(root / "generation_summary.json", summary)
    return summary
