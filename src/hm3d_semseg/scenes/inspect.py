"""Render a few deterministic scene views for human label inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image

from hm3d_semseg.camera.profile import CameraProfile
from hm3d_semseg.camera.resolve import resolve_camera_profile
from hm3d_semseg.config.schema import ProjectConfig
from hm3d_semseg.rendering.habitat import HabitatSceneRenderer
from hm3d_semseg.sampling.poses import PoseSampler, scene_seed
from hm3d_semseg.scenes.discovery import (
    discover_scenes,
    find_split_scene_dataset_config,
)
from hm3d_semseg.taxonomy.constants import ID2LABEL
from hm3d_semseg.taxonomy.mapping import MatterportMapping, TaxonomyMapper
from hm3d_semseg.types import NumpyArray
from hm3d_semseg.utils.hashing import atomic_write_json
from hm3d_semseg.visualization.masks import colorize_mask, overlay_mask


def _raw_semantic_colors(ids: NumpyArray) -> NumpyArray:
    values = ids.astype(np.uint64)
    return np.stack(
        [
            ((values * 37 + 17) % 255).astype(np.uint8),
            ((values * 67 + 43) % 255).astype(np.uint8),
            ((values * 97 + 89) % 255).astype(np.uint8),
        ],
        axis=-1,
    )


def inspect_scene(
    config: ProjectConfig,
    split: str,
    scene_id: str,
    num_views: int,
    output: Path,
) -> Dict[str, Any]:
    if (
        config.paths.hm3d_root is None
        or config.paths.objectnav_config is None
        or config.paths.taxonomy_mapping is None
    ):
        raise ValueError(
            "paths.hm3d_root, paths.objectnav_config, and paths.taxonomy_mapping are required"
        )
    camera = (
        CameraProfile.load(config.camera.profile)
        if config.camera.profile is not None
        else resolve_camera_profile(
            config.paths.objectnav_config, config.paths.habitat_lab_root
        )
    )
    mapping = MatterportMapping.from_file(
        config.paths.taxonomy_mapping, config.taxonomy.expected_mapping_sha256
    )
    mapper = TaxonomyMapper(mapping, config.taxonomy)
    scenes = discover_scenes(config.paths.hm3d_root, split, [scene_id], require_complete=True)
    scene = scenes[0]
    assert scene.rgb_mesh is not None
    scene_config = config.paths.scene_dataset_config or find_split_scene_dataset_config(
        config.paths.hm3d_root, split
    )
    output.mkdir(parents=True, exist_ok=True)
    camera.save(output / "camera_profile.yaml")
    report: Dict[str, Any] = {
        "scene_id": scene_id,
        "split": split,
        "camera_profile_hash": camera.profile_hash,
        "taxonomy_mapping_hash": mapping.sha256,
        "views": [],
    }
    pitches: Sequence[float] = config.camera.pitch_degrees or [0.0]
    with HabitatSceneRenderer(
        scene.rgb_mesh, scene_config, camera, store_depth=True
    ) as renderer:
        renderer.pathfinder.seed(scene_seed(config.sampling.seed, scene_id))
        poses = PoseSampler(config.sampling).sample(
            scene_id,
            renderer.pathfinder.get_random_navigable_point,
            pitches,
            num_views,
        )
        for index, pose in enumerate(poses[:num_views]):
            frame = renderer.render(pose)
            mask, decisions = mapper.map_semantic_observation(
                frame.semantic_ids, frame.semantic_id_to_raw_name
            )
            stem = f"view_{index:03d}"
            Image.fromarray(frame.rgb, mode="RGB").save(output / f"{stem}_rgb.png")
            Image.fromarray(_raw_semantic_colors(frame.semantic_ids), mode="RGB").save(
                output / f"{stem}_raw_semantic.png"
            )
            Image.fromarray(mask, mode="L").save(output / f"{stem}_mask.png")
            Image.fromarray(colorize_mask(mask), mode="RGB").save(
                output / f"{stem}_mask_color.png"
            )
            Image.fromarray(overlay_mask(frame.rgb, mask), mode="RGB").save(
                output / f"{stem}_overlay.png"
            )
            if frame.depth is not None:
                valid_depth = frame.depth[np.isfinite(frame.depth)]
                scale = float(np.percentile(valid_depth, 99)) if valid_depth.size else 1.0
                depth_image = np.clip(frame.depth / max(scale, 1e-6) * 255, 0, 255)
                Image.fromarray(depth_image.astype(np.uint8), mode="L").save(
                    output / f"{stem}_depth.png"
                )
            histogram = np.bincount(mask[mask != 255], minlength=41)
            report["views"].append(
                {
                    "index": index,
                    "pose": pose.to_dict(),
                    "class_histogram": {
                        ID2LABEL[class_id]: int(histogram[class_id])
                        for class_id in range(41)
                        if histogram[class_id]
                    },
                    "unknown_pixels": int(histogram[0]),
                    "ignored_pixels": int(np.sum(mask == 255)),
                    "semantic_id_decisions": {
                        str(key): vars(value) for key, value in decisions.items()
                    },
                }
            )
    atomic_write_json(output / "report.json", report)
    atomic_write_json(
        output / "legend.json", {str(key): value for key, value in ID2LABEL.items()}
    )
    return report
