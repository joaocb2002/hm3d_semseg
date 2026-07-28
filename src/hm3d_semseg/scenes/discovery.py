"""Filesystem-independent HM3D scene asset validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class SceneAssets:
    scene_id: str
    split: str
    directory: Path
    rgb_mesh: Optional[Path]
    semantic_mesh: Optional[Path]
    semantic_descriptor: Optional[Path]
    navmesh: Optional[Path]

    @property
    def complete(self) -> bool:
        return all(
            path is not None
            for path in (
                self.rgb_mesh,
                self.semantic_mesh,
                self.semantic_descriptor,
                self.navmesh,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


def _single(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[0].resolve() if len(matches) == 1 else None


def discover_scenes(
    hm3d_root: Path,
    split: str,
    scene_ids: Optional[Iterable[str]] = None,
    require_complete: bool = False,
) -> List[SceneAssets]:
    """Discover only annotated scene directories for a configured split."""
    split_name = "val" if split in {"validation", "val"} else split
    split_root = hm3d_root / split_name
    wanted = set(scene_ids) if scene_ids is not None else None
    scenes: List[SceneAssets] = []
    for descriptor in sorted(split_root.glob("*/*.semantic.txt")):
        directory = descriptor.parent
        scene_id = directory.name
        if wanted is not None and scene_id not in wanted:
            continue
        assets = SceneAssets(
            scene_id=scene_id,
            split=split_name,
            directory=directory.resolve(),
            rgb_mesh=_single(directory, "*.basis.glb"),
            semantic_mesh=_single(directory, "*.semantic.glb"),
            semantic_descriptor=descriptor.resolve(),
            navmesh=_single(directory, "*.basis.navmesh"),
        )
        if require_complete and not assets.complete:
            missing = [
                name
                for name, path in (
                    ("RGB mesh", assets.rgb_mesh),
                    ("semantic mesh", assets.semantic_mesh),
                    ("semantic descriptor", assets.semantic_descriptor),
                    ("navmesh", assets.navmesh),
                )
                if path is None
            ]
            raise FileNotFoundError(f"Scene {scene_id} is incomplete: {', '.join(missing)}")
        scenes.append(assets)
    if wanted is not None:
        found = {scene.scene_id for scene in scenes}
        missing_ids = sorted(wanted - found)
        if missing_ids:
            raise FileNotFoundError(
                f"Annotated scene(s) not found in {split_root}: {', '.join(missing_ids)}"
            )
    return scenes


def find_split_scene_dataset_config(hm3d_root: Path, split: str) -> Path:
    """Find the unique annotated config for a split."""
    split_name = "val" if split in {"validation", "val"} else split
    matches = sorted(
        (hm3d_root / split_name).glob(
            f"hm3d_annotated_{split_name}_*.scene_dataset_config.json"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one annotated scene-dataset config for {split_name}, "
            f"found {len(matches)} under {hm3d_root / split_name}"
        )
    return matches[0].resolve()
