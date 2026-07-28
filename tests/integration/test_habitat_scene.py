from __future__ import annotations

import os
from pathlib import Path

import pytest

from hm3d_semseg.camera.resolve import resolve_camera_profile
from hm3d_semseg.config import load_config
from hm3d_semseg.rendering.habitat import HabitatSceneRenderer
from hm3d_semseg.sampling.poses import CameraPose
from hm3d_semseg.scenes.discovery import discover_scenes, find_split_scene_dataset_config
from hm3d_semseg.taxonomy.mapping import MatterportMapping, TaxonomyMapper
from hm3d_semseg.visualization.masks import overlay_mask

pytestmark = pytest.mark.habitat


def configured() -> object:
    path = os.environ.get("HM3D_SEMSEG_LOCAL_CONFIG")
    if not path or not Path(path).is_file():
        pytest.skip("Set HM3D_SEMSEG_LOCAL_CONFIG to run proprietary HM3D tests")
    return load_config(local_config=Path(path))


def test_semantic_scene_ids_match_descriptor_without_offset() -> None:
    habitat_sim = pytest.importorskip("habitat_sim")
    config = configured()
    if config.paths.hm3d_root is None:
        pytest.skip("paths.hm3d_root is unset")
    scene = discover_scenes(config.paths.hm3d_root, "minival", ["00800-TEEsavR23oF"], True)[0]
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene.rgb_mesh)
    backend.scene_dataset_config_file = str(
        find_split_scene_dataset_config(config.paths.hm3d_root, "minival")
    )
    backend.create_renderer = False
    backend.load_semantic_mesh = True
    simulator = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [habitat_sim.AgentConfiguration()])
    )
    try:
        objects = [item for item in simulator.semantic_scene.objects if item is not None]
        assert objects
        assert objects[1].semantic_id == 1
        assert objects[1].category.name() == "ceiling"
        point = simulator.pathfinder.get_random_navigable_point()
        assert len(point) == 3
    finally:
        simulator.close(destroy=True)
    # Reopen to catch leaked simulator/context state.
    simulator = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [habitat_sim.AgentConfiguration()])
    )
    simulator.close(destroy=True)


@pytest.mark.gpu
def test_aligned_rgb_semantic_depth_render_and_mapping(tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image

    config = configured()
    if (
        config.paths.hm3d_root is None
        or config.paths.objectnav_config is None
        or config.paths.taxonomy_mapping is None
    ):
        pytest.skip("HM3D, ObjectNav, or taxonomy path is unset")
    scene = discover_scenes(config.paths.hm3d_root, "minival", ["00800-TEEsavR23oF"], True)[0]
    camera = resolve_camera_profile(
        config.paths.objectnav_config, config.paths.habitat_lab_root
    )
    mapping = MatterportMapping.from_file(
        config.paths.taxonomy_mapping, config.taxonomy.expected_mapping_sha256
    )
    mapper = TaxonomyMapper(mapping, config.taxonomy)
    with HabitatSceneRenderer(
        scene.rgb_mesh,
        find_split_scene_dataset_config(config.paths.hm3d_root, "minival"),
        camera,
        store_depth=True,
    ) as renderer:
        point = renderer.pathfinder.get_random_navigable_point()
        frame = renderer.render(CameraPose(list(np.asarray(point, dtype=float)), 0.0, 0.0))
        mask, decisions = mapper.map_semantic_observation(
            frame.semantic_ids, frame.semantic_id_to_raw_name
        )
        assert frame.rgb.shape == (camera.rgb.height, camera.rgb.width, 3)
        assert frame.rgb.dtype == np.uint8
        assert frame.semantic_ids.shape == frame.rgb.shape[:2]
        assert frame.depth is not None and frame.depth.shape == frame.rgb.shape[:2]
        assert decisions
        assert set(np.unique(mask)) <= set(range(41)) | {255}
        Image.fromarray(overlay_mask(frame.rgb, mask)).save(tmp_path / "inspection_panel.png")
