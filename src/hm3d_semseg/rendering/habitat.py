"""One-scene-at-a-time Habitat-Sim renderer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hm3d_semseg.camera.profile import CameraProfile
from hm3d_semseg.exceptions import OptionalDependencyError
from hm3d_semseg.sampling.poses import CameraPose
from hm3d_semseg.types import NumpyArray


def postprocess_depth(depth: NumpyArray, camera: CameraProfile) -> NumpyArray:
    """Match Habitat-Lab's configured depth clipping and normalization."""
    result = np.asarray(depth, dtype=np.float32).copy()
    profile = camera.depth
    if profile is None:
        return result
    if profile.min_depth is not None:
        result = np.maximum(result, profile.min_depth)
    if profile.max_depth is not None:
        result = np.minimum(result, profile.max_depth)
    if profile.normalize_depth:
        if (
            profile.min_depth is None
            or profile.max_depth is None
            or profile.max_depth <= profile.min_depth
        ):
            raise ValueError(
                "Normalized depth requires finite max_depth greater than min_depth"
            )
        result = (result - profile.min_depth) / (profile.max_depth - profile.min_depth)
    return result.astype(np.float32, copy=False)


@dataclass
class RenderedFrame:
    rgb: NumpyArray
    semantic_ids: NumpyArray
    depth: Optional[NumpyArray]
    semantic_id_to_raw_name: Dict[int, str]


class HabitatSceneRenderer:
    """Own exactly one simulator and close its GPU context deterministically."""

    def __init__(
        self,
        scene_mesh: Path,
        scene_dataset_config: Path,
        camera: CameraProfile,
        store_depth: bool,
        gpu_device_id: int = 0,
        create_renderer: bool = True,
    ) -> None:
        try:
            import habitat_sim
        except ImportError as error:
            raise OptionalDependencyError(
                "Habitat-Sim is required for rendering. Use the documented render environment."
            ) from error
        self._habitat_sim = habitat_sim
        self.camera = camera
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = str(scene_mesh)
        backend.scene_dataset_config_file = str(scene_dataset_config)
        backend.load_semantic_mesh = True
        backend.gpu_device_id = gpu_device_id
        backend.create_renderer = create_renderer
        backend.random_seed = 0

        rgb = self._sensor("rgb", habitat_sim.SensorType.COLOR)
        semantic = self._sensor("semantic", habitat_sim.SensorType.SEMANTIC)
        sensors = [rgb, semantic]
        if store_depth:
            sensors.append(self._sensor("depth", habitat_sim.SensorType.DEPTH))
        agent = habitat_sim.AgentConfiguration()
        agent.height = camera.agent.height
        agent.radius = camera.agent.radius
        agent.sensor_specifications = sensors
        try:
            self.simulator = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
        except Exception as error:
            raise RuntimeError(
                "Habitat-Sim could not create the scene/renderer. Check the annotated "
                "scene config, semantic assets, EGL/OpenGL, NVIDIA driver, and GPU ID. "
                f"Original error: {error}"
            ) from error
        self.agent = self.simulator.initialize_agent(0)
        self.semantic_id_to_raw_name = {
            int(obj.semantic_id): str(obj.category.name())
            for obj in self.simulator.semantic_scene.objects
            if obj is not None
        }

    def _sensor(self, uuid: str, sensor_type: Any) -> Any:
        habitat_sim = self._habitat_sim
        source = self.camera.rgb
        if uuid == "depth" and self.camera.depth is not None:
            source = self.camera.depth
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [source.height, source.width]
        spec.hfov = source.hfov
        spec.position = source.position
        spec.orientation = source.orientation
        return spec

    @property
    def pathfinder(self) -> Any:
        return self.simulator.pathfinder

    def render(self, pose: CameraPose) -> RenderedFrame:
        habitat_sim = self._habitat_sim
        state = habitat_sim.AgentState()
        state.position = np.asarray(pose.position, dtype=np.float32)
        common = habitat_sim.utils.common
        yaw = common.quat_from_angle_axis(
            np.deg2rad(pose.yaw_degrees), np.array([0.0, 1.0, 0.0])
        )
        state.rotation = yaw
        self.agent.set_state(state, reset_sensors=True)

        # ObjectNav look actions rotate each sensor about its own optical center.
        # Pitching the agent body would also rotate the sensor's position offset.
        pitch = common.quat_from_angle_axis(
            np.deg2rad(pose.pitch_degrees), np.array([1.0, 0.0, 0.0])
        )
        pitched_state = self.agent.get_state()
        for sensor_state in pitched_state.sensor_states.values():
            sensor_state.rotation = sensor_state.rotation * pitch
        self.agent.set_state(
            pitched_state,
            reset_sensors=False,
            infer_sensor_states=False,
        )
        observations = self.simulator.get_sensor_observations()
        rgb = np.asarray(observations["rgb"])[..., :3].copy()
        semantic = np.asarray(observations["semantic"]).astype(np.int64, copy=True)
        depth = (
            postprocess_depth(np.asarray(observations["depth"]), self.camera)
            if "depth" in observations
            else None
        )
        if rgb.shape[:2] != semantic.shape:
            raise RuntimeError(
                f"Aligned sensors returned different shapes: RGB {rgb.shape}, "
                f"semantic {semantic.shape}"
            )
        expected = (self.camera.rgb.height, self.camera.rgb.width)
        if rgb.shape[:2] != expected:
            raise RuntimeError(
                f"Renderer returned {rgb.shape[:2]}, expected camera shape {expected}"
            )
        if depth is not None and depth.shape != expected:
            raise RuntimeError(
                f"Depth returned {depth.shape}, expected aligned shape {expected}"
            )
        return RenderedFrame(rgb, semantic, depth, self.semantic_id_to_raw_name)

    def close(self) -> None:
        simulator = getattr(self, "simulator", None)
        if simulator is not None:
            simulator.close(destroy=True)
            self.simulator = None

    def __enter__(self) -> "HabitatSceneRenderer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
