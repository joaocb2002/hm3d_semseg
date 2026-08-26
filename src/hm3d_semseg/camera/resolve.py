"""Resolve Habitat Hydra configs into a frozen camera profile."""

from __future__ import annotations

import importlib.metadata
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import yaml

from hm3d_semseg.camera.profile import (
    AgentProfile,
    CameraProfile,
    PitchProfile,
    SensorProfile,
)
from hm3d_semseg.exceptions import CameraContractError, OptionalDependencyError
from hm3d_semseg.utils.hashing import sha256_file


def _as_plain_mapping(config: Any) -> Dict[str, Any]:
    if isinstance(config, dict):
        return config
    try:
        from omegaconf import OmegaConf
    except ImportError as error:
        raise OptionalDependencyError(
            "OmegaConf is required to resolve a composed Habitat configuration"
        ) from error
    container = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(container, dict):
        raise CameraContractError("Resolved Habitat configuration is not a mapping")
    return cast(Dict[str, Any], container)


def _sensor_profile(uuid: str, values: Mapping[str, Any]) -> SensorProfile:
    required = ("type", "width", "height", "hfov", "position", "orientation")
    missing = [key for key in required if key not in values]
    if missing:
        raise CameraContractError(
            f"Sensor {uuid!r} is missing required values: {', '.join(missing)}"
        )
    return SensorProfile(
        uuid=uuid,
        sensor_type=str(values["type"]),
        width=int(values["width"]),
        height=int(values["height"]),
        hfov=float(values["hfov"]),
        position=[float(value) for value in values["position"]],
        orientation=[float(value) for value in values["orientation"]],
        min_depth=(float(values["min_depth"]) if values.get("min_depth") is not None else None),
        max_depth=(float(values["max_depth"]) if values.get("max_depth") is not None else None),
        normalize_depth=(
            bool(values["normalize_depth"])
            if values.get("normalize_depth") is not None
            else None
        ),
    )


def _choose_agent(simulator: Mapping[str, Any]) -> Tuple[str, Mapping[str, Any]]:
    agents = simulator.get("agents")
    if not isinstance(agents, Mapping) or not agents:
        raise CameraContractError("No agents found at habitat.simulator.agents")
    order = simulator.get("agents_order")
    default_id = int(simulator.get("default_agent_id", 0))
    if isinstance(order, list) and 0 <= default_id < len(order):
        name = str(order[default_id])
    elif len(agents) == 1:
        name = str(next(iter(agents)))
    else:
        raise CameraContractError(
            "Cannot identify active agent: agents_order/default_agent_id are unresolved"
        )
    agent = agents.get(name)
    if not isinstance(agent, Mapping):
        raise CameraContractError(f"Active agent {name!r} is absent")
    return name, agent


def _choose_sensor(
    sensors: Mapping[str, Any], kind: str, required: bool
) -> Optional[SensorProfile]:
    candidates: List[Tuple[str, Mapping[str, Any]]] = []
    needle = kind.casefold()
    for name, values in sensors.items():
        if not isinstance(values, Mapping):
            continue
        sensor_type = str(values.get("type", "")).casefold()
        if needle in str(name).casefold() or needle in sensor_type:
            candidates.append((str(name), values))
    if len(candidates) == 1:
        return _sensor_profile(*candidates[0])
    if not candidates and not required:
        return None
    if not candidates:
        raise CameraContractError(f"No active {kind} sensor found")
    raise CameraContractError(
        f"Multiple active {kind} sensors found; specify one explicitly: "
        + ", ".join(name for name, _ in candidates)
    )


def extract_camera_profile(
    config: Any,
    *,
    provenance: Optional[Dict[str, Any]] = None,
    profile_warnings: Optional[List[str]] = None,
) -> CameraProfile:
    """Extract camera geometry from an already composed Habitat configuration."""
    plain = _as_plain_mapping(config)
    habitat = plain.get("habitat")
    if not isinstance(habitat, Mapping):
        raise CameraContractError("Missing resolved 'habitat' configuration")
    simulator = habitat.get("simulator")
    if not isinstance(simulator, Mapping):
        raise CameraContractError("Missing resolved 'habitat.simulator' configuration")
    agent_name, agent_values = _choose_agent(simulator)
    sensors = agent_values.get("sim_sensors")
    if not isinstance(sensors, Mapping):
        raise CameraContractError(f"Active agent {agent_name!r} has no sim_sensors")
    rgb = _choose_sensor(sensors, "rgb", required=True)
    assert rgb is not None
    depth = _choose_sensor(sensors, "depth", required=False)
    semantic = _choose_sensor(sensors, "semantic", required=False)

    actions = habitat.get("task", {}).get("actions", {})
    look_values = []
    if isinstance(actions, Mapping):
        for name in ("look_up", "look_down"):
            action = actions.get(name)
            if isinstance(action, Mapping) and action.get("tilt_angle") is not None:
                look_values.append(float(action["tilt_angle"]))
    pitch = PitchProfile(
        supported=bool(look_values),
        increment_degrees=(look_values[0] if look_values else None),
        minimum_degrees=None,
        maximum_degrees=None,
        source=(
            "resolved task look actions; repeat bounds are not represented in config"
            if look_values
            else "no resolved look actions"
        ),
    )
    transforms = (
        plain.get("habitat_baselines", {})
        .get("rl", {})
        .get("policy", {})
        .get("obs_transforms", {})
    )
    return CameraProfile(
        schema_version="1.0",
        rgb=rgb,
        depth=depth,
        semantic=semantic,
        agent=AgentProfile(
            name=agent_name,
            height=float(agent_values["height"]),
            radius=float(agent_values["radius"]),
        ),
        pitch=pitch,
        observation_transforms=transforms if isinstance(transforms, dict) else {},
        provenance=provenance or {},
        warnings=profile_warnings or [],
    )


def _git_commit(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_camera_profile(
    objectnav_config: Path,
    habitat_lab_root: Optional[Path] = None,
    allow_raw_yaml_fallback: bool = False,
) -> CameraProfile:
    """Compose a Habitat config with the installed API and freeze its camera."""
    source = objectnav_config.resolve()
    provenance: Dict[str, Any] = {
        "source_config": str(source),
        "source_config_sha256": sha256_file(source),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "habitat_lab_commit": _git_commit(habitat_lab_root),
        "fallback": False,
    }
    try:
        import habitat

        provenance["habitat_lab_version"] = getattr(
            habitat, "__version__", importlib.metadata.version("habitat-lab")
        )
        composed = habitat.get_config(str(source))
        return extract_camera_profile(composed, provenance=provenance)
    except Exception as error:
        if not allow_raw_yaml_fallback:
            raise CameraContractError(
                f"Habitat could not compose {source}: {error}. Raw YAML is not a "
                "valid substitute for a full generation run."
            ) from error
        warning = (
            "PROMINENT WARNING: used raw YAML diagnostic fallback; Hydra defaults and "
            "interpolations may be unresolved."
        )
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        provenance["fallback"] = True
        provenance["compose_error"] = repr(error)
        return extract_camera_profile(raw, provenance=provenance, profile_warnings=[warning])
