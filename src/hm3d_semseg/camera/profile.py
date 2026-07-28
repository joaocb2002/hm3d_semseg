"""Frozen camera profile data model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hm3d_semseg.exceptions import CameraContractError
from hm3d_semseg.utils.hashing import atomic_write_text, canonical_hash


@dataclass(frozen=True)
class SensorProfile:
    uuid: str
    sensor_type: str
    width: int
    height: int
    hfov: float
    position: List[float]
    orientation: List[float]
    min_depth: Optional[float] = None
    max_depth: Optional[float] = None
    normalize_depth: Optional[bool] = None


@dataclass(frozen=True)
class AgentProfile:
    name: str
    height: float
    radius: float


@dataclass(frozen=True)
class PitchProfile:
    supported: bool
    increment_degrees: Optional[float]
    minimum_degrees: Optional[float]
    maximum_degrees: Optional[float]
    source: str


@dataclass(frozen=True)
class CameraProfile:
    schema_version: str
    rgb: SensorProfile
    agent: AgentProfile
    depth: Optional[SensorProfile]
    semantic: Optional[SensorProfile]
    pitch: PitchProfile
    observation_transforms: Dict[str, Any]
    provenance: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self, include_hash: bool = True) -> Dict[str, Any]:
        value = asdict(self)
        if include_hash:
            value["camera_profile_hash"] = self.profile_hash
        return value

    @property
    def profile_hash(self) -> str:
        """Hash the stable contract, excluding extraction time and diagnostics."""
        value = self.to_dict(include_hash=False)
        provenance = value["provenance"]
        value["provenance"] = {
            key: provenance.get(key)
            for key in (
                "source",
                "source_config",
                "source_config_sha256",
                "habitat_lab_version",
                "habitat_lab_commit",
                "fallback",
            )
            if key in provenance
        }
        value.pop("warnings", None)
        return canonical_hash(value)

    def save(self, path: Path) -> None:
        atomic_write_text(path, yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> "CameraProfile":
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        expected_hash = value.pop("camera_profile_hash", None)
        profile = cls(
            schema_version=str(value["schema_version"]),
            rgb=SensorProfile(**value["rgb"]),
            agent=AgentProfile(**value["agent"]),
            depth=SensorProfile(**value["depth"]) if value.get("depth") else None,
            semantic=SensorProfile(**value["semantic"]) if value.get("semantic") else None,
            pitch=PitchProfile(**value["pitch"]),
            observation_transforms=value.get("observation_transforms", {}),
            provenance=value.get("provenance", {}),
            warnings=value.get("warnings", []),
        )
        if expected_hash is not None and expected_hash != profile.profile_hash:
            raise CameraContractError(
                f"Camera profile hash mismatch in {path}: expected {expected_hash}, "
                f"computed {profile.profile_hash}"
            )
        return profile


GEOMETRY_FIELDS = ("width", "height", "hfov", "position", "orientation")


def camera_differences(
    expected: CameraProfile, actual: CameraProfile, tolerance: float = 1e-6
) -> List[str]:
    differences: List[str] = []
    for field_name in GEOMETRY_FIELDS:
        left = getattr(expected.rgb, field_name)
        right = getattr(actual.rgb, field_name)
        if isinstance(left, list):
            mismatch = len(left) != len(right) or any(
                abs(float(a) - float(b)) > tolerance for a, b in zip(left, right)
            )
        elif isinstance(left, (float, int)) and isinstance(right, (float, int)):
            mismatch = abs(float(left) - float(right)) > tolerance
        else:
            mismatch = left != right
        if mismatch:
            differences.append(f"rgb.{field_name}: expected {left!r}, found {right!r}")
    for field_name in ("height", "radius"):
        left = getattr(expected.agent, field_name)
        right = getattr(actual.agent, field_name)
        if abs(left - right) > tolerance:
            differences.append(f"agent.{field_name}: expected {left!r}, found {right!r}")
    if expected.observation_transforms != actual.observation_transforms:
        differences.append(
            "observation_transforms: expected "
            f"{expected.observation_transforms!r}, found {actual.observation_transforms!r}"
        )
    return differences


def assert_camera_compatible(
    expected: CameraProfile, actual: CameraProfile, allow_mismatch: bool = False
) -> List[str]:
    """Fail on accidental projection/extrinsic mismatch."""
    differences = camera_differences(expected, actual)
    if differences and not allow_mismatch:
        raise CameraContractError(
            "Camera profiles are incompatible:\n- " + "\n- ".join(differences)
        )
    return differences


def official_2023_profile() -> CameraProfile:
    """Return the checked portrait fallback/regression profile."""
    rgb = SensorProfile(
        uuid="rgb_sensor",
        sensor_type="HabitatSimRGBSensor",
        width=480,
        height=640,
        hfov=42.0,
        position=[0.0, 1.31, 0.0],
        orientation=[0.0, 0.0, 0.0],
    )
    depth = SensorProfile(
        uuid="depth_sensor",
        sensor_type="HabitatSimDepthSensor",
        width=480,
        height=640,
        hfov=42.0,
        position=[0.0, 1.31, 0.0],
        orientation=[0.0, 0.0, 0.0],
        min_depth=0.5,
        max_depth=5.0,
        normalize_depth=False,
    )
    return CameraProfile(
        schema_version="1.0",
        rgb=rgb,
        depth=depth,
        semantic=None,
        agent=AgentProfile(name="main_agent", height=1.41, radius=0.17),
        pitch=PitchProfile(True, None, None, None, "official profile: bounds unresolved"),
        observation_transforms={},
        provenance={
            "source": (
                "https://github.com/facebookresearch/habitat-challenge/blob/main/"
                "configs/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch_challenge.yaml"
            ),
            "fallback": True,
        },
        warnings=["Official 2023 fixture; the actual composed local configuration wins."],
    )
