from dataclasses import replace
from pathlib import Path

import pytest

from hm3d_semseg.camera.profile import (
    CameraProfile,
    assert_camera_compatible,
    official_2023_profile,
)
from hm3d_semseg.camera.resolve import extract_camera_profile
from hm3d_semseg.exceptions import CameraContractError

pytestmark = pytest.mark.unit


def fixture_config(width: int = 480) -> dict:
    sensor = {
        "type": "HabitatSimRGBSensor",
        "width": width,
        "height": 640,
        "hfov": 42,
        "position": [0, 1.31, 0],
        "orientation": [0, 0, 0],
    }
    return {
        "habitat": {
            "simulator": {
                "default_agent_id": 0,
                "agents_order": ["main_agent"],
                "agents": {
                    "main_agent": {
                        "height": 1.41,
                        "radius": 0.17,
                        "sim_sensors": {
                            "rgb_sensor": sensor,
                            "depth_sensor": {
                                **sensor,
                                "type": "HabitatSimDepthSensor",
                                "min_depth": 0.5,
                                "max_depth": 5.0,
                                "normalize_depth": False,
                            },
                        },
                    }
                },
            },
            "task": {
                "actions": {
                    "look_up": {"tilt_angle": 30},
                    "look_down": {"tilt_angle": 30},
                }
            },
        }
    }


def test_official_camera_fixture_extraction_is_portrait() -> None:
    profile = extract_camera_profile(fixture_config())
    assert profile.rgb.width == 480
    assert profile.rgb.height == 640
    assert profile.rgb.hfov == 42
    assert profile.depth is not None
    assert profile.depth.max_depth == 5.0
    assert profile.pitch.increment_degrees == 30


def test_camera_hash_roundtrip_and_mismatch(tmp_path: Path) -> None:
    profile = official_2023_profile()
    path = tmp_path / "camera.yaml"
    profile.save(path)
    loaded = CameraProfile.load(path)
    assert loaded.profile_hash == profile.profile_hash
    changed = extract_camera_profile(fixture_config(width=640))
    with pytest.raises(CameraContractError, match=r"rgb\.width"):
        assert_camera_compatible(profile, changed)
    assert assert_camera_compatible(profile, changed, allow_mismatch=True)


def test_camera_hash_excludes_extraction_timestamp() -> None:
    profile = extract_camera_profile(
        fixture_config(),
        provenance={"source_config_sha256": "abc", "extracted_at": "first"},
    )
    later = replace(
        profile,
        provenance={"source_config_sha256": "abc", "extracted_at": "later"},
    )
    assert later.profile_hash == profile.profile_hash


def test_camera_compatibility_includes_observation_transforms() -> None:
    profile = official_2023_profile()
    transformed = replace(
        profile,
        observation_transforms={"center_cropper": {"height": 512, "width": 384}},
    )
    with pytest.raises(CameraContractError, match="observation_transforms"):
        assert_camera_compatible(profile, transformed)
