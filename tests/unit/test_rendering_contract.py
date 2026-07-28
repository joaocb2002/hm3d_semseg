from __future__ import annotations

import numpy as np
import pytest

from hm3d_semseg.camera.profile import (
    AgentProfile,
    CameraProfile,
    PitchProfile,
    SensorProfile,
)
from hm3d_semseg.rendering.habitat import postprocess_depth

pytestmark = pytest.mark.unit


def camera(normalize_depth: bool) -> CameraProfile:
    sensor = SensorProfile(
        uuid="rgb",
        sensor_type="HabitatSimRGBSensor",
        width=2,
        height=2,
        hfov=79.0,
        position=[0.0, 0.88, 0.0],
        orientation=[0.0, 0.0, 0.0],
    )
    depth = SensorProfile(
        uuid="depth",
        sensor_type="HabitatSimDepthSensor",
        width=2,
        height=2,
        hfov=79.0,
        position=[0.0, 0.88, 0.0],
        orientation=[0.0, 0.0, 0.0],
        min_depth=0.5,
        max_depth=5.0,
        normalize_depth=normalize_depth,
    )
    return CameraProfile(
        schema_version="1.0",
        rgb=sensor,
        depth=depth,
        semantic=None,
        agent=AgentProfile("main_agent", 0.88, 0.18),
        pitch=PitchProfile(True, 30.0, None, None, "test"),
        observation_transforms={},
        provenance={},
    )


def test_metric_depth_is_clipped_like_habitat_lab() -> None:
    depth = np.asarray([[0.1, 0.5], [2.0, 8.0]], dtype=np.float32)
    result = postprocess_depth(depth, camera(normalize_depth=False))
    assert result.dtype == np.float32
    assert np.array_equal(result, np.asarray([[0.5, 0.5], [2.0, 5.0]]))


def test_normalized_depth_matches_objectnav_range() -> None:
    depth = np.asarray([[0.1, 0.5], [2.75, 8.0]], dtype=np.float32)
    result = postprocess_depth(depth, camera(normalize_depth=True))
    assert np.allclose(result, np.asarray([[0.0, 0.0], [0.5, 1.0]]))
