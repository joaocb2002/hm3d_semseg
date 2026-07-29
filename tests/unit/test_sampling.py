import numpy as np
import pytest

from hm3d_semseg.config.schema import SamplingConfig
from hm3d_semseg.sampling.poses import PoseSampler

pytestmark = pytest.mark.unit


def test_pose_sampling_is_reproducible_and_spatial() -> None:
    points = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.1, 0.0, 0.1]),
        np.asarray([2.0, 0.0, 0.0]),
    ]

    def run() -> list:
        index = 0

        def point() -> np.ndarray:
            nonlocal index
            value = points[index % len(points)]
            index += 1
            return value

        config = SamplingConfig(
            positions_per_scene=2,
            yaws_per_position=2,
            min_position_distance_m=1.0,
            max_attempts_per_position=10,
        )
        return PoseSampler(config).sample("scene", point, [-30, 0, 30], 4)

    assert run() == run()
    poses = run()
    assert len(poses) == 4
    assert (poses[2].yaw_degrees - poses[0].yaw_degrees) % 360.0 == pytest.approx(30.0)
