"""Spatially separated navigable pose sampling."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from hm3d_semseg.config.schema import SamplingConfig


@dataclass(frozen=True)
class CameraPose:
    position: List[float]
    yaw_degrees: float
    pitch_degrees: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def scene_seed(global_seed: int, scene_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{scene_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


class PoseSampler:
    """Poisson-like position filtering with deterministic headings."""

    def __init__(self, config: SamplingConfig) -> None:
        self.config = config

    def sample(
        self,
        scene_id: str,
        random_navigable_point: Callable[[], Sequence[float]],
        pitch_degrees: Sequence[float],
        max_samples: int,
    ) -> List[CameraPose]:
        if not pitch_degrees:
            raise ValueError("At least one explicit pitch value is required")
        rng = np.random.default_rng(scene_seed(self.config.seed, scene_id))
        positions: List[np.ndarray] = []
        attempts = 0
        max_attempts = self.config.positions_per_scene * self.config.max_attempts_per_position
        while len(positions) < self.config.positions_per_scene and attempts < max_attempts:
            candidate = np.asarray(random_navigable_point(), dtype=np.float64)
            attempts += 1
            if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
                continue
            if all(
                abs(candidate[1] - prior[1]) >= self.config.floor_separation_m
                or np.linalg.norm(candidate[[0, 2]] - prior[[0, 2]])
                >= self.config.min_position_distance_m
                for prior in positions
            ):
                positions.append(candidate)
        poses: List[CameraPose] = []
        yaw_offset = float(rng.uniform(0.0, 360.0))
        yaw_step = 360.0 / self.config.yaws_per_position
        for position_index, position in enumerate(positions):
            for heading in range(self.config.yaws_per_position):
                yaw = math.fmod(
                    yaw_offset + position_index * yaw_step / 2.0 + heading * yaw_step,
                    360.0,
                )
                pitch = float(pitch_degrees[(position_index + heading) % len(pitch_degrees)])
                poses.append(
                    CameraPose(position=position.tolist(), yaw_degrees=yaw, pitch_degrees=pitch)
                )
                if len(poses) >= max_samples:
                    return poses
        return poses
