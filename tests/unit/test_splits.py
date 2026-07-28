import json
from pathlib import Path

import pytest

from hm3d_semseg.data.splits import (
    make_calibration_split,
    make_development_split,
)

pytestmark = pytest.mark.unit


def test_scene_splits_are_disjoint_and_reproducible(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {"scene_class_coverage": {f"scene-{index:03d}": {"wall": 1} for index in range(40)}}
        )
    )
    first = make_development_split(audit, tmp_path / "dev-a", 7, 5)
    second = make_development_split(audit, tmp_path / "dev-b", 7, 5)
    assert first["development"] == second["development"]
    assert not set(first["fit"]) & set(first["development"])
    calibration = make_calibration_split(audit, tmp_path / "calibration", 8, 12)
    assert calibration["disjoint"]
    assert len(calibration["calibration_fit"]) == 12
    assert len(calibration["calibration_evaluation"]) == 28
