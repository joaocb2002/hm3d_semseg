from pathlib import Path

import pytest

from hm3d_semseg.training.run_directory import allocate_run_directory

pytestmark = pytest.mark.unit


def test_fresh_runs_get_collision_safe_sequence_names(tmp_path: Path) -> None:
    first = allocate_run_directory(tmp_path, "overfit_tiny", resuming=False)
    second = allocate_run_directory(tmp_path, "overfit_tiny", resuming=False)
    third = allocate_run_directory(tmp_path, "overfit_tiny", resuming=False)

    assert first.name == "overfit_tiny"
    assert second.name == "overfit_tiny-002"
    assert third.name == "overfit_tiny-003"


def test_resume_reuses_requested_run_directory(tmp_path: Path) -> None:
    expected = tmp_path / "baseline"
    expected.mkdir()

    assert (
        allocate_run_directory(tmp_path, "baseline", resuming=True)
        == expected
    )
