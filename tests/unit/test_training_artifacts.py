from pathlib import Path

import pytest

from hm3d_semseg.training.loop import _prepare_run_directories

pytestmark = pytest.mark.unit


def test_training_artifact_hierarchy_separates_progress_and_subset_diagnostics(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"

    _prepare_run_directories(run)

    assert (run / "checkpoints").is_dir()
    assert (run / "tensorboard").is_dir()
    assert (run / "plots").is_dir()
    assert (run / "diagnostics" / "training_progress" / "qualitative").is_dir()
    assert not (run / "qualitative").exists()
