from pathlib import Path

import pytest

from hm3d_semseg.training.loop import (
    _configure_tensorboard_layout,
    _prepare_run_directories,
)

pytestmark = pytest.mark.unit


def test_training_artifact_hierarchy_separates_progress_and_subset_diagnostics(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"

    _prepare_run_directories(run)

    assert (run / "checkpoints").is_dir()
    assert (run / "tensorboard").is_dir()
    assert (run / "plots").is_dir()
    qualitative = run / "diagnostics" / "training_progress" / "qualitative"
    assert (qualitative / "train").is_dir()
    assert (qualitative / "development").is_dir()
    assert (run / "report" / "tables").is_dir()
    assert (run / "report" / "plots").is_dir()
    assert not (run / "qualitative").exists()


def test_tensorboard_layout_groups_generalization_and_optimizer_health() -> None:
    class Writer:
        layout = None

        def add_custom_scalars(self, layout: object) -> None:
            self.layout = layout

    writer = Writer()

    _configure_tensorboard_layout(writer)

    assert writer.layout is not None
    assert "Generalization" in writer.layout
    assert "Optimization" in writer.layout
