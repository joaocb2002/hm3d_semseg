from pathlib import Path

import pytest

from hm3d_semseg.training.loop import (
    _configure_tensorboard_layout,
    _learning_rate_scale,
    _prepare_run_directories,
    _resolve_optimizer_steps,
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


def test_official_iteration_schedule_warms_up_then_decays_to_zero() -> None:
    arguments = {
        "total_steps": 160_000,
        "warmup_steps": 1_500,
        "warmup_start_factor": 1e-6,
        "schedule": "polynomial",
        "polynomial_power": 1.0,
    }

    assert _learning_rate_scale(0, **arguments) == pytest.approx(1e-6)
    assert _learning_rate_scale(1_499, **arguments) == pytest.approx(1.0)
    assert 0.0 < _learning_rate_scale(80_000, **arguments) < 1.0
    assert _learning_rate_scale(160_000, **arguments) == pytest.approx(0.0)
    assert _resolve_optimizer_steps(
        epochs=50, steps_per_epoch=3_201, maximum=160_000
    ) == 160_000
