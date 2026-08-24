import json
from pathlib import Path

import pytest

from hm3d_semseg.training.reporting import summarize_training_metrics

pytestmark = pytest.mark.unit


def _write_records(path: Path) -> None:
    records = [
        {
            "kind": "train_step",
            "epoch": 0,
            "step": 1,
            "loss": 2.0,
            "gradient_norm": 3.0,
            "learning_rates": [1e-4, 1e-3],
            "samples": 2,
            "step_seconds": 2.0,
            "samples_per_second": 1.0,
            "gpu_peak_memory_bytes": 1024,
        },
        {
            "kind": "train_step",
            "epoch": 0,
            "step": 2,
            "loss": 1.0,
            "gradient_norm": 4.0,
            "learning_rates": [5e-5, 5e-4],
            "samples": 2,
            "step_seconds": 1.0,
            "samples_per_second": 2.0,
            "gpu_peak_memory_bytes": 2048,
        },
        {"kind": "train_epoch", "epoch": 0, "loss": 1.5},
        {
            "kind": "development_epoch",
            "epoch": 0,
            "loss": 1.25,
            "known_class_miou": 0.4,
        },
        {"kind": "train_epoch", "epoch": 1, "loss": 0.75},
        {
            "kind": "development_epoch",
            "epoch": 1,
            "loss": 1.0,
            "known_class_miou": 0.5,
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_training_metric_summary_preserves_important_extrema(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_records(metrics)

    summary = summarize_training_metrics(metrics)

    assert summary["training"]["optimizer_steps_recorded"] == 2
    assert summary["training"]["step_cross_entropy"]["initial"] == 2.0
    assert summary["training"]["step_cross_entropy"]["minimum"] == {
        "step": 2,
        "value": 1.0,
    }
    assert summary["training"]["gradient_norm"]["maximum"] == 4.0
    assert summary["training"]["optimization"][
        "overall_samples_per_second"
    ] == pytest.approx(4 / 3)
    assert summary["training"]["optimization"]["peak_gpu_memory_bytes"] == 2048
    assert summary["development"]["best_known_class_miou"] == {
        "epoch": 1,
        "value": 0.5,
    }


def test_training_plots_cover_optimization_and_development(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    from hm3d_semseg.training.plots import save_training_plots

    metrics = tmp_path / "metrics.jsonl"
    output = tmp_path / "plots"
    _write_records(metrics)

    created = save_training_plots(metrics, output)

    assert {path.name for path in created} == {
        "loss_and_learning_rate.png",
        "optimization_diagnostics.png",
        "development_metrics.png",
    }
    assert all(path.is_file() for path in created)
