import json
from pathlib import Path

import pytest

from hm3d_semseg.taxonomy.constants import ID2LABEL, OBJECTNAV_SIX
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
    assert summary["training"]["gradient_norm"]["nonfinite_count"] == 0
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


def test_static_training_report_keeps_raw_metrics_authoritative(tmp_path: Path) -> None:
    from hm3d_semseg.training.report import generate_training_report

    metrics = tmp_path / "metrics.jsonl"
    _write_records(metrics)

    result = generate_training_report(tmp_path)

    assert metrics.is_file()
    assert Path(result["report"]).is_file()
    assert (tmp_path / "report" / "summary.md").is_file()
    assert (tmp_path / "report" / "tables" / "epoch_metrics.csv").is_file()
    assert (tmp_path / "report" / "report_manifest.json").is_file()
    assert "No development evaluation exists" in " ".join(result["warnings"])


def _write_evaluation(run: Path, miou: float, scene_value: float) -> None:
    classes = [
        {
            "id": class_id,
            "name": ID2LABEL[class_id],
            "support": 100,
            "predicted": 100,
            "iou": miou,
            "precision": miou,
            "recall": miou,
            "f1": miou,
        }
        for class_id in range(41)
    ]
    normalized = [[0.0] * 41 for _ in range(41)]
    for class_id in range(41):
        normalized[class_id][class_id] = miou
        normalized[class_id][(class_id + 1) % 41] = 1.0 - miou
    report = {
        "checkpoint": str(run / "checkpoints" / "best"),
        "dataset": str(run / "development"),
        "evaluation_samples": 10,
        "evaluation_scenes": 1,
        "temperature": 1.0,
        "mean_cross_entropy_loss": 1.0,
        "global": {
            "known_class_miou": miou,
            "miou_41": miou,
            "objectnav_six_miou": miou,
            "overall_pixel_accuracy": miou,
            "frequency_weighted_iou": miou,
            "mean_class_recall": miou,
            "unknown": {"iou": miou, "prevalence": 0.1},
            "per_class": classes,
            "row_normalized_confusion_matrix": normalized,
            "objectnav_six": {
                goal: {
                    "model_class": ID2LABEL[class_id],
                    "iou": miou,
                    "precision": miou,
                    "recall": miou,
                    "support": 100,
                }
                for goal, class_id in OBJECTNAV_SIX.items()
            },
        },
        "scene_macro": {
            "mean": scene_value,
            "median": scene_value,
            "bootstrap_95_percent_ci": [scene_value, scene_value],
            "per_scene_known_class_miou": {"scene": scene_value},
        },
        "probability_quality": {
            "nll": 1.0,
            "multiclass_brier": 0.3,
            "ece": 0.1,
        },
    }
    output = run / "evaluation-epoch-000"
    output.mkdir()
    (output / "summary.json").write_text(json.dumps(report), encoding="utf-8")


def test_explicit_evaluation_report_exposes_metrics_tables_and_confusions(
    tmp_path: Path,
) -> None:
    from hm3d_semseg.evaluation.reporting import generate_evaluation_report

    _write_evaluation(tmp_path, 0.4, 0.35)
    evaluation = tmp_path / "evaluation-epoch-000"

    result = generate_evaluation_report(evaluation)

    assert Path(result["report"]).is_file()
    assert (evaluation / "report" / "tables" / "per_class.csv").is_file()
    assert (evaluation / "report" / "tables" / "top_confusions.csv").is_file()


def test_compare_runs_uses_held_out_metrics_and_paired_scenes(tmp_path: Path) -> None:
    from hm3d_semseg.training.report import compare_training_runs

    runs = [tmp_path / "baseline", tmp_path / "balanced"]
    for index, run in enumerate(runs):
        run.mkdir()
        _write_records(run / "metrics.jsonl")
        _write_evaluation(run, 0.4 + index * 0.1, 0.35 + index * 0.1)
        (run / "provenance.json").write_text(
            json.dumps(
                {
                    "seed": 2027,
                    "model_id": "model",
                    "model_revision": "revision",
                    "hm3d_semseg_commit": "commit",
                    "train_dataset_validation": {"manifest_hash": "train"},
                    "development_dataset_validation": {"manifest_hash": "dev"},
                    "training_sample_ids": None,
                    "development_sample_ids": None,
                }
            ),
            encoding="utf-8",
        )

    result = compare_training_runs(runs, tmp_path / "comparison")

    assert result["comparable"] is True
    comparison = json.loads((tmp_path / "comparison" / "summary.json").read_text())
    assert comparison["training_loss_compared"] is False
    assert comparison["paired_comparison"]["mean_paired_scene_delta"] == pytest.approx(
        0.1
    )
    assert (tmp_path / "comparison" / "per_class_differences.csv").is_file()
