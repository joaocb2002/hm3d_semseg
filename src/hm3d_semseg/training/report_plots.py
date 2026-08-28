"""Human-facing plots assembled from complete training/evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from hm3d_semseg.taxonomy.constants import OBJECTNAV_SIX
from hm3d_semseg.training.plots import save_training_plots


def save_report_plots(
    metrics_path: Path,
    evaluations: Sequence[Tuple[int, Dict[str, Any]]],
    output: Path,
) -> List[Path]:
    """Create a compact plot suite; return nothing when matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plot
    except ImportError:
        return []
    created = save_training_plots(metrics_path, output)
    if not evaluations:
        return created
    output.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray([epoch + 1 for epoch, _ in evaluations])

    figure, axes = plot.subplots(3, 1, figsize=(12, 12), sharex=True)
    training_epochs = _training_epoch_losses(metrics_path)
    if training_epochs:
        axes[0].plot(
            [epoch + 1 for epoch, _ in training_epochs],
            [value for _, value in training_epochs],
            marker="o",
            label="training objective CE (mean over minibatches)",
        )
    axes[0].plot(
        epochs,
        [_number(report.get("mean_cross_entropy_loss")) for _, report in evaluations],
        marker="o",
        label="development CE (pooled valid pixels)",
    )
    axes[0].set_ylabel("Cross-entropy")
    axes[0].set_title(
        "Loss: training optimization objective versus unweighted held-out pixel NLL"
    )
    axes[0].legend()
    axes[1].plot(
        epochs,
        [_global(report, "known_class_miou") for _, report in evaluations],
        marker="o",
        label="global known-class mIoU (macro over classes)",
    )
    axes[1].plot(
        epochs,
        [_global(report, "objectnav_six_miou") for _, report in evaluations],
        marker="o",
        label="global ObjectNav-six mIoU (macro over goal classes)",
    )
    axes[1].plot(
        epochs,
        [_number(report["scene_macro"].get("mean")) for _, report in evaluations],
        marker="o",
        label="scene-macro known-class mIoU (macro over scenes)",
    )
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0, 1)
    axes[1].set_title(
        "Global pools every development pixel first; scene macro equally weights "
        "per-scene mIoUs"
    )
    axes[1].legend(fontsize=9)
    axes[2].plot(
        epochs,
        [_global(report, "overall_pixel_accuracy") for _, report in evaluations],
        marker="o",
        label="global pixel accuracy (micro over valid pixels)",
    )
    axes[2].plot(
        epochs,
        [_global(report, "mean_class_recall") for _, report in evaluations],
        marker="o",
        label="global mean class recall (macro over present classes)",
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Both values come from one pooled development confusion matrix")
    axes[2].legend(fontsize=9)
    _mark_checkpoints(axes, evaluations)
    figure.suptitle("Learning and held-out generalization")
    figure.tight_layout()
    created.append(_save(plot, figure, output / "learning_and_generalization.png"))

    figure, axes = plot.subplots(3, 1, figsize=(10, 10), sharex=True)
    probability_keys = (
        ("nll", "Negative log-likelihood"),
        ("multiclass_brier", "Multiclass Brier score"),
        ("ece", "Expected calibration error"),
    )
    for axis, (key, label) in zip(axes, probability_keys):
        axis.plot(
            epochs,
            [_number(report["probability_quality"].get(key)) for _, report in evaluations],
            marker="o",
        )
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Epoch")
    figure.suptitle("Development probability quality (temperature = 1)")
    figure.tight_layout()
    created.append(_save(plot, figure, output / "probability_quality.png"))

    per_class = [report["global"]["per_class"] for _, report in evaluations]
    class_names = [item["name"] for item in per_class[0]][1:]
    heatmap = np.asarray(
        [
            [np.nan if item["iou"] is None else float(item["iou"]) for item in row[1:]]
            for row in per_class
        ]
    )
    figure, axes = plot.subplots(figsize=(16, max(5, len(evaluations) * 0.35)))
    image = axes.imshow(heatmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    axes.set_xticks(np.arange(len(class_names)))
    axes.set_xticklabels(class_names, rotation=70, ha="right")
    axes.set_yticks(np.arange(len(epochs)))
    axes.set_yticklabels(epochs)
    axes.set_xlabel("Known class")
    axes.set_ylabel("Epoch")
    axes.set_title("Development class IoU through training")
    figure.colorbar(image, ax=axes, label="IoU")
    figure.tight_layout()
    created.append(_save(plot, figure, output / "class_iou_heatmap.png"))

    figure, axes = plot.subplots(figsize=(11, 6))
    for goal in OBJECTNAV_SIX:
        axes.plot(
            epochs,
            [
                _number(report["global"]["objectnav_six"][goal].get("iou"))
                for _, report in evaluations
            ],
            marker="o",
            label=goal,
        )
    axes.set_xlabel("Epoch")
    axes.set_ylabel("IoU")
    axes.set_ylim(0, 1)
    axes.set_title("ObjectNav goal-class IoU through training")
    axes.legend(ncol=3)
    figure.tight_layout()
    created.append(_save(plot, figure, output / "objectnav_six_over_time.png"))

    scene_names = sorted(
        {
            scene
            for _, report in evaluations
            for scene in report["scene_macro"]["per_scene_known_class_miou"]
        }
    )
    scene_heatmap = np.asarray(
        [
            [
                _number(report["scene_macro"]["per_scene_known_class_miou"].get(scene))
                for scene in scene_names
            ]
            for _, report in evaluations
        ]
    )
    figure, axes = plot.subplots(
        figsize=(max(10, len(scene_names) * 0.55), max(5, len(evaluations) * 0.35))
    )
    image = axes.imshow(scene_heatmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    axes.set_xticks(np.arange(len(scene_names)))
    axes.set_xticklabels(scene_names, rotation=70, ha="right")
    axes.set_yticks(np.arange(len(epochs)))
    axes.set_yticklabels(epochs)
    axes.set_xlabel("Development scene")
    axes.set_ylabel("Epoch")
    axes.set_title("Scene known-class mIoU through training")
    figure.colorbar(image, ax=axes, label="mIoU")
    figure.tight_layout()
    created.append(_save(plot, figure, output / "scene_miou_heatmap.png"))

    best_epoch, best = max(
        evaluations,
        key=lambda item: float(item[1]["global"]["known_class_miou"]),
    )
    classes = best["global"]["per_class"][1:]
    figure, axes = plot.subplots(figsize=(15, 6))
    axes.bar(
        [item["name"] for item in classes],
        [np.nan if item["iou"] is None else item["iou"] for item in classes],
    )
    axes.tick_params(axis="x", rotation=70)
    axes.set_ylabel("IoU")
    axes.set_ylim(0, 1)
    axes.set_title(f"Per-class development IoU at selected best epoch {best_epoch + 1}")
    figure.tight_layout()
    created.append(_save(plot, figure, output / "best_per_class_iou.png"))
    normalized = best["global"].get("row_normalized_confusion_matrix")
    if normalized is not None:
        matrix = np.asarray(normalized, dtype=np.float64)
        labels = [item["name"] for item in best["global"]["per_class"]]
        figure, axes = plot.subplots(figsize=(14, 12))
        image = axes.imshow(matrix, interpolation="nearest", cmap="magma", vmin=0, vmax=1)
        axes.set_xticks(np.arange(len(labels)))
        axes.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        axes.set_yticks(np.arange(len(labels)))
        axes.set_yticklabels(labels, fontsize=7)
        axes.set_xlabel("Predicted class")
        axes.set_ylabel("Ground-truth class")
        axes.set_title(
            f"Row-normalized development confusion at selected epoch {best_epoch + 1}"
        )
        figure.colorbar(image, ax=axes, label="Fraction of ground-truth row")
        figure.tight_layout()
        created.append(
            _save(plot, figure, output / "best_confusion_row_normalized.png")
        )
    return created


def _training_epoch_losses(metrics_path: Path) -> List[Tuple[int, float]]:
    import json

    result = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("kind") == "train_epoch":
                result.append((int(item["epoch"]), float(item["loss"])))
    return result


def _global(report: Dict[str, Any], key: str) -> float:
    return _number(report["global"].get(key))


def _number(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _mark_checkpoints(axes: Any, evaluations: Sequence[Tuple[int, Dict[str, Any]]]) -> None:
    losses = [
        item for item in evaluations if item[1].get("mean_cross_entropy_loss") is not None
    ]
    if losses:
        minimum_loss_epoch = min(
            losses, key=lambda item: float(item[1]["mean_cross_entropy_loss"])
        )[0]
        axes[0].axvline(
            minimum_loss_epoch + 1,
            linestyle="--",
            color="tab:orange",
            alpha=0.6,
            label="minimum development CE",
        )
        axes[0].legend()
    best_epoch = max(
        evaluations,
        key=lambda item: float(item[1]["global"]["known_class_miou"]),
    )[0]
    axes[1].axvline(
        best_epoch + 1,
        linestyle="--",
        color="tab:green",
        alpha=0.6,
        label="selected best mIoU",
    )
    axes[1].legend()


def _save(plot: Any, figure: Any, path: Path) -> Path:
    figure.savefig(path, dpi=160)
    plot.close(figure)
    return path
