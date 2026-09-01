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
    """Create the organized plot suite; omit plots when matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plot
    except ImportError:
        return []

    roots = {
        name: output / name
        for name in (
            "overview",
            "segmentation",
            "classes_and_scenes",
            "probability",
            "optimization",
        )
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    created = save_training_plots(metrics_path, roots["optimization"])
    if not evaluations:
        return created

    created.extend(_save_overview(plot, metrics_path, evaluations, roots["overview"]))
    created.extend(_save_segmentation(plot, evaluations, roots["segmentation"]))
    created.extend(_save_probability(plot, evaluations, roots["probability"]))
    created.extend(
        _save_classes_and_scenes(
            plot, evaluations, roots["classes_and_scenes"]
        )
    )
    return created


def _save_overview(
    plot: Any,
    metrics_path: Path,
    evaluations: Sequence[Tuple[int, Dict[str, Any]]],
    output: Path,
) -> List[Path]:
    epochs = np.asarray([epoch + 1 for epoch, _ in evaluations])
    figure, axes = plot.subplots(3, 1, figsize=(12, 12), sharex=True)
    training_epochs = _training_epoch_losses(metrics_path)
    if training_epochs:
        axes[0].plot(
            [item["epoch"] + 1 for item in training_epochs],
            [item["objective"] for item in training_epochs],
            marker="o",
            label="training objective (mean over minibatches)",
        )
        if any(item["has_components"] for item in training_epochs):
            axes[0].plot(
                [item["epoch"] + 1 for item in training_epochs],
                [item["cross_entropy"] for item in training_epochs],
                marker=".",
                linestyle="--",
                label="training CE component (unscaled)",
            )
            axes[0].plot(
                [item["epoch"] + 1 for item in training_epochs],
                [item["lovasz"] for item in training_epochs],
                marker=".",
                linestyle=":",
                label="training Lovasz component (unscaled)",
            )
    axes[0].plot(
        epochs,
        [_number(report.get("mean_cross_entropy_loss")) for _, report in evaluations],
        marker="o",
        label="development CE / NLL (pooled valid pixels)",
    )
    axes[0].set_ylabel("Loss")
    axes[0].set_title(
        "Optimization objective versus unweighted held-out probability loss"
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
        label="scene-macro known-class mIoU (equal scene weight)",
    )
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Global pools pixels first; scene macro averages scene scores")
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
    axes[2].set_title("Both values use one pooled development confusion matrix")
    axes[2].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
    _mark_checkpoints(axes, evaluations)
    figure.suptitle("Learning and held-out generalization")
    figure.tight_layout()
    return [_save(plot, figure, output / "learning_and_generalization.png")]


def _save_segmentation(
    plot: Any,
    evaluations: Sequence[Tuple[int, Dict[str, Any]]],
    output: Path,
) -> List[Path]:
    epochs = np.asarray([epoch + 1 for epoch, _ in evaluations])
    global_miou = np.asarray(
        [_global(report, "known_class_miou") for _, report in evaluations]
    )
    scene_miou = np.asarray(
        [_number(report["scene_macro"].get("mean")) for _, report in evaluations]
    )
    objectnav = np.asarray(
        [_global(report, "objectnav_six_miou") for _, report in evaluations]
    )
    figure, axes = plot.subplots(3, 1, figsize=(12, 12), sharex=True)
    for values, label in (
        (global_miou, "global known-class mIoU"),
        (scene_miou, "scene-macro known-class mIoU"),
        (objectnav, "global ObjectNav-six mIoU"),
    ):
        axes[0].plot(epochs, values, marker="o", label=label)
        axes[1].plot(epochs, values, marker="o", label=label)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("IoU")
    axes[0].set_title("Development segmentation scores (full 0-1 scale)")
    finite_parts = [
        values[np.isfinite(values)]
        for values in (global_miou, scene_miou, objectnav)
        if np.isfinite(values).any()
    ]
    finite = np.concatenate(finite_parts) if finite_parts else np.asarray([])
    if finite.size:
        spread = max(float(finite.max() - finite.min()), 0.02)
        axes[1].set_ylim(
            max(0.0, float(finite.min()) - 0.15 * spread),
            min(1.0, float(finite.max()) + 0.15 * spread),
        )
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Same mIoU curves (zoomed to expose plateaus and regressions)")
    axes[2].plot(
        epochs,
        [_global(report, "overall_pixel_accuracy") for _, report in evaluations],
        marker="o",
        label="global pixel accuracy",
    )
    axes[2].plot(
        epochs,
        [_global(report, "mean_class_recall") for _, report in evaluations],
        marker="o",
        label="global mean class recall",
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title("Pooled-pixel accuracy versus macro class recall")
    best_epoch = max(
        evaluations, key=lambda item: float(item[1]["global"]["known_class_miou"])
    )[0]
    for axis in axes:
        axis.axvline(best_epoch + 1, ls="--", color="tab:green", alpha=0.55)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    figure.tight_layout()
    created = [
        _save(
            plot,
            figure,
            output / "development_segmentation_metrics_over_time.png",
        )
    ]

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
    axes.set_ylabel("Global per-class IoU")
    axes.set_ylim(0, 1)
    axes.set_title("ObjectNav goal-class IoU through training")
    axes.grid(alpha=0.2)
    axes.legend(ncol=3)
    figure.tight_layout()
    created.append(_save(plot, figure, output / "objectnav_six_over_time.png"))
    return created


def _save_probability(
    plot: Any,
    evaluations: Sequence[Tuple[int, Dict[str, Any]]],
    output: Path,
) -> List[Path]:
    epochs = np.asarray([epoch + 1 for epoch, _ in evaluations])
    figure, axes = plot.subplots(4, 1, figsize=(11, 12), sharex=True)
    for axis, key, label in (
        (axes[0], "nll", "NLL / development cross-entropy"),
        (axes[1], "multiclass_brier", "Multiclass Brier score"),
        (axes[2], "ece", "Expected calibration error"),
    ):
        axis.plot(
            epochs,
            [_probability(report, key) for _, report in evaluations],
            marker="o",
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[3].plot(
        epochs,
        [_probability(report, "mean_entropy_correct") for _, report in evaluations],
        marker="o",
        label="correct pixels",
    )
    axes[3].plot(
        epochs,
        [_probability(report, "mean_entropy_incorrect") for _, report in evaluations],
        marker="o",
        label="incorrect pixels",
    )
    axes[3].set_ylabel("Mean predictive entropy")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(alpha=0.2)
    axes[3].legend()
    figure.suptitle("Development probability quality at temperature T = 1")
    figure.tight_layout()
    created = [
        _save(
            plot,
            figure,
            output / "development_probability_quality_over_time.png",
        )
    ]

    target_coverages = (0.70, 0.80, 0.90, 0.95, 1.00)
    figure, axes = plot.subplots(figsize=(11, 6))
    plotted = False
    for coverage in target_coverages:
        values = [_risk_near_coverage(report, coverage) for _, report in evaluations]
        if np.isfinite(np.asarray(values, dtype=np.float64)).any():
            axes.plot(epochs, values, marker="o", label=f"coverage ≈ {coverage:.0%}")
            plotted = True
    if plotted:
        axes.set_xlabel("Epoch")
        axes.set_ylabel("Risk (1 - pixel accuracy among retained pixels)")
        axes.set_title("Selective prediction risk at fixed development coverage")
        axes.grid(alpha=0.2)
        axes.legend(ncol=2)
        figure.tight_layout()
        created.append(
            _save(plot, figure, output / "risk_at_fixed_coverage_over_time.png")
        )
    else:
        plot.close(figure)

    _, best = max(
        evaluations, key=lambda item: float(item[1]["global"]["known_class_miou"])
    )
    bins = best.get("probability_quality", {}).get("bins", [])
    if bins:
        figure, axes = plot.subplots(figsize=(7, 7))
        axes.plot([0, 1], [0, 1], "--", color="black", label="perfect calibration")
        axes.plot(
            [_number(item.get("confidence")) for item in bins],
            [_number(item.get("accuracy")) for item in bins],
            marker="o",
            label="development bins",
        )
        axes.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy")
        axes.set_title("Best-mIoU checkpoint reliability diagram (T = 1)")
        axes.grid(alpha=0.2)
        axes.legend()
        figure.tight_layout()
        created.append(_save(plot, figure, output / "best_reliability.png"))

    risk = best.get("probability_quality", {}).get("risk_coverage", [])
    if risk:
        figure, (axis, table_axis) = plot.subplots(
            1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.55, 1.0]}
        )
        thresholds = np.asarray(
            [_number(item.get("minimum_confidence")) for item in risk]
        )
        scatter = axis.scatter(
            [_number(item.get("coverage")) for item in risk],
            [_number(item.get("risk")) for item in risk],
            c=thresholds,
            cmap="viridis",
            vmin=0,
            vmax=1,
            zorder=3,
        )
        axis.plot(
            [_number(item.get("coverage")) for item in risk],
            [_number(item.get("risk")) for item in risk],
            alpha=0.55,
        )
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Risk (1 - accuracy)")
        axis.set_title("Retain pixels with softmax confidence ≥ threshold t")
        axis.grid(alpha=0.2)
        figure.colorbar(scatter, ax=axis, label="Minimum confidence threshold t")
        table_axis.axis("off")
        table_axis.set_title("Exact operating points")
        table_axis.table(
            cellText=[
                [
                    f"{_number(item.get('minimum_confidence')):.2f}",
                    f"{_number(item.get('coverage')):.3f}",
                    f"{_number(item.get('risk')):.3f}",
                ]
                for item in risk
            ],
            colLabels=["threshold t", "coverage", "risk"],
            loc="center",
            cellLoc="center",
        )
        figure.tight_layout()
        created.append(_save(plot, figure, output / "best_risk_coverage.png"))
    return created


def _save_classes_and_scenes(
    plot: Any,
    evaluations: Sequence[Tuple[int, Dict[str, Any]]],
    output: Path,
) -> List[Path]:
    epochs = np.asarray([epoch + 1 for epoch, _ in evaluations])
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
    axes.set_title("Global development per-class IoU (pixels pooled across scenes)")
    figure.colorbar(image, ax=axes, label="IoU")
    figure.tight_layout()
    created = [_save(plot, figure, output / "class_iou_heatmap.png")]

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
    image = axes.imshow(
        scene_heatmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1
    )
    axes.set_xticks(np.arange(len(scene_names)))
    axes.set_xticklabels(scene_names, rotation=70, ha="right")
    axes.set_yticks(np.arange(len(epochs)))
    axes.set_yticklabels(epochs)
    axes.set_xlabel("Held-out development scene")
    axes.set_ylabel("Epoch")
    axes.set_title("Per-scene known-class mIoU (frames pooled within each scene)")
    figure.colorbar(image, ax=axes, label="mIoU")
    figure.tight_layout()
    created.append(_save(plot, figure, output / "scene_miou_heatmap.png"))

    best_epoch, best = max(
        evaluations, key=lambda item: float(item[1]["global"]["known_class_miou"])
    )
    classes = [
        item for item in best["global"]["per_class"][1:] if item["iou"] is not None
    ]
    classes.sort(key=lambda item: float(item["iou"]))
    figure, axes = plot.subplots(figsize=(11, max(7, len(classes) * 0.28)))
    y = np.arange(len(classes))
    axes.barh(y - 0.24, [item["iou"] for item in classes], height=0.24, label="IoU")
    axes.barh(
        y,
        [_number(item.get("precision")) for item in classes],
        height=0.24,
        label="precision",
    )
    axes.barh(
        y + 0.24,
        [_number(item.get("recall")) for item in classes],
        height=0.24,
        label="recall",
    )
    axes.set_yticks(y)
    axes.set_yticklabels([item["name"] for item in classes])
    axes.set_xlim(0, 1)
    axes.set_xlabel("Global pooled-class score")
    axes.set_title(f"Class precision, recall, and IoU at selected epoch {best_epoch + 1}")
    axes.grid(axis="x", alpha=0.2)
    axes.legend(ncol=3)
    figure.tight_layout()
    created.append(_save(plot, figure, output / "best_class_precision_recall_iou.png"))

    matrices = (
        ("confusion_matrix", "best_confusion_raw.png", "log1p pixel count", True),
        (
            "row_normalized_confusion_matrix",
            "best_confusion_row_normalized.png",
            "Fraction of ground-truth row",
            False,
        ),
    )
    labels = [item["name"] for item in best["global"]["per_class"]]
    for key, filename, color_label, logarithmic in matrices:
        value = best["global"].get(key)
        if value is None:
            continue
        matrix = np.asarray(value, dtype=np.float64)
        shown = np.log1p(matrix) if logarithmic else matrix
        figure, axes = plot.subplots(figsize=(14, 12))
        image = axes.imshow(
            shown,
            interpolation="nearest",
            cmap="magma",
            vmin=0,
            vmax=None if logarithmic else 1,
        )
        axes.set_xticks(np.arange(len(labels)))
        axes.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        axes.set_yticks(np.arange(len(labels)))
        axes.set_yticklabels(labels, fontsize=7)
        axes.set_xlabel("Predicted class")
        axes.set_ylabel("Ground-truth class")
        axes.set_title(f"Development confusion at selected epoch {best_epoch + 1}")
        figure.colorbar(image, ax=axes, label=color_label)
        figure.tight_layout()
        created.append(_save(plot, figure, output / filename))

    best_scenes = best["scene_macro"]["per_scene_known_class_miou"]
    ordered = sorted(
        ((name, value) for name, value in best_scenes.items() if value is not None),
        key=lambda item: float(item[1]),
    )
    if ordered:
        figure, axes = plot.subplots(figsize=(11, max(5, len(ordered) * 0.42)))
        axes.barh([name for name, _ in ordered], [value for _, value in ordered])
        axes.set_xlim(0, 1)
        axes.set_xlabel("Per-scene known-class mIoU")
        axes.set_title(f"Held-out scenes at selected epoch {best_epoch + 1}")
        axes.grid(axis="x", alpha=0.2)
        figure.tight_layout()
        created.append(_save(plot, figure, output / "best_per_scene_miou.png"))

        figure, axes = plot.subplots(figsize=(9, 5))
        axes.hist([value for _, value in ordered], bins=min(12, len(ordered)))
        axes.set_xlabel("Per-scene known-class mIoU")
        axes.set_ylabel("Scenes")
        axes.set_title("Distribution across held-out scenes at selected checkpoint")
        axes.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        created.append(
            _save(plot, figure, output / "best_scene_miou_distribution.png")
        )

    probability_epoch, probability_best = min(
        evaluations,
        key=lambda item: float(item[1].get("mean_cross_entropy_loss", float("inf"))),
    )
    if probability_epoch != best_epoch:
        first = probability_best["scene_macro"]["per_scene_known_class_miou"]
        common = [
            name
            for name in sorted(set(first) & set(best_scenes))
            if first[name] is not None and best_scenes[name] is not None
        ]
        if common:
            figure, axes = plot.subplots(figsize=(12, max(6, len(common) * 0.43)))
            y = np.arange(len(common))
            left = np.asarray([first[name] for name in common], dtype=np.float64)
            right = np.asarray([best_scenes[name] for name in common], dtype=np.float64)
            for index in range(len(common)):
                axes.plot(
                    [left[index], right[index]],
                    [y[index], y[index]],
                    color="#a9b5c2",
                )
            axes.scatter(
                left, y, label=f"epoch {probability_epoch + 1} (minimum NLL)"
            )
            axes.scatter(right, y, label=f"epoch {best_epoch + 1} (best global mIoU)")
            axes.set_yticks(y)
            axes.set_yticklabels(common)
            axes.set_xlim(0, 1)
            axes.set_xlabel("Per-scene known-class mIoU")
            axes.set_title("Held-out scene changes between checkpoint criteria")
            axes.grid(axis="x", alpha=0.2)
            axes.legend()
            figure.tight_layout()
            created.append(
                _save(plot, figure, output / "per_scene_checkpoint_change.png")
            )
    return created


def _training_epoch_losses(metrics_path: Path) -> List[Dict[str, Any]]:
    import json

    result = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("kind") == "train_epoch":
                result.append(
                    {
                        "epoch": int(item["epoch"]),
                        "objective": float(
                            item.get("objective_loss", item["loss"])
                        ),
                        "cross_entropy": float(
                            item.get("cross_entropy_loss", item["loss"])
                        ),
                        "lovasz": float(item.get("lovasz_loss", 0.0)),
                        "has_components": float(
                            item.get("lovasz_weight", 0.0)
                        ) > 0.0,
                    }
                )
    return result


def _global(report: Dict[str, Any], key: str) -> float:
    return _number(report["global"].get(key))


def _probability(report: Dict[str, Any], key: str) -> float:
    return _number(report.get("probability_quality", {}).get(key))


def _number(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _risk_near_coverage(report: Dict[str, Any], target: float) -> float:
    rows = report.get("probability_quality", {}).get("risk_coverage", [])
    usable = [
        item
        for item in rows
        if item.get("coverage") is not None and item.get("risk") is not None
    ]
    if not usable:
        return float("nan")
    selected = min(usable, key=lambda item: abs(float(item["coverage"]) - target))
    return float(selected["risk"])


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
