"""Deterministic report plots generated from structured metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_evaluation_plots(report: Dict[str, Any], output: Path) -> None:
    """Save confusion, IoU, scene, reliability, and risk-coverage plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plot
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        return
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(report["global"]["confusion_matrix"])
    normalized = np.asarray(report["global"]["row_normalized_confusion_matrix"])
    for name, matrix in (("confusion_raw", raw), ("confusion_row_normalized", normalized)):
        figure, axes = plot.subplots(figsize=(12, 10))
        image = axes.imshow(matrix, interpolation="nearest", cmap="magma")
        axes.set_xlabel("Predicted class")
        axes.set_ylabel("Ground-truth class")
        axes.set_title(name.replace("_", " "))
        figure.colorbar(image, ax=axes)
        figure.tight_layout()
        figure.savefig(plots / f"{name}.png", dpi=160)
        plot.close(figure)

    classes = report["global"]["per_class"]
    figure, axes = plot.subplots(figsize=(14, 5))
    axes.bar(
        [item["name"] for item in classes],
        [np.nan if item["iou"] is None else item["iou"] for item in classes],
    )
    axes.tick_params(axis="x", rotation=75)
    axes.set_ylabel("IoU")
    axes.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(plots / "per_class_iou.png", dpi=160)
    plot.close(figure)

    scene_items = sorted(
        (
            (scene_id, value)
            for scene_id, value in report["scene_macro"][
                "per_scene_known_class_miou"
            ].items()
            if value is not None
        ),
        key=lambda item: float(item[1]),
    )
    scene_values = [float(value) for _, value in scene_items]
    scene_height = min(12.0, max(4.5, 1.8 + 0.28 * len(scene_items)))
    figure, axes = plot.subplots(figsize=(9, scene_height))
    positions = np.arange(len(scene_items))
    axes.hlines(positions, 0, scene_values, color="tab:blue", alpha=0.35)
    axes.scatter(scene_values, positions, color="tab:blue", zorder=3)
    axes.set_yticks(positions)
    axes.set_yticklabels([scene_id for scene_id, _ in scene_items], fontsize=8)
    axes.set_xlim(0, 1)
    axes.set_xlabel("Known-class mIoU within one scene")
    axes.set_ylabel("Development scene (sorted by score)")
    axes.set_title(
        "Per-scene known-class mIoU\n"
        "Pool frames within a scene; macro-average its present known classes"
    )
    if scene_values:
        axes.axvline(
            float(np.mean(scene_values)),
            color="tab:orange",
            linestyle="--",
            label="equal-scene mean",
        )
        axes.axvline(
            float(np.median(scene_values)),
            color="tab:green",
            linestyle=":",
            label="scene median",
        )
        axes.legend()
    axes.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(plots / "per_scene_miou.png", dpi=160)
    plot.close(figure)

    figure, axes = plot.subplots(figsize=(7, 4))
    axes.hist(scene_values, bins=min(20, max(1, len(scene_values))))
    axes.set_xlabel("Known-class mIoU within one scene")
    axes.set_ylabel("Number of scenes")
    axes.set_title(
        f"Distribution across {len(scene_values)} scenes "
        "(one aggregate score per scene)"
    )
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(plots / "per_scene_miou_distribution.png", dpi=160)
    plot.close(figure)

    bins = report["probability_quality"]["bins"]
    figure, axes = plot.subplots(figsize=(6, 6))
    axes.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes.plot(
        [item["confidence"] for item in bins if item["count"]],
        [item["accuracy"] for item in bins if item["count"]],
        marker="o",
    )
    axes.set_xlabel("Confidence")
    axes.set_ylabel("Accuracy")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(plots / "reliability.png", dpi=160)
    plot.close(figure)

    curve = [
        item
        for item in report["probability_quality"]["risk_coverage"]
        if item["risk"] is not None
    ]
    coverage = [float(item["coverage"]) for item in curve]
    risk = [float(item["risk"]) for item in curve]
    thresholds = [float(item["minimum_confidence"]) for item in curve]
    figure, (axes, table_axes) = plot.subplots(
        1,
        2,
        figsize=(12, 5.8),
        gridspec_kw={"width_ratios": [3.0, 1.25]},
    )
    axes.plot(coverage, risk, color="tab:blue", alpha=0.55)
    points = axes.scatter(
        coverage,
        risk,
        c=thresholds,
        cmap="viridis",
        vmin=0,
        vmax=1,
        zorder=3,
    )
    axes.set_xlabel("Coverage")
    axes.set_ylabel("Risk (1 - accuracy)")
    axes.set_title("Retain pixels with softmax confidence ≥ threshold t")
    axes.grid(alpha=0.2)
    colorbar = figure.colorbar(points, ax=axes, pad=0.02)
    colorbar.set_label("Minimum confidence threshold t")
    table_axes.axis("off")
    table_axes.set_title("Exact operating points", fontsize=10, pad=8)
    operating_points = table_axes.table(
        cellText=[
            [f"{threshold:.2f}", f"{x_value:.3f}", f"{y_value:.3f}"]
            for threshold, x_value, y_value in zip(thresholds, coverage, risk)
        ],
        colLabels=["threshold t", "coverage", "risk"],
        cellLoc="center",
        loc="center",
    )
    operating_points.auto_set_font_size(False)
    operating_points.set_fontsize(8)
    operating_points.scale(1.0, 1.18)
    figure.tight_layout()
    figure.savefig(plots / "risk_coverage.png", dpi=160)
    plot.close(figure)
