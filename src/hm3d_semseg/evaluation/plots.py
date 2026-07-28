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

    scene_values = [
        value
        for value in report["scene_macro"]["per_scene_known_class_miou"].values()
        if value is not None
    ]
    figure, axes = plot.subplots(figsize=(7, 4))
    axes.hist(scene_values, bins=min(20, max(1, len(scene_values))))
    axes.set_xlabel("Known-class mIoU")
    axes.set_ylabel("Scenes")
    figure.tight_layout()
    figure.savefig(plots / "per_scene_miou.png", dpi=160)
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

    curve = report["probability_quality"]["risk_coverage"]
    figure, axes = plot.subplots(figsize=(7, 4))
    axes.plot(
        [item["coverage"] for item in curve if item["risk"] is not None],
        [item["risk"] for item in curve if item["risk"] is not None],
        marker="o",
    )
    axes.set_xlabel("Coverage")
    axes.set_ylabel("Risk (1 - accuracy)")
    figure.tight_layout()
    figure.savefig(plots / "risk_coverage.png", dpi=160)
    plot.close(figure)
