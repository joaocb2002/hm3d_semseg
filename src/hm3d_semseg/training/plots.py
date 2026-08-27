"""Readable training plots derived from the append-only JSONL log."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from hm3d_semseg.types import NumpyArray


def save_training_plots(metrics_path: Path, output: Path) -> List[Path]:
    """Save robustly aggregated optimization and epoch-level development plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plot
    except ImportError:
        return []
    output.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    data = load_plot_data(metrics_path)
    steps = data["step_bins"]
    if not steps:
        return created

    x = np.asarray([item["step"] for item in steps])
    figure, axes = plot.subplots(2, 1, figsize=(10, 8), sharex=True)
    loss_median = np.asarray([item["loss_median"] for item in steps])
    axes[0].plot(x, loss_median, label="median within step bin")
    axes[0].fill_between(
        x,
        [item["loss_p10"] for item in steps],
        [item["loss_p90"] for item in steps],
        alpha=0.22,
        label="10th-90th percentile",
    )
    axes[0].set_ylabel("Cross-entropy")
    axes[0].set_title("Training loss (aggregated; raw values remain in metrics.jsonl)")
    axes[0].legend()
    group_count = len(steps[0]["learning_rates"])
    for index in range(group_count):
        group_name = (
            ("pretrained", "classifier")[index] if group_count == 2 else f"group {index}"
        )
        axes[1].plot(
            x,
            [item["learning_rates"][index] for item in steps],
            label=f"{group_name} (group {index})",
        )
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Learning rate")
    axes[1].legend()
    figure.tight_layout()
    loss_and_lr = output / "loss_and_learning_rate.png"
    figure.savefig(loss_and_lr, dpi=160)
    plot.close(figure)
    created.append(loss_and_lr)

    has_gpu_memory = any(item.get("gpu_memory_median") is not None for item in steps)
    rows = 4 if has_gpu_memory else 3
    figure, axes = plot.subplots(rows, 1, figsize=(10, 2.7 * rows), sharex=True)
    finite_gradient = np.asarray([item["gradient_median"] for item in steps], dtype=np.float64)
    axes[0].plot(x, finite_gradient, label="finite median")
    axes[0].fill_between(
        x,
        [item["gradient_p10"] for item in steps],
        [item["gradient_p90"] for item in steps],
        alpha=0.22,
        label="finite 10th-90th percentile",
    )
    if np.isfinite(finite_gradient).any() and np.nanmax(finite_gradient) > 0:
        axes[0].set_yscale("log")
    nonfinite = sum(int(item["gradient_nonfinite_count"]) for item in steps)
    skipped = sum(int(item["optimizer_steps_skipped"]) for item in steps)
    axes[0].set_ylabel("Gradient norm")
    axes[0].set_title(
        f"Finite gradient norms; {nonfinite} non-finite records, "
        f"{skipped} AMP-skipped updates"
    )
    axes[0].legend()
    _plot_band(axes[1], x, steps, "throughput", "Samples/s")
    _plot_band(axes[2], x, steps, "step_seconds", "Seconds/step")
    if has_gpu_memory:
        axes[3].plot(x, [item["gpu_memory_median"] for item in steps])
        axes[3].set_ylabel("Peak GPU GiB")
    axes[-1].set_xlabel("Optimizer step")
    figure.tight_layout()
    diagnostics = output / "optimization_diagnostics.png"
    figure.savefig(diagnostics, dpi=160)
    plot.close(figure)
    created.append(diagnostics)

    development = data["development"]
    if development:
        train_epochs = data["train_epochs"]
        figure, axes = plot.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(
            [item["epoch"] + 1 for item in train_epochs],
            [item["loss"] for item in train_epochs],
            marker="o",
            label="training objective cross-entropy",
        )
        axes[0].plot(
            [item["epoch"] + 1 for item in development],
            [item["loss"] for item in development],
            marker="o",
            label="development cross-entropy",
        )
        axes[0].set_ylabel("Cross-entropy")
        axes[0].legend()
        axes[1].plot(
            [item["epoch"] + 1 for item in development],
            [item["known_class_miou"] for item in development],
            marker="o",
        )
        _mark_epoch_extrema(axes, development)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Development known-class mIoU")
        axes[1].set_ylim(0, 1)
        figure.tight_layout()
        development_metrics = output / "development_metrics.png"
        figure.savefig(development_metrics, dpi=160)
        plot.close(figure)
        created.append(development_metrics)
    return created


def load_plot_data(metrics_path: Path, *, maximum_step_bins: int = 1000) -> Dict[str, Any]:
    """Load small epoch series and aggregate a potentially huge step log."""
    step_count = 0
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"kind": "train_step"' in line or '"kind":"train_step"' in line:
                step_count += 1
    bin_size = max(1, math.ceil(step_count / maximum_step_bins))
    step_bins: List[Dict[str, Any]] = []
    bucket: List[Dict[str, Any]] = []
    train_epochs: List[Dict[str, Any]] = []
    development: List[Dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            kind = item.get("kind")
            if kind == "train_step":
                bucket.append(item)
                if len(bucket) >= bin_size:
                    step_bins.append(_aggregate_step_bucket(bucket))
                    bucket = []
            elif kind == "train_epoch":
                train_epochs.append(item)
            elif kind == "development_epoch":
                development.append(item)
    if bucket:
        step_bins.append(_aggregate_step_bucket(bucket))
    return {
        "step_bins": step_bins,
        "train_epochs": train_epochs,
        "development": development,
        "raw_step_count": step_count,
        "step_bin_size": bin_size,
    }


def _aggregate_step_bucket(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    loss = np.asarray([float(item["loss"]) for item in records])
    gradients = np.asarray([float(item["gradient_norm"]) for item in records])
    finite_gradients = gradients[np.isfinite(gradients)]
    throughput = np.asarray([float(item["samples_per_second"]) for item in records])
    durations = np.asarray([float(item["step_seconds"]) for item in records])
    memory = [
        float(item["gpu_peak_memory_bytes"]) / (1024**3)
        for item in records
        if item.get("gpu_peak_memory_bytes") is not None
    ]
    midpoint = records[len(records) // 2]
    return {
        "step": int(midpoint["step"]),
        "loss_median": float(np.median(loss)),
        "loss_p10": float(np.percentile(loss, 10)),
        "loss_p90": float(np.percentile(loss, 90)),
        "gradient_median": _finite_percentile(finite_gradients, 50),
        "gradient_p10": _finite_percentile(finite_gradients, 10),
        "gradient_p90": _finite_percentile(finite_gradients, 90),
        "gradient_nonfinite_count": int((~np.isfinite(gradients)).sum()),
        "optimizer_steps_skipped": sum(
            int(bool(item.get("optimizer_step_skipped", False))) for item in records
        ),
        "throughput_median": float(np.median(throughput)),
        "throughput_p10": float(np.percentile(throughput, 10)),
        "throughput_p90": float(np.percentile(throughput, 90)),
        "step_seconds_median": float(np.median(durations)),
        "step_seconds_p10": float(np.percentile(durations, 10)),
        "step_seconds_p90": float(np.percentile(durations, 90)),
        "gpu_memory_median": float(np.median(memory)) if memory else None,
        "learning_rates": [float(value) for value in midpoint["learning_rates"]],
    }


def _finite_percentile(values: NumpyArray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if len(values) else float("nan")


def _plot_band(
    axes: Any,
    x: NumpyArray,
    records: List[Dict[str, Any]],
    key: str,
    label: str,
) -> None:
    axes.plot(x, [item[f"{key}_median"] for item in records])
    axes.fill_between(
        x,
        [item[f"{key}_p10"] for item in records],
        [item[f"{key}_p90"] for item in records],
        alpha=0.22,
    )
    axes.set_ylabel(label)


def _mark_epoch_extrema(axes: Any, development: List[Dict[str, Any]]) -> None:
    losses = [item for item in development if item.get("loss") is not None]
    mious = [item for item in development if item.get("known_class_miou") is not None]
    if losses:
        best_loss = min(losses, key=lambda item: float(item["loss"]))
        axes[0].axvline(
            int(best_loss["epoch"]) + 1,
            linestyle="--",
            color="tab:orange",
            alpha=0.55,
            label="minimum development loss",
        )
        axes[0].legend()
    if mious:
        best_miou = max(mious, key=lambda item: float(item["known_class_miou"]))
        axes[1].axvline(
            int(best_miou["epoch"]) + 1,
            linestyle="--",
            color="tab:green",
            alpha=0.55,
            label="best development mIoU",
        )
        axes[1].legend()
