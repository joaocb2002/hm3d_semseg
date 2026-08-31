"""Readable training plots derived from the append-only JSONL log."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from hm3d_semseg.types import NumpyArray


def save_training_plots(metrics_path: Path, output: Path) -> List[Path]:
    """Save readable optimization plots while retaining every raw JSONL record."""
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
    figure, axes = plot.subplots(figsize=(12, 5))
    loss_median = np.asarray([item["loss_median"] for item in steps])
    axes.plot(x, loss_median, label="median within step bin")
    axes.fill_between(
        x,
        [item["loss_p10"] for item in steps],
        [item["loss_p90"] for item in steps],
        alpha=0.22,
        label="10th-90th percentile",
    )
    axes.set_xlabel("Optimizer step")
    axes.set_ylabel("Training objective cross-entropy")
    axes.set_title(
        "Training loss (aggregated; every raw step remains in records/metrics.jsonl)"
    )
    axes.grid(alpha=0.2)
    axes.legend()
    figure.tight_layout()
    training_loss = output / "training_loss.png"
    figure.savefig(training_loss, dpi=160)
    plot.close(figure)
    created.append(training_loss)

    figure, axes = plot.subplots(figsize=(12, 5))
    group_count = len(steps[0]["learning_rates"])
    standard_group_names = (
        "pretrained decay",
        "pretrained no-decay",
        "decode head decay",
        "decode head no-decay",
    )
    for index in range(group_count):
        group_name = (
            ("pretrained", "classifier")[index]
            if group_count == 2
            else standard_group_names[index]
            if group_count == 4
            else f"group {index}"
        )
        axes.plot(
            x,
            [item["learning_rates"][index] for item in steps],
            label=f"{group_name} (group {index})",
        )
    axes.set_xlabel("Optimizer step")
    axes.set_ylabel("Learning rate")
    axes.set_title("Scheduled learning rates by optimizer parameter group")
    axes.grid(alpha=0.2)
    axes.legend()
    figure.tight_layout()
    learning_rates = output / "learning_rates.png"
    figure.savefig(learning_rates, dpi=160)
    plot.close(figure)
    created.append(learning_rates)

    figure, axes = plot.subplots(figsize=(12, 5))
    finite_gradient = np.asarray([item["gradient_median"] for item in steps], dtype=np.float64)
    axes.plot(x, finite_gradient, label="finite median")
    axes.fill_between(
        x,
        [item["gradient_p10"] for item in steps],
        [item["gradient_p90"] for item in steps],
        alpha=0.22,
        label="finite 10th-90th percentile",
    )
    if np.isfinite(finite_gradient).any() and np.nanmax(finite_gradient) > 0:
        axes.set_yscale("log")
    nonfinite = sum(int(item["gradient_nonfinite_count"]) for item in steps)
    skipped = sum(int(item["optimizer_steps_skipped"]) for item in steps)
    axes.set_xlabel("Optimizer step")
    axes.set_ylabel("Gradient norm")
    axes.set_title(
        f"Finite gradient norms; {nonfinite} non-finite records, "
        f"{skipped} AMP-skipped updates"
    )
    axes.grid(alpha=0.2)
    axes.legend()
    figure.tight_layout()
    diagnostics = output / "gradient_and_amp_health.png"
    figure.savefig(diagnostics, dpi=160)
    plot.close(figure)
    created.append(diagnostics)

    has_gpu_memory = any(item.get("gpu_memory_median") is not None for item in steps)
    rows = 3 if has_gpu_memory else 2
    figure, axes = plot.subplots(rows, 1, figsize=(12, 3.3 * rows), sharex=True)
    _plot_band(axes[0], x, steps, "throughput", "Samples/s")
    axes[0].set_title("Throughput")
    _plot_band(axes[1], x, steps, "step_seconds", "Seconds/step")
    axes[1].set_title("Step duration")
    if has_gpu_memory:
        _plot_band(axes[2], x, steps, "gpu_memory", "GiB")
        axes[2].set_title("Peak allocated CUDA memory")
        axes[2].ticklabel_format(style="plain", axis="y", useOffset=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Optimizer step")
    figure.suptitle(
        "Runtime efficiency (aggregated; every raw step remains in records/metrics.jsonl)"
    )
    figure.tight_layout()
    runtime = output / "throughput_and_memory.png"
    figure.savefig(runtime, dpi=160)
    plot.close(figure)
    created.append(runtime)
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
        "gpu_memory_p10": float(np.percentile(memory, 10)) if memory else None,
        "gpu_memory_p90": float(np.percentile(memory, 90)) if memory else None,
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
    median = [
        np.nan if item.get(f"{key}_median") is None else item[f"{key}_median"]
        for item in records
    ]
    p10 = [
        np.nan if item.get(f"{key}_p10") is None else item[f"{key}_p10"]
        for item in records
    ]
    p90 = [
        np.nan if item.get(f"{key}_p90") is None else item[f"{key}_p90"]
        for item in records
    ]
    axes.plot(x, median, label="median within step bin")
    axes.fill_between(
        x,
        p10,
        p90,
        alpha=0.22,
        label="10th-90th percentile",
    )
    axes.set_ylabel(label)
    axes.legend()


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
