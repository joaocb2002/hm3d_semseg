"""Training curve plots from the append-only JSONL log."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from hm3d_semseg.training.reporting import load_training_records


def save_training_plots(metrics_path: Path, output: Path) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plot
    except ImportError:
        return []
    output.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    records: List[Dict[str, Any]] = load_training_records(metrics_path)
    steps = [item for item in records if item["kind"] == "train_step"]
    if not steps:
        return created
    figure, axes = plot.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot([item["step"] for item in steps], [item["loss"] for item in steps])
    axes[0].set_ylabel("Cross-entropy")
    group_count = len(steps[0]["learning_rates"])
    for index in range(group_count):
        group_name = (
            ("pretrained", "classifier")[index]
            if group_count == 2
            else f"group {index}"
        )
        axes[1].plot(
            [item["step"] for item in steps],
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

    has_gpu_memory = any(item.get("gpu_peak_memory_bytes") is not None for item in steps)
    rows = 4 if has_gpu_memory else 3
    figure, axes = plot.subplots(rows, 1, figsize=(9, 2.5 * rows), sharex=True)
    step_values = [item["step"] for item in steps]
    axes[0].plot(step_values, [item["gradient_norm"] for item in steps])
    axes[0].set_ylabel("Gradient norm")
    axes[1].plot(step_values, [item["samples_per_second"] for item in steps])
    axes[1].set_ylabel("Samples/s")
    axes[2].plot(step_values, [item["step_seconds"] for item in steps])
    axes[2].set_ylabel("Seconds/step")
    if has_gpu_memory:
        axes[3].plot(
            step_values,
            [
                (
                    float(item["gpu_peak_memory_bytes"]) / (1024**3)
                    if item.get("gpu_peak_memory_bytes") is not None
                    else float("nan")
                )
                for item in steps
            ],
        )
        axes[3].set_ylabel("Peak GPU GiB")
    axes[-1].set_xlabel("Optimizer step")
    figure.tight_layout()
    diagnostics = output / "optimization_diagnostics.png"
    figure.savefig(diagnostics, dpi=160)
    plot.close(figure)
    created.append(diagnostics)

    development = [
        item
        for item in records
        if item["kind"] == "development_epoch" and item.get("loss") is not None
    ]
    if development:
        train_epochs = [item for item in records if item["kind"] == "train_epoch"]
        figure, axes = plot.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(
            [item["epoch"] + 1 for item in train_epochs],
            [item["loss"] for item in train_epochs],
            marker="o",
            label="training cross-entropy",
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
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Development known-class mIoU")
        axes[1].set_ylim(0, 1)
        figure.tight_layout()
        development_metrics = output / "development_metrics.png"
        figure.savefig(development_metrics, dpi=160)
        plot.close(figure)
        created.append(development_metrics)
    return created
