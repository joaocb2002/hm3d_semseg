"""Training curve plots from the append-only JSONL log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def save_training_plots(metrics_path: Path, output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plot
    except ImportError:
        return
    records: List[Dict[str, Any]] = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [item for item in records if item["kind"] == "train_step"]
    if not steps:
        return
    figure, axes = plot.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot([item["step"] for item in steps], [item["loss"] for item in steps])
    axes[0].set_ylabel("Cross-entropy")
    group_count = len(steps[0]["learning_rates"])
    for index in range(group_count):
        axes[1].plot(
            [item["step"] for item in steps],
            [item["learning_rates"][index] for item in steps],
            label=f"group {index}",
        )
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Learning rate")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "loss_and_learning_rate.png", dpi=160)
    plot.close(figure)

    development = [
        item
        for item in records
        if item["kind"] == "development_epoch" and item.get("loss") is not None
    ]
    if development:
        figure, axes = plot.subplots(figsize=(8, 4))
        axes.plot(
            [item["epoch"] for item in development],
            [item["loss"] for item in development],
            marker="o",
            label="development cross-entropy",
        )
        axes.set_xlabel("Epoch")
        axes.set_ylabel("Cross-entropy")
        axes.legend()
        figure.tight_layout()
        figure.savefig(output / "development_loss.png", dpi=160)
        plot.close(figure)
