"""Compact, deterministic summaries of append-only training metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_training_records(metrics_path: Path) -> List[Dict[str, Any]]:
    """Load non-empty JSONL records without changing their append order."""
    return [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_training_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Produce a human-sized digest while leaving step-level JSONL authoritative."""
    records = load_training_records(metrics_path)
    steps = [item for item in records if item.get("kind") == "train_step"]
    epochs = [item for item in records if item.get("kind") == "train_epoch"]
    development = [item for item in records if item.get("kind") == "development_epoch"]
    early_stopping = [item for item in records if item.get("kind") == "early_stopping"]

    total_step_seconds = sum(float(item.get("step_seconds", 0.0)) for item in steps)
    processed_samples = sum(
        float(
            item.get(
                "samples",
                float(item.get("samples_per_second", 0.0))
                * float(item.get("step_seconds", 0.0)),
            )
        )
        for item in steps
    )
    gpu_peaks = [
        int(item["gpu_peak_memory_bytes"])
        for item in steps
        if item.get("gpu_peak_memory_bytes") is not None
    ]
    gradient_norms = [float(item["gradient_norm"]) for item in steps]

    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "source": str(metrics_path.resolve()),
        "training": {
            "optimizer_steps_recorded": len(steps),
            "epochs_recorded": len(epochs),
            "step_cross_entropy": _series_summary(steps, "loss", "step"),
            "epoch_cross_entropy": _series_summary(epochs, "loss", "epoch"),
            "epoch_cross_entropy_history": [
                {"epoch": int(item["epoch"]), "value": float(item["loss"])}
                for item in epochs
            ],
            "gradient_norm": {
                "final": gradient_norms[-1] if gradient_norms else None,
                "maximum": max(gradient_norms) if gradient_norms else None,
            },
            "optimization": {
                "total_step_seconds": total_step_seconds,
                "mean_step_seconds": (
                    total_step_seconds / len(steps) if steps else None
                ),
                "overall_samples_per_second": (
                    processed_samples / total_step_seconds
                    if total_step_seconds > 0.0
                    else None
                ),
                "peak_gpu_memory_bytes": max(gpu_peaks) if gpu_peaks else None,
                "initial_learning_rates": (
                    [float(value) for value in steps[0]["learning_rates"]]
                    if steps
                    else []
                ),
                "final_learning_rates": (
                    [float(value) for value in steps[-1]["learning_rates"]]
                    if steps
                    else []
                ),
            },
        },
        "development": _development_summary(development),
        "early_stopping": early_stopping[-1] if early_stopping else None,
    }
    return result


def _series_summary(
    records: List[Dict[str, Any]], value_key: str, position_key: str
) -> Dict[str, Any]:
    if not records:
        return {"initial": None, "final": None, "minimum": None}
    minimum = min(records, key=lambda item: float(item[value_key]))
    return {
        "initial": float(records[0][value_key]),
        "final": float(records[-1][value_key]),
        "minimum": {
            position_key: int(minimum[position_key]),
            "value": float(minimum[value_key]),
        },
    }


def _development_summary(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    losses = [item for item in records if item.get("loss") is not None]
    mious = [item for item in records if item.get("known_class_miou") is not None]
    best_loss = min(losses, key=lambda item: float(item["loss"])) if losses else None
    best_miou = (
        max(mious, key=lambda item: float(item["known_class_miou"])) if mious else None
    )
    return {
        "evaluations_recorded": len(records),
        "history": [
            {
                "epoch": int(item["epoch"]),
                "mean_cross_entropy_loss": (
                    float(item["loss"]) if item.get("loss") is not None else None
                ),
                "known_class_miou": (
                    float(item["known_class_miou"])
                    if item.get("known_class_miou") is not None
                    else None
                ),
            }
            for item in records
        ],
        "final": {
            "epoch": int(records[-1]["epoch"]),
            "mean_cross_entropy_loss": (
                float(records[-1]["loss"])
                if records[-1].get("loss") is not None
                else None
            ),
            "known_class_miou": (
                float(records[-1]["known_class_miou"])
                if records[-1].get("known_class_miou") is not None
                else None
            ),
        },
        "minimum_cross_entropy": (
            {"epoch": int(best_loss["epoch"]), "value": float(best_loss["loss"])}
            if best_loss is not None
            else None
        ),
        "best_known_class_miou": (
            {
                "epoch": int(best_miou["epoch"]),
                "value": float(best_miou["known_class_miou"]),
            }
            if best_miou is not None
            else None
        ),
    }
