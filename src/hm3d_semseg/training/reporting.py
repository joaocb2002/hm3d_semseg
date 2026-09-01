"""Compact, deterministic summaries of append-only training metrics."""

from __future__ import annotations

import json
import math
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
    """Produce a human-sized digest without retaining step records in memory."""
    step_count = 0
    step_objective: Dict[str, Any] = {"initial": None, "final": None, "minimum": None}
    step_cross_entropy: Dict[str, Any] = {
        "initial": None,
        "final": None,
        "minimum": None,
    }
    step_lovasz: Dict[str, Any] = {"initial": None, "final": None, "minimum": None}
    epochs: List[Dict[str, Any]] = []
    development: List[Dict[str, Any]] = []
    early_stopping: Optional[Dict[str, Any]] = None
    total_step_seconds = 0.0
    processed_samples = 0.0
    peak_gpu_memory: Optional[int] = None
    initial_learning_rates: List[float] = []
    final_learning_rates: List[float] = []
    maximum_finite_gradient: Optional[float] = None
    final_gradient: Optional[float] = None
    nonfinite_gradients = 0
    skipped_optimizer_steps = 0
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            kind = item.get("kind")
            if kind == "train_step":
                step_count += 1
                position = int(item["step"])
                objective = float(item.get("objective_loss", item["loss"]))
                cross_entropy = float(item.get("cross_entropy_loss", item["loss"]))
                lovasz = float(item.get("lovasz_loss", 0.0))
                _update_series_summary(step_objective, position, objective)
                _update_series_summary(step_cross_entropy, position, cross_entropy)
                _update_series_summary(step_lovasz, position, lovasz)
                if step_count == 1:
                    initial_learning_rates = [
                        float(value) for value in item.get("learning_rates", [])
                    ]
                final_learning_rates = [
                    float(value) for value in item.get("learning_rates", [])
                ]
                seconds = float(item.get("step_seconds", 0.0))
                total_step_seconds += seconds
                processed_samples += float(
                    item.get(
                        "samples",
                        float(item.get("samples_per_second", 0.0)) * seconds,
                    )
                )
                memory = item.get("gpu_peak_memory_bytes")
                if memory is not None:
                    peak_gpu_memory = max(peak_gpu_memory or 0, int(memory))
                gradient = float(item["gradient_norm"])
                final_gradient = gradient
                if math.isfinite(gradient):
                    maximum_finite_gradient = max(maximum_finite_gradient or gradient, gradient)
                else:
                    nonfinite_gradients += 1
                skipped_optimizer_steps += int(bool(item.get("optimizer_step_skipped", False)))
            elif kind == "train_epoch":
                epochs.append(
                    {
                        **item,
                        "objective_loss": float(
                            item.get("objective_loss", item["loss"])
                        ),
                        "cross_entropy_loss": float(
                            item.get("cross_entropy_loss", item["loss"])
                        ),
                        "lovasz_loss": float(item.get("lovasz_loss", 0.0)),
                    }
                )
            elif kind == "development_epoch":
                development.append(item)
            elif kind == "early_stopping":
                early_stopping = item

    result: Dict[str, Any] = {
        "schema_version": "1.1",
        "source": str(metrics_path.resolve()),
        "training": {
            "optimizer_steps_recorded": step_count,
            "epochs_recorded": len(epochs),
            "step_objective": step_objective,
            "step_cross_entropy": step_cross_entropy,
            "step_lovasz": step_lovasz,
            "epoch_objective": _series_summary(
                epochs, "objective_loss", "epoch"
            ),
            "epoch_cross_entropy": _series_summary(
                epochs, "cross_entropy_loss", "epoch"
            ),
            "epoch_lovasz": _series_summary(epochs, "lovasz_loss", "epoch"),
            "epoch_objective_history": [
                {"epoch": int(item["epoch"]), "value": float(item["objective_loss"])}
                for item in epochs
            ],
            "epoch_cross_entropy_history": [
                {
                    "epoch": int(item["epoch"]),
                    "value": float(item["cross_entropy_loss"]),
                }
                for item in epochs
            ],
            "epoch_lovasz_history": [
                {"epoch": int(item["epoch"]), "value": float(item["lovasz_loss"])}
                for item in epochs
            ],
            "gradient_norm": {
                "final": final_gradient,
                "maximum": maximum_finite_gradient,
                "nonfinite_count": nonfinite_gradients,
                "nonfinite_fraction": (
                    nonfinite_gradients / step_count if step_count else None
                ),
            },
            "optimization": {
                "total_step_seconds": total_step_seconds,
                "mean_step_seconds": (total_step_seconds / step_count if step_count else None),
                "overall_samples_per_second": (
                    processed_samples / total_step_seconds if total_step_seconds > 0.0 else None
                ),
                "peak_gpu_memory_bytes": peak_gpu_memory,
                "initial_learning_rates": initial_learning_rates,
                "final_learning_rates": final_learning_rates,
                "optimizer_steps_skipped": skipped_optimizer_steps,
            },
        },
        "development": _development_summary(development),
        "early_stopping": early_stopping,
    }
    return result


def _update_series_summary(
    summary: Dict[str, Any], position: int, value: float
) -> None:
    if summary["initial"] is None:
        summary["initial"] = value
    summary["final"] = value
    minimum = summary["minimum"]
    if minimum is None or value < float(minimum["value"]):
        summary["minimum"] = {"step": position, "value": value}


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
    best_miou = max(mious, key=lambda item: float(item["known_class_miou"])) if mious else None
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
                "overall_pixel_accuracy": _optional_number(item.get("overall_pixel_accuracy")),
                "objectnav_six_miou": _optional_number(item.get("objectnav_six_miou")),
                "scene_macro_mean_miou": _optional_number(item.get("scene_macro_mean_miou")),
                "nll": _optional_number(item.get("nll")),
                "ece": _optional_number(item.get("ece")),
                "multiclass_brier": _optional_number(item.get("multiclass_brier")),
            }
            for item in records
        ],
        "final": {
            "epoch": int(records[-1]["epoch"]),
            "mean_cross_entropy_loss": (
                float(records[-1]["loss"]) if records[-1].get("loss") is not None else None
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


def _optional_number(value: Any) -> Optional[float]:
    return float(value) if value is not None else None
