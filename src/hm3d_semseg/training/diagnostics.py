"""Quantitative and visual diagnostics for deliberately small training subsets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from hm3d_semseg.config.schema import ModelConfig
from hm3d_semseg.data.dataset import OfflineSegmentationDataset
from hm3d_semseg.evaluation.confusion import StreamingConfusionMatrix
from hm3d_semseg.evaluation.metrics import metrics_from_confusion
from hm3d_semseg.models.segformer import predict
from hm3d_semseg.types import NumpyArray
from hm3d_semseg.utils.hashing import atomic_write_json
from hm3d_semseg.visualization.masks import overlay_mask


def evaluate_training_subset(
    model: Any,
    dataset: OfflineSegmentationDataset,
    output: Path,
    model_config: ModelConfig,
    *,
    checkpoint: Path,
    device: str,
    ignore_index: int = 255,
) -> Dict[str, Any]:
    """Measure memorization of every selected sample and save inspectable artifacts."""
    import torch

    output.mkdir(parents=True, exist_ok=True)
    qualitative = output / "qualitative"
    qualitative.mkdir(exist_ok=True)
    confusion = StreamingConfusionMatrix(ignore_index=ignore_index)
    samples = []
    loss_sum = 0.0
    valid_pixels = 0
    model.eval()
    with torch.inference_mode():
        for index, record in enumerate(dataset.records):
            item = dataset[index]
            labels = item["labels"].to(device)
            result = predict(
                model,
                item["pixel_values"].unsqueeze(0).to(device),
                output_size=tuple(labels.shape),
                align_corners=model_config.align_corners,
            )
            prediction = result.labels[0]
            valid = labels != ignore_index
            sample_valid_pixels = int(valid.sum().cpu())
            sample_loss_sum = float(
                torch.nn.functional.cross_entropy(
                    result.logits,
                    labels.unsqueeze(0),
                    ignore_index=ignore_index,
                    reduction="sum",
                ).cpu()
            )
            local = StreamingConfusionMatrix(ignore_index=ignore_index)
            local.update(prediction, labels)
            confusion.update(prediction, labels)
            local_metrics = metrics_from_confusion(local.matrix)
            sample_report = {
                "sample_id": record.sample_id,
                "scene_id": record.scene_id,
                "valid_pixels": sample_valid_pixels,
                "mean_cross_entropy_loss": (
                    sample_loss_sum / sample_valid_pixels
                    if sample_valid_pixels
                    else None
                ),
                "overall_pixel_accuracy": local_metrics["overall_pixel_accuracy"],
                "known_class_miou": local_metrics["known_class_miou"],
                "classes_present": local_metrics["known_classes_included"],
            }
            samples.append(sample_report)
            loss_sum += sample_loss_sum
            valid_pixels += sample_valid_pixels
            with Image.open(dataset.root / record.rgb) as handle:
                rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
            target = labels.to(torch.uint8).cpu().numpy()
            predicted = prediction.to(torch.uint8).cpu().numpy()
            _save_qualitative_panel(
                rgb,
                target,
                predicted,
                qualitative / f"{record.sample_id}.png",
            )

    global_metrics = metrics_from_confusion(confusion.matrix)
    report = {
        "schema_version": "1.0",
        "evaluation_kind": "training_subset_memorization",
        "checkpoint": str(checkpoint.resolve()),
        "dataset": str(dataset.root),
        "sample_ids": [record.sample_id for record in dataset.records],
        "scene_ids": [record.scene_id for record in dataset.records],
        "samples": samples,
        "mean_cross_entropy_loss": (
            loss_sum / valid_pixels if valid_pixels else None
        ),
        "global": global_metrics,
    }
    atomic_write_json(output / "summary.json", report)
    np.save(output / "confusion_matrix.npy", confusion.matrix, allow_pickle=False)
    _save_subset_plots(report, output / "plots")
    return report


def _save_qualitative_panel(
    rgb: NumpyArray,
    target: NumpyArray,
    prediction: NumpyArray,
    output: Path,
) -> None:
    target_overlay = overlay_mask(rgb, target)
    prediction_overlay = overlay_mask(rgb, prediction)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plot
    except ImportError:
        panel = np.concatenate([rgb, target_overlay, prediction_overlay], axis=1)
        Image.fromarray(panel, mode="RGB").save(output)
        return
    figure, axes = plot.subplots(1, 3, figsize=(15, 4))
    for axis, image, title in zip(
        axes,
        (rgb, target_overlay, prediction_overlay),
        ("RGB", "Ground truth", "Prediction"),
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plot.close(figure)


def _save_subset_plots(report: Dict[str, Any], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plot
    except ImportError:
        return
    output.mkdir(parents=True, exist_ok=True)
    samples = report["samples"]
    figure, axes = plot.subplots(figsize=(8, 4))
    axes.bar(
        [item["sample_id"][-12:] for item in samples],
        [item["overall_pixel_accuracy"] for item in samples],
    )
    axes.set_ylabel("Pixel accuracy")
    axes.set_ylim(0, 1)
    axes.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output / "per_sample_pixel_accuracy.png", dpi=160)
    plot.close(figure)

    present = [item for item in report["global"]["per_class"] if item["support"] > 0]
    figure, axes = plot.subplots(figsize=(12, 5))
    axes.bar(
        [item["name"] for item in present],
        [item["iou"] for item in present],
    )
    axes.set_ylabel("IoU")
    axes.set_ylim(0, 1)
    axes.tick_params(axis="x", rotation=65)
    figure.tight_layout()
    figure.savefig(output / "per_class_iou.png", dpi=160)
    plot.close(figure)

    normalized = np.asarray(report["global"]["row_normalized_confusion_matrix"])
    figure, axes = plot.subplots(figsize=(10, 8))
    image = axes.imshow(normalized, interpolation="nearest", cmap="magma", vmin=0, vmax=1)
    axes.set_xlabel("Predicted class ID")
    axes.set_ylabel("Ground-truth class ID")
    axes.set_title("Row-normalized confusion matrix")
    figure.colorbar(image, ax=axes)
    figure.tight_layout()
    figure.savefig(output / "confusion_row_normalized.png", dpi=160)
    plot.close(figure)
