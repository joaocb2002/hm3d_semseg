"""Model evaluation over one fixed manifest."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hm3d_semseg.calibration.metrics import StreamingCalibrationMetrics
from hm3d_semseg.camera.profile import CameraProfile, assert_camera_compatible
from hm3d_semseg.config.schema import ProjectConfig
from hm3d_semseg.data.dataset import OfflineSegmentationDataset
from hm3d_semseg.data.validate import validate_dataset
from hm3d_semseg.evaluation.confusion import StreamingConfusionMatrix
from hm3d_semseg.evaluation.metrics import bootstrap_scene_metric, metrics_from_confusion
from hm3d_semseg.evaluation.plots import save_evaluation_plots
from hm3d_semseg.models.segformer import build_segformer, predict
from hm3d_semseg.utils.device import select_torch_device
from hm3d_semseg.utils.hashing import atomic_write_json


def evaluate_model(
    checkpoint: Path,
    dataset_root: Path,
    output: Path,
    config: ProjectConfig,
    *,
    temperature: float = 1.0,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    validation = validate_dataset(dataset_root)
    checkpoint_camera = CameraProfile.load(checkpoint / "camera_profile.yaml")
    dataset_camera = CameraProfile.load(dataset_root / "camera_profile.yaml")
    assert_camera_compatible(checkpoint_camera, dataset_camera, config.camera.allow_mismatch)
    dataset = OfflineSegmentationDataset(dataset_root, augment=False)
    calibration_file = checkpoint / "calibration.json"
    if temperature != 1.0 and calibration_file.is_file():
        calibration_provenance = json.loads(calibration_file.read_text(encoding="utf-8"))
        fitted_scenes = set(calibration_provenance.get("calibration_scenes", []))
        evaluated_scenes = {record.scene_id for record in dataset.records}
        overlap = fitted_scenes & evaluated_scenes
        if overlap:
            raise ValueError(
                "Calibrated probability metrics would leak temperature-fit scenes: "
                + ", ".join(sorted(overlap)[:10])
            )
    device_selection = select_torch_device(device)
    device = device_selection.device
    loader = DataLoader(
        dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.evaluation.workers,
        pin_memory=device.startswith("cuda"),
    )
    model = build_segformer(config.model, checkpoint=checkpoint).to(device).eval()
    confusion = StreamingConfusionMatrix()
    calibration_metrics = StreamingCalibrationMetrics(config.evaluation.calibration_bins)
    scene_confusions: Dict[str, StreamingConfusionMatrix] = defaultdict(
        StreamingConfusionMatrix
    )
    loss_sum = 0.0
    valid_pixels = 0
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            result = predict(
                model,
                pixels,
                output_size=tuple(labels.shape[-2:]),
                align_corners=config.model.align_corners,
                temperature=temperature,
            )
            loss_sum += float(
                torch.nn.functional.cross_entropy(
                    result.logits,
                    labels,
                    ignore_index=config.taxonomy.ignore_index,
                    reduction="sum",
                ).cpu()
            )
            valid_pixels += int((labels != config.taxonomy.ignore_index).sum().cpu())
            confusion.update(result.labels, labels)
            calibration_metrics.update(result.probabilities, labels)
            for item, scene_id in enumerate(batch["scene_id"]):
                scene_confusions[str(scene_id)].update(result.labels[item], labels[item])
    global_metrics = metrics_from_confusion(confusion.matrix)
    scene_metrics: Dict[str, Optional[float]] = {}
    for scene_id, accumulator in sorted(scene_confusions.items()):
        scene_metrics[scene_id] = metrics_from_confusion(accumulator.matrix)["known_class_miou"]
    values = [value for value in scene_metrics.values() if value is not None]
    confidence_interval = bootstrap_scene_metric(
        values,
        config.evaluation.bootstrap_samples,
        config.evaluation.bootstrap_seed,
    )
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "dataset": str(dataset_root.resolve()),
        "dataset_validation": validation,
        "temperature": temperature,
        "device_selection": device_selection.to_dict(),
        "mean_cross_entropy_loss": (loss_sum / valid_pixels if valid_pixels else None),
        "global": global_metrics,
        "scene_macro": {
            "per_scene_known_class_miou": scene_metrics,
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "bootstrap_95_percent_ci": list(confidence_interval),
            "bootstrap_seed": config.evaluation.bootstrap_seed,
            "bootstrap_samples": config.evaluation.bootstrap_samples,
        },
        "probability_quality": calibration_metrics.compute(),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", report)
    np.save(output / "confusion_matrix.npy", confusion.matrix, allow_pickle=False)
    save_evaluation_plots(report, output)
    return report
