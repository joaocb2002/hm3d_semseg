"""Deterministic, storage-conscious qualitative segmentation diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import numpy as np
from PIL import Image, ImageDraw

from hm3d_semseg.data.dataset import OfflineSegmentationDataset
from hm3d_semseg.data.schema import ManifestRecord
from hm3d_semseg.evaluation.confusion import StreamingConfusionMatrix
from hm3d_semseg.evaluation.metrics import metrics_from_confusion
from hm3d_semseg.models.segformer import predict
from hm3d_semseg.taxonomy.constants import ID2LABEL, OBJECTNAV_SIX
from hm3d_semseg.types import NumpyArray
from hm3d_semseg.utils.hashing import atomic_write_json
from hm3d_semseg.visualization.masks import colorize_mask


def select_qualitative_records(
    records: Sequence[ManifestRecord], limit: int, *, seed: int
) -> List[ManifestRecord]:
    """Choose stable, scene-diverse views with broad/rare class coverage.

    Selection uses manifest metadata and ground truth only. Model predictions never
    influence the panel set, which prevents qualitative cherry-picking.
    """
    available = list(records)
    if limit <= 0:
        raise ValueError("Qualitative sample limit must be positive")
    if len(available) <= limit:
        return available

    support = np.asarray(
        [record.class_histogram for record in available], dtype=np.float64
    ).sum(axis=0)
    inverse_frequency = np.zeros(41, dtype=np.float64)
    present = support > 0
    inverse_frequency[present] = 1.0 / np.sqrt(support[present])
    if inverse_frequency[1:].max(initial=0.0) > 0:
        inverse_frequency /= inverse_frequency[1:].max()

    objectnav_ids: Set[int] = set(OBJECTNAV_SIX.values())
    selected: List[ManifestRecord] = []
    selected_ids: Set[str] = set()
    selected_scenes: Set[str] = set()
    covered_classes: Set[int] = set()
    while len(selected) < min(limit, len(available)):
        remaining = [item for item in available if item.sample_id not in selected_ids]
        unseen_scenes = [item for item in remaining if item.scene_id not in selected_scenes]
        candidates = unseen_scenes or remaining

        def score(record: ManifestRecord) -> Any:
            record_classes = {
                index
                for index, count in enumerate(record.class_histogram)
                if index > 0 and count > 0
            }
            new_classes = record_classes - covered_classes
            new_objectnav = len(new_classes & objectnav_ids)
            rare_coverage = float(inverse_frequency[list(record_classes)].sum())
            digest = hashlib.sha256(f"{seed}:{record.sample_id}".encode("utf-8")).hexdigest()
            tie_breaker = int(digest[:16], 16)
            return (new_objectnav, len(new_classes), rare_coverage, tie_breaker)

        chosen = max(candidates, key=score)
        selected.append(chosen)
        selected_ids.add(chosen.sample_id)
        selected_scenes.add(chosen.scene_id)
        covered_classes.update(
            index
            for index, count in enumerate(chosen.class_histogram)
            if index > 0 and count > 0
        )
    return selected


def selection_report(
    train_records: Sequence[ManifestRecord],
    development_records: Sequence[ManifestRecord],
    *,
    seed: int,
    requested_per_split: int,
    evaluation_records: Sequence[ManifestRecord] = (),
) -> Dict[str, Any]:
    """Describe the fixed qualitative panel set before predictions are produced."""

    def describe(records: Sequence[ManifestRecord]) -> List[Dict[str, Any]]:
        return [
            {
                "sample_id": record.sample_id,
                "scene_id": record.scene_id,
                "classes_present": [
                    ID2LABEL[index]
                    for index, count in enumerate(record.class_histogram)
                    if index in ID2LABEL and count > 0
                ],
            }
            for record in records
        ]

    return {
        "schema_version": "1.0",
        "selection_policy": "ground_truth_class_coverage_then_scene_diversity",
        "prediction_independent": True,
        "seed": seed,
        "requested_per_split": requested_per_split,
        "train": describe(train_records),
        "development": describe(development_records),
        "evaluation": describe(evaluation_records),
    }


def save_qualitative_prediction(
    *,
    dataset_root: Path,
    record: ManifestRecord,
    target: NumpyArray,
    prediction: NumpyArray,
    confidence: NumpyArray,
    output: Path,
    epoch: int,
    ignore_index: int = 255,
) -> Dict[str, Any]:
    """Save static inputs once and compact per-epoch prediction diagnostics."""
    sample_directory = output / "samples" / _safe_component(record.sample_id)
    sample_directory.mkdir(parents=True, exist_ok=True)
    rgb_path = sample_directory / "rgb.png"
    target_path = sample_directory / "ground_truth.png"
    if not rgb_path.is_file():
        with Image.open(dataset_root / record.rgb) as handle:
            _atomic_save_image(handle.convert("RGB"), rgb_path)
    if not target_path.is_file():
        _atomic_save_image(Image.fromarray(colorize_mask(target), mode="RGB"), target_path)

    prefix = f"epoch_{epoch:03d}"
    prediction_path = sample_directory / f"{prefix}_prediction.png"
    error_path = sample_directory / f"{prefix}_error.png"
    confidence_path = sample_directory / f"{prefix}_confidence.png"
    _atomic_save_image(Image.fromarray(colorize_mask(prediction), mode="RGB"), prediction_path)
    error = np.zeros((*target.shape, 3), dtype=np.uint8)
    valid = target != ignore_index
    error[valid & (target == prediction)] = [28, 105, 55]
    error[valid & (target != prediction)] = [210, 48, 48]
    error[~valid] = [90, 90, 90]
    _atomic_save_image(Image.fromarray(error, mode="RGB"), error_path)
    confidence_uint8 = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
    _atomic_save_image(Image.fromarray(confidence_uint8, mode="L"), confidence_path)

    local = StreamingConfusionMatrix(ignore_index=ignore_index)
    local.update(prediction, target)
    metrics = metrics_from_confusion(local.matrix)
    result = {
        "sample_id": record.sample_id,
        "scene_id": record.scene_id,
        "epoch": epoch,
        "overall_pixel_accuracy": metrics["overall_pixel_accuracy"],
        "known_class_miou": metrics["known_class_miou"],
        "mean_confidence": float(confidence[valid].mean()) if valid.any() else None,
        "rgb": str(rgb_path.relative_to(output)),
        "ground_truth": str(target_path.relative_to(output)),
        "prediction": str(prediction_path.relative_to(output)),
        "error": str(error_path.relative_to(output)),
        "confidence": str(confidence_path.relative_to(output)),
    }
    atomic_write_json(sample_directory / f"{prefix}_metrics.json", result)
    return result


def capture_model_qualitative(
    model: Any,
    dataset: OfflineSegmentationDataset,
    output: Path,
    *,
    epoch: int,
    device: str,
    align_corners: bool,
    ignore_index: int = 255,
) -> Dict[str, Any]:
    """Run a small fixed unaugmented set and write one inspectable contact sheet."""
    import torch

    reports = []
    model.eval()
    with torch.inference_mode():
        for index, record in enumerate(dataset.records):
            item = dataset[index]
            labels = item["labels"]
            result = predict(
                model,
                item["pixel_values"].unsqueeze(0).to(device),
                output_size=tuple(labels.shape),
                align_corners=align_corners,
            )
            reports.append(
                save_qualitative_prediction(
                    dataset_root=dataset.root,
                    record=record,
                    target=labels.cpu().numpy(),
                    prediction=result.labels[0].to(torch.uint8).cpu().numpy(),
                    confidence=result.confidence[0].float().cpu().numpy(),
                    output=output,
                    epoch=epoch,
                    ignore_index=ignore_index,
                )
            )
    contact_sheet = save_contact_sheet(output, reports, epoch=epoch)
    return {
        "epoch": epoch,
        "samples": reports,
        "contact_sheet": str(contact_sheet.relative_to(output)),
    }


def save_contact_sheet(output: Path, reports: Sequence[Dict[str, Any]], *, epoch: int) -> Path:
    """Compose a compact RGB/target/prediction/error/confidence overview."""
    tile_width = 192
    tile_height = 144
    header_height = 22
    columns = ("RGB", "Ground truth", "Prediction", "Error", "Confidence")
    sheet = Image.new(
        "RGB",
        (
            tile_width * len(columns),
            header_height + (header_height + tile_height) * len(reports),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(columns):
        draw.text((column * tile_width + 4, 4), label, fill="black")
    for row, report in enumerate(reports):
        y = header_height + row * (tile_height + header_height)
        short_id = str(report["sample_id"])[-16:]
        draw.text(
            (4, y),
            f"{short_id} | {report['scene_id']}",
            fill="black",
        )
        paths = (
            report["rgb"],
            report["ground_truth"],
            report["prediction"],
            report["error"],
            report["confidence"],
        )
        for column, relative in enumerate(paths):
            with Image.open(output / relative) as handle:
                tile = handle.convert("RGB")
                resampling = (
                    Image.Resampling.NEAREST
                    if column in {1, 2, 3}
                    else Image.Resampling.BILINEAR
                )
                tile.thumbnail((tile_width, tile_height), resampling)
                x = column * tile_width + (tile_width - tile.width) // 2
                image_y = y + header_height + (tile_height - tile.height) // 2
                sheet.paste(tile, (x, image_y))
    sheets = output / "contact_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    path = sheets / f"epoch_{epoch:03d}.png"
    _atomic_save_image(sheet, path)
    return path


def append_qualitative_epoch(output: Path, report: Dict[str, Any]) -> None:
    """Append a compact epoch record for report generation and interruption safety."""
    path = output / "epochs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip() and json.loads(line).get("epoch") == report.get("epoch"):
                    return
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    if not cleaned or cleaned in {".", ".."}:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"sample_{digest}"
    if cleaned != value:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return f"{cleaned}_{digest}"
    return cleaned


def _atomic_save_image(image: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=output.suffix, dir=str(output.parent)
    )
    os.close(descriptor)
    try:
        image.save(temporary)
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
