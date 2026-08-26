"""Required semantic segmentation metrics from one global confusion matrix."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple, cast

import numpy as np

from hm3d_semseg.taxonomy.constants import ID2LABEL, OBJECTNAV_SIX
from hm3d_semseg.types import NumpyArray


def _safe_divide(numerator: NumpyArray, denominator: NumpyArray) -> NumpyArray:
    output = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator != 0)
    return output


def metrics_from_confusion(confusion: NumpyArray) -> Dict[str, Any]:
    """Calculate global metrics, excluding absent classes from macro averages."""
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (41, 41):
        raise ValueError(f"Expected 41x41 confusion matrix, got {matrix.shape}")
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    union = support + predicted - true_positive
    iou = _safe_divide(true_positive, union)
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    known_present = np.asarray([index for index in range(1, 41) if support[index] > 0])
    all_present = np.asarray([index for index in range(41) if support[index] > 0])
    total = int(matrix.sum())
    per_class = []
    for index in range(41):
        per_class.append(
            {
                "id": index,
                "name": ID2LABEL[index],
                "intersection": int(true_positive[index]),
                "union": int(union[index]),
                "support": int(support[index]),
                "predicted": int(predicted[index]),
                "iou": _optional_float(iou[index]),
                "precision": _optional_float(precision[index]),
                "recall": _optional_float(recall[index]),
                "f1": _optional_float(f1[index]),
            }
        )
    objectnav_indices = list(OBJECTNAV_SIX.values())
    objectnav_present = [index for index in objectnav_indices if support[index] > 0]
    unknown_metrics = dict(per_class[0])
    unknown_metrics["prevalence"] = float(support[0] / total) if total else None
    return {
        "known_class_miou": _mean_at(iou, known_present),
        "known_classes_included": [ID2LABEL[int(index)] for index in known_present],
        "miou_41": _mean_at(iou, all_present),
        "unknown": unknown_metrics,
        "mean_class_recall": _mean_at(recall, all_present),
        "overall_pixel_accuracy": (float(true_positive.sum() / total) if total else None),
        "frequency_weighted_iou": (
            float(np.nansum((support / total) * iou)) if total else None
        ),
        "objectnav_six_miou": _mean_at(iou, objectnav_present),
        "objectnav_six": {
            goal: {
                "model_class": ID2LABEL[index],
                "iou": _optional_float(iou[index]),
                "precision": _optional_float(precision[index]),
                "recall": _optional_float(recall[index]),
                "support": int(support[index]),
            }
            for goal, index in OBJECTNAV_SIX.items()
        },
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "row_normalized_confusion_matrix": _row_normalize(matrix).tolist(),
    }


def _row_normalize(matrix: NumpyArray) -> NumpyArray:
    rows = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        rows,
        out=np.zeros(matrix.shape, dtype=np.float64),
        where=rows != 0,
    )


def _optional_float(value: float) -> Any:
    return None if np.isnan(value) else float(value)


def _mean_at(values: NumpyArray, indices: Iterable[int]) -> Any:
    selected = list(indices)
    return float(np.nanmean(values[selected])) if selected else None


def bootstrap_scene_metric(
    scene_values: Sequence[float], samples: int, seed: int
) -> Tuple[float, float]:
    """Scene-level percentile bootstrap confidence interval."""
    values = np.asarray(scene_values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    bounds = cast(NumpyArray, np.percentile(means, [2.5, 97.5]))
    return float(bounds[0]), float(bounds[1])
