"""Memory-safe whole-manifest confusion accumulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from hm3d_semseg.types import NumpyArray


@dataclass
class StreamingConfusionMatrix:
    num_classes: int = 41
    ignore_index: int = 255
    matrix: NumpyArray = field(init=False)

    def __post_init__(self) -> None:
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, prediction: Any, target: Any) -> None:
        prediction_array = _to_numpy(prediction).astype(np.int64, copy=False)
        target_array = _to_numpy(target).astype(np.int64, copy=False)
        if prediction_array.shape != target_array.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: {prediction_array.shape} "
                f"versus {target_array.shape}"
            )
        valid = target_array != self.ignore_index
        valid &= (target_array >= 0) & (target_array < self.num_classes)
        valid &= (prediction_array >= 0) & (prediction_array < self.num_classes)
        encoded = self.num_classes * target_array[valid] + prediction_array[valid]
        self.matrix += np.bincount(encoded, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )


def _to_numpy(value: Any) -> NumpyArray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return cast(NumpyArray, value.detach().cpu().numpy())
    return np.asarray(value)
