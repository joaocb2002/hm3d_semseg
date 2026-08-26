"""Memory-safe NLL, Brier, ECE, confidence, and entropy statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, cast

import numpy as np

from hm3d_semseg.types import NumpyArray


def _numpy(value: Any) -> NumpyArray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return cast(NumpyArray, value.detach().cpu().numpy())
    return np.asarray(value)


@dataclass
class StreamingCalibrationMetrics:
    bins: int = 15
    ignore_index: int = 255
    pixels: int = 0
    nll_sum: float = 0.0
    brier_sum: float = 0.0
    correct_entropy_sum: float = 0.0
    incorrect_entropy_sum: float = 0.0
    correct_pixels: int = 0
    incorrect_pixels: int = 0
    bin_count: NumpyArray = field(init=False)
    bin_confidence: NumpyArray = field(init=False)
    bin_correct: NumpyArray = field(init=False)

    def __post_init__(self) -> None:
        self.bin_count = np.zeros(self.bins, dtype=np.int64)
        self.bin_confidence = np.zeros(self.bins, dtype=np.float64)
        self.bin_correct = np.zeros(self.bins, dtype=np.float64)

    def update(self, probabilities: Any, targets: Any) -> None:
        probs = _numpy(probabilities).astype(np.float64, copy=False)
        labels = _numpy(targets).astype(np.int64, copy=False)
        if probs.ndim != labels.ndim + 1:
            raise ValueError("Probabilities must include one class dimension")
        # Accept [B,C,H,W] or [C,H,W].
        class_axis = 1 if probs.ndim == 4 else 0
        probs = np.moveaxis(probs, class_axis, -1)
        valid = labels != self.ignore_index
        flat_probs = probs[valid]
        flat_labels = labels[valid]
        if not len(flat_labels):
            return
        row = np.arange(len(flat_labels))
        true_probability = flat_probs[row, flat_labels]
        self.nll_sum += float(-np.log(np.clip(true_probability, 1e-12, 1.0)).sum())
        self.brier_sum += float(
            (np.square(flat_probs).sum(axis=1) - 2.0 * true_probability + 1.0).sum()
        )
        predictions = flat_probs.argmax(axis=1)
        confidence = flat_probs.max(axis=1)
        correct = predictions == flat_labels
        entropy = -(flat_probs * np.log(np.clip(flat_probs, 1e-12, 1.0))).sum(axis=1)
        self.correct_entropy_sum += float(entropy[correct].sum())
        self.incorrect_entropy_sum += float(entropy[~correct].sum())
        self.correct_pixels += int(correct.sum())
        self.incorrect_pixels += int((~correct).sum())
        indices = np.minimum((confidence * self.bins).astype(int), self.bins - 1)
        self.bin_count += np.bincount(indices, minlength=self.bins)
        self.bin_confidence += np.bincount(indices, weights=confidence, minlength=self.bins)
        self.bin_correct += np.bincount(
            indices, weights=correct.astype(float), minlength=self.bins
        )
        self.pixels += len(flat_labels)

    def compute(self) -> Dict[str, Any]:
        denominator = max(1, self.pixels)
        average_confidence = np.divide(
            self.bin_confidence,
            self.bin_count,
            out=np.zeros(self.bins),
            where=self.bin_count > 0,
        )
        average_accuracy = np.divide(
            self.bin_correct,
            self.bin_count,
            out=np.zeros(self.bins),
            where=self.bin_count > 0,
        )
        ece = float(
            np.sum(
                self.bin_count
                / denominator
                * np.abs(average_accuracy - average_confidence)
            )
        )
        high_to_low_count = np.cumsum(self.bin_count[::-1])
        high_to_low_correct = np.cumsum(self.bin_correct[::-1])
        retained_accuracy = np.divide(
            high_to_low_correct,
            high_to_low_count,
            out=np.zeros(self.bins),
            where=high_to_low_count > 0,
        )
        return {
            "pixels": self.pixels,
            "nll": self.nll_sum / denominator,
            "multiclass_brier": self.brier_sum / denominator,
            "ece": ece,
            "bins": [
                {
                    "lower": index / self.bins,
                    "upper": (index + 1) / self.bins,
                    "count": int(self.bin_count[index]),
                    "confidence": float(average_confidence[index]),
                    "accuracy": float(average_accuracy[index]),
                }
                for index in range(self.bins)
            ],
            "risk_coverage": [
                {
                    "minimum_confidence": (self.bins - index - 1) / self.bins,
                    "coverage": float(high_to_low_count[index] / denominator),
                    "risk": (
                        float(1.0 - retained_accuracy[index])
                        if high_to_low_count[index]
                        else None
                    ),
                }
                for index in range(self.bins)
            ],
            "mean_entropy_correct": (
                self.correct_entropy_sum / self.correct_pixels if self.correct_pixels else None
            ),
            "mean_entropy_incorrect": (
                self.incorrect_entropy_sum / self.incorrect_pixels
                if self.incorrect_pixels
                else None
            ),
        }
