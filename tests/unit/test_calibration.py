import numpy as np
import pytest

from hm3d_semseg.calibration.metrics import StreamingCalibrationMetrics

pytestmark = pytest.mark.unit


def test_perfect_categorical_probabilities() -> None:
    probabilities = np.zeros((1, 41, 1, 2), dtype=np.float64)
    probabilities[0, 1, 0, 0] = 1.0
    probabilities[0, 2, 0, 1] = 1.0
    target = np.asarray([[[1, 2]]])
    metric = StreamingCalibrationMetrics(bins=10)
    metric.update(probabilities, target)
    result = metric.compute()
    assert result["nll"] == pytest.approx(0.0)
    assert result["multiclass_brier"] == pytest.approx(0.0)
    assert result["ece"] == pytest.approx(0.0)


def test_ignored_probability_pixel_is_excluded() -> None:
    probabilities = np.full((1, 41, 1, 1), 1 / 41)
    metric = StreamingCalibrationMetrics()
    metric.update(probabilities, np.asarray([[[255]]]))
    assert metric.compute()["pixels"] == 0
