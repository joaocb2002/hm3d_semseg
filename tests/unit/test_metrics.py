import numpy as np
import pytest

from hm3d_semseg.evaluation.confusion import StreamingConfusionMatrix
from hm3d_semseg.evaluation.metrics import bootstrap_scene_metric, metrics_from_confusion
from hm3d_semseg.taxonomy.constants import OBJECTNAV_SIX

pytestmark = pytest.mark.unit


def test_ignore_does_not_affect_confusion() -> None:
    accumulator = StreamingConfusionMatrix()
    accumulator.update(np.asarray([[1, 2, 40, 0]]), np.asarray([[1, 255, 2, 0]]))
    assert accumulator.matrix.sum() == 3
    assert accumulator.matrix[1, 1] == 1
    assert accumulator.matrix[2, 40] == 1


def test_hand_calculated_iou_and_absent_classes() -> None:
    matrix = np.zeros((41, 41), dtype=np.int64)
    matrix[1, 1] = 2
    matrix[1, 2] = 1
    matrix[2, 1] = 1
    matrix[2, 2] = 1
    metrics = metrics_from_confusion(matrix)
    assert metrics["per_class"][1]["iou"] == pytest.approx(0.5)
    assert metrics["per_class"][2]["iou"] == pytest.approx(1 / 3)
    assert metrics["per_class"][3]["iou"] is None
    assert metrics["known_class_miou"] == pytest.approx(5 / 12)
    assert metrics["known_classes_included"] == ["wall", "floor"]


def test_objectnav_six_indices_are_known_and_unique() -> None:
    assert set(OBJECTNAV_SIX) == {
        "chair",
        "couch",
        "potted plant",
        "bed",
        "toilet",
        "tv",
    }
    assert len(set(OBJECTNAV_SIX.values())) == 6
    assert all(index > 0 for index in OBJECTNAV_SIX.values())


def test_bootstrap_is_reproducible() -> None:
    first = bootstrap_scene_metric([0.1, 0.2, 0.3], 100, 7)
    second = bootstrap_scene_metric([0.1, 0.2, 0.3], 100, 7)
    assert first == second
