from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hm3d_semseg.data.storage import load_mask, save_mask
from hm3d_semseg.data.transforms import horizontal_flip, resize_pair

pytestmark = pytest.mark.unit


def test_mask_round_trip_preserves_unknown_and_ignore(tmp_path: Path) -> None:
    mask = np.asarray([[0, 1, 40, 255]], dtype=np.uint8)
    path = tmp_path / "mask.png"
    save_mask(mask, path)
    assert np.array_equal(load_mask(path), mask)


def test_alignment_and_class_zero_survive_flip() -> None:
    mask = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
    rgb = np.repeat(mask[:, :, None], 3, axis=2)
    flipped_rgb, flipped_mask = horizontal_flip(rgb, mask)
    assert np.array_equal(flipped_rgb[:, :, 0], flipped_mask)
    assert 0 in flipped_mask


def test_mask_resize_is_nearest_neighbor() -> None:
    mask = np.asarray([[0, 40], [255, 1]], dtype=np.uint8)
    rgb = np.repeat(np.where(mask == 255, 0, mask)[:, :, None], 3, axis=2)
    _, resized = resize_pair(rgb, mask, 4, 4)
    assert set(np.unique(resized)) == {0, 1, 40, 255}


def test_alignment_visualization_before_and_after_augmentation(
    tmp_path: Path,
) -> None:
    mask = np.tile(np.arange(8, dtype=np.uint8), (6, 1))
    rgb = np.repeat((mask * 20)[:, :, None], 3, axis=2)
    flipped_rgb, flipped_mask = horizontal_flip(rgb, mask)
    assert np.array_equal(flipped_rgb[:, :, 0] // 20, flipped_mask)
    panel = np.concatenate(
        [
            rgb,
            np.repeat(mask[:, :, None], 3, axis=2) * 20,
            flipped_rgb,
            np.repeat(flipped_mask[:, :, None], 3, axis=2) * 20,
        ],
        axis=1,
    )
    output = tmp_path / "augmentation_alignment.png"
    Image.fromarray(panel).save(output)
    assert output.is_file()
