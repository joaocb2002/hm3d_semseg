from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hm3d_semseg.data.schema import ManifestRecord
from hm3d_semseg.diagnostics.qualitative import (
    save_contact_sheet,
    save_qualitative_prediction,
    select_qualitative_records,
)

pytestmark = pytest.mark.unit


def _record(sample_id: str, scene_id: str, classes: list[int]) -> ManifestRecord:
    histogram = [0] * 41
    for class_id in classes:
        histogram[class_id] = 100
    return ManifestRecord(
        sample_id=sample_id,
        split="train",
        scene_id=scene_id,
        rgb=f"rgb/{sample_id}.png",
        mask=f"mask/{sample_id}.png",
        metadata=f"metadata/{sample_id}.json",
        depth=None,
        width=8,
        height=6,
        camera_profile_hash="camera",
        taxonomy_mapping_hash="taxonomy",
        class_histogram=histogram,
        ignored_pixels=0,
        unknown_pixels=0,
    )


def test_qualitative_selection_is_deterministic_scene_diverse_and_prediction_free(
    tmp_path: Path,
) -> None:
    del tmp_path
    records = [
        _record("a", "scene-1", [1, 2]),
        _record("b", "scene-1", [3]),
        _record("c", "scene-2", [14]),
        _record("d", "scene-3", [18]),
    ]

    first = select_qualitative_records(records, 3, seed=2027)
    second = select_qualitative_records(records, 3, seed=2027)

    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert len({item.scene_id for item in first}) == 3
    assert {item.sample_id for item in first} <= {item.sample_id for item in records}


def test_qualitative_outputs_store_static_inputs_and_compact_epoch_maps(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "rgb").mkdir(parents=True)
    record = _record("sample", "scene", [1, 2])
    Image.fromarray(np.full((6, 8, 3), 128, dtype=np.uint8), mode="RGB").save(
        dataset / record.rgb
    )
    target = np.ones((6, 8), dtype=np.uint8)
    target[0, 0] = 255
    prediction = target.copy()
    prediction[1, 1] = 2
    confidence = np.full((6, 8), 0.75, dtype=np.float32)
    output = tmp_path / "qualitative"

    report = save_qualitative_prediction(
        dataset_root=dataset,
        record=record,
        target=target,
        prediction=prediction,
        confidence=confidence,
        output=output,
        epoch=3,
    )
    sheet = save_contact_sheet(output, [report], epoch=3)

    assert (output / report["rgb"]).is_file()
    assert (output / report["ground_truth"]).is_file()
    assert (output / report["prediction"]).is_file()
    assert (output / report["error"]).is_file()
    assert (output / report["confidence"]).is_file()
    with Image.open(sheet) as handle:
        assert handle.size == (960, 188)
