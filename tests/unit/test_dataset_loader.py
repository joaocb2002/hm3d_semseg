import json
from pathlib import Path

import pytest

from hm3d_semseg.data.dataset import OfflineSegmentationDataset, select_manifest_records
from hm3d_semseg.data.schema import ManifestRecord

pytestmark = pytest.mark.unit


def _record(index: int, scene_id: str = "scene") -> ManifestRecord:
    return ManifestRecord(
        sample_id=f"sample-{index}",
        split="fit",
        scene_id=scene_id,
        rgb=f"rgb/{index}.png",
        mask=f"mask/{index}.png",
        metadata=f"metadata/{index}.json",
        depth=None,
        width=3,
        height=2,
        camera_profile_hash="camera",
        taxonomy_mapping_hash="mapping",
        class_histogram=[6] + [0] * 40,
        ignored_pixels=0,
        unknown_pixels=0,
    )


def test_max_samples_uses_deterministic_manifest_prefix(tmp_path: Path) -> None:
    records = [_record(index) for index in range(6)]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(record.to_dict()) + "\n" for record in records),
        encoding="utf-8",
    )

    dataset = OfflineSegmentationDataset(tmp_path, max_samples=4)

    assert len(dataset) == 4
    assert [record.sample_id for record in dataset.records] == [
        "sample-0",
        "sample-1",
        "sample-2",
        "sample-3",
    ]


def test_max_samples_must_be_positive(tmp_path: Path) -> None:
    (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="max_samples must be positive"):
        OfflineSegmentationDataset(tmp_path, max_samples=0)


def test_scene_diverse_selection_is_deterministic_and_uses_distinct_scenes() -> None:
    records = [
        _record(scene * 10 + view, scene_id=f"scene-{scene}")
        for scene in range(6)
        for view in range(3)
    ]

    first = select_manifest_records(records, 4, strategy="scene_diverse", seed=2027)
    repeated = select_manifest_records(records, 4, strategy="scene_diverse", seed=2027)

    assert [record.sample_id for record in first] == [
        record.sample_id for record in repeated
    ]
    assert len({record.scene_id for record in first}) == 4


def test_explicit_sample_ids_preserve_requested_order(tmp_path: Path) -> None:
    records = [_record(index) for index in range(4)]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(record.to_dict()) + "\n" for record in records),
        encoding="utf-8",
    )

    dataset = OfflineSegmentationDataset(
        tmp_path,
        sample_ids=["sample-3", "sample-1"],
    )

    assert [record.sample_id for record in dataset.records] == ["sample-3", "sample-1"]
