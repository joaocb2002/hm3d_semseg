import json
from pathlib import Path

import pytest

from hm3d_semseg.data.dataset import OfflineSegmentationDataset
from hm3d_semseg.data.schema import ManifestRecord

pytestmark = pytest.mark.unit


def _record(index: int) -> ManifestRecord:
    return ManifestRecord(
        sample_id=f"sample-{index}",
        split="fit",
        scene_id="scene",
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
