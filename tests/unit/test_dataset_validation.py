import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from hm3d_semseg.data.schema import ManifestRecord
from hm3d_semseg.data.validate import validate_dataset

pytestmark = pytest.mark.unit


def test_scene_split_leakage_is_reported(tmp_path: Path) -> None:
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "camera_profile_hash": "camera",
                "taxonomy_mapping_hash": "mapping",
            }
        )
    )
    records = []
    for index, split in enumerate(("fit", "development")):
        base = tmp_path / split / "scene"
        for name in ("rgb", "mask", "metadata"):
            (base / name).mkdir(parents=True, exist_ok=True)
        sample = f"sample-{index}"
        Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(
            base / "rgb" / f"{sample}.png"
        )
        Image.fromarray(np.zeros((2, 3), dtype=np.uint8)).save(base / "mask" / f"{sample}.png")
        (base / "metadata" / f"{sample}.json").write_text(json.dumps({"sample_id": sample}))
        records.append(
            ManifestRecord(
                sample,
                split,
                "scene",
                str((base / "rgb" / f"{sample}.png").relative_to(tmp_path)),
                str((base / "mask" / f"{sample}.png").relative_to(tmp_path)),
                str((base / "metadata" / f"{sample}.json").relative_to(tmp_path)),
                None,
                3,
                2,
                "camera",
                "mapping",
                [6] + [0] * 40,
                0,
                6,
            )
        )
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(record.to_dict()) + "\n" for record in records)
    )
    report = validate_dataset(tmp_path, raise_on_error=False)
    assert not report["valid"]
    assert any("scene split leakage" in error for error in report["errors"])


def test_selected_validation_does_not_decode_unselected_files(tmp_path: Path) -> None:
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "camera_profile_hash": "camera",
                "taxonomy_mapping_hash": "mapping",
            }
        )
    )
    base = tmp_path / "fit" / "scene"
    for name in ("rgb", "mask", "metadata"):
        (base / name).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(base / "rgb" / "kept.png")
    Image.fromarray(np.zeros((2, 3), dtype=np.uint8)).save(base / "mask" / "kept.png")
    (base / "metadata" / "kept.json").write_text(json.dumps({"sample_id": "kept"}))
    records = [
        ManifestRecord(
            sample_id=sample_id,
            split="fit",
            scene_id=f"scene-{sample_id}",
            rgb=f"fit/scene/rgb/{sample_id}.png",
            mask=f"fit/scene/mask/{sample_id}.png",
            metadata=f"fit/scene/metadata/{sample_id}.json",
            depth=None,
            width=3,
            height=2,
            camera_profile_hash="camera",
            taxonomy_mapping_hash="mapping",
            class_histogram=[6] + [0] * 40,
            ignored_pixels=0,
            unknown_pixels=6,
        )
        for sample_id in ("kept", "not-on-disk")
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(record.to_dict()) + "\n" for record in records)
    )

    report = validate_dataset(tmp_path, raise_on_error=False, sample_ids=["kept"])

    assert report["validation_scope"] == "selected"
    assert report["samples"] == 1
    assert report["manifest_samples"] == 2
    assert not any("not-on-disk: missing" in error for error in report["errors"])
