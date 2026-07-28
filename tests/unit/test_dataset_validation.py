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
