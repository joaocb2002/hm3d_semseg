"""Versioned directory-backed manifest schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DATASET_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    split: str
    scene_id: str
    rgb: str
    mask: str
    metadata: str
    depth: Optional[str]
    width: int
    height: int
    camera_profile_hash: str
    taxonomy_mapping_hash: str
    class_histogram: List[int]
    ignored_pixels: int
    unknown_pixels: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ManifestRecord":
        return cls(**value)


def load_manifest(path: Path) -> List[ManifestRecord]:
    records: List[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(ManifestRecord.from_dict(value))
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid manifest record") from error
    return records


def iter_manifest(path: Path) -> Iterator[ManifestRecord]:
    for record in load_manifest(path):
        yield record
