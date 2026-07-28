"""HM3D ``*.semantic.txt`` parsing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from hm3d_semseg.exceptions import HM3DSemsegError


@dataclass(frozen=True)
class SemanticDescriptorEntry:
    """One scene-specific semantic instance."""

    semantic_id: int
    color_hex: str
    raw_category: str
    region_id: int


def parse_semantic_descriptor(path: Path) -> Dict[int, SemanticDescriptorEntry]:
    """Parse quoted category names without assuming a pixel-ID offset."""
    if not path.is_file():
        raise FileNotFoundError(f"Semantic descriptor does not exist: {path}")
    entries: Dict[int, SemanticDescriptorEntry] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first = handle.readline().strip()
        if first != "HM3D Semantic Annotations":
            raise HM3DSemsegError(f"Unexpected semantic descriptor header in {path}: {first!r}")
        reader = csv.reader(handle, delimiter=",", quotechar='"', strict=True)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 4:
                raise HM3DSemsegError(
                    f"{path}:{line_number}: expected 4 CSV fields, found {len(row)}"
                )
            semantic_id = int(row[0])
            if semantic_id in entries:
                raise HM3DSemsegError(
                    f"{path}:{line_number}: duplicate semantic ID {semantic_id}"
                )
            entries[semantic_id] = SemanticDescriptorEntry(
                semantic_id=semantic_id,
                color_hex=row[1].strip(),
                raw_category=row[2],
                region_id=int(row[3]),
            )
    return entries
