from __future__ import annotations

from pathlib import Path

import pytest

from hm3d_semseg.taxonomy.constants import MPCAT40_NAMES


@pytest.fixture
def mapping_file(tmp_path: Path) -> Path:
    """Small complete authoritative-shape mapping with policy rows."""
    path = tmp_path / "mapping.tsv"
    lines = ["index\traw_category\tcategory\tmpcat40index\tmpcat40"]
    for index, name in enumerate(MPCAT40_NAMES, start=1):
        lines.append(f"{index}\t{name}\t{name}\t{index}\t{name}")
    lines.extend(
        [
            '41\t"quoted, raw label"\tobject\t39\tobjects',
            "42\tremove\tremove\t0\tvoid",
            "43\tunlabeled\tunlabeled\t41\tunlabeled",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
