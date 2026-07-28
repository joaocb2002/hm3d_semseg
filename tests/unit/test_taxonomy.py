from pathlib import Path

import numpy as np
import pytest

from hm3d_semseg.config.schema import TaxonomyConfig
from hm3d_semseg.taxonomy.constants import ID2LABEL, LABEL2ID, NUM_CLASSES
from hm3d_semseg.taxonomy.mapping import MatterportMapping, TaxonomyMapper
from hm3d_semseg.taxonomy.semantic_descriptor import parse_semantic_descriptor

pytestmark = pytest.mark.unit


def test_quoted_semantic_descriptor_parsing(tmp_path: Path) -> None:
    path = tmp_path / "scene.semantic.txt"
    path.write_text(
        'HM3D Semantic Annotations\n1,ABCDEF,"chair, dining",2\n2,123456,"wall",1\n'
    )
    entries = parse_semantic_descriptor(path)
    assert entries[1].raw_category == "chair, dining"
    assert entries[2].semantic_id == 2


def test_real_parser_handles_quoted_mapping(mapping_file: Path) -> None:
    mapping = MatterportMapping.from_file(mapping_file)
    row = mapping.lookup("quoted, raw label")
    assert row is not None
    assert row.mpcat40index == 39


def test_semantic_id_mapping_has_no_offset(mapping_file: Path) -> None:
    mapper = TaxonomyMapper(MatterportMapping.from_file(mapping_file), TaxonomyConfig())
    semantic = np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.int32)
    mask, decisions = mapper.map_semantic_observation(
        semantic, {1: "wall", 2: "floor", 3: "chair"}
    )
    assert mask.tolist() == [[1, 2, 3], [3, 2, 1]]
    assert decisions[1].target_id == 1


def test_unknown_and_ignore_are_distinct(mapping_file: Path) -> None:
    mapper = TaxonomyMapper(MatterportMapping.from_file(mapping_file), TaxonomyConfig())
    assert mapper.map_raw_name("unlabeled").target_id == 0
    assert mapper.map_raw_name("remove").target_id == 255
    assert mapper.map_raw_name("not in mapping").target_id == 255
    assert mapper.map_raw_name(None).target_id == 255


def test_output_ids_are_contiguous() -> None:
    assert NUM_CLASSES == 41
    assert set(ID2LABEL) == set(range(41))
    assert set(LABEL2ID.values()) == set(range(41))
    assert LABEL2ID["unknown"] == 0
