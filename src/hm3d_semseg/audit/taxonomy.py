"""Split-wide semantic descriptor and rendered-pixel census."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from hm3d_semseg.config.schema import ProjectConfig
from hm3d_semseg.data.schema import load_manifest
from hm3d_semseg.scenes.discovery import discover_scenes
from hm3d_semseg.taxonomy.constants import ID2LABEL, OBJECTNAV_SIX
from hm3d_semseg.taxonomy.mapping import MatterportMapping, TaxonomyMapper
from hm3d_semseg.taxonomy.semantic_descriptor import parse_semantic_descriptor
from hm3d_semseg.utils.hashing import atomic_write_json


def audit_taxonomy(
    config: ProjectConfig,
    split: str,
    output: Path,
    rendered_dataset: Optional[Path] = None,
) -> Dict[str, Any]:
    if config.paths.hm3d_root is None or config.paths.taxonomy_mapping is None:
        raise ValueError("paths.hm3d_root and paths.taxonomy_mapping are required")
    mapping = MatterportMapping.from_file(
        config.paths.taxonomy_mapping, config.taxonomy.expected_mapping_sha256
    )
    mapper = TaxonomyMapper(mapping, config.taxonomy)
    scenes = discover_scenes(config.paths.hm3d_root, split, require_complete=False)
    raw_counts: Counter = Counter()
    target_objects: Counter = Counter()
    status_counts: Counter = Counter()
    scene_coverage: Dict[str, Counter] = {}
    raw_decisions: Dict[str, Dict[str, Any]] = {}
    for scene in scenes:
        if scene.semantic_descriptor is None:
            continue
        entries = parse_semantic_descriptor(scene.semantic_descriptor)
        coverage: Counter = Counter()
        for entry in entries.values():
            raw_counts[entry.raw_category] += 1
            decision = mapper.map_raw_name(entry.raw_category)
            status_counts[decision.status] += 1
            target_objects[decision.target_id] += 1
            coverage[decision.target_id] += 1
            raw_decisions.setdefault(entry.raw_category, vars(decision))
        scene_coverage[scene.scene_id] = coverage
    pixel_counts = [0] * 41
    image_count = 0
    if rendered_dataset is not None:
        records = load_manifest(rendered_dataset / "manifest.jsonl")
        image_count = len(records)
        for record in records:
            pixel_counts = [
                left + right for left, right in zip(pixel_counts, record.class_histogram)
            ]
    report = {
        "split": split,
        "discovered_scenes": len(scenes),
        "complete_scenes": sum(scene.complete for scene in scenes),
        "semantic_objects": int(sum(raw_counts.values())),
        "mapping_sha256": mapping.sha256,
        "mapping_source_url": mapping.source_url,
        "raw_label_counts": dict(sorted(raw_counts.items())),
        "raw_label_decisions": dict(sorted(raw_decisions.items())),
        "mapping_status_counts": dict(status_counts),
        "object_instance_counts": {
            ID2LABEL[index] if index in ID2LABEL else f"ignore_{index}": count
            for index, count in sorted(target_objects.items())
        },
        "zero_support_classes": [
            ID2LABEL[index] for index in range(1, 41) if target_objects[index] == 0
        ],
        "objectnav_six_support": {
            goal: int(target_objects[class_id]) for goal, class_id in OBJECTNAV_SIX.items()
        },
        "rendered_image_count": image_count,
        "rendered_pixel_counts": {ID2LABEL[index]: pixel_counts[index] for index in range(41)},
        "scene_class_coverage": {
            scene_id: {
                ID2LABEL[index]: int(count)
                for index, count in sorted(coverage.items())
                if index in ID2LABEL
            }
            for scene_id, coverage in sorted(scene_coverage.items())
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "audit.json", report)
    with (output / "scene_by_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene_id"] + [ID2LABEL[index] for index in range(41)])
        for scene_id, coverage in sorted(scene_coverage.items()):
            writer.writerow([scene_id] + [coverage[index] for index in range(41)])
    return report
