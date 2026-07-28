"""Deterministic scene-disjoint fit/development split creation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from hm3d_semseg.utils.hashing import atomic_write_json, atomic_write_text


def make_development_split(
    audit_path: Path, output: Path, seed: int = 2027, development_scenes: int = 15
) -> Dict[str, Any]:
    audit_file = audit_path / "audit.json" if audit_path.is_dir() else audit_path
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    scene_ids = sorted(audit["scene_class_coverage"])
    if len(scene_ids) <= development_scenes:
        raise ValueError(
            f"Need more than {development_scenes} scenes; audit has {len(scene_ids)}"
        )
    ranked = sorted(
        scene_ids,
        key=lambda scene: hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest(),
    )
    development = sorted(ranked[:development_scenes])
    fit = sorted(set(scene_ids) - set(development))
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output / "fit.txt", "\n".join(fit) + "\n")
    atomic_write_text(output / "development.txt", "\n".join(development) + "\n")
    coverage = audit["scene_class_coverage"]
    report = {
        "seed": seed,
        "fit_scenes": len(fit),
        "development_scenes": len(development),
        "fit": fit,
        "development": development,
        "development_class_scene_counts": {
            name: sum(name in coverage[scene] for scene in development)
            for name in sorted({name for scene in development for name in coverage[scene]})
        },
    }
    atomic_write_json(output / "split_report.json", report)
    return report


def make_calibration_split(
    audit_path: Path, output: Path, seed: int = 42027, fit_scenes: int = 12
) -> Dict[str, Any]:
    """Freeze disjoint temperature-fit and calibration-evaluation scene lists."""
    audit_file = audit_path / "audit.json" if audit_path.is_dir() else audit_path
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    scene_ids = sorted(audit["scene_class_coverage"])
    if len(scene_ids) <= fit_scenes:
        raise ValueError(
            f"Need more than {fit_scenes} validation scenes; audit has {len(scene_ids)}"
        )
    ranked = sorted(
        scene_ids,
        key=lambda scene: hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest(),
    )
    calibration_fit = sorted(ranked[:fit_scenes])
    calibration_evaluation = sorted(set(scene_ids) - set(calibration_fit))
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output / "calibration_fit.txt", "\n".join(calibration_fit) + "\n")
    atomic_write_text(
        output / "calibration_evaluation.txt",
        "\n".join(calibration_evaluation) + "\n",
    )
    report = {
        "seed": seed,
        "calibration_fit": calibration_fit,
        "calibration_evaluation": calibration_evaluation,
        "disjoint": not bool(set(calibration_fit) & set(calibration_evaluation)),
    }
    atomic_write_json(output / "calibration_split_report.json", report)
    return report
