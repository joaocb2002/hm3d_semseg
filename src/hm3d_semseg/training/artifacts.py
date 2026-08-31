"""Stable paths for organized training-run artifacts.

Writers use the version-2 layout. Readers accept both layouts so ``report-run``
and ``compare-runs`` continue to work for historical runs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional, Tuple

LAYOUT_NAME = "organized-training-run-v2"


def records_root(run: Path) -> Path:
    return run / "records"


def provenance_root(run: Path) -> Path:
    return run / "provenance"


def report_root(run: Path) -> Path:
    return run / "report"


def plots_root(run: Path) -> Path:
    return report_root(run) / "summary_metrics_plots"


def qualitative_root(run: Path) -> Path:
    return run / "diagnostics" / "qualitative"


def development_evaluations_root(run: Path) -> Path:
    return run / "diagnostics" / "epoch_evaluations" / "development"


def development_evaluation_root(run: Path, epoch: int) -> Path:
    return development_evaluations_root(run) / f"epoch_{epoch:03d}"


def existing_artifact(run: Path, current: str, legacy: str) -> Path:
    """Return the current path when present, otherwise the historical path."""
    current_path = run / current
    return current_path if current_path.is_file() else run / legacy


def metrics_path(run: Path) -> Path:
    return existing_artifact(run, "records/metrics.jsonl", "metrics.jsonl")


def metrics_summary_path(run: Path) -> Path:
    return existing_artifact(
        run, "records/metrics_summary.json", "metrics_summary.json"
    )


def run_summary_path(run: Path) -> Path:
    return existing_artifact(run, "records/run_summary.json", "summary.json")


def provenance_path(run: Path) -> Path:
    return existing_artifact(run, "provenance/provenance.json", "provenance.json")


def parameter_counts_path(run: Path) -> Path:
    return existing_artifact(
        run, "provenance/parameter_counts.json", "parameter_counts.json"
    )


def resolved_config_path(run: Path) -> Path:
    return existing_artifact(
        run, "provenance/resolved_config.yaml", "resolved_config.yaml"
    )


def existing_qualitative_root(run: Path) -> Path:
    current = qualitative_root(run)
    if current.is_dir():
        return current
    return run / "diagnostics" / "training_progress" / "qualitative"


def iter_development_evaluations(run: Path) -> Iterator[Tuple[int, Path]]:
    """Yield ``(zero-based epoch, summary path)`` in deterministic order."""
    current = development_evaluations_root(run)
    pattern = re.compile(r"epoch_(\d+)$")
    paths = sorted(current.glob("epoch_*/summary.json")) if current.is_dir() else []
    if not paths:
        pattern = re.compile(r"evaluation-epoch-(\d+)$")
        paths = sorted(run.glob("evaluation-epoch-*/summary.json"))
    for path in paths:
        match = pattern.match(path.parent.name)
        if match:
            yield int(match.group(1)), path


def existing_evaluation_path(run: Path, epoch: int) -> Optional[Path]:
    current = development_evaluation_root(run, epoch)
    if current.is_dir():
        return current
    legacy = run / f"evaluation-epoch-{epoch:03d}"
    return legacy if legacy.is_dir() else None
