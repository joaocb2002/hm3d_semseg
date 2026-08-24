"""Collision-safe allocation of experiment output directories."""

from __future__ import annotations

from pathlib import Path


def allocate_run_directory(
    runs_root: Path,
    requested_name: str,
    *,
    resuming: bool,
) -> Path:
    """Create a fresh run directory, or reuse its explicit resume target."""
    runs_root.mkdir(parents=True, exist_ok=True)
    requested = runs_root / requested_name
    if resuming:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    candidate = requested
    sequence = 2
    while True:
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = runs_root / f"{requested_name}-{sequence:03d}"
            sequence += 1
