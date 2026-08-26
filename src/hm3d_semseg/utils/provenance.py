"""Runtime and Git provenance collection."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hm3d_semseg.utils.hashing import sha256_file


def package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    """Return installed versions without importing heavy packages."""
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit(path: Optional[Path]) -> Optional[str]:
    """Return a repository commit, or ``None`` for an absent/non-Git path."""
    if path is None or not path.exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def collect_provenance(habitat_lab_root: Optional[Path] = None) -> Dict[str, Any]:
    """Collect reproducibility metadata without initializing CUDA or Habitat."""
    project_root = Path(__file__).resolve().parents[3]
    constraint_files = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name in ("training.txt", "quality.txt")
        if (path := project_root / "constraints" / name).is_file()
    }
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(
            [
                "hm3d-semseg",
                "habitat-lab",
                "habitat-sim",
                "torch",
                "torchvision",
                "transformers",
                "huggingface-hub",
                "safetensors",
                "numpy",
                "Pillow",
                "PyYAML",
                "scipy",
                "matplotlib",
                "tensorboard",
                "tqdm",
                "typer",
            ]
        ),
        "dependency_constraints": constraint_files,
        "hm3d_semseg_commit": git_commit(project_root),
        "habitat_lab_commit": git_commit(habitat_lab_root),
    }
