"""Recoverable training-state checkpoint helpers."""

from __future__ import annotations

import os
import random
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

from hm3d_semseg.utils.hashing import atomic_write_json


@contextmanager
def _without_model_save_progress() -> Iterator[None]:
    """Hide Transformers' shard bar while preserving the caller's prior setting."""
    try:
        from transformers.utils import logging as transformers_logging
    except ImportError:
        yield
        return

    is_enabled = getattr(transformers_logging, "is_progress_bar_enabled", None)
    was_enabled = bool(is_enabled()) if is_enabled is not None else True
    if was_enabled:
        transformers_logging.disable_progress_bar()
    try:
        yield
    finally:
        if was_enabled:
            transformers_logging.enable_progress_bar()


def atomic_torch_save(value: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_checkpoint(
    target: Path,
    model: Any,
    *,
    optimizer: Optional[Any],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    step: int,
    primary_metric: Optional[float],
    camera_profile_path: Path,
    epochs_without_improvement: int = 0,
    model_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Build a complete directory, then atomically swap it into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent)))
    backup = target.parent / f".{target.name}.previous"
    try:
        with _without_model_save_progress():
            model.save_pretrained(temporary, safe_serialization=True)
        state: Dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "primary_metric": primary_metric,
            "epochs_without_improvement": epochs_without_improvement,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch_random_state(),
        }
        atomic_torch_save(state, temporary / "training_state.pt")
        shutil.copy2(camera_profile_path, temporary / "camera_profile.yaml")
        checkpoint_metadata = {
            "epoch": epoch,
            "step": step,
            "primary_metric": primary_metric,
            "epochs_without_improvement": epochs_without_improvement,
        }
        checkpoint_metadata.update(model_metadata or {})
        atomic_write_json(temporary / "checkpoint.json", checkpoint_metadata)
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        os.replace(temporary, target)
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise


def load_training_state(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    import torch

    return torch.load(path / "training_state.pt", map_location=map_location, weights_only=False)


def torch_random_state() -> Dict[str, Any]:
    """Capture CPU and available CUDA generators at an epoch boundary."""
    import torch

    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state: Dict[str, Any]) -> None:
    """Restore all saved generators so epoch-boundary resume is reproducible."""
    import torch

    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])
    if state.get("numpy_random_state") is not None:
        np.random.set_state(state["numpy_random_state"])
    torch_state = state.get("torch_random_state") or {}
    if torch_state.get("cpu") is not None:
        torch.set_rng_state(torch_state["cpu"])
    if torch.cuda.is_available() and torch_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(torch_state["cuda"])


def update_checkpoint_progress(
    checkpoint: Path,
    *,
    primary_metric: Optional[float],
    epochs_without_improvement: int,
) -> None:
    """Atomically update post-evaluation state without rewriting model weights."""
    state = load_training_state(checkpoint)
    state["primary_metric"] = primary_metric
    state["epochs_without_improvement"] = epochs_without_improvement
    state["python_random_state"] = random.getstate()
    state["numpy_random_state"] = np.random.get_state()
    state["torch_random_state"] = torch_random_state()
    atomic_torch_save(state, checkpoint / "training_state.pt")
    metadata_path = checkpoint / "checkpoint.json"
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["primary_metric"] = primary_metric
    metadata["epochs_without_improvement"] = epochs_without_improvement
    atomic_write_json(metadata_path, metadata)
