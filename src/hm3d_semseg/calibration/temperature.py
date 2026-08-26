"""Fit one scalar temperature on dedicated calibration scenes."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

from hm3d_semseg.camera.profile import CameraProfile, assert_camera_compatible
from hm3d_semseg.config.schema import ProjectConfig
from hm3d_semseg.data.dataset import OfflineSegmentationDataset
from hm3d_semseg.data.validate import validate_dataset
from hm3d_semseg.models.segformer import build_segformer, upsample_logits
from hm3d_semseg.utils.device import select_torch_device
from hm3d_semseg.utils.hashing import atomic_write_json


def fit_temperature(
    checkpoint: Path,
    dataset_root: Path,
    output: Path,
    config: ProjectConfig,
    *,
    epochs: int = 5,
    learning_rate: float = 0.05,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Optimize log-temperature with pixels streamed from fixed scenes."""
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    validate_dataset(dataset_root)
    assert_camera_compatible(
        CameraProfile.load(checkpoint / "camera_profile.yaml"),
        CameraProfile.load(dataset_root / "camera_profile.yaml"),
        config.camera.allow_mismatch,
    )
    dataset = OfflineSegmentationDataset(dataset_root, augment=False)
    device_selection = select_torch_device(device)
    device = device_selection.device
    loader: Any = DataLoader(
        dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.evaluation.workers,
    )
    model = build_segformer(config.model, checkpoint=checkpoint).to(device).eval()
    log_temperature = torch.nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.Adam([log_temperature], lr=learning_rate)
    losses: List[float] = []
    for _ in range(epochs):
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                raw = model(pixel_values=pixels).logits
                logits = upsample_logits(
                    raw, tuple(labels.shape[-2:]), config.model.align_corners
                )
            temperature_tensor = log_temperature.exp().clamp(0.05, 20.0)
            loss: torch.Tensor = functional.cross_entropy(
                logits / temperature_tensor,
                labels,
                ignore_index=config.taxonomy.ignore_index,
            )
            optimizer.zero_grad()
            cast(Callable[[], None], loss.backward)()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    fitted_temperature = float(
        log_temperature.detach().exp().clamp(0.05, 20.0).cpu()
    )
    scenes = sorted({record.scene_id for record in dataset.records})
    provenance = {
        "temperature": fitted_temperature,
        "checkpoint": str(checkpoint.resolve()),
        "calibration_dataset": str(dataset_root.resolve()),
        "calibration_scenes": scenes,
        "optimization_epochs": epochs,
        "learning_rate": learning_rate,
        "device_selection": device_selection.to_dict(),
        "initial_nll": losses[0] if losses else None,
        "final_nll": losses[-1] if losses else None,
    }
    if output.exists():
        raise FileExistsError(
            f"Calibration output already exists; refusing to overwrite: {output}"
        )
    shutil.copytree(checkpoint, output)
    atomic_write_json(output / "calibration.json", provenance)
    return provenance
