"""Native-resolution 41-way semantic inference."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from hm3d_semseg.camera.profile import CameraProfile, assert_camera_compatible
from hm3d_semseg.config.schema import ModelConfig
from hm3d_semseg.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from hm3d_semseg.exceptions import OptionalDependencyError
from hm3d_semseg.models.segformer import build_segformer, predict
from hm3d_semseg.taxonomy.constants import ID2LABEL
from hm3d_semseg.types import NumpyArray
from hm3d_semseg.utils.device import select_torch_device
from hm3d_semseg.utils.hashing import atomic_write_json
from hm3d_semseg.visualization.masks import colorize_mask, overlay_mask


class SemanticSegmenter:
    """Small reusable API returning probabilities and labels, never instances."""

    def __init__(
        self,
        model: Any,
        model_config: ModelConfig,
        *,
        device: str,
        temperature: float = 1.0,
        camera_profile: Optional[CameraProfile] = None,
        checkpoint: Optional[Path] = None,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise OptionalDependencyError("PyTorch is required for inference") from error
        self.torch = torch
        self.model = model.to(device).eval()
        self.model_config = model_config
        self.device = device
        self.temperature = temperature
        self.camera_profile = camera_profile
        self.checkpoint = checkpoint

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        device: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> "SemanticSegmenter":
        checkpoint = checkpoint.resolve()
        checkpoint_metadata_path = checkpoint / "checkpoint.json"
        checkpoint_metadata = (
            json.loads(checkpoint_metadata_path.read_text(encoding="utf-8"))
            if checkpoint_metadata_path.is_file()
            else {}
        )
        model_config = ModelConfig(
            model_id=str(
                checkpoint_metadata.get("model_id", "nvidia/segformer-b2-finetuned-ade-512-512")
            ),
            revision=checkpoint_metadata.get("model_revision"),
            num_labels=int(checkpoint_metadata.get("num_labels", 41)),
            align_corners=bool(checkpoint_metadata.get("align_corners", False)),
            local_files_only=True,
        )
        model = build_segformer(model_config, checkpoint=checkpoint)
        device = select_torch_device(device).device
        calibration_path = checkpoint / "calibration.json"
        resolved_temperature = 1.0 if temperature is None else temperature
        if temperature is None and calibration_path.is_file():
            resolved_temperature = float(
                json.loads(calibration_path.read_text(encoding="utf-8"))["temperature"]
            )
        camera_path = checkpoint / "camera_profile.yaml"
        camera = CameraProfile.load(camera_path) if camera_path.is_file() else None
        return cls(
            model,
            model_config,
            device=device,
            temperature=resolved_temperature,
            camera_profile=camera,
            checkpoint=checkpoint,
        )

    def assert_camera(
        self, camera_profile: CameraProfile, allow_mismatch: bool = False
    ) -> None:
        if self.camera_profile is None:
            raise ValueError("Checkpoint has no frozen camera profile")
        assert_camera_compatible(self.camera_profile, camera_profile, allow_mismatch)

    def __call__(self, rgb: NumpyArray) -> Dict[str, NumpyArray]:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB [H,W,3], got {rgb.shape} {rgb.dtype}")
        normalized = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = (
            self.torch.from_numpy(normalized.transpose(2, 0, 1).copy())
            .unsqueeze(0)
            .to(self.device)
        )
        with self.torch.inference_mode():
            output = predict(
                self.model,
                tensor,
                output_size=(int(rgb.shape[0]), int(rgb.shape[1])),
                align_corners=self.model_config.align_corners,
                temperature=self.temperature,
            )
        return {
            "probabilities": output.probabilities[0].cpu().numpy(),
            "labels": output.labels[0].to(self.torch.uint8).cpu().numpy(),
            "confidence": output.confidence[0].cpu().numpy(),
            "entropy": output.entropy[0].cpu().numpy(),
        }

    def infer_file(
        self,
        image: Path,
        output: Path,
        *,
        save_probabilities: bool = False,
    ) -> Dict[str, Any]:
        with Image.open(image) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        start = time.perf_counter()
        result = self(rgb)
        elapsed = time.perf_counter() - start
        output.mkdir(parents=True, exist_ok=True)
        labels = result["labels"]
        Image.fromarray(labels, mode="L").save(output / "class_ids.png")
        Image.fromarray(colorize_mask(labels), mode="RGB").save(output / "colorized.png")
        Image.fromarray(overlay_mask(rgb, labels), mode="RGB").save(output / "overlay.png")
        Image.fromarray(
            np.clip(result["confidence"] * 255, 0, 255).astype(np.uint8), mode="L"
        ).save(output / "confidence.png")
        entropy_max = np.log(41.0)
        Image.fromarray(
            np.clip(result["entropy"] / entropy_max * 255, 0, 255).astype(np.uint8),
            mode="L",
        ).save(output / "entropy.png")
        if save_probabilities:
            np.save(
                output / "probabilities.npy",
                result["probabilities"].astype(np.float32),
                allow_pickle=False,
            )
        metadata = {
            "source_image": str(image.resolve()),
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "model_id": self.model_config.model_id,
            "model_revision": self.model_config.revision,
            "class_names": ID2LABEL,
            "temperature": self.temperature,
            "calibrated": self.temperature != 1.0,
            "camera_profile_hash": (
                self.camera_profile.profile_hash if self.camera_profile is not None else None
            ),
            "preprocessing": {
                "resize": None,
                "aspect_ratio_preserved": True,
                "mean": IMAGENET_MEAN.tolist(),
                "std": IMAGENET_STD.tolist(),
                "reduce_labels": False,
            },
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "inference_seconds": elapsed,
            "probabilities_saved": save_probabilities,
        }
        atomic_write_json(output / "metadata.json", metadata)
        return metadata
