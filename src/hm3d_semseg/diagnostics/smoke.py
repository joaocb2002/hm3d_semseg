"""Tiny generation-to-inference smoke workflow."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

from hm3d_semseg.config.schema import ProjectConfig
from hm3d_semseg.data.generate import generate_dataset
from hm3d_semseg.data.schema import load_manifest
from hm3d_semseg.data.validate import validate_dataset
from hm3d_semseg.evaluation.run import evaluate_model
from hm3d_semseg.inference.api import SemanticSegmenter
from hm3d_semseg.training.loop import train


def run_smoke_test(config: ProjectConfig) -> Dict[str, Any]:
    """Render four frames, train one epoch, reload, evaluate, and infer."""
    if config.paths.generated_data_root is None or config.paths.runs_root is None:
        raise ValueError("paths.generated_data_root and paths.runs_root are required")
    smoke = copy.deepcopy(config)
    dataset_root = config.paths.generated_data_root / "smoke-pilot"
    smoke.dataset.name = "smoke-pilot"
    smoke.dataset.output_dir = dataset_root
    smoke.dataset.split = "minival"
    smoke.dataset.rgb_codec = "png"
    smoke.dataset.store_depth = False
    smoke.dataset.max_scenes = 1
    smoke.dataset.max_samples_per_scene = 4
    smoke.camera.pitch_degrees = [0.0]
    smoke.camera.require_explicit_pitch = False
    smoke.sampling.positions_per_scene = 1
    smoke.sampling.yaws_per_position = 4
    smoke.sampling.class_aware_fraction = 0.0
    generation = generate_dataset(smoke, max_scenes=1, max_samples_per_scene=4)
    validation = validate_dataset(dataset_root)
    if validation["samples"] < 2:
        raise RuntimeError(
            f"Smoke generation produced {validation['samples']} valid samples; need at least 2"
        )

    smoke.training.train_dataset = dataset_root
    smoke.training.development_dataset = None
    smoke.training.run_name = "smoke"
    smoke.training.epochs = 1
    smoke.training.batch_size = 1
    smoke.training.workers = 0
    smoke.training.amp = False
    smoke.training.warmup_fraction = 0.0
    smoke.training.resume = None
    smoke.training.class_weights = None
    smoke.training.class_weighting = "none"
    smoke.augmentation.horizontal_flip_probability = 0.0
    smoke.augmentation.color_jitter = 0.0
    smoke.augmentation.blur_probability = 0.0
    smoke.augmentation.sensor_noise_std = 0.0
    smoke.model.local_files_only = True
    training = train(smoke)
    run_root = Path(training["run"])
    checkpoint = run_root / "checkpoints" / "last"
    evaluation = evaluate_model(
        checkpoint,
        dataset_root,
        run_root / "evaluation-diagnostic",
        smoke,
    )
    record = load_manifest(dataset_root / "manifest.jsonl")[0]
    inference = SemanticSegmenter.from_checkpoint(checkpoint).infer_file(
        dataset_root / record.rgb,
        run_root / "inference-diagnostic",
        save_probabilities=False,
    )
    return {
        "generation": generation,
        "validation": validation,
        "training": training,
        "evaluation_known_class_miou": evaluation["global"]["known_class_miou"],
        "inference": inference,
        "diagnostic_only": True,
    }
