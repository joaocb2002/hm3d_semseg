"""Dataclass-backed configuration schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PathsConfig:
    """External paths. Every configured value must resolve to an absolute path."""

    habitat_lab_root: Optional[Path] = None
    hm3d_root: Optional[Path] = None
    scene_dataset_config: Optional[Path] = None
    objectnav_config: Optional[Path] = None
    taxonomy_mapping: Optional[Path] = None
    generated_data_root: Optional[Path] = None
    runs_root: Optional[Path] = None
    cache_root: Optional[Path] = None


@dataclass
class CameraConfig:
    profile: Optional[Path] = None
    pitch_degrees: Optional[List[float]] = None
    require_explicit_pitch: bool = False
    allow_mismatch: bool = False


@dataclass
class TaxonomyPolicyConfig:
    void: str = "ignore"
    remove: str = "ignore"
    unlabeled: str = "unknown"
    unknown: str = "unknown"
    missing_id: str = "ignore"
    unmapped_name: str = "ignore"


@dataclass
class TaxonomyConfig:
    ignore_index: int = 255
    expected_mapping_sha256: Optional[str] = (
        "36e40c25cbe32c8bf34ef55f199f194671045106914dd09b1581aeedcf051a05"
    )
    policy: TaxonomyPolicyConfig = field(default_factory=TaxonomyPolicyConfig)


@dataclass
class SamplingConfig:
    seed: int = 2027
    positions_per_scene: int = 32
    yaws_per_position: int = 4
    min_position_distance_m: float = 1.0
    floor_separation_m: float = 1.5
    max_attempts_per_position: int = 100
    min_valid_pixel_fraction: float = 0.05
    max_unknown_fraction: float = 0.95
    class_aware_fraction: float = 0.0


@dataclass
class DatasetConfig:
    schema_version: str = "1.0"
    name: str = "pilot"
    split: str = "minival"
    output_dir: Optional[Path] = None
    rgb_codec: str = "png"
    jpeg_quality: int = 95
    store_depth: bool = False
    max_scenes: Optional[int] = 1
    max_samples_per_scene: Optional[int] = 8
    validation_only: bool = False


@dataclass
class AugmentationConfig:
    horizontal_flip_probability: float = 0.5
    color_jitter: float = 0.1
    blur_probability: float = 0.05
    sensor_noise_std: float = 0.01


@dataclass
class ModelConfig:
    model_id: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    revision: Optional[str] = None
    num_labels: int = 41
    align_corners: bool = False
    local_files_only: bool = False


@dataclass
class TrainingConfig:
    seed: int = 2027
    device: str = "auto"
    train_dataset: Optional[Path] = None
    development_dataset: Optional[Path] = None
    run_name: str = "segformer_b2_baseline"
    epochs: int = 20
    batch_size: int = 2
    workers: int = 4
    encoder_learning_rate: float = 6e-5
    classifier_learning_rate: float = 6e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    amp: bool = True
    resume: Optional[Path] = None
    class_weights: Optional[Path] = None
    class_weighting: str = "none"
    class_weight_cap: float = 5.0
    early_stopping_patience: Optional[int] = None


@dataclass
class EvaluationConfig:
    batch_size: int = 1
    workers: int = 2
    bootstrap_seed: int = 2027
    bootstrap_samples: int = 1000
    calibration_bins: int = 15


@dataclass
class ProjectConfig:
    """Fully resolved project configuration."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Return a YAML/JSON-compatible representation."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))
