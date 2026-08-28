"""Deterministic strict YAML configuration merging."""

from __future__ import annotations

import math
import re
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

from hm3d_semseg.config.schema import ProjectConfig
from hm3d_semseg.exceptions import ConfigurationError
from hm3d_semseg.utils.hashing import atomic_write_text

T = TypeVar("T")


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Top-level configuration must be a mapping: {path}")
    return loaded


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _unwrap_optional(field_type: Any) -> Any:
    origin = get_origin(field_type)
    if origin is Union:
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        return args[0] if len(args) == 1 else field_type
    return field_type


def _coerce(field_type: Any, value: Any, dotted_key: str) -> Any:
    if value is None:
        return None
    concrete = _unwrap_optional(field_type)
    origin = get_origin(concrete)
    if concrete is Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            raise ConfigurationError(
                f"Path '{dotted_key}' must be absolute, got {value!r}. "
                "External data roots never use the current directory implicitly."
            )
        return path.resolve(strict=False)
    if origin in (list, Sequence):
        if not isinstance(value, list):
            raise ConfigurationError(f"'{dotted_key}' must be a list")
        item_type = get_args(concrete)[0]
        return [_coerce(item_type, item, f"{dotted_key}[]") for item in value]
    if is_dataclass(concrete):
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"'{dotted_key}' must be a mapping")
        return _construct(cast(Type[Any], concrete), value, dotted_key)
    if concrete is bool and not isinstance(value, bool):
        raise ConfigurationError(f"'{dotted_key}' must be true or false")
    if concrete is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise ConfigurationError(f"'{dotted_key}' must be an integer")
    if concrete is float and not isinstance(value, (int, float)):
        raise ConfigurationError(f"'{dotted_key}' must be numeric")
    if concrete is str and not isinstance(value, str):
        raise ConfigurationError(f"'{dotted_key}' must be a string")
    return concrete(value) if concrete in (int, float, str) else value


def _construct(cls: Type[T], values: Mapping[str, Any], prefix: str = "") -> T:
    known = {field.name: field for field in fields(cast(Any, cls))}
    type_hints = get_type_hints(cls)
    unknown = sorted(set(values) - set(known))
    if unknown:
        names = ", ".join(f"{prefix}.{name}".strip(".") for name in unknown)
        raise ConfigurationError(f"Unknown configuration key(s): {names}")
    kwargs: Dict[str, Any] = {}
    for name, field_info in known.items():
        dotted = f"{prefix}.{name}".strip(".")
        if name in values:
            kwargs[name] = _coerce(type_hints[name], values[name], dotted)
        elif field_info.default is MISSING and field_info.default_factory is MISSING:
            raise ConfigurationError(f"Missing required configuration key: {dotted}")
    return cls(**kwargs)


def load_config(
    command_config: Optional[Path] = None,
    local_config: Optional[Path] = None,
    cli_overrides: Optional[Mapping[str, Any]] = None,
) -> ProjectConfig:
    """Load config with precedence CLI > local > command > dataclass defaults."""
    defaults = ProjectConfig().to_dict()
    merged = defaults
    if command_config is not None:
        merged = _deep_merge(merged, _read_yaml(command_config))
    if local_config is not None:
        merged = _deep_merge(merged, _read_yaml(local_config))
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)
    config = _construct(ProjectConfig, merged)
    _validate(config)
    _resolve_training_dataset_names(config)
    return config


def _resolve_training_dataset_names(config: ProjectConfig) -> None:
    datasets = config.training.datasets
    if datasets.train is None:
        return
    root = config.paths.generated_data_root
    if root is None:
        raise ConfigurationError(
            "training.datasets.train requires paths.generated_data_root"
        )
    config.training.train_dataset = (root / datasets.train).resolve(strict=False)
    config.training.development_dataset = (
        (root / datasets.development).resolve(strict=False)
        if datasets.development is not None
        else None
    )


def _validate(config: ProjectConfig) -> None:
    if config.taxonomy.ignore_index in range(config.model.num_labels):
        raise ConfigurationError("ignore_index must not overlap a model output class")
    if config.model.num_labels != 41:
        raise ConfigurationError("The baseline contract requires exactly 41 output classes")
    valid_policies = {"unknown", "ignore"}
    policy = config.taxonomy.policy
    for key, value in vars(policy).items():
        if value not in valid_policies:
            raise ConfigurationError(
                f"taxonomy.policy.{key} must be one of {sorted(valid_policies)}, got {value!r}"
            )
    if config.dataset.rgb_codec not in {"png", "jpeg", "webp"}:
        raise ConfigurationError("dataset.rgb_codec must be png, jpeg, or webp")
    if config.dataset.split not in {"train", "val", "minival"}:
        raise ConfigurationError(
            "dataset.split must be train, val, or minival; private test is inaccessible"
        )
    augmentation = config.augmentation
    paired_resize = (
        augmentation.resize_base_width,
        augmentation.resize_base_height,
    )
    if (paired_resize[0] is None) != (paired_resize[1] is None):
        raise ConfigurationError(
            "augmentation.resize_base_width and resize_base_height must be set together"
        )
    if any(value is not None and value <= 0 for value in paired_resize):
        raise ConfigurationError("augmentation resize dimensions must be positive")
    paired_crop = (augmentation.crop_width, augmentation.crop_height)
    if (paired_crop[0] is None) != (paired_crop[1] is None):
        raise ConfigurationError(
            "augmentation.crop_width and crop_height must be set together"
        )
    if any(value is not None and value <= 0 for value in paired_crop):
        raise ConfigurationError("augmentation crop dimensions must be positive")
    if augmentation.random_scale_min <= 0:
        raise ConfigurationError("augmentation.random_scale_min must be positive")
    if augmentation.random_scale_max < augmentation.random_scale_min:
        raise ConfigurationError(
            "augmentation.random_scale_max must be at least random_scale_min"
        )
    if not 0.0 < augmentation.crop_max_class_fraction <= 1.0:
        raise ConfigurationError(
            "augmentation.crop_max_class_fraction must be in (0, 1]"
        )
    if augmentation.crop_attempts <= 0:
        raise ConfigurationError("augmentation.crop_attempts must be positive")
    for key in ("horizontal_flip_probability", "blur_probability"):
        if not 0.0 <= getattr(augmentation, key) <= 1.0:
            raise ConfigurationError(f"augmentation.{key} must be in [0, 1]")
    if augmentation.color_jitter < 0:
        raise ConfigurationError("augmentation.color_jitter must be nonnegative")
    if augmentation.sensor_noise_std < 0:
        raise ConfigurationError("augmentation.sensor_noise_std must be nonnegative")
    if not 0.0 <= config.sampling.class_aware_fraction <= 1.0:
        raise ConfigurationError("sampling.class_aware_fraction must be in [0, 1]")
    for key in ("positions_per_scene", "yaws_per_position", "max_attempts_per_position"):
        if getattr(config.sampling, key) <= 0:
            raise ConfigurationError(f"sampling.{key} must be positive")
    yaw_offset = config.sampling.yaw_offset_per_position_degrees
    if not math.isfinite(yaw_offset) or not 0.0 <= yaw_offset < 360.0:
        raise ConfigurationError(
            "sampling.yaw_offset_per_position_degrees must be finite and in [0, 360)"
        )
    if config.sampling.min_position_distance_m < 0:
        raise ConfigurationError("sampling.min_position_distance_m must be nonnegative")
    if config.sampling.floor_separation_m <= 0:
        raise ConfigurationError("sampling.floor_separation_m must be positive")
    if not 0.0 <= config.training.warmup_fraction < 1.0:
        raise ConfigurationError("training.warmup_fraction must be in [0, 1)")
    if config.training.warmup_steps is not None:
        if config.training.warmup_steps < 0:
            raise ConfigurationError("training.warmup_steps must be nonnegative")
        if config.training.warmup_fraction != 0.0:
            raise ConfigurationError(
                "Set either training.warmup_steps or a nonzero warmup_fraction, not both"
            )
    if not 0.0 <= config.training.warmup_start_factor <= 1.0:
        raise ConfigurationError("training.warmup_start_factor must be in [0, 1]")
    if config.training.learning_rate_schedule not in {"cosine", "polynomial"}:
        raise ConfigurationError(
            "training.learning_rate_schedule must be 'cosine' or 'polynomial'"
        )
    if (
        config.training.learning_rate_schedule_steps is not None
        and config.training.learning_rate_schedule_steps <= 0
    ):
        raise ConfigurationError(
            "training.learning_rate_schedule_steps must be positive"
        )
    if (
        config.training.learning_rate_schedule_steps is not None
        and config.training.max_optimizer_steps is not None
        and config.training.learning_rate_schedule_steps
        < config.training.max_optimizer_steps
    ):
        raise ConfigurationError(
            "training.learning_rate_schedule_steps must be at least "
            "max_optimizer_steps"
        )
    if (
        config.training.learning_rate_schedule_steps is not None
        and config.training.warmup_steps is not None
        and config.training.warmup_steps >= config.training.learning_rate_schedule_steps
    ):
        raise ConfigurationError(
            "training.warmup_steps must be smaller than learning_rate_schedule_steps"
        )
    if config.training.polynomial_power <= 0:
        raise ConfigurationError("training.polynomial_power must be positive")
    if (
        config.training.head_learning_rate is not None
        and config.training.head_learning_rate <= 0
    ):
        raise ConfigurationError("training.head_learning_rate must be positive")
    if (
        config.training.max_optimizer_steps is not None
        and config.training.max_optimizer_steps <= 0
    ):
        raise ConfigurationError("training.max_optimizer_steps must be positive")
    dataset_name_pattern = r"[A-Za-z0-9][A-Za-z0-9._-]*"
    for key, value in (
        ("train", config.training.datasets.train),
        ("development", config.training.datasets.development),
    ):
        if value is not None and not re.fullmatch(dataset_name_pattern, value):
            raise ConfigurationError(
                f"training.datasets.{key} must be a simple dataset directory name"
            )
    if (
        config.training.datasets.development is not None
        and config.training.datasets.train is None
    ):
        raise ConfigurationError(
            "training.datasets.development requires training.datasets.train"
        )
    if (
        config.training.max_train_samples is not None
        and config.training.max_train_samples <= 0
    ):
        raise ConfigurationError("training.max_train_samples must be positive")
    if (
        config.training.max_development_samples is not None
        and config.training.max_development_samples <= 0
    ):
        raise ConfigurationError("training.max_development_samples must be positive")
    if config.training.sample_selection not in {"manifest_order", "scene_diverse"}:
        raise ConfigurationError(
            "training.sample_selection must be 'manifest_order' or 'scene_diverse'"
        )
    if config.training.development_sample_selection not in {
        "manifest_order",
        "scene_diverse",
    }:
        raise ConfigurationError(
            "training.development_sample_selection must be 'manifest_order' "
            "or 'scene_diverse'"
        )
    if (
        config.training.evaluate_train_subset
        and config.training.max_train_samples is None
    ):
        raise ConfigurationError(
            "training.evaluate_train_subset requires training.max_train_samples"
        )
    if config.training.qualitative_samples <= 0:
        raise ConfigurationError("training.qualitative_samples must be positive")
    if config.training.qualitative_every_epochs <= 0:
        raise ConfigurationError("training.qualitative_every_epochs must be positive")
    if not re.fullmatch(r"(auto|cpu|cuda(?::\d+)?)", config.training.device.lower()):
        raise ConfigurationError("training.device must be auto, cpu, cuda, or cuda:N")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", config.training.run_name):
        raise ConfigurationError(
            "training.run_name must be a simple directory name containing only "
            "letters, numbers, dots, underscores, and hyphens"
        )
    if config.training.class_weighting not in {"none", "inverse_sqrt"}:
        raise ConfigurationError("training.class_weighting must be 'none' or 'inverse_sqrt'")
    if config.training.class_weight_cap < 1.0:
        raise ConfigurationError("training.class_weight_cap must be at least 1")
    if config.training.class_weights is not None and config.training.class_weighting != "none":
        raise ConfigurationError(
            "Set either training.class_weights or training.class_weighting, not both"
        )
    if (
        config.training.early_stopping_patience is not None
        and config.training.early_stopping_patience <= 0
    ):
        raise ConfigurationError("training.early_stopping_patience must be positive")
    if config.evaluation.bootstrap_samples <= 0:
        raise ConfigurationError("evaluation.bootstrap_samples must be positive")
    if config.evaluation.qualitative_samples <= 0:
        raise ConfigurationError("evaluation.qualitative_samples must be positive")
    if config.evaluation.calibration_bins <= 0:
        raise ConfigurationError("evaluation.calibration_bins must be positive")


def save_resolved_config(config: ProjectConfig, output: Path) -> None:
    """Write the exact resolved configuration used by a command."""
    atomic_write_text(output, yaml.safe_dump(config.to_dict(), sort_keys=False))
