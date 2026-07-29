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
        return _construct(concrete, value, dotted_key)
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
    known = {field.name: field for field in fields(cls)}
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
    return config


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
    if config.augmentation.sensor_noise_std < 0:
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
    if not re.fullmatch(r"(auto|cpu|cuda(?::\d+)?)", config.training.device.lower()):
        raise ConfigurationError("training.device must be auto, cpu, cuda, or cuda:N")
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


def save_resolved_config(config: ProjectConfig, output: Path) -> None:
    """Write the exact resolved configuration used by a command."""
    atomic_write_text(output, yaml.safe_dump(config.to_dict(), sort_keys=False))
