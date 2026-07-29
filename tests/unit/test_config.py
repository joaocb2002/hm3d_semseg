from pathlib import Path

import pytest

from hm3d_semseg.config import load_config
from hm3d_semseg.config.schema import ModelConfig
from hm3d_semseg.exceptions import ConfigurationError
from hm3d_semseg.models.segformer import build_segformer

pytestmark = pytest.mark.unit


def test_config_precedence_and_types(tmp_path: Path) -> None:
    command = tmp_path / "command.yaml"
    local = tmp_path / "local.yaml"
    command.write_text("sampling:\n  seed: 1\n  positions_per_scene: 20\n")
    local.write_text("sampling:\n  seed: 2\n")
    config = load_config(
        command,
        local,
        {"sampling": {"seed": 3}, "paths": {"hm3d_root": "/tmp/hm3d"}},
    )
    assert config.sampling.seed == 3
    assert config.sampling.positions_per_scene == 20
    assert config.paths.hm3d_root == Path("/tmp/hm3d")


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("sampling:\n  typo_seed: 1\n")
    with pytest.raises(ConfigurationError, match=r"sampling\.typo_seed"):
        load_config(command_config=path)


def test_relative_external_path_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="must be absolute"):
        load_config(cli_overrides={"paths": {"hm3d_root": "data/hm3d"}})


def test_num_labels_and_ignore_must_not_overlap() -> None:
    with pytest.raises(ConfigurationError, match="exactly 41"):
        load_config(cli_overrides={"model": {"num_labels": 40}})
    with pytest.raises(ConfigurationError, match="ignore_index"):
        load_config(cli_overrides={"taxonomy": {"ignore_index": 40}})


def test_weighting_and_warmup_are_validated() -> None:
    with pytest.raises(ConfigurationError, match="class_weighting"):
        load_config(cli_overrides={"training": {"class_weighting": "extreme"}})
    with pytest.raises(ConfigurationError, match="warmup_fraction"):
        load_config(cli_overrides={"training": {"warmup_fraction": 1.0}})
    with pytest.raises(ConfigurationError, match=r"training\.device"):
        load_config(cli_overrides={"training": {"device": "gpu"}})
    with pytest.raises(ConfigurationError, match=r"training\.max_train_samples"):
        load_config(cli_overrides={"training": {"max_train_samples": 0}})


def test_yaw_offset_per_position_is_validated() -> None:
    config = load_config(
        cli_overrides={"sampling": {"yaw_offset_per_position_degrees": 30.0}}
    )
    assert config.sampling.yaw_offset_per_position_degrees == 30.0
    with pytest.raises(ConfigurationError, match="yaw_offset_per_position_degrees"):
        load_config(
            cli_overrides={"sampling": {"yaw_offset_per_position_degrees": 360.0}}
        )


def test_unpinned_remote_model_is_rejected_before_import_or_download() -> None:
    with pytest.raises(ConfigurationError, match=r"model\.revision"):
        build_segformer(ModelConfig(revision=None))
