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


def _experiment(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "experiments" / name


@pytest.mark.parametrize(
    ("name", "weighting"),
    [
        ("segformer_b2_baseline_smoke.yaml", "none"),
        ("segformer_b2_moderately_balanced_smoke.yaml", "inverse_sqrt"),
    ],
)
def test_development_smoke_recipes_are_bounded_and_scene_diverse(
    name: str, weighting: str
) -> None:
    generated = Path("/tmp/generated")
    config = load_config(
        command_config=_experiment(name),
        cli_overrides={"paths": {"generated_data_root": str(generated)}},
    )
    assert config.training.max_train_samples == 1024
    assert config.training.max_development_samples == 256
    assert config.training.sample_selection == "scene_diverse"
    assert config.training.development_sample_selection == "scene_diverse"
    assert config.training.evaluate_train_subset is False
    assert config.training.qualitative_samples == 10
    assert config.training.qualitative_every_epochs == 1
    assert config.training.deterministic_algorithms is False
    assert config.training.epochs == 10
    assert config.training.class_weighting == weighting
    assert config.training.train_dataset == generated / "train-v1"
    assert config.training.development_dataset == generated / "development-v1"
    assert config.evaluation.qualitative_samples == 10


def test_tiny_overfit_explicitly_disables_development_dataset() -> None:
    config = load_config(
        command_config=_experiment("overfit_tiny.yaml"),
        cli_overrides={
            "paths": {"generated_data_root": "/tmp/generated"},
            "training": {"development_dataset": "/tmp/legacy-development"},
        },
    )
    assert config.training.datasets.train == "train-v1"
    assert config.training.datasets.development is None
    assert config.training.train_dataset == Path("/tmp/generated/train-v1")
    assert config.training.development_dataset is None


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
    with pytest.raises(ConfigurationError, match=r"training\.max_development_samples"):
        load_config(cli_overrides={"training": {"max_development_samples": 0}})
    with pytest.raises(ConfigurationError, match=r"training\.sample_selection"):
        load_config(cli_overrides={"training": {"sample_selection": "random"}})
    with pytest.raises(
        ConfigurationError, match=r"training\.development_sample_selection"
    ):
        load_config(
            cli_overrides={"training": {"development_sample_selection": "random"}}
        )
    with pytest.raises(ConfigurationError, match=r"training\.evaluate_train_subset"):
        load_config(cli_overrides={"training": {"evaluate_train_subset": True}})
    with pytest.raises(ConfigurationError, match=r"training\.qualitative_samples"):
        load_config(cli_overrides={"training": {"qualitative_samples": 0}})
    with pytest.raises(ConfigurationError, match=r"training\.qualitative_every_epochs"):
        load_config(cli_overrides={"training": {"qualitative_every_epochs": 0}})
    with pytest.raises(ConfigurationError, match=r"training\.run_name"):
        load_config(cli_overrides={"training": {"run_name": "../outside"}})
    with pytest.raises(ConfigurationError, match=r"evaluation\.bootstrap_samples"):
        load_config(cli_overrides={"evaluation": {"bootstrap_samples": 0}})
    with pytest.raises(ConfigurationError, match=r"evaluation\.qualitative_samples"):
        load_config(cli_overrides={"evaluation": {"qualitative_samples": 0}})
    with pytest.raises(ConfigurationError, match=r"evaluation\.calibration_bins"):
        load_config(cli_overrides={"evaluation": {"calibration_bins": 0}})


def test_portable_training_dataset_names_are_validated() -> None:
    with pytest.raises(ConfigurationError, match=r"training\.datasets\.train"):
        load_config(cli_overrides={"training": {"datasets": {"train": "../escape"}}})
    with pytest.raises(ConfigurationError, match=r"training\.datasets\.development"):
        load_config(
            cli_overrides={"training": {"datasets": {"development": "development-v1"}}}
        )


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
