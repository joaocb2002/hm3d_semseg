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


def test_ade20k_recipe_is_explicit_and_iteration_bounded() -> None:
    generated = Path("/tmp/generated")
    config = load_config(
        command_config=_experiment("segformer_b2_ade20k_recipe.yaml"),
        cli_overrides={"paths": {"generated_data_root": str(generated)}},
    )

    assert config.training.train_dataset == generated / "train-v1"
    assert config.training.development_dataset == generated / "development-v1"
    assert config.training.class_weighting == "none"
    assert config.training.batch_size == 16
    assert config.training.head_learning_rate == pytest.approx(6e-4)
    assert config.training.max_optimizer_steps == 160_000
    assert config.training.learning_rate_schedule == "polynomial"
    assert config.training.learning_rate_schedule_steps == 160_000
    assert config.training.warmup_steps == 1_500
    assert config.training.warmup_start_factor == pytest.approx(1e-6)
    assert config.augmentation.resize_base_width == 2_048
    assert config.augmentation.resize_base_height == 512
    assert config.augmentation.random_scale_min == 0.5
    assert config.augmentation.random_scale_max == 2.0
    assert config.augmentation.crop_width == 512
    assert config.augmentation.crop_height == 512
    assert config.augmentation.crop_max_class_fraction == 0.75
    assert config.augmentation.photometric_distortion is True


def test_generalization_probe_uses_full_training_and_bounded_diagnostics() -> None:
    generated = Path("/tmp/generated")
    config = load_config(
        command_config=_experiment("segformer_b2_generalization_probe.yaml"),
        cli_overrides={"paths": {"generated_data_root": str(generated)}},
    )

    assert config.training.train_dataset == generated / "train-v1"
    assert config.training.development_dataset == generated / "development-v1"
    assert config.training.max_train_samples is None
    assert config.training.max_development_samples is None
    assert config.training.evaluate_train_subset is True
    assert config.training.train_subset_evaluation_samples == 2_048
    assert config.training.save_min_development_loss_checkpoint is True
    assert config.training.encoder_learning_rate == pytest.approx(6e-6)
    assert config.training.head_learning_rate == pytest.approx(6e-4)
    assert config.training.max_optimizer_steps == 48_000
    assert config.training.learning_rate_schedule_steps == 48_000
    assert config.training.warmup_steps == 500
    assert config.training.early_stopping_patience == 5
    assert config.training.loss.cross_entropy_weight == pytest.approx(1.0)
    assert config.training.loss.lovasz_weight == pytest.approx(0.0)


def test_intermediate_lr_probe_changes_only_encoder_lr_and_run_name() -> None:
    generated = Path("/tmp/generated")
    overrides = {"paths": {"generated_data_root": str(generated)}}
    stable = load_config(
        command_config=_experiment("segformer_b2_generalization_probe.yaml"),
        cli_overrides=overrides,
    ).to_dict()
    intermediate = load_config(
        command_config=_experiment(
            "segformer_b2_generalization_probe_intermediate_lr.yaml"
        ),
        cli_overrides=overrides,
    ).to_dict()

    assert stable["training"]["encoder_learning_rate"] == pytest.approx(6e-6)
    assert intermediate["training"]["encoder_learning_rate"] == pytest.approx(2e-5)
    assert intermediate["training"]["run_name"] == (
        "segformer_b2_generalization_probe_intermediate_lr"
    )

    stable["training"]["encoder_learning_rate"] = intermediate["training"][
        "encoder_learning_rate"
    ]
    stable["training"]["run_name"] = intermediate["training"]["run_name"]
    assert intermediate == stable


def test_ce_lovasz_probe_changes_only_loss_and_run_name() -> None:
    generated = Path("/tmp/generated")
    overrides = {"paths": {"generated_data_root": str(generated)}}
    stable = load_config(
        command_config=_experiment("segformer_b2_generalization_probe.yaml"),
        cli_overrides=overrides,
    ).to_dict()
    ce_lovasz = load_config(
        command_config=_experiment(
            "segformer_b2_generalization_probe_ce_lovasz.yaml"
        ),
        cli_overrides=overrides,
    ).to_dict()

    assert ce_lovasz["training"]["loss"] == {
        "cross_entropy_weight": pytest.approx(0.8),
        "lovasz_weight": pytest.approx(0.2),
        "lovasz_include_unknown": False,
        "lovasz_resolution": "native",
    }
    assert ce_lovasz["training"]["encoder_learning_rate"] == pytest.approx(6e-6)

    stable["training"]["loss"] = ce_lovasz["training"]["loss"]
    stable["training"]["run_name"] = ce_lovasz["training"]["run_name"]
    assert ce_lovasz == stable


def test_ce_lovasz_smoke_is_bounded_and_exercises_mixed_loss() -> None:
    config = load_config(
        command_config=_experiment(
            "segformer_b2_generalization_probe_ce_lovasz_smoke.yaml"
        ),
        cli_overrides={"paths": {"generated_data_root": "/tmp/generated"}},
    )

    assert config.training.max_train_samples == 256
    assert config.training.max_development_samples == 64
    assert config.training.epochs == 2
    assert config.training.max_optimizer_steps == 256
    assert config.training.loss.cross_entropy_weight == pytest.approx(0.8)
    assert config.training.loss.lovasz_weight == pytest.approx(0.2)


def test_all_preexisting_recipes_remain_pure_cross_entropy() -> None:
    for path in sorted(_experiment(".").glob("*.yaml")):
        if path.name in {
            "segformer_b2_generalization_probe_ce_lovasz.yaml",
            "segformer_b2_generalization_probe_ce_lovasz_smoke.yaml",
        }:
            continue
        config = load_config(
            command_config=path,
            cli_overrides={"paths": {"generated_data_root": "/tmp/generated"}},
        )
        assert config.training.loss.cross_entropy_weight == pytest.approx(1.0)
        assert config.training.loss.lovasz_weight == pytest.approx(0.0)


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
    with pytest.raises(ConfigurationError, match=r"either training\.warmup_steps"):
        load_config(cli_overrides={"training": {"warmup_steps": 10}})
    with pytest.raises(ConfigurationError, match="learning_rate_schedule"):
        load_config(
            cli_overrides={"training": {"learning_rate_schedule": "triangle"}}
        )
    with pytest.raises(ConfigurationError, match="max_optimizer_steps"):
        load_config(cli_overrides={"training": {"max_optimizer_steps": 0}})
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
    with pytest.raises(
        ConfigurationError, match=r"training\.train_subset_evaluation_samples"
    ):
        load_config(
            cli_overrides={"training": {"train_subset_evaluation_samples": 0}}
        )
    with pytest.raises(
        ConfigurationError, match=r"training\.train_subset_evaluation_samples"
    ):
        load_config(
            cli_overrides={"training": {"train_subset_evaluation_samples": 128}}
        )
    with pytest.raises(
        ConfigurationError,
        match=r"training\.save_min_development_loss_checkpoint",
    ):
        load_config(
            cli_overrides={
                "training": {"save_min_development_loss_checkpoint": True}
            }
        )
    with pytest.raises(ConfigurationError, match=r"training\.qualitative_samples"):
        load_config(cli_overrides={"training": {"qualitative_samples": 0}})
    with pytest.raises(ConfigurationError, match=r"training\.qualitative_every_epochs"):
        load_config(cli_overrides={"training": {"qualitative_every_epochs": 0}})
    with pytest.raises(ConfigurationError, match=r"training\.run_name"):
        load_config(cli_overrides={"training": {"run_name": "../outside"}})
    with pytest.raises(ConfigurationError, match=r"cross_entropy_weight"):
        load_config(
            cli_overrides={
                "training": {
                    "loss": {
                        "cross_entropy_weight": 0.0,
                        "lovasz_weight": 1.0,
                    }
                }
            }
        )
    with pytest.raises(ConfigurationError, match=r"weights must sum to 1"):
        load_config(
            cli_overrides={
                "training": {
                    "loss": {
                        "cross_entropy_weight": 1.0,
                        "lovasz_weight": 0.2,
                    }
                }
            }
        )
    with pytest.raises(ConfigurationError, match=r"lovasz_resolution"):
        load_config(
            cli_overrides={
                "training": {"loss": {"lovasz_resolution": "quarter"}}
            }
        )
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
