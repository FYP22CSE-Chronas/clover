from __future__ import annotations

import os

import pytest

from config import (
    CONFIG_ROOT,
    EvalConfig,
    ModelConfig,
    TrainConfig,
    available_datasets,
    load_config,
)

DATASETS = available_datasets()


def test_every_shipped_dataset_has_a_config() -> None:
    assert set(DATASETS) == {
        "favorita",
        "labour",
        "m5",
        "police",
        "prison",
        "tourism_large",
        "tourism_small",
        "traffic",
        "wiki2",
    }


@pytest.mark.parametrize("dataset", DATASETS)
def test_config_loads_and_validates(dataset: str) -> None:
    config = load_config(dataset)
    assert config.name == dataset
    assert config.data.h >= 1
    assert config.data.input_size >= config.model.receptive_field


@pytest.mark.parametrize("dataset", DATASETS)
def test_data_dir_resolves_to_an_existing_directory(dataset: str) -> None:
    config = load_config(dataset)
    assert os.path.isabs(config.data.dir)
    assert os.path.isdir(config.data.dir), config.data.dir


def test_receptive_field() -> None:
    assert ModelConfig(dilations=(1, 2, 4), kernel_size=2).receptive_field == 8
    assert ModelConfig(dilations=(1, 7, 14, 28), kernel_size=2).receptive_field == 51


def test_lr_step_size_is_floored_at_one() -> None:
    assert TrainConfig(max_steps=1000, num_lr_decays=4).lr_step_size == 250
    assert TrainConfig(max_steps=2, num_lr_decays=100).lr_step_size == 1


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        ModelConfig.from_dict({"nope": 1})


def test_override_applies_and_is_typed() -> None:
    config = load_config("tourism_small", overrides=["train.learning_rate=1e-3"])
    assert config.train.learning_rate == pytest.approx(1e-3)


def test_override_of_an_unknown_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="targets no field"):
        load_config("tourism_small", overrides=["train.nope=1"])


def test_override_needs_an_equals_sign() -> None:
    with pytest.raises(ValueError, match="section.key=value"):
        load_config("tourism_small", overrides=["train.learning_rate"])


def test_missing_config_file_is_reported() -> None:
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_config("does_not_exist")


def test_config_root_points_at_the_repo() -> None:
    assert os.path.isfile(os.path.join(CONFIG_ROOT, "default.yaml"))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"sigma_activation": "relu"}, "sigma_activation"),
        ({"dilations": ()}, "non-empty"),
        ({"kernel_size": 0}, "kernel_size"),
        ({"n_factors": 0}, "n_factors"),
        ({"cross_series_hidden": -1}, "cross_series_hidden"),
    ],
)
def test_model_config_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ModelConfig(**kwargs)


def test_train_config_validation() -> None:
    with pytest.raises(ValueError, match="requires objective"):
        TrainConfig(loss_weighting="per_series", objective="energy")
    with pytest.raises(ValueError, match="Monte Carlo"):
        TrainConfig(train_mc_samples=1)


def test_eval_config_validation() -> None:
    with pytest.raises(ValueError, match="estimator"):
        EvalConfig(estimator="nope")
    with pytest.raises(ValueError, match="quantile_step"):
        EvalConfig(quantile_step=1.5)
