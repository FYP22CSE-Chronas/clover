from __future__ import annotations

import os

import numpy as np
import pytest

from experiments.ec_clover.config import BetaInitConfig, ECConfig, load_ec_config


def test_defaults_disable_ec() -> None:
    config = ECConfig()
    assert config.enabled is False
    assert config.beta_init.type == "random"


def test_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        ECConfig.from_dict({"not_a_field": 1})


def test_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="rank must be >= 1"):
        ECConfig(rank=0)


def test_rejects_bad_encoder_channels() -> None:
    with pytest.raises(ValueError, match="encoder_channels must be >= 1"):
        ECConfig(encoder_channels=0)


def test_beta_init_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="type must be 'random' or 'johansen'"):
        BetaInitConfig(type="xavier")


def test_beta_init_johansen_requires_values() -> None:
    with pytest.raises(ValueError, match="requires 'values' or 'values_path'"):
        BetaInitConfig(type="johansen")


def test_beta_init_rejects_both_values_and_path() -> None:
    with pytest.raises(ValueError, match="only one of"):
        BetaInitConfig(type="johansen", values=((1.0,),), values_path="beta.npy")


def test_load_ec_config_none_path_disables() -> None:
    assert load_ec_config(None) == ECConfig()


def test_load_ec_config_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ec_config(str(tmp_path / "missing.yaml"))


def test_load_ec_config_rejects_unknown_top_level_key(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  n_factors: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level key"):
        load_ec_config(str(path))


def test_load_ec_config_parses_ec_section(tmp_path) -> None:
    path = tmp_path / "ec.yaml"
    path.write_text(
        "ec:\n  enabled: true\n  rank: 2\n  beta_init:\n    type: random\n",
        encoding="utf-8",
    )
    config = load_ec_config(str(path))
    assert config.enabled is True
    assert config.rank == 2


def test_shipped_example_config_loads() -> None:
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "configs", "tourism_small.yaml"
    )
    config = load_ec_config(path)
    assert config.enabled is True
    assert config.rank == 2


def test_values_path_beta_loads_from_npy(tmp_path) -> None:
    beta = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 1.0]], dtype=np.float32)
    path = tmp_path / "beta.npy"
    np.save(path, beta)
    config = BetaInitConfig(type="johansen", values_path=str(path))
    assert config.values_path == str(path)
