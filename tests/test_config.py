from types import SimpleNamespace

import pytest

from main import (
    EXPERIMENT_CONFIG_KEYS,
    REQUIRED_CONFIG_KEYS,
    RUNTIME_CONFIG_KEYS,
    apply_cli_config,
    build_wandb_config,
    default_checkpoint_path,
    ensure_config_keys,
    resolve_checkpoint_path,
)
from utils.parser import Parser


def _args(**overrides):
    values: dict[str, object] = {key: f"{key}_value" for key in REQUIRED_CONFIG_KEYS}
    values["config"] = None
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_wandb_config_keeps_wandb_yaml_config_path() -> None:
    args = _args(config="configs/NF-UNSW-NB15-v3.yaml")

    assert build_wandb_config(args) == "configs/NF-UNSW-NB15-v3.yaml"


def test_parser_defaults_transformer_feedforward_dim_to_256() -> None:
    args = Parser().parse_args(["--data_dir", "/tmp/data"])

    assert args.ae_feedforward_dim == 256


def test_build_wandb_config_uses_experiment_args_without_config_file() -> None:
    args = _args(config=None, dataset="NF-UNSW-NB15-v3", data_dir="/tmp/data")

    config = build_wandb_config(args)

    assert set(config) == set(EXPERIMENT_CONFIG_KEYS)
    assert config["dataset"] == "NF-UNSW-NB15-v3"
    assert "data_dir" not in config


def test_apply_cli_config_adds_runtime_values_and_missing_defaults() -> None:
    run_config = {"split_mode": "temporal"}
    args = _args(
        data_dir="/tmp/data",
        checkpoint=None,
        reload_dataset=True,
        test=True,
        save_curve=True,
        seed=123,
        wandb=False,
        split_mode="stratified",
        distribution_segment="post_shift",
        ae_feedforward_dim=256,
    )

    apply_cli_config(run_config, args)

    for key in RUNTIME_CONFIG_KEYS:
        assert run_config[key] == getattr(args, key)
    assert run_config["split_mode"] == "temporal"
    assert run_config["distribution_segment"] == "post_shift"
    assert run_config["ae_feedforward_dim"] == 256


def test_ensure_config_keys_reports_missing_values() -> None:
    complete_config = {key: f"{key}_value" for key in REQUIRED_CONFIG_KEYS}
    ensure_config_keys(complete_config)

    with pytest.raises(KeyError, match="dataset"):
        ensure_config_keys({"data_dir": "/tmp/data"})


def test_resolve_checkpoint_path_preserves_explicit_new_path() -> None:
    config = _args(
        checkpoint="checkpoints/custom_validation.ckpt",
        dataset="NF-UNSW-NB15-v3",
        seed=42,
        wandb=False,
        split_mode="stratified",
        distribution_segment="pre_shift",
    )

    assert resolve_checkpoint_path(config, run_name="ignored") == (
        "checkpoints/custom_validation.ckpt"
    )


def test_default_checkpoint_path_includes_non_default_split() -> None:
    config = _args(
        checkpoint=None,
        dataset="NF-CSE-CIC-IDS2018-v3",
        seed=42,
        wandb=False,
        split_mode="temporal_shift_aware",
        distribution_segment="post_shift",
    )

    assert default_checkpoint_path(config, run_name="ignored") == (
        "checkpoints/GraphIDS_NF-CSE-CIC-IDS2018-v3_"
        "temporal_shift_aware_post_shift_42.ckpt"
    )
