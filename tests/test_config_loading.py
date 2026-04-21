from __future__ import annotations

from pathlib import Path

from src.utils.io import load_config


def test_debug_config_loads() -> None:
    config = load_config(Path("configs/debug_tiny.yaml"))
    assert config.model.debug_random_init is True
    assert config.model.large_model_name == "debug/gemma2-large-tiny"
    assert config.split.large_prefix_end == 23
    assert config.split.small_delegate_end == 19
    assert config.training.seq_len == 32


def test_pilot_config_loads() -> None:
    config = load_config(Path("configs/gemma2_conservative_pilot_256.yaml"))
    assert config.training.seq_len == 256
    assert config.training.stage_a.max_steps == 200
    assert config.training.stage_b.max_steps == 200


def test_output_aware_stage_b_config_loads() -> None:
    config = load_config(Path("configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml"))
    assert config.training.seq_len == 256
    assert config.training.stage_b.kl_weight == 5.0
    assert config.training.stage_b.ce_weight == 1.0
    assert config.training.stage_b.delta_reg_weight == 1.0e-4
