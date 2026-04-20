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
