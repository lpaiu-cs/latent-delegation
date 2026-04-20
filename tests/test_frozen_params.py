from __future__ import annotations

from pathlib import Path

from src.models.backbone_loader import load_backbones
from src.models.hooks import assert_frozen
from src.models.hybrid_gemma import HybridDelegationModel
from src.utils.io import load_config


def test_backbones_are_frozen_in_debug_mode() -> None:
    config = load_config(Path("configs/debug_tiny.yaml"))
    backbones = load_backbones(config)
    assert_frozen(backbones.large_model, "large_model")
    assert_frozen(backbones.small_model, "small_model")


def test_only_adapters_are_trainable_after_freezing() -> None:
    config = load_config(Path("configs/debug_tiny.yaml"))
    backbones = load_backbones(config)
    model = HybridDelegationModel(config, backbones.large_model, backbones.small_model)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable_names
    assert all(name.startswith(("entry_projector", "return_adapter", "gate")) for name in trainable_names)
