from __future__ import annotations

from pathlib import Path

from src.models.backbone_loader import load_backbones
from src.train.trainer_utils import build_model_variant, required_backbones_for_variant
from src.utils.io import load_config


def test_large_only_loading_avoids_small_backbone() -> None:
    config = load_config(Path("configs/debug_tiny.yaml"))
    backbones = load_backbones(config, load_large=True, load_small=False, load_tokenizer=False)
    assert backbones.large_model is not None
    assert backbones.small_model is None
    model = build_model_variant("bridge_only", config, backbones)
    assert model.large_model is backbones.large_model


def test_variant_backbone_requirements() -> None:
    assert required_backbones_for_variant("full_large") == (True, False)
    assert required_backbones_for_variant("skip_only") == (True, False)
    assert required_backbones_for_variant("bridge_only") == (True, False)
    assert required_backbones_for_variant("hybrid") == (True, True)
