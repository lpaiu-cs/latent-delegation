"""Adaptive bridge post-paper research modules."""

from src.adaptive_bridge.common import (
    AdaptiveBridgeSettings,
    AdaptiveEvalSpec,
    adaptive_bridge_gate_settings,
    adaptive_bridge_settings,
    adaptive_bridge_trainable_prefixes,
    clone_config_with_seed,
    matched_bridge_rank,
)
from src.adaptive_bridge.models import (
    BridgeAwareResidualMoE,
    BridgeAwareResidualMoENoSmall,
)

__all__ = [
    "AdaptiveBridgeSettings",
    "AdaptiveEvalSpec",
    "BridgeAwareResidualMoE",
    "BridgeAwareResidualMoENoSmall",
    "adaptive_bridge_gate_settings",
    "adaptive_bridge_settings",
    "adaptive_bridge_trainable_prefixes",
    "clone_config_with_seed",
    "matched_bridge_rank",
]
