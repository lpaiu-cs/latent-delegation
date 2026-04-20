"""Shape and frozen-parameter contract helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.io import ExperimentConfig


@dataclass
class ParameterSummary:
    """Trainable parameter summary."""

    total_params: int
    trainable_params: int


def count_parameters(module: nn.Module) -> ParameterSummary:
    """Count total and trainable parameters."""

    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return ParameterSummary(total_params=total, trainable_params=trainable)


def assert_split_fits_model(config: ExperimentConfig, large_num_layers: int, small_num_layers: int) -> None:
    """Validate that configured layer windows fit inside the loaded backbones."""

    split = config.split
    assert split.large_prefix_end < large_num_layers
    assert split.large_removed_end < large_num_layers
    assert split.large_suffix_start < large_num_layers
    assert split.small_entry_target_layer < small_num_layers
    assert split.small_delegate_end < small_num_layers


def assert_hidden_size(expected: int, hidden_states: torch.Tensor, label: str) -> None:
    """Assert a hidden tensor's final dimension."""

    actual = hidden_states.shape[-1]
    assert actual == expected, f"{label} hidden size mismatch: expected {expected}, got {actual}"


def assert_frozen(module: nn.Module, label: str) -> None:
    """Assert that all parameters of a module are frozen."""

    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    assert not trainable, f"{label} has trainable parameters: {trainable[:8]}"
