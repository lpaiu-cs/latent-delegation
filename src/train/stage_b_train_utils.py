"""Helpers for Stage B trainable modules and diagnostics."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch

from src.utils.io import ExperimentConfig


def stage_b_train_entry_projector(config: ExperimentConfig) -> bool:
    """Return whether Stage B should keep adapting the entry projector."""

    return bool(config.training.stage_b.train_entry_projector)


def stage_b_trainable_prefixes(variant: str, config: ExperimentConfig) -> list[str]:
    """Return the trainable parameter prefixes for one Stage B variant."""

    if variant in {"bridge_only", "bridge_only_param_matched"}:
        return ["bridge", "gate"]
    if variant in {"hybrid", "hybrid_no_small"}:
        prefixes = ["return_adapter", "gate"]
        if stage_b_train_entry_projector(config):
            prefixes.insert(0, "entry_projector")
        return prefixes
    raise ValueError(f"Unsupported Stage B variant: {variant}")


def compute_hybrid_prediction(
    model: Any,
    hidden_after_prefix: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    train_entry_projector: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return delegated small hidden and large-space delta for hybrid variants."""

    if train_entry_projector:
        projected_hidden = model.entry_projector(hidden_after_prefix)
        delegated_small_hidden = model.run_delegated_small_block(projected_hidden, attention_mask)
        delta_large = model.return_adapter(delegated_small_hidden)
        return delegated_small_hidden, delta_large

    with torch.no_grad():
        projected_hidden = model.entry_projector(hidden_after_prefix)
        delegated_small_hidden = model.run_delegated_small_block(projected_hidden, attention_mask)
    delta_large = model.return_adapter(delegated_small_hidden.detach())
    return delegated_small_hidden.detach(), delta_large


def capture_entry_projector_init(model: Any) -> OrderedDict[str, torch.Tensor] | None:
    """Capture the Stage A initialization of the entry projector for later drift metrics."""

    entry_projector = getattr(model, "entry_projector", None)
    if entry_projector is None:
        return None
    return OrderedDict((name, parameter.detach().cpu().clone()) for name, parameter in entry_projector.named_parameters())


def entry_projector_update_norm(
    model: Any,
    init_state: OrderedDict[str, torch.Tensor] | None,
) -> float | None:
    """Return the L2 distance from the Stage A entry-projector initialization."""

    entry_projector = getattr(model, "entry_projector", None)
    if entry_projector is None or init_state is None:
        return None

    total = 0.0
    for name, parameter in entry_projector.named_parameters():
        if name not in init_state:
            continue
        diff = parameter.detach().cpu().float() - init_state[name].float()
        total += float(diff.pow(2).sum().item())
    return math.sqrt(total)


def entry_projector_grad_norm(model: Any) -> float | None:
    """Return the entry-projector gradient L2 norm if gradients are present."""

    entry_projector = getattr(model, "entry_projector", None)
    if entry_projector is None:
        return None

    total = 0.0
    found_grad = False
    for parameter in entry_projector.parameters():
        if parameter.grad is None:
            continue
        found_grad = True
        grad = parameter.grad.detach().float()
        total += float(grad.pow(2).sum().item())
    if not found_grad:
        return None
    return math.sqrt(total)
