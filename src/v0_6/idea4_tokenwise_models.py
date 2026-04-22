"""Minimal token-wise Idea 4 models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.adapters import RMSNorm
from src.models.hybrid_gemma import HybridForwardOutput, _module_device
from src.utils.io import ExperimentConfig
from src.v0_6.idea4_models import TwoPathStaticMixtureHybrid


def tokenwise_gate_settings(config: ExperimentConfig) -> dict[str, Any]:
    """Return the raw token-wise gate settings."""

    idea4 = config.raw.get("idea4", {})
    values = dict(idea4.get("tokenwise_gate", {}))
    return {
        "hidden_dim": int(values.get("hidden_dim", 0)),
        "use_rmsnorm": bool(values.get("use_rmsnorm", True)),
        "entropy_reg_weight": float(values.get("entropy_reg_weight", 0.0)),
        "prior_kl_weight": float(values.get("prior_kl_weight", 0.0)),
        "smoothness_weight": float(values.get("smoothness_weight", 0.0)),
        "collapse_threshold": float(values.get("collapse_threshold", 0.9)),
    }


def tokenwise_mixture_trainable_prefixes(config: ExperimentConfig) -> list[str]:
    """Return the trainable parameter prefixes for the token-wise mixture."""

    prefixes = [
        "return_adapter_b",
        "return_adapter_a",
        "gate_network",
    ]
    if config.training.stage_b.train_entry_projector:
        prefixes = ["entry_projector_b", "entry_projector_a"] + prefixes
    return prefixes


class TokenwiseGateNetwork(nn.Module):
    """Low-capacity per-token 2-logit gate head."""

    def __init__(self, input_dim: int, hidden_dim: int, *, use_rmsnorm: bool, rms_norm_eps: float) -> None:
        super().__init__()
        self.norm = RMSNorm(input_dim, eps=rms_norm_eps) if use_rmsnorm else nn.Identity()
        self.hidden_dim = hidden_dim
        if hidden_dim > 0:
            self.down = nn.Linear(input_dim, hidden_dim, bias=True)
            self.activation = nn.GELU()
            self.out = nn.Linear(hidden_dim, 2, bias=True)
        else:
            self.down = None
            self.activation = nn.Identity()
            self.out = nn.Linear(input_dim, 2, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def initialize_from_static_logits(self, static_logits: torch.Tensor) -> None:
        """Initialize the gate bias from the learned static-mixture logits."""

        with torch.no_grad():
            self.out.bias.copy_(static_logits.to(device=self.out.bias.device, dtype=self.out.bias.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = self.norm(hidden_states)
        if self.down is not None:
            hidden_states = self.down(hidden_states.to(self.down.weight.dtype))
            hidden_states = self.activation(hidden_states)
        else:
            hidden_states = hidden_states.to(self.out.weight.dtype)
        logits = self.out(hidden_states)
        return logits.to(input_dtype)


class TwoPathTokenwiseMixtureHybrid(TwoPathStaticMixtureHybrid):
    """Two shortlisted delegated paths mixed by a per-token 2-logit gate."""

    def __init__(
        self,
        config: ExperimentConfig,
        large_model: nn.Module,
        small_model: nn.Module,
        path_specs: list[Any],
    ) -> None:
        super().__init__(config, large_model, small_model, path_specs)
        settings = tokenwise_gate_settings(config)
        self.gate_network = TokenwiseGateNetwork(
            input_dim=self.large_runner.hidden_size,
            hidden_dim=settings["hidden_dim"],
            use_rmsnorm=settings["use_rmsnorm"],
            rms_norm_eps=config.adapters.rms_norm_eps,
        )
        adapter_device = _module_device(large_model)
        self.gate_network.to(adapter_device)
        self.register_buffer("static_prior_logits", torch.zeros(2, device=adapter_device), persistent=True)

    def set_static_prior_logits(self, static_logits: torch.Tensor) -> None:
        """Initialize and cache the static-mixture prior logits."""

        static_logits = static_logits.detach().to(device=self.static_prior_logits.device, dtype=self.static_prior_logits.dtype)
        self.static_prior_logits.copy_(static_logits)
        self.gate_network.initialize_from_static_logits(static_logits)

    def static_prior_weights(self) -> torch.Tensor:
        """Return the stored static-mixture prior weights."""

        return torch.softmax(self.static_prior_logits, dim=0)

    def compute_mixed_delta(
        self,
        hidden_after_prefix: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        train_entry_projector: bool,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return path outputs, mixed delta, token-wise weights, and gate logits."""

        path_outputs = self.compute_path_outputs(
            hidden_after_prefix,
            attention_mask,
            train_entry_projector=train_entry_projector,
        )
        gate_logits = self.gate_network(hidden_after_prefix)
        gate_weights = F.softmax(gate_logits, dim=-1)
        delta_mix = (
            gate_weights[..., 0:1] * path_outputs["path_b"].delta_large
            + gate_weights[..., 1:2] * path_outputs["path_a"].delta_large
        )
        return path_outputs, delta_mix, gate_weights, gate_logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> HybridForwardOutput:
        prefix_state = self.large_runner.prepare_from_input_ids(input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            prefix_state = self.large_runner.run_layers(
                prefix_state,
                start=0,
                end=self.config.split.large_prefix_end,
            )
            hidden_after_prefix = prefix_state.hidden_states.detach()

        _, delta_mix, _weights, _gate_logits = self.compute_mixed_delta(
            hidden_after_prefix,
            prefix_state.attention_mask_2d,
            train_entry_projector=True,
        )
        hidden_after_removed = hidden_after_prefix + delta_mix
        suffix_state = prefix_state.with_hidden(hidden_after_removed)
        suffix_state = self.large_runner.run_layers(
            suffix_state,
            start=self.config.split.large_suffix_start,
            end=self.large_runner.num_layers - 1,
        )
        logits = self.large_runner.logits_from_hidden(suffix_state.hidden_states)
        final_hidden = self.large_runner.finalize_hidden(suffix_state.hidden_states)
        return HybridForwardOutput(
            logits=logits,
            hidden_after_prefix=hidden_after_prefix,
            hidden_after_removed=hidden_after_removed,
            final_hidden=final_hidden,
            delta_large=delta_mix,
        )


class TwoPathTokenwiseMixtureNoSmallModel(TwoPathTokenwiseMixtureHybrid):
    """Token-wise mixture control that removes both delegated small-model computations."""

    def run_delegated_small_block_for_path(
        self,
        path_spec: Any,
        projected_small: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del path_spec, attention_mask
        return projected_small
