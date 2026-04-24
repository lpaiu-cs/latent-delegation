"""Bridge-aware residual mixture-of-experts models for the adaptive fork."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.adaptive_bridge.common import adaptive_bridge_gate_settings, adaptive_bridge_settings
from src.models.adapters import LowRankAdapter, RMSNorm
from src.models.hybrid_gemma import HybridForwardOutput, _module_device
from src.utils.io import ExperimentConfig
from src.v0_6.idea4_models import MixturePathOutputs, TwoPathStaticMixtureHybrid


@dataclass
class AdaptiveExpertOutputs:
    """Per-expert tensors before the adaptive mixture."""

    delta_large: torch.Tensor
    projected_small_hidden: torch.Tensor | None = None
    delegated_small_hidden: torch.Tensor | None = None


class AdaptiveBridgeGateNetwork(nn.Module):
    """Low-capacity per-token 3-logit gate head."""

    def __init__(self, input_dim: int, hidden_dim: int, *, use_rmsnorm: bool, rms_norm_eps: float) -> None:
        super().__init__()
        self.norm = RMSNorm(input_dim, eps=rms_norm_eps) if use_rmsnorm else nn.Identity()
        self.hidden_dim = hidden_dim
        if hidden_dim > 0:
            self.down = nn.Linear(input_dim, hidden_dim, bias=True)
            self.activation = nn.GELU()
            self.out = nn.Linear(hidden_dim, 3, bias=True)
        else:
            self.down = None
            self.activation = nn.Identity()
            self.out = nn.Linear(input_dim, 3, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def initialize_from_prior_logits(self, prior_logits: torch.Tensor) -> None:
        """Initialize the gate bias from cached prior logits."""

        with torch.no_grad():
            self.out.bias.copy_(prior_logits.to(device=self.out.bias.device, dtype=self.out.bias.dtype))

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


class BridgeAwareResidualMoE(TwoPathStaticMixtureHybrid):
    """Bridge-aware token-wise residual MoE over bridge, path B, and path A."""

    def __init__(
        self,
        config: ExperimentConfig,
        large_model: nn.Module,
        small_model: nn.Module,
        path_specs: list[Any],
    ) -> None:
        super().__init__(config, large_model, small_model, path_specs)
        del self.alpha
        settings = adaptive_bridge_settings(config)
        gate_settings = adaptive_bridge_gate_settings(config)
        self.bridge_expert = LowRankAdapter(
            input_dim=self.large_runner.hidden_size,
            output_dim=self.large_runner.hidden_size,
            rank=settings.bridge_rank,
        )
        self.gate_network = AdaptiveBridgeGateNetwork(
            input_dim=self.large_runner.hidden_size,
            hidden_dim=gate_settings.hidden_dim,
            use_rmsnorm=gate_settings.use_rmsnorm,
            rms_norm_eps=config.adapters.rms_norm_eps,
        )
        adapter_device = _module_device(large_model)
        self.bridge_expert.to(adapter_device)
        self.gate_network.to(adapter_device)
        initial_prior = torch.tensor(
            [gate_settings.bridge_init_logit, 0.0, 0.0],
            device=adapter_device,
            dtype=torch.float32,
        )
        self.register_buffer("expert_prior_logits", initial_prior.clone(), persistent=True)
        self.gate_network.initialize_from_prior_logits(initial_prior)

    def set_expert_prior_logits(self, prior_logits: torch.Tensor) -> None:
        """Initialize and cache the three-expert prior logits."""

        prior_logits = prior_logits.detach().to(device=self.expert_prior_logits.device, dtype=self.expert_prior_logits.dtype)
        if prior_logits.shape != self.expert_prior_logits.shape:
            raise ValueError(f"Expected prior logits shape {tuple(self.expert_prior_logits.shape)}, got {tuple(prior_logits.shape)}.")
        self.expert_prior_logits.copy_(prior_logits)
        self.gate_network.initialize_from_prior_logits(prior_logits)

    def expert_prior_weights(self) -> torch.Tensor:
        """Return the stored three-expert prior weights."""

        return torch.softmax(self.expert_prior_logits, dim=0)

    def warm_start_from_tokenwise_payload(self, payload: dict[str, Any]) -> None:
        """Warm-start the delegated experts from a frozen v0.6.0 token-wise checkpoint."""

        gate_settings = adaptive_bridge_gate_settings(self.config)
        self.entry_projector_b.load_state_dict(payload["entry_projector_b"])
        self.entry_projector_a.load_state_dict(payload["entry_projector_a"])
        self.return_adapter_b.load_state_dict(payload["return_adapter_b"])
        self.return_adapter_a.load_state_dict(payload["return_adapter_a"])

        path_prior_logits = payload.get("static_prior_logits")
        if path_prior_logits is None:
            path_prior_logits = torch.zeros(2, dtype=torch.float32)
        combined_prior = torch.tensor(
            [gate_settings.bridge_init_logit, float(path_prior_logits[0]), float(path_prior_logits[1])],
            device=self.expert_prior_logits.device,
            dtype=self.expert_prior_logits.dtype,
        )
        self.set_expert_prior_logits(combined_prior)

        gate_state = payload.get("gate_network")
        if gate_state is None:
            return
        own_state = self.gate_network.state_dict()
        for key in ("norm.weight", "down.weight", "down.bias"):
            if key in gate_state and key in own_state and own_state[key].shape == gate_state[key].shape:
                own_state[key].copy_(gate_state[key].to(device=own_state[key].device, dtype=own_state[key].dtype))
        if "out.weight" in gate_state and own_state["out.weight"].shape[1] == gate_state["out.weight"].shape[1]:
            own_state["out.weight"].zero_()
            own_state["out.weight"][1:, :].copy_(
                gate_state["out.weight"].to(device=own_state["out.weight"].device, dtype=own_state["out.weight"].dtype)
            )
        if "out.bias" in gate_state:
            own_state["out.bias"].copy_(self.expert_prior_logits.to(device=own_state["out.bias"].device, dtype=own_state["out.bias"].dtype))
            own_state["out.bias"][1:].copy_(
                gate_state["out.bias"].to(device=own_state["out.bias"].device, dtype=own_state["out.bias"].dtype)
            )
        self.gate_network.load_state_dict(own_state)

    def compute_expert_outputs(
        self,
        hidden_after_prefix: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        train_entry_projector: bool,
    ) -> dict[str, AdaptiveExpertOutputs]:
        """Compute all three expert deltas before mixing."""

        path_outputs = self.compute_path_outputs(
            hidden_after_prefix,
            attention_mask,
            train_entry_projector=train_entry_projector,
        )
        bridge_delta = self.bridge_expert(hidden_after_prefix)
        outputs: dict[str, AdaptiveExpertOutputs] = {
            "bridge": AdaptiveExpertOutputs(delta_large=bridge_delta),
        }
        for name, value in path_outputs.items():
            outputs[name] = AdaptiveExpertOutputs(
                delta_large=value.delta_large,
                projected_small_hidden=value.projected_small_hidden,
                delegated_small_hidden=value.delegated_small_hidden,
            )
        return outputs

    def compute_mixed_delta(
        self,
        hidden_after_prefix: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        train_entry_projector: bool,
    ) -> tuple[dict[str, AdaptiveExpertOutputs], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return expert outputs, the mixed delta, token-wise weights, and gate logits."""

        expert_outputs = self.compute_expert_outputs(
            hidden_after_prefix,
            attention_mask,
            train_entry_projector=train_entry_projector,
        )
        gate_logits = self.gate_network(hidden_after_prefix)
        gate_weights = F.softmax(gate_logits, dim=-1)
        delta_mix = (
            gate_weights[..., 0:1] * expert_outputs["bridge"].delta_large
            + gate_weights[..., 1:2] * expert_outputs["path_b"].delta_large
            + gate_weights[..., 2:3] * expert_outputs["path_a"].delta_large
        )
        return expert_outputs, delta_mix, gate_weights, gate_logits

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

        _, delta_mix, _, _ = self.compute_mixed_delta(
            hidden_after_prefix,
            prefix_state.attention_mask_2d,
            train_entry_projector=bool(self.config.training.stage_b.train_entry_projector),
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


class BridgeAwareResidualMoENoSmall(BridgeAwareResidualMoE):
    """Adaptive control that keeps the same gate family but removes delegated small layers."""

    def run_delegated_small_block_for_path(
        self,
        path_spec: Any,
        projected_small: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del path_spec, attention_mask
        return projected_small
