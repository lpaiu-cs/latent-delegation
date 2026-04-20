"""Gemma-specific hybrid delegation model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from src.models.adapters import EntryProjector, LowRankAdapter, ScalarGate
from src.models.hooks import assert_hidden_size
from src.utils.io import ExperimentConfig


@dataclass
class LayerRunState:
    """Execution state for a Gemma layer range."""

    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    attention_masks: dict[str, torch.Tensor]
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    attention_mask_2d: torch.Tensor

    def with_hidden(self, hidden_states: torch.Tensor) -> "LayerRunState":
        return replace(self, hidden_states=hidden_states)


@dataclass
class HybridForwardOutput:
    """Structured output from baseline and hybrid models."""

    logits: torch.Tensor
    hidden_after_prefix: torch.Tensor
    hidden_after_removed: torch.Tensor
    final_hidden: torch.Tensor
    delta_large: torch.Tensor | None = None
    delegated_small_hidden: torch.Tensor | None = None
    projected_small_hidden: torch.Tensor | None = None
    gate_value: float | None = None


class GemmaCausalLMRunner:
    """Manual layer runner for Gemma-style decoder blocks."""

    def __init__(self, causal_lm: nn.Module) -> None:
        self.causal_lm = causal_lm
        self.model = causal_lm.model
        self.config = causal_lm.config

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def num_layers(self) -> int:
        return int(self.config.num_hidden_layers)

    def _position_ids(self, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        return position_ids, cache_position

    def prepare_from_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> LayerRunState:
        inputs_embeds = self.model.embed_tokens(input_ids)
        return self.prepare_from_hidden(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            apply_input_scaling=True,
        )

    def prepare_from_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        apply_input_scaling: bool = False,
    ) -> LayerRunState:
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)
        batch_size, seq_len, _ = hidden_states.shape
        position_ids, cache_position = self._position_ids(batch_size, seq_len, hidden_states.device)
        mask_kwargs = {
            "config": self.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        attention_masks = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        if apply_input_scaling:
            normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = hidden_states * normalizer
        return LayerRunState(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache_position=cache_position,
            attention_masks=attention_masks,
            position_embeddings=position_embeddings,
            attention_mask_2d=attention_mask,
        )

    def run_layers(self, state: LayerRunState, start: int, end: int) -> LayerRunState:
        hidden_states = state.hidden_states
        for layer_idx in range(start, end + 1):
            decoder_layer = self.model.layers[layer_idx]
            outputs = decoder_layer(
                hidden_states,
                position_embeddings=state.position_embeddings,
                attention_mask=state.attention_masks[decoder_layer.attention_type],
                position_ids=state.position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=state.cache_position,
            )
            hidden_states = outputs[0]
        return state.with_hidden(hidden_states)

    def finalize_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.norm(hidden_states)

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.finalize_hidden(hidden_states)
        logits = self.causal_lm.lm_head(hidden_states)
        if getattr(self.config, "final_logit_softcapping", None) is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return logits


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _resolve_gate_value(gate: ScalarGate, delta_large: torch.Tensor, gate_override: float | None = None) -> tuple[torch.Tensor, float]:
    if gate_override is not None:
        gate_value = torch.tensor(float(gate_override), dtype=delta_large.dtype, device=delta_large.device)
    else:
        gate_value = gate.value().to(delta_large.dtype)
    return gate_value * delta_large, float(gate_value.detach().cpu())


class HybridDelegationModel(nn.Module):
    """Large-prefix -> small delegated block -> large suffix hybrid."""

    def __init__(self, config: ExperimentConfig, large_model: nn.Module, small_model: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.large_model = large_model
        self.small_model = small_model
        self.large_runner = GemmaCausalLMRunner(large_model)
        self.small_runner = GemmaCausalLMRunner(small_model)

        self.entry_projector = EntryProjector(
            input_dim=self.large_runner.hidden_size,
            output_dim=self.small_runner.hidden_size,
            use_rmsnorm=config.adapters.use_rmsnorm_after_entry,
            rms_norm_eps=config.adapters.rms_norm_eps,
        )
        self.return_adapter = LowRankAdapter(
            input_dim=self.small_runner.hidden_size,
            output_dim=self.large_runner.hidden_size,
            rank=config.adapters.return_adapter_rank,
        )
        self.gate = ScalarGate(init_value=config.adapters.gate_init)
        adapter_device = _module_device(large_model)
        self.entry_projector.to(adapter_device)
        self.return_adapter.to(adapter_device)
        self.gate.to(adapter_device)

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        assert self.config.split.large_removed_start == self.config.split.large_prefix_end + 1
        assert self.config.split.small_delegate_start == self.config.split.small_entry_target_layer + 1

    def run_delegated_small_block(
        self,
        projected_small: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the configured delegated small-model layer window."""

        small_state = self.small_runner.prepare_from_hidden(
            hidden_states=projected_small,
            attention_mask=attention_mask,
            apply_input_scaling=False,
        )
        small_state = self.small_runner.run_layers(
            small_state,
            start=self.config.split.small_delegate_start,
            end=self.config.split.small_delegate_end,
        )
        return small_state.hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        gate_override: float | None = None,
    ) -> HybridForwardOutput:
        large_prefix_state = self.large_runner.prepare_from_input_ids(input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            large_prefix_state = self.large_runner.run_layers(
                large_prefix_state,
                start=0,
                end=self.config.split.large_prefix_end,
            )
            hidden_after_prefix = large_prefix_state.hidden_states.detach()

        projected_small = self.entry_projector(hidden_after_prefix)
        assert_hidden_size(self.small_runner.hidden_size, projected_small, "entry_projector")
        delegated_small_hidden = self.run_delegated_small_block(projected_small, large_prefix_state.attention_mask_2d)

        delta_large = self.return_adapter(delegated_small_hidden)
        assert_hidden_size(self.large_runner.hidden_size, delta_large, "return_adapter")
        gated_delta_large, gate_value = _resolve_gate_value(self.gate, delta_large, gate_override=gate_override)
        hidden_after_removed = hidden_after_prefix + gated_delta_large

        suffix_state = large_prefix_state.with_hidden(hidden_after_removed)
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
            delta_large=delta_large,
            delegated_small_hidden=delegated_small_hidden,
            projected_small_hidden=projected_small,
            gate_value=gate_value,
        )


class HybridNoSmallModel(HybridDelegationModel):
    """Hybrid control that keeps the interface modules but removes delegated small layers."""

    def run_delegated_small_block(
        self,
        projected_small: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask
        return projected_small
