"""Baseline models for the latent delegation experiments."""

from __future__ import annotations

import torch
from torch import nn

from src.models.adapters import LowRankAdapter, ScalarGate
from src.models.hybrid_gemma import GemmaCausalLMRunner, HybridForwardOutput, _module_device
from src.utils.io import ExperimentConfig


class FullLargeModel(nn.Module):
    """Frozen full large model baseline."""

    def __init__(self, config: ExperimentConfig, large_model: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.large_model = large_model
        self.large_runner = GemmaCausalLMRunner(large_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> HybridForwardOutput:
        state = self.large_runner.prepare_from_input_ids(input_ids, attention_mask=attention_mask)
        state = self.large_runner.run_layers(state, start=0, end=self.large_runner.num_layers - 1)
        logits = self.large_runner.logits_from_hidden(state.hidden_states)
        final_hidden = self.large_runner.finalize_hidden(state.hidden_states)
        return HybridForwardOutput(
            logits=logits,
            hidden_after_prefix=state.hidden_states,
            hidden_after_removed=state.hidden_states,
            final_hidden=final_hidden,
        )


class SkipOnlyLargeModel(nn.Module):
    """Large model with the removed middle block skipped entirely."""

    def __init__(self, config: ExperimentConfig, large_model: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.large_model = large_model
        self.large_runner = GemmaCausalLMRunner(large_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> HybridForwardOutput:
        state = self.large_runner.prepare_from_input_ids(input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            state = self.large_runner.run_layers(state, start=0, end=self.config.split.large_prefix_end)
            hidden_after_prefix = state.hidden_states.detach()
        skipped_state = state.with_hidden(hidden_after_prefix)
        skipped_state = self.large_runner.run_layers(
            skipped_state,
            start=self.config.split.large_suffix_start,
            end=self.large_runner.num_layers - 1,
        )
        logits = self.large_runner.logits_from_hidden(skipped_state.hidden_states)
        final_hidden = self.large_runner.finalize_hidden(skipped_state.hidden_states)
        return HybridForwardOutput(
            logits=logits,
            hidden_after_prefix=hidden_after_prefix,
            hidden_after_removed=hidden_after_prefix,
            final_hidden=final_hidden,
        )


class BridgeOnlyLargeModel(nn.Module):
    """Large-only learned low-rank bridge baseline."""

    def __init__(self, config: ExperimentConfig, large_model: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.large_model = large_model
        self.large_runner = GemmaCausalLMRunner(large_model)
        self.bridge = LowRankAdapter(
            input_dim=self.large_runner.hidden_size,
            output_dim=self.large_runner.hidden_size,
            rank=config.adapters.bridge_rank,
        )
        self.gate = ScalarGate(init_value=config.adapters.gate_init)
        adapter_device = _module_device(large_model)
        self.bridge.to(adapter_device)
        self.gate.to(adapter_device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> HybridForwardOutput:
        state = self.large_runner.prepare_from_input_ids(input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            state = self.large_runner.run_layers(state, start=0, end=self.config.split.large_prefix_end)
            hidden_after_prefix = state.hidden_states.detach()
        delta_large = self.bridge(hidden_after_prefix)
        hidden_after_removed = hidden_after_prefix + self.gate(delta_large)
        bridged_state = state.with_hidden(hidden_after_removed)
        bridged_state = self.large_runner.run_layers(
            bridged_state,
            start=self.config.split.large_suffix_start,
            end=self.large_runner.num_layers - 1,
        )
        logits = self.large_runner.logits_from_hidden(bridged_state.hidden_states)
        final_hidden = self.large_runner.finalize_hidden(bridged_state.hidden_states)
        return HybridForwardOutput(
            logits=logits,
            hidden_after_prefix=hidden_after_prefix,
            hidden_after_removed=hidden_after_removed,
            final_hidden=final_hidden,
            delta_large=delta_large,
        )
