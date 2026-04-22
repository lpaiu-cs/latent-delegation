"""Minimal two-path static-mixture models for Idea 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.models.adapters import EntryProjector, LowRankAdapter
from src.models.hooks import assert_hidden_size
from src.models.hybrid_gemma import GemmaCausalLMRunner, HybridForwardOutput, _module_device
from src.utils.io import ExperimentConfig
from src.v0_6.idea4_common import MixturePathSpec, path_spec_by_name


@dataclass
class MixturePathOutputs:
    """Per-path latent tensors before mixture aggregation."""

    projected_small_hidden: torch.Tensor
    delegated_small_hidden: torch.Tensor
    delta_large: torch.Tensor


def static_mixture_trainable_prefixes(config: ExperimentConfig) -> list[str]:
    """Return the trainable parameter prefixes for the static two-path mixture."""

    prefixes = [
        "return_adapter_b",
        "return_adapter_a",
        "alpha",
    ]
    if config.training.stage_b.train_entry_projector:
        prefixes = ["entry_projector_b", "entry_projector_a"] + prefixes
    return prefixes


class TwoPathStaticMixtureHybrid(nn.Module):
    """Two shortlisted delegated paths combined with one global softmax mixture."""

    def __init__(
        self,
        config: ExperimentConfig,
        large_model: nn.Module,
        small_model: nn.Module,
        path_specs: list[MixturePathSpec],
    ) -> None:
        super().__init__()
        self.config = config
        self.large_model = large_model
        self.small_model = small_model
        self.large_runner = GemmaCausalLMRunner(large_model)
        self.small_runner = GemmaCausalLMRunner(small_model)
        self.path_specs = list(path_specs)
        self.path_b = path_spec_by_name(path_specs, "path_b")
        self.path_a = path_spec_by_name(path_specs, "path_a")

        self.entry_projector_b = EntryProjector(
            input_dim=self.large_runner.hidden_size,
            output_dim=self.small_runner.hidden_size,
            use_rmsnorm=config.adapters.use_rmsnorm_after_entry,
            rms_norm_eps=config.adapters.rms_norm_eps,
        )
        self.entry_projector_a = EntryProjector(
            input_dim=self.large_runner.hidden_size,
            output_dim=self.small_runner.hidden_size,
            use_rmsnorm=config.adapters.use_rmsnorm_after_entry,
            rms_norm_eps=config.adapters.rms_norm_eps,
        )
        self.return_adapter_b = LowRankAdapter(
            input_dim=self.small_runner.hidden_size,
            output_dim=self.large_runner.hidden_size,
            rank=config.adapters.return_adapter_rank,
        )
        self.return_adapter_a = LowRankAdapter(
            input_dim=self.small_runner.hidden_size,
            output_dim=self.large_runner.hidden_size,
            rank=config.adapters.return_adapter_rank,
        )
        self.alpha = nn.Parameter(torch.zeros(2))
        adapter_device = _module_device(large_model)
        self.entry_projector_b.to(adapter_device)
        self.entry_projector_a.to(adapter_device)
        self.return_adapter_b.to(adapter_device)
        self.return_adapter_a.to(adapter_device)
        self.alpha.data = self.alpha.data.to(adapter_device)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.config.split.large_removed_start != self.path_a.candidate.large_start:
            raise ValueError("Idea 4 config and path A disagree on the large removed window.")
        if self.config.split.large_removed_start != self.path_b.candidate.large_start:
            raise ValueError("Idea 4 config and path B disagree on the large removed window.")
        if self.path_a.candidate.large_end != self.path_b.candidate.large_end:
            raise ValueError("Idea 4 path definitions must share the same large removed window.")

    def mixture_weights(self) -> torch.Tensor:
        """Return the static path weights in `[path_b, path_a]` order."""

        return torch.softmax(self.alpha, dim=0)

    def _entry_projector(self, path_name: str) -> EntryProjector:
        if path_name == "path_b":
            return self.entry_projector_b
        if path_name == "path_a":
            return self.entry_projector_a
        raise KeyError(path_name)

    def _return_adapter(self, path_name: str) -> LowRankAdapter:
        if path_name == "path_b":
            return self.return_adapter_b
        if path_name == "path_a":
            return self.return_adapter_a
        raise KeyError(path_name)

    def run_delegated_small_block_for_path(
        self,
        path_spec: MixturePathSpec,
        projected_small: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one shortlisted delegated small-model layer window."""

        small_state = self.small_runner.prepare_from_hidden(
            hidden_states=projected_small,
            attention_mask=attention_mask,
            apply_input_scaling=False,
        )
        small_state = self.small_runner.run_layers(
            small_state,
            start=path_spec.candidate.small_start,
            end=path_spec.candidate.small_end,
        )
        return small_state.hidden_states

    def compute_path_outputs(
        self,
        hidden_after_prefix: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        train_entry_projector: bool,
    ) -> dict[str, MixturePathOutputs]:
        """Compute both path deltas before the mixture is applied."""

        outputs: dict[str, MixturePathOutputs] = {}
        for path_spec in self.path_specs:
            entry_projector = self._entry_projector(path_spec.name)
            return_adapter = self._return_adapter(path_spec.name)
            if train_entry_projector:
                projected_small = entry_projector(hidden_after_prefix)
                delegated_small = self.run_delegated_small_block_for_path(path_spec, projected_small, attention_mask)
            else:
                with torch.no_grad():
                    projected_small = entry_projector(hidden_after_prefix)
                    delegated_small = self.run_delegated_small_block_for_path(path_spec, projected_small, attention_mask)
                projected_small = projected_small.detach()
                delegated_small = delegated_small.detach()

            assert_hidden_size(self.small_runner.hidden_size, projected_small, f"{path_spec.name}_entry_projector")
            assert_hidden_size(self.small_runner.hidden_size, delegated_small, f"{path_spec.name}_delegated_small")
            delta_large = return_adapter(delegated_small)
            assert_hidden_size(self.large_runner.hidden_size, delta_large, f"{path_spec.name}_return_adapter")
            outputs[path_spec.name] = MixturePathOutputs(
                projected_small_hidden=projected_small,
                delegated_small_hidden=delegated_small,
                delta_large=delta_large,
            )
        return outputs

    def compute_mixed_delta(
        self,
        hidden_after_prefix: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        train_entry_projector: bool,
    ) -> tuple[dict[str, MixturePathOutputs], torch.Tensor, torch.Tensor]:
        """Return per-path outputs, the mixed large-space delta, and the mixture weights."""

        path_outputs = self.compute_path_outputs(
            hidden_after_prefix,
            attention_mask,
            train_entry_projector=train_entry_projector,
        )
        weights = self.mixture_weights().to(hidden_after_prefix.dtype)
        delta_mix = (
            weights[0] * path_outputs["path_b"].delta_large
            + weights[1] * path_outputs["path_a"].delta_large
        )
        return path_outputs, delta_mix, weights

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

        _, delta_mix, _ = self.compute_mixed_delta(
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


class TwoPathStaticMixtureNoSmallModel(TwoPathStaticMixtureHybrid):
    """Static mixture control that removes both delegated small-model computations."""

    def run_delegated_small_block_for_path(
        self,
        path_spec: MixturePathSpec,
        projected_small: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del path_spec, attention_mask
        return projected_small
