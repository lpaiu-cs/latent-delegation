"""Sublayer attribution models for the v0.8 Idea 2 discovery track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.models.hybrid_gemma import HybridForwardOutput, LayerRunState
from src.v0_6.idea4_models import MixturePathOutputs
from src.v0_6.idea4_tokenwise_models import TwoPathTokenwiseMixtureHybrid


@dataclass(frozen=True)
class PathSuppressionSpec:
    """Per-path sublayer suppression flags."""

    suppress_attention: bool = False
    suppress_mlp: bool = False


@dataclass(frozen=True)
class Idea2AblationSpec:
    """Named ablation spec for the token-wise delegated paths."""

    name: str
    description: str
    path_controls: dict[str, PathSuppressionSpec]

    def control_for_path(self, path_name: str) -> PathSuppressionSpec:
        """Return the path-local suppression flags."""

        return self.path_controls.get(path_name, PathSuppressionSpec())


@dataclass
class AttributionPathOutputs(MixturePathOutputs):
    """Per-path tensors plus attribution diagnostics."""

    raw_attention_norm_mean: float
    applied_attention_norm_mean: float
    raw_mlp_norm_mean: float
    applied_mlp_norm_mean: float


@dataclass
class AttributionForwardOutput(HybridForwardOutput):
    """Extended forward output for the Idea 2 attribution runs."""

    gate_weights: torch.Tensor | None = None
    gate_logits: torch.Tensor | None = None
    path_outputs: dict[str, AttributionPathOutputs] | None = None


def idea2_ablation_specs(*, include_path_specific: bool = True) -> list[Idea2AblationSpec]:
    """Return the ordered Idea 2 attribution specs."""

    specs = [
        Idea2AblationSpec(
            name="tokenwise_full",
            description="Full v0.6.0 token-wise baseline with both delegated subcomponents active.",
            path_controls={},
        ),
        Idea2AblationSpec(
            name="tokenwise_attn_suppressed",
            description="Suppress delegated attention residuals on both paths.",
            path_controls={
                "path_b": PathSuppressionSpec(suppress_attention=True),
                "path_a": PathSuppressionSpec(suppress_attention=True),
            },
        ),
        Idea2AblationSpec(
            name="tokenwise_mlp_suppressed",
            description="Suppress delegated MLP residuals on both paths.",
            path_controls={
                "path_b": PathSuppressionSpec(suppress_mlp=True),
                "path_a": PathSuppressionSpec(suppress_mlp=True),
            },
        ),
        Idea2AblationSpec(
            name="tokenwise_both_suppressed",
            description="Suppress both delegated attention and delegated MLP residuals on both paths.",
            path_controls={
                "path_b": PathSuppressionSpec(suppress_attention=True, suppress_mlp=True),
                "path_a": PathSuppressionSpec(suppress_attention=True, suppress_mlp=True),
            },
        ),
    ]
    if include_path_specific:
        specs.extend(
            [
                Idea2AblationSpec(
                    name="tokenwise_attn_suppressed_path_b",
                    description="Suppress delegated attention on path B only.",
                    path_controls={"path_b": PathSuppressionSpec(suppress_attention=True)},
                ),
                Idea2AblationSpec(
                    name="tokenwise_attn_suppressed_path_a",
                    description="Suppress delegated attention on path A only.",
                    path_controls={"path_a": PathSuppressionSpec(suppress_attention=True)},
                ),
                Idea2AblationSpec(
                    name="tokenwise_mlp_suppressed_path_b",
                    description="Suppress delegated MLP on path B only.",
                    path_controls={"path_b": PathSuppressionSpec(suppress_mlp=True)},
                ),
                Idea2AblationSpec(
                    name="tokenwise_mlp_suppressed_path_a",
                    description="Suppress delegated MLP on path A only.",
                    path_controls={"path_a": PathSuppressionSpec(suppress_mlp=True)},
                ),
            ]
        )
    return specs


def ablation_spec_by_name(specs: list[Idea2AblationSpec], name: str) -> Idea2AblationSpec:
    """Return one ablation spec by name."""

    for spec in specs:
        if spec.name == name:
            return spec
    raise KeyError(name)


class SublayerAttributionTokenwiseHybrid(TwoPathTokenwiseMixtureHybrid):
    """Token-wise mixture with explicit attention/MLP suppression inside the delegated block."""

    def __init__(
        self,
        config: Any,
        large_model: torch.nn.Module,
        small_model: torch.nn.Module,
        path_specs: list[Any],
    ) -> None:
        super().__init__(config, large_model, small_model, path_specs)
        self.active_ablation = idea2_ablation_specs(include_path_specific=False)[0]

    def set_active_ablation(self, spec: Idea2AblationSpec) -> None:
        """Set the active delegated-sublayer suppression."""

        self.active_ablation = spec

    @staticmethod
    def _masked_norm_mean(values: torch.Tensor, attention_mask: torch.Tensor) -> float:
        mask = attention_mask.to(values.dtype)
        return float(((values * mask).sum() / mask.sum().clamp_min(1.0)).detach().cpu())

    def _run_one_small_layer(
        self,
        decoder_layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        *,
        small_state: LayerRunState,
        control: PathSuppressionSpec,
        attention_mask_2d: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        attention_delta, _self_attn_weights = decoder_layer.self_attn(
            hidden_states=hidden_states,
            position_embeddings=small_state.position_embeddings,
            attention_mask=small_state.attention_masks[decoder_layer.attention_type],
            position_ids=small_state.position_ids,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
            cache_position=small_state.cache_position,
        )
        attention_delta = decoder_layer.post_attention_layernorm(attention_delta)
        applied_attention_delta = torch.zeros_like(attention_delta) if control.suppress_attention else attention_delta
        hidden_states = residual + applied_attention_delta

        residual = hidden_states
        hidden_states = decoder_layer.pre_feedforward_layernorm(hidden_states)
        mlp_delta = decoder_layer.mlp(hidden_states)
        mlp_delta = decoder_layer.post_feedforward_layernorm(mlp_delta)
        applied_mlp_delta = torch.zeros_like(mlp_delta) if control.suppress_mlp else mlp_delta
        hidden_states = residual + applied_mlp_delta

        stats = {
            "raw_attention_norm_mean": self._masked_norm_mean(attention_delta.detach().float().norm(dim=-1), attention_mask_2d),
            "applied_attention_norm_mean": self._masked_norm_mean(applied_attention_delta.detach().float().norm(dim=-1), attention_mask_2d),
            "raw_mlp_norm_mean": self._masked_norm_mean(mlp_delta.detach().float().norm(dim=-1), attention_mask_2d),
            "applied_mlp_norm_mean": self._masked_norm_mean(applied_mlp_delta.detach().float().norm(dim=-1), attention_mask_2d),
        }
        return hidden_states, stats

    def run_delegated_small_block_for_path(
        self,
        path_spec: Any,
        projected_small: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run one delegated small block with explicit attention/MLP suppression."""

        small_state = self.small_runner.prepare_from_hidden(
            hidden_states=projected_small,
            attention_mask=attention_mask,
            apply_input_scaling=False,
        )
        hidden_states = small_state.hidden_states
        control = self.active_ablation.control_for_path(path_spec.name)
        totals = {
            "raw_attention_norm_mean": 0.0,
            "applied_attention_norm_mean": 0.0,
            "raw_mlp_norm_mean": 0.0,
            "applied_mlp_norm_mean": 0.0,
        }
        layer_count = 0
        for layer_idx in range(path_spec.candidate.small_start, path_spec.candidate.small_end + 1):
            decoder_layer = self.small_runner.model.layers[layer_idx]
            hidden_states, stats = self._run_one_small_layer(
                decoder_layer,
                hidden_states,
                small_state=small_state,
                control=control,
                attention_mask_2d=small_state.attention_mask_2d,
            )
            for key, value in stats.items():
                totals[key] += value
            layer_count += 1
        if layer_count > 0:
            for key in totals.keys():
                totals[key] /= float(layer_count)
        return hidden_states, totals

    def compute_path_outputs(
        self,
        hidden_after_prefix: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        train_entry_projector: bool,
    ) -> dict[str, AttributionPathOutputs]:
        """Compute both path outputs plus explicit sublayer attribution diagnostics."""

        outputs: dict[str, AttributionPathOutputs] = {}
        for path_spec in self.path_specs:
            entry_projector = self._entry_projector(path_spec.name)
            return_adapter = self._return_adapter(path_spec.name)
            if train_entry_projector:
                projected_small = entry_projector(hidden_after_prefix)
                delegated_small, stats = self.run_delegated_small_block_for_path(path_spec, projected_small, attention_mask)
            else:
                with torch.no_grad():
                    projected_small = entry_projector(hidden_after_prefix)
                    delegated_small, stats = self.run_delegated_small_block_for_path(path_spec, projected_small, attention_mask)
                projected_small = projected_small.detach()
                delegated_small = delegated_small.detach()

            delta_large = return_adapter(delegated_small)
            outputs[path_spec.name] = AttributionPathOutputs(
                projected_small_hidden=projected_small,
                delegated_small_hidden=delegated_small,
                delta_large=delta_large,
                raw_attention_norm_mean=stats["raw_attention_norm_mean"],
                applied_attention_norm_mean=stats["applied_attention_norm_mean"],
                raw_mlp_norm_mean=stats["raw_mlp_norm_mean"],
                applied_mlp_norm_mean=stats["applied_mlp_norm_mean"],
            )
        return outputs

    def forward_from_prefix_state(self, prefix_state: LayerRunState) -> AttributionForwardOutput:
        """Run the delegated block and suffix from an already-computed large-prefix state."""

        hidden_after_prefix = prefix_state.hidden_states.detach()
        path_outputs, delta_mix, gate_weights, gate_logits = self.compute_mixed_delta(
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
        return AttributionForwardOutput(
            logits=logits,
            hidden_after_prefix=hidden_after_prefix,
            hidden_after_removed=hidden_after_removed,
            final_hidden=final_hidden,
            delta_large=delta_mix,
            gate_weights=gate_weights,
            gate_logits=gate_logits,
            path_outputs=path_outputs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> AttributionForwardOutput:
        """Standard forward path using the active sublayer-ablation spec."""

        prefix_state = self.large_runner.prepare_from_input_ids(input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            prefix_state = self.large_runner.run_layers(
                prefix_state,
                start=0,
                end=self.config.split.large_prefix_end,
            )
        return self.forward_from_prefix_state(prefix_state)

