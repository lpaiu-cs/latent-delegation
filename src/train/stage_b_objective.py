"""Shared Stage B teacher-target preparation and loss composition."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.eval.metrics import (
    masked_hidden_cosine_loss,
    masked_hidden_mse,
    shifted_cross_entropy,
    shifted_kl_divergence,
)
from src.models.hybrid_gemma import GemmaCausalLMRunner, LayerRunState
from src.utils.io import ExperimentConfig


@dataclass
class StageBTeacherTargets:
    """Cached teacher targets for one Stage B batch."""

    prefix_state: LayerRunState
    hidden_after_prefix: torch.Tensor
    teacher_hidden: torch.Tensor
    teacher_logits: torch.Tensor | None


@dataclass
class StageBLossBreakdown:
    """Expanded Stage B loss terms."""

    total_loss: torch.Tensor
    mse_loss: torch.Tensor
    cosine_loss: torch.Tensor
    kl_loss: torch.Tensor
    ce_loss: torch.Tensor
    delta_reg: torch.Tensor


def _weight(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def stage_b_uses_output_objective(config: ExperimentConfig) -> bool:
    """Return whether Stage B should include teacher-logit supervision."""

    stage_b = config.training.stage_b
    return _weight(stage_b.kl_weight) > 0.0 or _weight(stage_b.ce_weight) > 0.0


def prepare_stage_b_teacher_targets(
    large_runner: GemmaCausalLMRunner,
    batch: dict[str, torch.Tensor],
    config: ExperimentConfig,
    include_teacher_logits: bool | None = None,
) -> StageBTeacherTargets:
    """Prepare the frozen large-model hidden targets for Stage B."""

    compute_teacher_logits = stage_b_uses_output_objective(config) if include_teacher_logits is None else include_teacher_logits
    with torch.no_grad():
        prefix_state = large_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
        prefix_state = large_runner.run_layers(prefix_state, 0, config.split.large_prefix_end)
        hidden_after_prefix = prefix_state.hidden_states.detach()

        teacher_state = prefix_state.with_hidden(hidden_after_prefix)
        teacher_state = large_runner.run_layers(
            teacher_state,
            config.split.large_removed_start,
            config.split.large_removed_end,
        )
        teacher_hidden = teacher_state.hidden_states.detach()
        teacher_logits = None
        if compute_teacher_logits:
            teacher_suffix_state = prefix_state.with_hidden(teacher_hidden)
            teacher_suffix_state = large_runner.run_layers(
                teacher_suffix_state,
                config.split.large_suffix_start,
                large_runner.num_layers - 1,
            )
            teacher_logits = large_runner.logits_from_hidden(teacher_suffix_state.hidden_states).detach()

    return StageBTeacherTargets(
        prefix_state=prefix_state,
        hidden_after_prefix=hidden_after_prefix,
        teacher_hidden=teacher_hidden,
        teacher_logits=teacher_logits,
    )


def compute_stage_b_loss_breakdown(
    large_runner: GemmaCausalLMRunner,
    config: ExperimentConfig,
    teacher_targets: StageBTeacherTargets,
    predicted_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    delta_large: torch.Tensor | None,
) -> StageBLossBreakdown:
    """Combine hidden recovery and optional output-aware Stage B losses."""

    stage_b = config.training.stage_b
    mse_loss = masked_hidden_mse(predicted_hidden, teacher_targets.teacher_hidden, attention_mask)
    cosine_loss = masked_hidden_cosine_loss(predicted_hidden, teacher_targets.teacher_hidden, attention_mask)

    zero = predicted_hidden.new_zeros(())
    kl_loss = zero
    ce_loss = zero
    delta_reg = zero

    kl_weight = _weight(stage_b.kl_weight)
    ce_weight = _weight(stage_b.ce_weight)
    delta_reg_weight = _weight(stage_b.delta_reg_weight)

    if kl_weight > 0.0 or ce_weight > 0.0:
        if teacher_targets.teacher_logits is None:
            raise ValueError("Teacher logits were not prepared for output-aware Stage B loss.")
        student_suffix_state = teacher_targets.prefix_state.with_hidden(predicted_hidden)
        student_suffix_state = large_runner.run_layers(
            student_suffix_state,
            config.split.large_suffix_start,
            large_runner.num_layers - 1,
        )
        student_logits = large_runner.logits_from_hidden(student_suffix_state.hidden_states)
        if kl_weight > 0.0:
            kl_loss = shifted_kl_divergence(student_logits, teacher_targets.teacher_logits, labels)
        if ce_weight > 0.0:
            ce_loss = shifted_cross_entropy(student_logits, labels)

    if delta_large is not None and delta_reg_weight > 0.0:
        delta_reg = delta_large.float().pow(2).mean()

    total_loss = mse_loss + cosine_loss
    if kl_weight > 0.0:
        total_loss = total_loss + kl_weight * kl_loss
    if ce_weight > 0.0:
        total_loss = total_loss + ce_weight * ce_loss
    if delta_reg_weight > 0.0:
        total_loss = total_loss + delta_reg_weight * delta_reg

    return StageBLossBreakdown(
        total_loss=total_loss,
        mse_loss=mse_loss,
        cosine_loss=cosine_loss,
        kl_loss=kl_loss,
        ce_loss=ce_loss,
        delta_reg=delta_reg,
    )
