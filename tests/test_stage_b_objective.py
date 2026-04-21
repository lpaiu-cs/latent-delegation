from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch

from src.train.stage_b_objective import StageBTeacherTargets, compute_stage_b_loss_breakdown
from src.utils.io import load_config


@dataclass
class _FakeState:
    hidden_states: torch.Tensor

    def with_hidden(self, hidden_states: torch.Tensor) -> "_FakeState":
        return _FakeState(hidden_states=hidden_states)


class _FakeRunner:
    num_layers = 4

    def run_layers(self, state: _FakeState, start: int, end: int) -> _FakeState:
        del start, end
        return state.with_hidden(state.hidden_states + 0.25)

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def test_stage_b_output_aware_loss_adds_logit_terms() -> None:
    config = copy.deepcopy(load_config(Path("configs/debug_tiny.yaml")))
    config.training.stage_b.kl_weight = 1.0
    config.training.stage_b.ce_weight = 1.0
    config.training.stage_b.delta_reg_weight = 1.0e-3

    predicted_hidden = torch.randn(2, 4, 8, requires_grad=True)
    teacher_hidden = predicted_hidden.detach() + 0.5
    teacher_logits = (teacher_hidden * 1.5) + torch.linspace(-0.3, 0.3, steps=8).view(1, 1, -1)
    attention_mask = torch.ones(2, 4, dtype=torch.long)
    labels = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    delta_large = torch.randn(2, 4, 8)

    teacher_targets = StageBTeacherTargets(
        prefix_state=_FakeState(hidden_states=torch.zeros_like(predicted_hidden)),
        hidden_after_prefix=torch.zeros_like(predicted_hidden),
        teacher_hidden=teacher_hidden,
        teacher_logits=teacher_logits,
    )
    breakdown = compute_stage_b_loss_breakdown(
        _FakeRunner(),
        config,
        teacher_targets,
        predicted_hidden,
        attention_mask,
        labels,
        delta_large,
    )

    assert float(breakdown.kl_loss.detach()) > 0.0
    assert float(breakdown.ce_loss.detach()) > 0.0
    assert float(breakdown.delta_reg.detach()) > 0.0
    breakdown.total_loss.backward()
    assert predicted_hidden.grad is not None
