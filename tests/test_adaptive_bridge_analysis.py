from __future__ import annotations

import torch

from src.adaptive_bridge.analysis_runtime import (
    _restrict_gate_weights,
    aggregate_stats,
    paired_bootstrap_summary,
    route_policies,
    task_gate_stats_from_batches,
)


def test_restrict_gate_weights_preserves_allowed_routes() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    weights = _restrict_gate_weights(logits, ("bridge", "path_a"))

    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    assert torch.allclose(weights[..., 1], torch.zeros_like(weights[..., 1]))
    assert float(weights[..., 2].item()) > float(weights[..., 0].item())


def test_paired_bootstrap_summary_returns_expected_sign() -> None:
    summary = paired_bootstrap_summary(
        [0.10, 0.20, 0.15, 0.12],
        [0.20, 0.30, 0.25, 0.22],
        weights=None,
        higher_is_better=False,
        num_samples=2000,
        seed=7,
    )

    assert summary["point_delta"] < 0.0
    assert summary["probability_of_improvement"] > 0.95


def test_task_gate_stats_aggregation_reports_mean_and_std() -> None:
    stats = task_gate_stats_from_batches(
        [torch.tensor([[[0.6, 0.3, 0.1], [0.5, 0.2, 0.3]]], dtype=torch.float32)],
        [torch.tensor([[1, 1]], dtype=torch.long)],
        collapse_threshold=0.9,
    )
    summary = aggregate_stats([stats, stats])

    assert stats.token_count == 2
    assert stats.weight_bridge > stats.weight_path_b > 0.0
    assert summary["weight_bridge_mean"] == stats.weight_bridge
    assert summary["weight_bridge_std"] == 0.0


def test_route_policy_names_cover_required_ablations() -> None:
    names = [policy.name for policy in route_policies()]

    assert names == [
        "full_adaptive_bridge_moe",
        "bridge_only_forced",
        "bridge_plus_path_a_only",
        "bridge_plus_path_b_only",
        "delegated_paths_only",
    ]
