"""Runtime helpers for adaptive-bridge evaluation hardening and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.adaptive_bridge.common import adaptive_bridge_gate_settings
from src.adaptive_bridge.models import BridgeAwareResidualMoE
from src.eval.metrics import perplexity_from_loss, shifted_cross_entropy, shifted_kl_divergence
from src.models.hybrid_gemma import HybridForwardOutput
from src.v0_9.task_scoring import ChoiceBatch, continuation_logprob_summaries


EXPERT_NAMES = ("bridge", "path_b", "path_a")
EXPERT_INDEX = {name: index for index, name in enumerate(EXPERT_NAMES)}


@dataclass(frozen=True)
class RoutePolicy:
    """One inference-only allowed-route policy for the adaptive MoE."""

    name: str
    allowed_experts: tuple[str, ...] | None


@dataclass(frozen=True)
class TaskGateStats:
    """Token-level gate statistics aggregated over one task slice."""

    token_count: int
    weight_bridge: float
    weight_path_b: float
    weight_path_a: float
    gate_entropy: float
    collapse_score: float
    bridge_usage_var: float
    path_b_usage_var: float
    path_a_usage_var: float


class RouteAblatedAdaptiveModel(nn.Module):
    """Thin inference-only wrapper that restricts an adaptive MoE to a route policy."""

    def __init__(self, base_model: BridgeAwareResidualMoE, policy: RoutePolicy) -> None:
        super().__init__()
        self.base_model = base_model
        self.policy = policy

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> HybridForwardOutput:
        output, _ = adaptive_forward_with_policy(
            self.base_model,
            input_ids,
            attention_mask=attention_mask,
            allowed_experts=self.policy.allowed_experts,
        )
        return output


def _restrict_gate_weights(gate_logits: torch.Tensor, allowed_experts: tuple[str, ...] | None) -> torch.Tensor:
    """Convert full gate logits into normalized weights over an allowed expert subset."""

    if allowed_experts is None:
        return F.softmax(gate_logits, dim=-1)
    if not allowed_experts:
        raise ValueError("allowed_experts may not be empty.")

    dtype = gate_logits.dtype
    device = gate_logits.device
    if len(allowed_experts) == 1:
        weights = torch.zeros_like(gate_logits)
        weights[..., EXPERT_INDEX[allowed_experts[0]]] = 1.0
        return weights

    allowed_indices = [EXPERT_INDEX[name] for name in allowed_experts]
    masked_logits = torch.full_like(gate_logits.float(), float("-inf"))
    masked_logits[..., allowed_indices] = gate_logits.float()[..., allowed_indices]
    return F.softmax(masked_logits, dim=-1).to(dtype=dtype, device=device)


def adaptive_forward_with_policy(
    model: BridgeAwareResidualMoE,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    allowed_experts: tuple[str, ...] | None = None,
) -> tuple[HybridForwardOutput, torch.Tensor]:
    """Run one adaptive MoE forward pass with an optional route restriction."""

    with torch.no_grad():
        prefix_state = model.large_runner.prepare_from_input_ids(input_ids, attention_mask=attention_mask)
        prefix_state = model.large_runner.run_layers(
            prefix_state,
            start=0,
            end=model.config.split.large_prefix_end,
        )
        hidden_after_prefix = prefix_state.hidden_states.detach()
        expert_outputs = model.compute_expert_outputs(
            hidden_after_prefix,
            prefix_state.attention_mask_2d,
            train_entry_projector=bool(model.config.training.stage_b.train_entry_projector),
        )
        gate_logits = model.gate_network(hidden_after_prefix)
        gate_weights = _restrict_gate_weights(gate_logits, allowed_experts)
        delta_mix = (
            gate_weights[..., 0:1] * expert_outputs["bridge"].delta_large
            + gate_weights[..., 1:2] * expert_outputs["path_b"].delta_large
            + gate_weights[..., 2:3] * expert_outputs["path_a"].delta_large
        )
        hidden_after_removed = hidden_after_prefix + delta_mix
        suffix_state = prefix_state.with_hidden(hidden_after_removed)
        suffix_state = model.large_runner.run_layers(
            suffix_state,
            start=model.config.split.large_suffix_start,
            end=model.large_runner.num_layers - 1,
        )
        logits = model.large_runner.logits_from_hidden(suffix_state.hidden_states)
        final_hidden = model.large_runner.finalize_hidden(suffix_state.hidden_states)
        return HybridForwardOutput(
            logits=logits,
            hidden_after_prefix=hidden_after_prefix,
            hidden_after_removed=hidden_after_removed,
            final_hidden=final_hidden,
            delta_large=delta_mix,
        ), gate_weights.detach()


def task_gate_stats_from_batches(
    gate_weight_batches: Iterable[torch.Tensor],
    attention_mask_batches: Iterable[torch.Tensor],
    *,
    collapse_threshold: float,
) -> TaskGateStats:
    """Aggregate token-level gate statistics from one or more batches."""

    valid_weights: list[torch.Tensor] = []
    for gate_weights, attention_mask in zip(gate_weight_batches, attention_mask_batches, strict=True):
        mask = attention_mask.bool()
        if mask.any():
            valid_weights.append(gate_weights.detach().float().cpu()[mask.cpu()])
    if not valid_weights:
        return TaskGateStats(
            token_count=0,
            weight_bridge=0.0,
            weight_path_b=0.0,
            weight_path_a=0.0,
            gate_entropy=0.0,
            collapse_score=0.0,
            bridge_usage_var=0.0,
            path_b_usage_var=0.0,
            path_a_usage_var=0.0,
        )

    weights = torch.cat(valid_weights, dim=0)
    entropy = -(weights.clamp_min(1.0e-8) * weights.clamp_min(1.0e-8).log()).sum(dim=-1)
    collapse = (weights.max(dim=-1).values > collapse_threshold).to(torch.float32)
    return TaskGateStats(
        token_count=int(weights.shape[0]),
        weight_bridge=float(weights[:, 0].mean().item()),
        weight_path_b=float(weights[:, 1].mean().item()),
        weight_path_a=float(weights[:, 2].mean().item()),
        gate_entropy=float(entropy.mean().item()),
        collapse_score=float(collapse.mean().item()),
        bridge_usage_var=float(weights[:, 0].var(unbiased=False).item()),
        path_b_usage_var=float(weights[:, 1].var(unbiased=False).item()),
        path_a_usage_var=float(weights[:, 2].var(unbiased=False).item()),
    )


def _example_tensors(tokenizer: Any, text: str, max_seq_len: int, device: torch.device) -> dict[str, torch.Tensor | bool]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "truncated": bool(input_ids.shape[1] >= max_seq_len),
    }


def lm_example_metrics(
    student_model: nn.Module,
    teacher_model: nn.Module,
    tokenizer: Any,
    text: str,
    *,
    max_seq_len: int,
    device: torch.device,
) -> dict[str, Any] | None:
    """Return per-example LM metrics for one model on one text sequence."""

    tensors = _example_tensors(tokenizer, text, max_seq_len=max_seq_len, device=device)
    labels = tensors["labels"]
    valid_tokens = int((labels[:, 1:] != -100).sum().item())
    if valid_tokens == 0:
        return None
    with torch.no_grad():
        teacher_logits = teacher_model(
            tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
        ).logits
        student_logits = student_model(
            tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
        ).logits
    nll = float(shifted_cross_entropy(student_logits, labels).cpu())
    kl = 0.0 if student_model is teacher_model else float(shifted_kl_divergence(student_logits, teacher_logits, labels).cpu())
    return {
        "nll": nll,
        "perplexity": perplexity_from_loss(nll),
        "logit_kl_to_teacher": kl,
        "valid_tokens": valid_tokens,
        "truncated": bool(tensors["truncated"]),
    }


def multichoice_metrics_from_logits(
    logits: torch.Tensor,
    batch: ChoiceBatch,
    *,
    label_index: int,
    length_normalize: bool,
) -> dict[str, Any]:
    """Return multiple-choice metrics from batched logits."""

    choice_scores = continuation_logprob_summaries(logits, batch.input_ids, batch.continuation_mask)
    ranking_scores = choice_scores["avg_logprob"] if length_normalize else choice_scores["sum_logprob"]
    winner = int(ranking_scores.argmax(dim=0).item())
    sorted_scores = torch.sort(ranking_scores.detach().float(), descending=True).values
    runner_up = float(sorted_scores[1].cpu()) if sorted_scores.numel() > 1 else float(sorted_scores[0].cpu())
    margin = float(sorted_scores[0].cpu()) - runner_up
    return {
        "predicted_index": winner,
        "correct": winner == label_index,
        "score_margin": margin,
        "ranking_score_name": "avg_logprob" if length_normalize else "sum_logprob",
        "choice_sum_logprob": [float(value) for value in choice_scores["sum_logprob"].detach().cpu().tolist()],
        "choice_avg_logprob": [float(value) for value in choice_scores["avg_logprob"].detach().cpu().tolist()],
        "choice_token_count": [int(value) for value in choice_scores["token_count"].detach().cpu().tolist()],
        "truncated_flags": list(batch.truncated_flags),
    }


def paired_bootstrap_summary(
    values_a: list[float],
    values_b: list[float],
    *,
    weights: list[float] | None,
    higher_is_better: bool,
    num_samples: int,
    seed: int,
) -> dict[str, float]:
    """Return paired bootstrap uncertainty for one scalar comparison."""

    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have the same length.")
    if not values_a:
        raise ValueError("paired bootstrap requires at least one observation.")

    tensor_a = torch.tensor(values_a, dtype=torch.float64)
    tensor_b = torch.tensor(values_b, dtype=torch.float64)
    if weights is None:
        tensor_w = torch.ones_like(tensor_a)
    else:
        if len(weights) != len(values_a):
            raise ValueError("weights must match the number of paired observations.")
        tensor_w = torch.tensor(weights, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    n = tensor_a.shape[0]
    indices = torch.randint(0, n, (num_samples, n), generator=generator)
    sample_a = tensor_a[indices]
    sample_b = tensor_b[indices]
    sample_w = tensor_w[indices]
    denom = sample_w.sum(dim=1).clamp_min(1.0)
    mean_a = (sample_a * sample_w).sum(dim=1) / denom
    mean_b = (sample_b * sample_w).sum(dim=1) / denom
    delta = mean_a - mean_b

    if higher_is_better:
        improvement = delta > 0
    else:
        improvement = delta < 0
    point_delta = float(((tensor_a * tensor_w).sum() / tensor_w.sum().clamp_min(1.0)) - ((tensor_b * tensor_w).sum() / tensor_w.sum().clamp_min(1.0)))
    ci_low, ci_high = torch.quantile(delta, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {
        "point_delta": point_delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "probability_of_improvement": float(improvement.double().mean().item()),
        "bootstrap_samples": int(num_samples),
        "observation_count": int(n),
    }


def aggregate_stats(items: list[TaskGateStats]) -> dict[str, float]:
    """Return a simple mean/std summary over per-seed task gate stats."""

    if not items:
        return {}
    numeric_keys = (
        "token_count",
        "weight_bridge",
        "weight_path_b",
        "weight_path_a",
        "gate_entropy",
        "collapse_score",
        "bridge_usage_var",
        "path_b_usage_var",
        "path_a_usage_var",
    )
    summary: dict[str, float] = {}
    for key in numeric_keys:
        values = torch.tensor([float(getattr(item, key)) for item in items], dtype=torch.float64)
        summary[f"{key}_mean"] = float(values.mean().item())
        summary[f"{key}_std"] = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    return summary


def route_policies() -> list[RoutePolicy]:
    """Return the fixed inference-only route ablation policies."""

    return [
        RoutePolicy(name="full_adaptive_bridge_moe", allowed_experts=None),
        RoutePolicy(name="bridge_only_forced", allowed_experts=("bridge",)),
        RoutePolicy(name="bridge_plus_path_a_only", allowed_experts=("bridge", "path_a")),
        RoutePolicy(name="bridge_plus_path_b_only", allowed_experts=("bridge", "path_b")),
        RoutePolicy(name="delegated_paths_only", allowed_experts=("path_b", "path_a")),
    ]


def task_family(task_name: str) -> str:
    """Return a coarse task-family label for reporting."""

    if task_name in {"development_holdout", "confirmation_holdout", "lambada_openai"}:
        return "lm_style"
    if task_name in {"piqa", "arc_easy"}:
        return "multichoice"
    return "other"


def gate_stats_for_model_task(
    model: BridgeAwareResidualMoE,
    tokenizer: Any,
    examples: list[Any],
    *,
    task_category: str,
    max_seq_len: int,
    length_normalize_choices: bool,
    device: torch.device,
) -> TaskGateStats:
    """Compute token-level gate statistics for one model on one task slice."""

    gate_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    collapse_threshold = adaptive_bridge_gate_settings(model.config).collapse_threshold
    for example in examples:
        if task_category == "multichoice":
            from src.v0_9.task_scoring import build_choice_batch

            batch = build_choice_batch(
                tokenizer,
                example.prompt,
                example.choices,
                max_seq_len=max_seq_len,
                device=device,
            )
            _, gate_weights = adaptive_forward_with_policy(
                model,
                batch.input_ids,
                attention_mask=batch.attention_mask,
            )
            gate_batches.append(gate_weights.cpu())
            mask_batches.append(batch.attention_mask.cpu())
        else:
            tensors = _example_tensors(tokenizer, example.text, max_seq_len=max_seq_len, device=device)
            _, gate_weights = adaptive_forward_with_policy(
                model,
                tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
            )
            gate_batches.append(gate_weights.cpu())
            mask_batches.append(tensors["attention_mask"].cpu())
    return task_gate_stats_from_batches(
        gate_batches,
        mask_batches,
        collapse_threshold=collapse_threshold,
    )
