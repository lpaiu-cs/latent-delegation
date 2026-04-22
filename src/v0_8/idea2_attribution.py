"""Run Idea 2 delegated-sublayer attribution on the frozen v0.6.0 token-wise model."""

from __future__ import annotations

import argparse
import copy
import statistics
from pathlib import Path
from typing import Any

import torch

from src.eval.metrics import (
    masked_hidden_cosine_loss,
    masked_hidden_mse,
    perplexity_from_loss,
    shifted_cross_entropy,
    shifted_kl_divergence,
)
from src.models.backbone_loader import LoadedBackbones, load_backbones
from src.models.baselines import BridgeOnlyLargeModel, BridgeOnlyParamMatchedModel
from src.models.hybrid_gemma import HybridForwardOutput, _resolve_gate_value
from src.train.stage_b_objective import prepare_stage_b_teacher_targets
from src.train.trainer_utils import load_checkpoint, move_batch_to_device
from src.utils.io import ensure_dir, export_run_metadata, load_config, save_config_snapshot, save_csv, save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything
from src.v0_6.idea4_common import load_mixture_path_specs
from src.v0_6.idea4_holdout import build_probe_holdout_slice
from src.v0_6.idea4_tokenwise_models import TwoPathTokenwiseMixtureNoSmallModel
from src.v0_8.idea2_models import (
    AttributionForwardOutput,
    Idea2AblationSpec,
    SublayerAttributionTokenwiseHybrid,
    idea2_ablation_specs,
)


LOGGER = get_logger(__name__)
PRIMARY_MODEL_ORDER = [
    "tokenwise_full",
    "tokenwise_attn_suppressed",
    "tokenwise_mlp_suppressed",
    "tokenwise_both_suppressed",
    "tokenwise_no_small",
    "bridge_only",
    "bridge_only_param_matched",
]
PATH_SPECIFIC_MODEL_ORDER = [
    "tokenwise_attn_suppressed_path_b",
    "tokenwise_attn_suppressed_path_a",
    "tokenwise_mlp_suppressed_path_b",
    "tokenwise_mlp_suppressed_path_a",
]
MODEL_ORDER = [*PRIMARY_MODEL_ORDER, *PATH_SPECIFIC_MODEL_ORDER, "full_large"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="artifacts/v0_8/idea2_attribution")
    parser.add_argument("--results-path", default="artifacts/v0_8/idea2_attribution/results.json")
    parser.add_argument("--summary-path", default="artifacts/v0_8/idea2_attribution/summary.csv")
    parser.add_argument("--report-path", default="notes/v0_8/idea2_attribution_report.md")
    parser.add_argument("--combined-decision-path", default="notes/v0_8/idea2_combined_decision.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def idea2_settings(config: Any) -> dict[str, Any]:
    """Load the raw Idea 2 settings."""

    values = dict(config.raw.get("idea2", {}))
    return {
        "static_stage_dir": str(values.get("static_stage_dir", "artifacts/v0_6/idea4_static_mixture/confirm/stage_b")),
        "tokenwise_stage_dir": str(values.get("tokenwise_stage_dir", "artifacts/v0_6/idea4_tokenwise/confirm/stage_b")),
        "holdout_policies": list(values.get("holdout_policies", ["main_validation", "fresh_untouched"])),
        "include_path_specific": bool(values.get("include_path_specific", True)),
    }


def _clone_config_with_seed(config: Any, seed: int) -> Any:
    cloned = copy.deepcopy(config)
    cloned.training.seed = seed
    cloned.raw["training"]["seed"] = seed
    return cloned


def _load_tokenwise_payload(model: Any, payload: dict[str, Any], device: torch.device) -> None:
    model.entry_projector_b.load_state_dict(payload["entry_projector_b"])
    model.entry_projector_a.load_state_dict(payload["entry_projector_a"])
    model.return_adapter_b.load_state_dict(payload["return_adapter_b"])
    model.return_adapter_a.load_state_dict(payload["return_adapter_a"])
    model.set_static_prior_logits(payload["static_prior_logits"].to(device=device, dtype=torch.float32))
    model.gate_network.load_state_dict(payload["gate_network"])


def _load_models_for_seed(
    config: Any,
    backbones: LoadedBackbones,
    *,
    path_specs: list[Any],
    seed: int,
    static_stage_dir: str | Path,
    tokenwise_stage_dir: str | Path,
) -> dict[str, Any]:
    models: dict[str, Any] = {}

    tokenwise_payload = load_checkpoint(
        Path(tokenwise_stage_dir) / f"seed_{seed}" / "tokenwise_mixture_checkpoint.pt",
        backbones.device,
    )
    tokenwise_model = SublayerAttributionTokenwiseHybrid(config, backbones.large_model, backbones.small_model, path_specs)
    _load_tokenwise_payload(tokenwise_model, tokenwise_payload, backbones.device)
    tokenwise_model.eval()
    models["tokenwise_full_model"] = tokenwise_model

    tokenwise_no_small_payload = load_checkpoint(
        Path(tokenwise_stage_dir) / f"seed_{seed}" / "tokenwise_mixture_no_small_checkpoint.pt",
        backbones.device,
    )
    tokenwise_no_small = TwoPathTokenwiseMixtureNoSmallModel(config, backbones.large_model, backbones.small_model, path_specs)
    _load_tokenwise_payload(tokenwise_no_small, tokenwise_no_small_payload, backbones.device)
    tokenwise_no_small.eval()
    models["tokenwise_no_small"] = tokenwise_no_small

    bridge_payload = load_checkpoint(
        Path(static_stage_dir) / f"seed_{seed}" / "bridge_only_checkpoint.pt",
        backbones.device,
    )
    bridge_only = BridgeOnlyLargeModel(config, backbones.large_model)
    bridge_only.bridge.load_state_dict(bridge_payload["bridge"])
    bridge_only.gate.load_state_dict(bridge_payload["gate"])
    bridge_only.eval()
    models["bridge_only"] = bridge_only

    bridge_param_payload = load_checkpoint(
        Path(tokenwise_stage_dir) / f"seed_{seed}" / "bridge_only_param_matched_checkpoint.pt",
        backbones.device,
    )
    bridge_param_rank = int(bridge_param_payload["bridge"]["down.weight"].shape[0])
    bridge_param = BridgeOnlyParamMatchedModel(config, backbones.large_model, rank=bridge_param_rank)
    bridge_param.bridge.load_state_dict(bridge_param_payload["bridge"])
    bridge_param.gate.load_state_dict(bridge_param_payload["gate"])
    bridge_param.eval()
    models["bridge_only_param_matched"] = bridge_param

    return models


def _valid_next_token_count(labels: torch.Tensor) -> int:
    return int((labels[:, 1:] != -100).sum().item())


def _empty_output_totals() -> dict[str, float]:
    return {
        "valid_tokens": 0.0,
        "hidden_positions": 0.0,
        "logit_kl_to_teacher_sum": 0.0,
        "nll_sum": 0.0,
        "top1_agreement_sum": 0.0,
        "top5_overlap_sum": 0.0,
        "hidden_mse_sum": 0.0,
        "hidden_cosine_sum": 0.0,
        "delta_norm_mean_sum": 0.0,
        "delta_norm_max_sum": 0.0,
        "batch_count": 0.0,
        "weight_path_b_sum": 0.0,
        "weight_path_a_sum": 0.0,
        "gate_entropy_sum": 0.0,
        "path_b_usage_var_sum": 0.0,
        "path_a_usage_var_sum": 0.0,
        "path_b_weighted_delta_norm_mean_sum": 0.0,
        "path_a_weighted_delta_norm_mean_sum": 0.0,
        "path_b_raw_attention_norm_mean_sum": 0.0,
        "path_b_applied_attention_norm_mean_sum": 0.0,
        "path_b_raw_mlp_norm_mean_sum": 0.0,
        "path_b_applied_mlp_norm_mean_sum": 0.0,
        "path_a_raw_attention_norm_mean_sum": 0.0,
        "path_a_applied_attention_norm_mean_sum": 0.0,
        "path_a_raw_mlp_norm_mean_sum": 0.0,
        "path_a_applied_mlp_norm_mean_sum": 0.0,
        "gate_value_sum": 0.0,
    }


def _compute_output_sums(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    top_k: int,
) -> dict[str, float]:
    valid_tokens = _valid_next_token_count(labels)
    if valid_tokens == 0:
        return {
            "valid_tokens": 0.0,
            "logit_kl_to_teacher_sum": 0.0,
            "nll_sum": 0.0,
            "top1_agreement_sum": 0.0,
            "top5_overlap_sum": 0.0,
        }
    nll_sum = float((shifted_cross_entropy(student_logits, labels) * valid_tokens).detach().cpu())
    kl_sum = float((shifted_kl_divergence(student_logits, teacher_logits, labels) * valid_tokens).detach().cpu())
    mask = labels[:, 1:] != -100
    student_top1 = student_logits[:, :-1, :].argmax(dim=-1)
    teacher_top1 = teacher_logits[:, :-1, :].argmax(dim=-1)
    top1_sum = float(((student_top1 == teacher_top1) & mask).sum().detach().cpu())
    student_topk = student_logits[:, :-1, :].topk(top_k, dim=-1).indices
    teacher_topk = teacher_logits[:, :-1, :].topk(top_k, dim=-1).indices
    overlap = (
        (student_topk.unsqueeze(-1) == teacher_topk.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float() / float(top_k)
    )
    top5_sum = float(overlap[mask].sum().detach().cpu())
    return {
        "valid_tokens": float(valid_tokens),
        "logit_kl_to_teacher_sum": kl_sum,
        "nll_sum": nll_sum,
        "top1_agreement_sum": top1_sum,
        "top5_overlap_sum": top5_sum,
    }


def _finalize_totals(totals: dict[str, float]) -> dict[str, float]:
    valid_tokens = max(1.0, totals["valid_tokens"])
    hidden_positions = max(1.0, totals["hidden_positions"])
    batch_count = max(1.0, totals["batch_count"])
    nll = totals["nll_sum"] / valid_tokens
    return {
        "logit_kl_to_teacher": totals["logit_kl_to_teacher_sum"] / valid_tokens,
        "nll": nll,
        "perplexity": perplexity_from_loss(nll),
        "top1_agreement": totals["top1_agreement_sum"] / valid_tokens,
        "top5_overlap": totals["top5_overlap_sum"] / valid_tokens,
        "hidden_mse": totals["hidden_mse_sum"] / hidden_positions,
        "hidden_cosine": totals["hidden_cosine_sum"] / hidden_positions,
        "delta_norm_mean": totals["delta_norm_mean_sum"] / batch_count,
        "delta_norm_max": totals["delta_norm_max_sum"] / batch_count,
        "weight_path_b": totals["weight_path_b_sum"] / batch_count,
        "weight_path_a": totals["weight_path_a_sum"] / batch_count,
        "gate_entropy": totals["gate_entropy_sum"] / batch_count,
        "path_b_usage_var": totals["path_b_usage_var_sum"] / batch_count,
        "path_a_usage_var": totals["path_a_usage_var_sum"] / batch_count,
        "path_b_weighted_delta_norm_mean": totals["path_b_weighted_delta_norm_mean_sum"] / batch_count,
        "path_a_weighted_delta_norm_mean": totals["path_a_weighted_delta_norm_mean_sum"] / batch_count,
        "path_b_raw_attention_norm_mean": totals["path_b_raw_attention_norm_mean_sum"] / batch_count,
        "path_b_applied_attention_norm_mean": totals["path_b_applied_attention_norm_mean_sum"] / batch_count,
        "path_b_raw_mlp_norm_mean": totals["path_b_raw_mlp_norm_mean_sum"] / batch_count,
        "path_b_applied_mlp_norm_mean": totals["path_b_applied_mlp_norm_mean_sum"] / batch_count,
        "path_a_raw_attention_norm_mean": totals["path_a_raw_attention_norm_mean_sum"] / batch_count,
        "path_a_applied_attention_norm_mean": totals["path_a_applied_attention_norm_mean_sum"] / batch_count,
        "path_a_raw_mlp_norm_mean": totals["path_a_raw_mlp_norm_mean_sum"] / batch_count,
        "path_a_applied_mlp_norm_mean": totals["path_a_applied_mlp_norm_mean_sum"] / batch_count,
        "gate_value": totals["gate_value_sum"] / batch_count,
        "valid_tokens": totals["valid_tokens"],
    }


def _delta_norm_stats(delta_large: torch.Tensor | None, attention_mask: torch.Tensor) -> dict[str, float]:
    if delta_large is None:
        return {"delta_norm_mean": 0.0, "delta_norm_max": 0.0}
    token_norms = delta_large.detach().float().norm(dim=-1)
    mask = attention_mask.to(token_norms.dtype)
    valid = token_norms[attention_mask.bool()]
    return {
        "delta_norm_mean": float(((token_norms * mask).sum() / mask.sum().clamp_min(1.0)).cpu()),
        "delta_norm_max": float(valid.max().cpu()) if valid.numel() else 0.0,
    }


def _tokenwise_stats(prediction: AttributionForwardOutput, attention_mask: torch.Tensor) -> dict[str, float]:
    if prediction.gate_weights is None or prediction.path_outputs is None:
        return {
            "weight_path_b": 0.0,
            "weight_path_a": 0.0,
            "gate_entropy": 0.0,
            "path_b_usage_var": 0.0,
            "path_a_usage_var": 0.0,
            "path_b_weighted_delta_norm_mean": 0.0,
            "path_a_weighted_delta_norm_mean": 0.0,
            "path_b_raw_attention_norm_mean": 0.0,
            "path_b_applied_attention_norm_mean": 0.0,
            "path_b_raw_mlp_norm_mean": 0.0,
            "path_b_applied_mlp_norm_mean": 0.0,
            "path_a_raw_attention_norm_mean": 0.0,
            "path_a_applied_attention_norm_mean": 0.0,
            "path_a_raw_mlp_norm_mean": 0.0,
            "path_a_applied_mlp_norm_mean": 0.0,
        }

    weights = prediction.gate_weights.detach().float()
    mask = attention_mask.bool()
    valid_weights = weights[mask]
    path_b = prediction.path_outputs["path_b"]
    path_a = prediction.path_outputs["path_a"]
    delta_b = path_b.delta_large.detach().float().norm(dim=-1)
    delta_a = path_a.delta_large.detach().float().norm(dim=-1)
    mask_float = attention_mask.to(delta_b.dtype)
    entropy = -(weights.clamp_min(1.0e-8) * weights.clamp_min(1.0e-8).log()).sum(dim=-1)
    return {
        "weight_path_b": float(valid_weights[:, 0].mean().cpu()),
        "weight_path_a": float(valid_weights[:, 1].mean().cpu()),
        "gate_entropy": float(entropy[mask].mean().cpu()),
        "path_b_usage_var": float(valid_weights[:, 0].var(unbiased=False).cpu()),
        "path_a_usage_var": float(valid_weights[:, 1].var(unbiased=False).cpu()),
        "path_b_weighted_delta_norm_mean": float(((weights[..., 0] * delta_b * mask_float).sum() / mask_float.sum().clamp_min(1.0)).cpu()),
        "path_a_weighted_delta_norm_mean": float(((weights[..., 1] * delta_a * mask_float).sum() / mask_float.sum().clamp_min(1.0)).cpu()),
        "path_b_raw_attention_norm_mean": path_b.raw_attention_norm_mean,
        "path_b_applied_attention_norm_mean": path_b.applied_attention_norm_mean,
        "path_b_raw_mlp_norm_mean": path_b.raw_mlp_norm_mean,
        "path_b_applied_mlp_norm_mean": path_b.applied_mlp_norm_mean,
        "path_a_raw_attention_norm_mean": path_a.raw_attention_norm_mean,
        "path_a_applied_attention_norm_mean": path_a.applied_attention_norm_mean,
        "path_a_raw_mlp_norm_mean": path_a.raw_mlp_norm_mean,
        "path_a_applied_mlp_norm_mean": path_a.applied_mlp_norm_mean,
    }


def _predict_bridge_from_prefix(model: Any, teacher_targets: Any) -> HybridForwardOutput:
    delta_large = model.bridge(teacher_targets.hidden_after_prefix)
    gated_delta, gate_value = _resolve_gate_value(model.gate, delta_large)
    hidden_after_removed = teacher_targets.hidden_after_prefix + gated_delta
    suffix_state = teacher_targets.prefix_state.with_hidden(hidden_after_removed)
    suffix_state = model.large_runner.run_layers(
        suffix_state,
        start=model.config.split.large_suffix_start,
        end=model.large_runner.num_layers - 1,
    )
    logits = model.large_runner.logits_from_hidden(suffix_state.hidden_states)
    final_hidden = model.large_runner.finalize_hidden(suffix_state.hidden_states)
    return HybridForwardOutput(
        logits=logits,
        hidden_after_prefix=teacher_targets.hidden_after_prefix,
        hidden_after_removed=hidden_after_removed,
        final_hidden=final_hidden,
        delta_large=delta_large,
        gate_value=gate_value,
    )


def _predict_no_small_from_prefix(model: Any, teacher_targets: Any) -> AttributionForwardOutput:
    path_outputs, delta_large, gate_weights, gate_logits = model.compute_mixed_delta(
        teacher_targets.hidden_after_prefix,
        teacher_targets.prefix_state.attention_mask_2d,
        train_entry_projector=True,
    )
    hidden_after_removed = teacher_targets.hidden_after_prefix + delta_large
    suffix_state = teacher_targets.prefix_state.with_hidden(hidden_after_removed)
    suffix_state = model.large_runner.run_layers(
        suffix_state,
        start=model.config.split.large_suffix_start,
        end=model.large_runner.num_layers - 1,
    )
    logits = model.large_runner.logits_from_hidden(suffix_state.hidden_states)
    final_hidden = model.large_runner.finalize_hidden(suffix_state.hidden_states)
    return AttributionForwardOutput(
        logits=logits,
        hidden_after_prefix=teacher_targets.hidden_after_prefix,
        hidden_after_removed=hidden_after_removed,
        final_hidden=final_hidden,
        delta_large=delta_large,
        gate_weights=gate_weights,
        gate_logits=gate_logits,
        path_outputs=None,
    )


def _accumulate_metrics(
    totals: dict[str, float],
    *,
    prediction: HybridForwardOutput | AttributionForwardOutput,
    teacher_targets: Any,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    top_k: int,
) -> None:
    output_sums = _compute_output_sums(prediction.logits, teacher_targets.teacher_logits, labels, top_k=top_k)
    for key, value in output_sums.items():
        totals[key] += value
    hidden_positions = float(attention_mask.sum().item())
    totals["hidden_positions"] += hidden_positions
    totals["hidden_mse_sum"] += float(masked_hidden_mse(prediction.hidden_after_removed, teacher_targets.teacher_hidden, attention_mask).cpu()) * hidden_positions
    hidden_cosine = 1.0 - float(masked_hidden_cosine_loss(prediction.hidden_after_removed, teacher_targets.teacher_hidden, attention_mask).cpu())
    totals["hidden_cosine_sum"] += hidden_cosine * hidden_positions
    delta_stats = _delta_norm_stats(prediction.delta_large, attention_mask)
    totals["delta_norm_mean_sum"] += delta_stats["delta_norm_mean"]
    totals["delta_norm_max_sum"] += delta_stats["delta_norm_max"]
    totals["batch_count"] += 1.0
    if isinstance(prediction, AttributionForwardOutput):
        tokenwise_stats = _tokenwise_stats(prediction, attention_mask)
        for key, value in tokenwise_stats.items():
            totals[f"{key}_sum"] += value
    if prediction.gate_value is not None:
        totals["gate_value_sum"] += float(prediction.gate_value)


def _teacher_reference_metrics(teacher_targets: Any, labels: torch.Tensor) -> dict[str, float]:
    valid_tokens = _valid_next_token_count(labels)
    if valid_tokens == 0:
        return {
            "logit_kl_to_teacher": 0.0,
            "nll": 0.0,
            "perplexity": 1.0,
            "top1_agreement": 1.0,
            "top5_overlap": 1.0,
            "hidden_mse": 0.0,
            "hidden_cosine": 1.0,
            "delta_norm_mean": 0.0,
            "delta_norm_max": 0.0,
            "weight_path_b": 0.0,
            "weight_path_a": 0.0,
            "gate_entropy": 0.0,
            "path_b_usage_var": 0.0,
            "path_a_usage_var": 0.0,
            "path_b_weighted_delta_norm_mean": 0.0,
            "path_a_weighted_delta_norm_mean": 0.0,
            "path_b_raw_attention_norm_mean": 0.0,
            "path_b_applied_attention_norm_mean": 0.0,
            "path_b_raw_mlp_norm_mean": 0.0,
            "path_b_applied_mlp_norm_mean": 0.0,
            "path_a_raw_attention_norm_mean": 0.0,
            "path_a_applied_attention_norm_mean": 0.0,
            "path_a_raw_mlp_norm_mean": 0.0,
            "path_a_applied_mlp_norm_mean": 0.0,
            "gate_value": 0.0,
            "valid_tokens": 0.0,
        }
    nll = float(shifted_cross_entropy(teacher_targets.teacher_logits, labels).cpu())
    return {
        "logit_kl_to_teacher": 0.0,
        "nll": nll,
        "perplexity": perplexity_from_loss(nll),
        "top1_agreement": 1.0,
        "top5_overlap": 1.0,
        "hidden_mse": 0.0,
        "hidden_cosine": 1.0,
        "delta_norm_mean": 0.0,
        "delta_norm_max": 0.0,
        "weight_path_b": 0.0,
        "weight_path_a": 0.0,
        "gate_entropy": 0.0,
        "path_b_usage_var": 0.0,
        "path_a_usage_var": 0.0,
        "path_b_weighted_delta_norm_mean": 0.0,
        "path_a_weighted_delta_norm_mean": 0.0,
        "path_b_raw_attention_norm_mean": 0.0,
        "path_b_applied_attention_norm_mean": 0.0,
        "path_b_raw_mlp_norm_mean": 0.0,
        "path_b_applied_mlp_norm_mean": 0.0,
        "path_a_raw_attention_norm_mean": 0.0,
        "path_a_applied_attention_norm_mean": 0.0,
        "path_a_raw_mlp_norm_mean": 0.0,
        "path_a_applied_mlp_norm_mean": 0.0,
        "gate_value": 0.0,
        "valid_tokens": float(valid_tokens),
    }


def _accumulate_teacher_metrics(
    totals: dict[str, float],
    *,
    teacher_targets: Any,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    top_k: int,
) -> None:
    output_sums = _compute_output_sums(teacher_targets.teacher_logits, teacher_targets.teacher_logits, labels, top_k=top_k)
    for key, value in output_sums.items():
        totals[key] += value
    hidden_positions = float(attention_mask.sum().item())
    totals["hidden_positions"] += hidden_positions
    totals["hidden_cosine_sum"] += hidden_positions
    totals["batch_count"] += 1.0


def _evaluate_one_holdout(
    config: Any,
    backbones: LoadedBackbones,
    *,
    seed: int,
    holdout_policy: str,
    top_k: int,
    models: dict[str, Any],
    ablation_specs: list[Idea2AblationSpec],
) -> dict[str, Any]:
    seed_config = _clone_config_with_seed(config, seed)
    holdout_slice = build_probe_holdout_slice(
        seed_config,
        backbones.tokenizer,
        holdout_policy=holdout_policy,
        seed=seed,
    )
    dataloader = holdout_slice.dataloader
    totals = {name: _empty_output_totals() for name in [spec.name for spec in ablation_specs]}
    totals["tokenwise_no_small"] = _empty_output_totals()
    totals["bridge_only"] = _empty_output_totals()
    totals["bridge_only_param_matched"] = _empty_output_totals()
    teacher_totals = _empty_output_totals()

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, backbones.device)
            teacher_targets = prepare_stage_b_teacher_targets(
                models["tokenwise_full_model"].large_runner,
                batch,
                seed_config,
                include_teacher_logits=True,
            )
            if teacher_targets.teacher_logits is None:
                raise RuntimeError("Teacher logits are required for Idea 2 attribution evaluation.")
            _accumulate_teacher_metrics(
                teacher_totals,
                teacher_targets=teacher_targets,
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                top_k=top_k,
            )

            for spec in ablation_specs:
                models["tokenwise_full_model"].set_active_ablation(spec)
                prediction = models["tokenwise_full_model"].forward_from_prefix_state(teacher_targets.prefix_state)
                _accumulate_metrics(
                    totals[spec.name],
                    prediction=prediction,
                    teacher_targets=teacher_targets,
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    top_k=top_k,
                )

            no_small_prediction = _predict_no_small_from_prefix(models["tokenwise_no_small"], teacher_targets)
            _accumulate_metrics(
                totals["tokenwise_no_small"],
                prediction=no_small_prediction,
                teacher_targets=teacher_targets,
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                top_k=top_k,
            )

            for bridge_name in ["bridge_only", "bridge_only_param_matched"]:
                bridge_prediction = _predict_bridge_from_prefix(models[bridge_name], teacher_targets)
                _accumulate_metrics(
                    totals[bridge_name],
                    prediction=bridge_prediction,
                    teacher_targets=teacher_targets,
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    top_k=top_k,
                )

    if teacher_totals["batch_count"] == 0.0:
        raise RuntimeError("Idea 2 attribution received no evaluation batches.")

    metrics_by_model = {name: _finalize_totals(model_totals) for name, model_totals in totals.items()}
    metrics_by_model["full_large"] = _finalize_totals(teacher_totals)

    pairwise_deltas: dict[str, dict[str, float]] = {}
    full = metrics_by_model["tokenwise_full"]
    for baseline in PRIMARY_MODEL_ORDER[1:]:
        row = metrics_by_model[baseline]
        pairwise_deltas[baseline] = {
            "logit_kl_to_teacher": row["logit_kl_to_teacher"] - full["logit_kl_to_teacher"],
            "nll": row["nll"] - full["nll"],
            "perplexity": row["perplexity"] - full["perplexity"],
            "top1_agreement": row["top1_agreement"] - full["top1_agreement"],
            "top5_overlap": row["top5_overlap"] - full["top5_overlap"],
        }

    path_deltas: dict[str, dict[str, float]] = {}
    for variant in PATH_SPECIFIC_MODEL_ORDER:
        row = metrics_by_model[variant]
        path_deltas[variant] = {
            "logit_kl_to_teacher": row["logit_kl_to_teacher"] - full["logit_kl_to_teacher"],
            "nll": row["nll"] - full["nll"],
            "perplexity": row["perplexity"] - full["perplexity"],
            "top1_agreement": row["top1_agreement"] - full["top1_agreement"],
            "top5_overlap": row["top5_overlap"] - full["top5_overlap"],
        }

    return {
        "seed": seed,
        "holdout_policy": holdout_policy,
        "sample_metadata": holdout_slice.sample_metadata,
        "slice_definition": holdout_slice.slice_definition,
        "metrics_by_model": metrics_by_model,
        "pairwise_deltas_from_full": pairwise_deltas,
        "path_specific_deltas_from_full": path_deltas,
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _aggregate_holdout(seed_results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {"per_model": {}, "pairwise_deltas_from_full": {}, "path_specific_deltas_from_full": {}}
    rows: list[dict[str, Any]] = []
    metric_names = [
        "logit_kl_to_teacher",
        "nll",
        "perplexity",
        "top1_agreement",
        "top5_overlap",
        "hidden_mse",
        "hidden_cosine",
        "weight_path_b",
        "weight_path_a",
        "path_b_raw_attention_norm_mean",
        "path_a_raw_attention_norm_mean",
        "path_b_raw_mlp_norm_mean",
        "path_a_raw_mlp_norm_mean",
        "path_b_applied_attention_norm_mean",
        "path_a_applied_attention_norm_mean",
        "path_b_applied_mlp_norm_mean",
        "path_a_applied_mlp_norm_mean",
        "path_b_weighted_delta_norm_mean",
        "path_a_weighted_delta_norm_mean",
        "delta_norm_mean",
        "delta_norm_max",
    ]

    for model_name in MODEL_ORDER:
        model_summary: dict[str, float] = {}
        for metric_name in metric_names:
            values = [row["metrics_by_model"][model_name][metric_name] for row in seed_results]
            model_summary[f"{metric_name}_mean"] = _mean(values)
            model_summary[f"{metric_name}_std"] = _std(values)
        summary["per_model"][model_name] = model_summary
        rows.append({"row_type": "model", "label": model_name, **model_summary})

    for baseline in PRIMARY_MODEL_ORDER[1:]:
        delta_summary: dict[str, float] = {}
        for metric_name in ["logit_kl_to_teacher", "nll", "perplexity", "top1_agreement", "top5_overlap"]:
            values = [row["pairwise_deltas_from_full"][baseline][metric_name] for row in seed_results]
            delta_summary[f"{metric_name}_mean"] = _mean(values)
            delta_summary[f"{metric_name}_std"] = _std(values)
        summary["pairwise_deltas_from_full"][baseline] = delta_summary
        rows.append({"row_type": "delta", "label": baseline, **delta_summary})

    for variant in PATH_SPECIFIC_MODEL_ORDER:
        delta_summary = {}
        for metric_name in ["logit_kl_to_teacher", "nll", "perplexity", "top1_agreement", "top5_overlap"]:
            values = [row["path_specific_deltas_from_full"][variant][metric_name] for row in seed_results]
            delta_summary[f"{metric_name}_mean"] = _mean(values)
            delta_summary[f"{metric_name}_std"] = _std(values)
        summary["path_specific_deltas_from_full"][variant] = delta_summary
        rows.append({"row_type": "path_delta", "label": variant, **delta_summary})

    return summary, rows


def _write_report(report_path: str | Path, results_payload: dict[str, Any]) -> None:
    main_summary = results_payload["holdouts"]["main_validation"]["summary"]
    fresh_summary = results_payload["holdouts"]["fresh_untouched"]["summary"]

    def row(summary: dict[str, Any], label: str) -> dict[str, float]:
        return summary["per_model"][label]

    lines = [
        "# Idea 2 Attribution Report",
        "",
        "## Scope",
        "",
        "- Subject model: frozen `v0.6.0` token-wise Idea 4 checkpoint family.",
        "- Attribution method: explicit suppression of delegated attention residuals and delegated MLP residuals inside the frozen small delegated block.",
        "- Holdout policies: original `main_validation` and the `fresh_untouched` slice from `v0.6.0`.",
        "",
        "## Main Holdout",
        "",
        f"- tokenwise_full: KL={row(main_summary, 'tokenwise_full')['logit_kl_to_teacher_mean']:.6f}, NLL={row(main_summary, 'tokenwise_full')['nll_mean']:.6f}, top1={row(main_summary, 'tokenwise_full')['top1_agreement_mean']:.6f}, top5={row(main_summary, 'tokenwise_full')['top5_overlap_mean']:.6f}",
        f"- attn_suppressed delta from full: dKL={main_summary['pairwise_deltas_from_full']['tokenwise_attn_suppressed']['logit_kl_to_teacher_mean']:.6f}, dNLL={main_summary['pairwise_deltas_from_full']['tokenwise_attn_suppressed']['nll_mean']:.6f}",
        f"- mlp_suppressed delta from full: dKL={main_summary['pairwise_deltas_from_full']['tokenwise_mlp_suppressed']['logit_kl_to_teacher_mean']:.6f}, dNLL={main_summary['pairwise_deltas_from_full']['tokenwise_mlp_suppressed']['nll_mean']:.6f}",
        f"- both_suppressed delta from full: dKL={main_summary['pairwise_deltas_from_full']['tokenwise_both_suppressed']['logit_kl_to_teacher_mean']:.6f}, dNLL={main_summary['pairwise_deltas_from_full']['tokenwise_both_suppressed']['nll_mean']:.6f}",
        "",
        "## Fresh Holdout",
        "",
        f"- tokenwise_full: KL={row(fresh_summary, 'tokenwise_full')['logit_kl_to_teacher_mean']:.6f}, NLL={row(fresh_summary, 'tokenwise_full')['nll_mean']:.6f}, top1={row(fresh_summary, 'tokenwise_full')['top1_agreement_mean']:.6f}, top5={row(fresh_summary, 'tokenwise_full')['top5_overlap_mean']:.6f}",
        f"- attn_suppressed delta from full: dKL={fresh_summary['pairwise_deltas_from_full']['tokenwise_attn_suppressed']['logit_kl_to_teacher_mean']:.6f}, dNLL={fresh_summary['pairwise_deltas_from_full']['tokenwise_attn_suppressed']['nll_mean']:.6f}",
        f"- mlp_suppressed delta from full: dKL={fresh_summary['pairwise_deltas_from_full']['tokenwise_mlp_suppressed']['logit_kl_to_teacher_mean']:.6f}, dNLL={fresh_summary['pairwise_deltas_from_full']['tokenwise_mlp_suppressed']['nll_mean']:.6f}",
        f"- both_suppressed delta from full: dKL={fresh_summary['pairwise_deltas_from_full']['tokenwise_both_suppressed']['logit_kl_to_teacher_mean']:.6f}, dNLL={fresh_summary['pairwise_deltas_from_full']['tokenwise_both_suppressed']['nll_mean']:.6f}",
        "",
        "## Path-Specific Deltas",
        "",
        f"- main holdout attention suppression: path B dKL={main_summary['path_specific_deltas_from_full']['tokenwise_attn_suppressed_path_b']['logit_kl_to_teacher_mean']:.6f}, path A dKL={main_summary['path_specific_deltas_from_full']['tokenwise_attn_suppressed_path_a']['logit_kl_to_teacher_mean']:.6f}",
        f"- main holdout MLP suppression: path B dKL={main_summary['path_specific_deltas_from_full']['tokenwise_mlp_suppressed_path_b']['logit_kl_to_teacher_mean']:.6f}, path A dKL={main_summary['path_specific_deltas_from_full']['tokenwise_mlp_suppressed_path_a']['logit_kl_to_teacher_mean']:.6f}",
        f"- fresh holdout attention suppression: path B dKL={fresh_summary['path_specific_deltas_from_full']['tokenwise_attn_suppressed_path_b']['logit_kl_to_teacher_mean']:.6f}, path A dKL={fresh_summary['path_specific_deltas_from_full']['tokenwise_attn_suppressed_path_a']['logit_kl_to_teacher_mean']:.6f}",
        f"- fresh holdout MLP suppression: path B dKL={fresh_summary['path_specific_deltas_from_full']['tokenwise_mlp_suppressed_path_b']['logit_kl_to_teacher_mean']:.6f}, path A dKL={fresh_summary['path_specific_deltas_from_full']['tokenwise_mlp_suppressed_path_a']['logit_kl_to_teacher_mean']:.6f}",
        "",
        "- Interpretive answers are finalized in `notes/v0_8/idea2_combined_decision.md` after inspection of these aggregates.",
    ]
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    settings = idea2_settings(config)
    path_specs = load_mixture_path_specs(config)
    ablation_specs = idea2_ablation_specs(include_path_specific=settings["include_path_specific"])
    seed_everything(config.training.seed)

    output_dir = ensure_dir(args.output_dir)
    save_config_snapshot(output_dir / "config_snapshot.yaml", config)
    export_run_metadata(
        output_dir / "metadata.json",
        config,
        {
            "stage": "idea2_attribution",
            "seeds": args.seeds,
            "top_k": args.top_k,
            "static_stage_dir": settings["static_stage_dir"],
            "tokenwise_stage_dir": settings["tokenwise_stage_dir"],
            "holdout_policies": settings["holdout_policies"],
            "ablation_specs": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "path_controls": {
                        path_name: {
                            "suppress_attention": control.suppress_attention,
                            "suppress_mlp": control.suppress_mlp,
                        }
                        for path_name, control in spec.path_controls.items()
                    },
                }
                for spec in ablation_specs
            ],
        },
    )

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    holdout_payloads: dict[str, Any] = {}

    for holdout_policy in settings["holdout_policies"]:
        LOGGER.info("idea2_attribution holdout=%s", holdout_policy)
        seed_results: list[dict[str, Any]] = []
        for seed in args.seeds:
            LOGGER.info("idea2_attribution seed=%s holdout=%s", seed, holdout_policy)
            models = _load_models_for_seed(
                _clone_config_with_seed(config, seed),
                backbones,
                path_specs=path_specs,
                seed=seed,
                static_stage_dir=settings["static_stage_dir"],
                tokenwise_stage_dir=settings["tokenwise_stage_dir"],
            )
            seed_result = _evaluate_one_holdout(
                config,
                backbones,
                seed=seed,
                holdout_policy=holdout_policy,
                top_k=args.top_k,
                models=models,
                ablation_specs=ablation_specs,
            )
            seed_results.append(seed_result)
            seed_dir = ensure_dir(output_dir / holdout_policy / f"seed_{seed}")
            save_json(seed_dir / "metrics.json", seed_result)
            save_json(seed_dir / "sample_ids.json", seed_result["sample_metadata"])
            save_json(seed_dir / "slice_definition.json", seed_result["slice_definition"])
            torch.cuda.empty_cache()

        summary, summary_rows = _aggregate_holdout(seed_results)
        holdout_payload = {
            "holdout_policy": holdout_policy,
            "seed_count": len(args.seeds),
            "seeds": args.seeds,
            "slice_definition": seed_results[0]["slice_definition"],
            "seed_results": seed_results,
            "summary": summary,
        }
        holdout_payloads[holdout_policy] = holdout_payload
        holdout_dir = ensure_dir(output_dir / holdout_policy)
        save_json(holdout_dir / "results.json", holdout_payload)
        save_csv(holdout_dir / "summary.csv", summary_rows)
        save_json(holdout_dir / "slice_definition.json", seed_results[0]["slice_definition"])

    combined_rows: list[dict[str, Any]] = []
    for holdout_policy, payload in holdout_payloads.items():
        for row in payload["summary"]["per_model"].keys():
            combined_rows.append(
                {
                    "holdout_policy": holdout_policy,
                    "label": row,
                    **payload["summary"]["per_model"][row],
                }
            )
    results_payload = {
        "config_path": args.config,
        "seeds": args.seeds,
        "top_k": args.top_k,
        "static_stage_dir": settings["static_stage_dir"],
        "tokenwise_stage_dir": settings["tokenwise_stage_dir"],
        "holdouts": holdout_payloads,
    }
    save_json(args.results_path, results_payload)
    save_csv(args.summary_path, combined_rows)
    _write_report(args.report_path, results_payload)
    decision_path = Path(args.combined_decision_path)
    if not decision_path.exists():
        decision_path.write_text("# Idea 2 Combined Decision\n\nPending interpretation.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
