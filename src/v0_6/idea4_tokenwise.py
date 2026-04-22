"""Train and summarize the minimal Idea 4 token-wise mixture."""

from __future__ import annotations

import argparse
import copy
import itertools
import statistics
from pathlib import Path
from typing import Any

import torch

from src.eval.metrics import masked_hidden_cosine_loss, masked_hidden_mse
from src.models.backbone_loader import LoadedBackbones, load_backbones
from src.models.baselines import BridgeOnlyLargeModel, BridgeOnlyParamMatchedModel, SkipOnlyLargeModel
from src.models.hooks import count_parameters
from src.models.hybrid_gemma import _resolve_gate_value
from src.train.stage_b_objective import compute_stage_b_loss_breakdown, prepare_stage_b_teacher_targets
from src.train.trainer_utils import (
    build_dataloader,
    build_stage_b_optimizer,
    load_checkpoint,
    move_batch_to_device,
    save_checkpoint,
    zero_requires_grad,
)
from src.utils.io import ensure_dir, export_run_metadata, load_config, save_config_snapshot, save_csv, save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything
from src.v0_6.idea4_models import (
    TwoPathStaticMixtureHybrid,
    TwoPathStaticMixtureNoSmallModel,
    static_mixture_trainable_prefixes,
)
from src.v0_6.idea4_common import load_mixture_path_specs
from src.v0_6.idea4_tokenwise_models import (
    TwoPathTokenwiseMixtureHybrid,
    TwoPathTokenwiseMixtureNoSmallModel,
    tokenwise_gate_settings,
    tokenwise_mixture_trainable_prefixes,
)


LOGGER = get_logger(__name__)
TRAINED_VARIANTS = [
    "bridge_only_param_matched",
    "tokenwise_mixture_no_small",
    "tokenwise_mixture",
]
REFERENCE_VARIANTS = [
    "skip_only",
    "bridge_only",
    "static_mixture_no_small",
    "static_mixture",
]
EVAL_VARIANTS = REFERENCE_VARIANTS + TRAINED_VARIANTS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Idea 4 token-wise Stage B runner."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--static-stage-dir", default="artifacts/v0_6/idea4_static_mixture/confirm/stage_b")
    parser.add_argument("--output-dir", default="artifacts/v0_6/idea4_tokenwise/stage_b")
    parser.add_argument("--results-path", default="artifacts/v0_6/idea4_tokenwise/results.json")
    parser.add_argument("--summary-path", default="artifacts/v0_6/idea4_tokenwise/summary.csv")
    parser.add_argument("--diagnostics-path", default="artifacts/v0_6/idea4_tokenwise/diagnostics.json")
    parser.add_argument("--report-path", default="notes/v0_6/idea4_tokenwise_report.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--variants", nargs="+", choices=TRAINED_VARIANTS, default=TRAINED_VARIANTS)
    return parser.parse_args()


def _clone_config_with_seed(config: Any, seed: int) -> Any:
    cloned = copy.deepcopy(config)
    cloned.training.seed = seed
    cloned.raw["training"]["seed"] = seed
    return cloned


def _matched_bridge_rank(large_hidden_size: int, target_trainable_params: int) -> int:
    base = max(1, int((target_trainable_params - 1) / max(1, 2 * large_hidden_size)))
    candidates = sorted({max(1, base), max(1, base + 1)})
    return min(candidates, key=lambda rank: abs((2 * large_hidden_size * rank + 1) - target_trainable_params))


def _delta_norm_stats(delta_large: torch.Tensor | None, attention_mask: torch.Tensor) -> dict[str, float]:
    if delta_large is None:
        return {
            "delta_norm_mean": 0.0,
            "delta_norm_max": 0.0,
        }
    token_norms = delta_large.detach().float().norm(dim=-1)
    mask = attention_mask.to(token_norms.dtype)
    valid = token_norms[attention_mask.bool()]
    return {
        "delta_norm_mean": float(((token_norms * mask).sum() / mask.sum().clamp_min(1.0)).cpu()),
        "delta_norm_max": float(valid.max().cpu()) if valid.numel() else 0.0,
    }


def _masked_mean(values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _gate_regularizers(
    model: Any,
    gate_weights: torch.Tensor,
    attention_mask: torch.Tensor,
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    eps = 1.0e-8
    gate_weights = gate_weights.clamp_min(eps)
    entropy = -(gate_weights * gate_weights.log()).sum(dim=-1)
    entropy_loss = _masked_mean(entropy, attention_mask)

    prior = model.static_prior_weights().to(device=gate_weights.device, dtype=gate_weights.dtype).clamp_min(eps)
    prior_log = prior.log().view(1, 1, -1)
    prior_kl = (gate_weights * (gate_weights.log() - prior_log)).sum(dim=-1)
    prior_kl_loss = _masked_mean(prior_kl, attention_mask)

    smoothness = gate_weights.new_zeros(())
    if float(settings["smoothness_weight"]) > 0.0 and gate_weights.shape[1] > 1:
        pair_mask = (attention_mask[:, 1:] * attention_mask[:, :-1]).to(gate_weights.dtype)
        diffs = (gate_weights[:, 1:, :] - gate_weights[:, :-1, :]).abs().sum(dim=-1)
        smoothness = (diffs * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    return {
        "entropy_loss": entropy_loss,
        "prior_kl_loss": prior_kl_loss,
        "smoothness_loss": smoothness,
    }


def _tokenwise_gate_stats(
    gate_weights: torch.Tensor | None,
    path_outputs: dict[str, Any] | None,
    attention_mask: torch.Tensor,
    collapse_threshold: float,
) -> dict[str, float]:
    if gate_weights is None or path_outputs is None:
        return {
            "weight_path_b": 0.0,
            "weight_path_a": 0.0,
            "gate_entropy": 0.0,
            "path_b_usage_var": 0.0,
            "path_a_usage_var": 0.0,
            "collapse_score": 0.0,
            "path_b_weighted_delta_norm_mean": 0.0,
            "path_a_weighted_delta_norm_mean": 0.0,
        }

    weights = gate_weights.detach().float()
    mask = attention_mask.bool()
    valid_weights = weights[mask]
    if valid_weights.numel() == 0:
        return {
            "weight_path_b": 0.0,
            "weight_path_a": 0.0,
            "gate_entropy": 0.0,
            "path_b_usage_var": 0.0,
            "path_a_usage_var": 0.0,
            "collapse_score": 0.0,
            "path_b_weighted_delta_norm_mean": 0.0,
            "path_a_weighted_delta_norm_mean": 0.0,
        }

    delta_b = path_outputs["path_b"].delta_large.detach().float().norm(dim=-1)
    delta_a = path_outputs["path_a"].delta_large.detach().float().norm(dim=-1)
    mask_float = attention_mask.to(delta_b.dtype)
    entropy = -(weights.clamp_min(1.0e-8) * weights.clamp_min(1.0e-8).log()).sum(dim=-1)
    collapse = (weights.max(dim=-1).values > collapse_threshold).to(delta_b.dtype)
    return {
        "weight_path_b": float(valid_weights[:, 0].mean().cpu()),
        "weight_path_a": float(valid_weights[:, 1].mean().cpu()),
        "gate_entropy": float(entropy[mask].mean().cpu()),
        "path_b_usage_var": float(valid_weights[:, 0].var(unbiased=False).cpu()),
        "path_a_usage_var": float(valid_weights[:, 1].var(unbiased=False).cpu()),
        "collapse_score": float(collapse[mask].mean().cpu()),
        "path_b_weighted_delta_norm_mean": float(((weights[..., 0] * delta_b * mask_float).sum() / mask_float.sum().clamp_min(1.0)).cpu()),
        "path_a_weighted_delta_norm_mean": float(((weights[..., 1] * delta_a * mask_float).sum() / mask_float.sum().clamp_min(1.0)).cpu()),
    }


def _parameter_budget_summary(
    config: Any,
    backbones: LoadedBackbones,
    path_specs: list[Any],
) -> dict[str, Any]:
    static_mixture = TwoPathStaticMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
    zero_requires_grad(static_mixture, except_prefixes=static_mixture_trainable_prefixes(config))
    static_mixture_no_small = TwoPathStaticMixtureNoSmallModel(config, backbones.large_model, backbones.small_model, path_specs)
    zero_requires_grad(static_mixture_no_small, except_prefixes=static_mixture_trainable_prefixes(config))

    tokenwise = TwoPathTokenwiseMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
    zero_requires_grad(tokenwise, except_prefixes=tokenwise_mixture_trainable_prefixes(config))
    tokenwise_no_small = TwoPathTokenwiseMixtureNoSmallModel(config, backbones.large_model, backbones.small_model, path_specs)
    zero_requires_grad(tokenwise_no_small, except_prefixes=tokenwise_mixture_trainable_prefixes(config))

    tokenwise_trainable = count_parameters(tokenwise).trainable_params
    param_matched_rank = _matched_bridge_rank(tokenwise.large_runner.hidden_size, tokenwise_trainable)

    bridge_only = BridgeOnlyLargeModel(config, backbones.large_model)
    zero_requires_grad(bridge_only, except_prefixes=["bridge", "gate"])
    bridge_param = BridgeOnlyParamMatchedModel(config, backbones.large_model, rank=param_matched_rank)
    zero_requires_grad(bridge_param, except_prefixes=["bridge", "gate"])

    return {
        "static_mixture_trainable_params": count_parameters(static_mixture).trainable_params,
        "static_mixture_no_small_trainable_params": count_parameters(static_mixture_no_small).trainable_params,
        "tokenwise_mixture_trainable_params": tokenwise_trainable,
        "tokenwise_mixture_no_small_trainable_params": count_parameters(tokenwise_no_small).trainable_params,
        "bridge_only_trainable_params": count_parameters(bridge_only).trainable_params,
        "bridge_only_param_matched_rank": param_matched_rank,
        "bridge_only_param_matched_trainable_params": count_parameters(bridge_param).trainable_params,
    }


def _load_static_mixture_payload(model: Any, payload: dict[str, Any], device: torch.device) -> torch.Tensor:
    model.entry_projector_b.load_state_dict(payload["entry_projector_b"])
    model.entry_projector_a.load_state_dict(payload["entry_projector_a"])
    model.return_adapter_b.load_state_dict(payload["return_adapter_b"])
    model.return_adapter_a.load_state_dict(payload["return_adapter_a"])
    static_logits = payload["alpha"].to(device=device, dtype=torch.float32)
    return static_logits


def _warm_start_tokenwise_model(
    model: Any,
    seed: int,
    *,
    static_stage_dir: str | Path,
    static_variant: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(static_stage_dir) / f"seed_{seed}" / f"{static_variant}_checkpoint.pt"
    payload = load_checkpoint(checkpoint_path, device)
    static_logits = _load_static_mixture_payload(model, payload, device)
    model.set_static_prior_logits(static_logits)
    return {
        "seed": seed,
        "static_variant": static_variant,
        "checkpoint_path": str(checkpoint_path),
        "initial_static_prior_logits": [float(value) for value in static_logits.detach().cpu().tolist()],
        "initial_static_prior_weights": [float(value) for value in torch.softmax(static_logits, dim=0).detach().cpu().tolist()],
    }


def _load_static_reference_models(
    config: Any,
    backbones: LoadedBackbones,
    path_specs: list[Any],
    seed: int,
    static_stage_dir: str | Path,
    parameter_budget: dict[str, Any],
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "skip_only": SkipOnlyLargeModel(config, backbones.large_model),
    }

    bridge_payload = load_checkpoint(Path(static_stage_dir) / f"seed_{seed}" / "bridge_only_checkpoint.pt", backbones.device)
    bridge_only = BridgeOnlyLargeModel(config, backbones.large_model)
    bridge_only.bridge.load_state_dict(bridge_payload["bridge"])
    bridge_only.gate.load_state_dict(bridge_payload["gate"])
    zero_requires_grad(bridge_only, except_prefixes=[])
    models["bridge_only"] = bridge_only

    static_no_small_payload = load_checkpoint(
        Path(static_stage_dir) / f"seed_{seed}" / "static_mixture_no_small_checkpoint.pt",
        backbones.device,
    )
    static_no_small = TwoPathStaticMixtureNoSmallModel(config, backbones.large_model, backbones.small_model, path_specs)
    _load_static_mixture_payload(static_no_small, static_no_small_payload, backbones.device)
    zero_requires_grad(static_no_small, except_prefixes=[])
    models["static_mixture_no_small"] = static_no_small

    static_payload = load_checkpoint(Path(static_stage_dir) / f"seed_{seed}" / "static_mixture_checkpoint.pt", backbones.device)
    static_mixture = TwoPathStaticMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
    _load_static_mixture_payload(static_mixture, static_payload, backbones.device)
    zero_requires_grad(static_mixture, except_prefixes=[])
    models["static_mixture"] = static_mixture

    updated_bridge = BridgeOnlyParamMatchedModel(
        config,
        backbones.large_model,
        rank=parameter_budget["bridge_only_param_matched_rank"],
    )
    zero_requires_grad(updated_bridge, except_prefixes=["bridge", "gate"])
    models["bridge_only_param_matched"] = updated_bridge
    return models


def _build_variant_models(
    config: Any,
    backbones: LoadedBackbones,
    path_specs: list[Any],
    seed: int,
    parameter_budget: dict[str, Any],
    static_stage_dir: str | Path,
    trained_variants: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    models = _load_static_reference_models(
        config,
        backbones,
        path_specs,
        seed,
        static_stage_dir,
        parameter_budget,
    )
    warm_start_metadata: dict[str, Any] = {}

    if "tokenwise_mixture_no_small" in trained_variants:
        model = TwoPathTokenwiseMixtureNoSmallModel(config, backbones.large_model, backbones.small_model, path_specs)
        warm_start_metadata["tokenwise_mixture_no_small"] = _warm_start_tokenwise_model(
            model,
            seed,
            static_stage_dir=static_stage_dir,
            static_variant="static_mixture_no_small",
            device=backbones.device,
        )
        zero_requires_grad(model, except_prefixes=tokenwise_mixture_trainable_prefixes(config))
        models["tokenwise_mixture_no_small"] = model

    if "tokenwise_mixture" in trained_variants:
        model = TwoPathTokenwiseMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
        warm_start_metadata["tokenwise_mixture"] = _warm_start_tokenwise_model(
            model,
            seed,
            static_stage_dir=static_stage_dir,
            static_variant="static_mixture",
            device=backbones.device,
        )
        zero_requires_grad(model, except_prefixes=tokenwise_mixture_trainable_prefixes(config))
        models["tokenwise_mixture"] = model

    return models, warm_start_metadata


def _variant_prediction(
    variant: str,
    model: Any,
    hidden_after_prefix: torch.Tensor,
    attention_mask: torch.Tensor,
    train_entry_projector: bool,
) -> dict[str, Any]:
    if variant == "skip_only":
        return {
            "hidden_after_removed": hidden_after_prefix,
            "delta_large": None,
            "static_weights": None,
            "tokenwise_weights": None,
            "gate_logits": None,
            "path_outputs": None,
            "gate_value": 0.0,
        }
    if variant in {"bridge_only", "bridge_only_param_matched"}:
        delta_large = model.bridge(hidden_after_prefix)
        gated_delta_large, gate_value = _resolve_gate_value(model.gate, delta_large)
        return {
            "hidden_after_removed": hidden_after_prefix + gated_delta_large,
            "delta_large": delta_large,
            "static_weights": None,
            "tokenwise_weights": None,
            "gate_logits": None,
            "path_outputs": None,
            "gate_value": gate_value,
        }
    if variant in {"static_mixture", "static_mixture_no_small"}:
        path_outputs, delta_large, weights = model.compute_mixed_delta(
            hidden_after_prefix,
            attention_mask,
            train_entry_projector=train_entry_projector,
        )
        return {
            "hidden_after_removed": hidden_after_prefix + delta_large,
            "delta_large": delta_large,
            "static_weights": [float(weights[0].detach().cpu()), float(weights[1].detach().cpu())],
            "tokenwise_weights": None,
            "gate_logits": None,
            "path_outputs": path_outputs,
            "gate_value": None,
        }
    if variant in {"tokenwise_mixture", "tokenwise_mixture_no_small"}:
        path_outputs, delta_large, gate_weights, gate_logits = model.compute_mixed_delta(
            hidden_after_prefix,
            attention_mask,
            train_entry_projector=train_entry_projector,
        )
        return {
            "hidden_after_removed": hidden_after_prefix + delta_large,
            "delta_large": delta_large,
            "static_weights": None,
            "tokenwise_weights": gate_weights,
            "gate_logits": gate_logits,
            "path_outputs": path_outputs,
            "gate_value": None,
        }
    raise ValueError(f"Unsupported Idea 4 tokenwise variant: {variant}")


def _train_variant(
    variant: str,
    model: Any,
    config: Any,
    backbones: LoadedBackbones,
    train_dataloader: Any,
    seed: int,
) -> tuple[Any, list[dict[str, float]], dict[str, Any]]:
    seed_everything(seed)
    optimizer = build_stage_b_optimizer(model, config)
    batch_iterator = itertools.cycle(train_dataloader)
    history: list[dict[str, float]] = []
    model.train()
    train_entry_projector = bool(config.training.stage_b.train_entry_projector)
    gate_settings = tokenwise_gate_settings(config)

    for step in range(1, config.training.stage_b.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        row_snapshot: dict[str, float] | None = None
        for _ in range(config.training.grad_accum_steps):
            batch = move_batch_to_device(next(batch_iterator), backbones.device)
            teacher_targets = prepare_stage_b_teacher_targets(model.large_runner, batch, config)
            hidden_after_prefix = teacher_targets.hidden_after_prefix
            prediction = _variant_prediction(
                variant,
                model,
                hidden_after_prefix=hidden_after_prefix,
                attention_mask=batch["attention_mask"],
                train_entry_projector=train_entry_projector,
            )
            loss_terms = compute_stage_b_loss_breakdown(
                model.large_runner,
                config,
                teacher_targets,
                prediction["hidden_after_removed"],
                batch["attention_mask"],
                batch["labels"],
                prediction["delta_large"],
            )
            total_loss = loss_terms.total_loss
            entropy_loss = total_loss.new_zeros(())
            prior_kl_loss = total_loss.new_zeros(())
            smoothness_loss = total_loss.new_zeros(())
            if prediction["tokenwise_weights"] is not None:
                regularizers = _gate_regularizers(
                    model,
                    prediction["tokenwise_weights"],
                    batch["attention_mask"],
                    gate_settings,
                )
                entropy_loss = regularizers["entropy_loss"]
                prior_kl_loss = regularizers["prior_kl_loss"]
                smoothness_loss = regularizers["smoothness_loss"]
                total_loss = total_loss + gate_settings["entropy_reg_weight"] * entropy_loss
                total_loss = total_loss + gate_settings["prior_kl_weight"] * prior_kl_loss
                total_loss = total_loss + gate_settings["smoothness_weight"] * smoothness_loss

            (total_loss / config.training.grad_accum_steps).backward()

            delta_stats = _delta_norm_stats(prediction["delta_large"], batch["attention_mask"])
            row_snapshot = {
                "seed": float(seed),
                "variant": variant,
                "step": float(step),
                "loss": float(total_loss.detach().cpu()),
                "mse_loss": float(loss_terms.mse_loss.detach().cpu()),
                "cosine_loss": float(loss_terms.cosine_loss.detach().cpu()),
                "kl_loss": float(loss_terms.kl_loss.detach().cpu()),
                "ce_loss": float(loss_terms.ce_loss.detach().cpu()),
                "delta_reg": float(loss_terms.delta_reg.detach().cpu()),
                "delta_norm_mean": delta_stats["delta_norm_mean"],
                "delta_norm_max": delta_stats["delta_norm_max"],
                "gate_entropy_loss": float(entropy_loss.detach().cpu()),
                "gate_prior_kl_loss": float(prior_kl_loss.detach().cpu()),
                "gate_smoothness_loss": float(smoothness_loss.detach().cpu()),
            }
            if prediction["tokenwise_weights"] is not None:
                gate_stats = _tokenwise_gate_stats(
                    prediction["tokenwise_weights"],
                    prediction["path_outputs"],
                    batch["attention_mask"],
                    gate_settings["collapse_threshold"],
                )
                row_snapshot.update(gate_stats)
            elif prediction["static_weights"] is not None:
                row_snapshot["weight_path_b"] = prediction["static_weights"][0]
                row_snapshot["weight_path_a"] = prediction["static_weights"][1]
            else:
                row_snapshot["gate_value"] = float(prediction["gate_value"] or 0.0)

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()
        if row_snapshot is None:
            raise RuntimeError("Token-wise training did not record a step snapshot.")
        history.append(row_snapshot)

        if step % config.training.log_every == 0 or step == config.training.stage_b.max_steps:
            LOGGER.info(
                "idea4_tokenwise seed=%s variant=%s step=%s loss=%.6f mse=%.6f cosine=%.6f kl=%.6f ce=%.6f",
                seed,
                variant,
                step,
                row_snapshot["loss"],
                row_snapshot["mse_loss"],
                row_snapshot["cosine_loss"],
                row_snapshot["kl_loss"],
                row_snapshot["ce_loss"],
            )

    checkpoint_payload = {
        "seed": seed,
        "variant": variant,
        "step": config.training.stage_b.max_steps,
    }
    if variant == "bridge_only_param_matched":
        checkpoint_payload["bridge"] = model.bridge.state_dict()
        checkpoint_payload["gate"] = model.gate.state_dict()
    else:
        checkpoint_payload["entry_projector_b"] = model.entry_projector_b.state_dict()
        checkpoint_payload["entry_projector_a"] = model.entry_projector_a.state_dict()
        checkpoint_payload["return_adapter_b"] = model.return_adapter_b.state_dict()
        checkpoint_payload["return_adapter_a"] = model.return_adapter_a.state_dict()
        checkpoint_payload["gate_network"] = model.gate_network.state_dict()
        checkpoint_payload["static_prior_logits"] = model.static_prior_logits.detach().cpu()
    return model, history, checkpoint_payload


@torch.no_grad()
def _evaluate_models(
    models: dict[str, Any],
    config: Any,
    backbones: LoadedBackbones,
    val_dataloader: Any,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    gate_settings = tokenwise_gate_settings(config)
    for model in models.values():
        model.eval()

    totals: dict[str, float] = {}
    for variant in EVAL_VARIANTS:
        totals[f"{variant}_hidden_mse"] = 0.0
        totals[f"{variant}_cosine_loss"] = 0.0
        totals[f"{variant}_delta_norm_mean"] = 0.0
        totals[f"{variant}_delta_norm_max"] = 0.0
        if variant in {"bridge_only", "bridge_only_param_matched"}:
            totals[f"{variant}_gate_value"] = 0.0
        if variant in {"static_mixture", "static_mixture_no_small"}:
            totals[f"{variant}_weight_path_b"] = 0.0
            totals[f"{variant}_weight_path_a"] = 0.0
        if variant in {"tokenwise_mixture", "tokenwise_mixture_no_small"}:
            totals[f"{variant}_weight_path_b"] = 0.0
            totals[f"{variant}_weight_path_a"] = 0.0
            totals[f"{variant}_gate_entropy"] = 0.0
            totals[f"{variant}_path_b_usage_var"] = 0.0
            totals[f"{variant}_path_a_usage_var"] = 0.0
            totals[f"{variant}_collapse_score"] = 0.0
            totals[f"{variant}_path_b_weighted_delta_norm_mean"] = 0.0
            totals[f"{variant}_path_a_weighted_delta_norm_mean"] = 0.0

    batches = 0
    for batch in val_dataloader:
        batch = move_batch_to_device(batch, backbones.device)
        teacher_runner = next(iter(models.values())).large_runner
        teacher_targets = prepare_stage_b_teacher_targets(teacher_runner, batch, config, include_teacher_logits=False)
        for variant in EVAL_VARIANTS:
            prediction = _variant_prediction(
                variant,
                models[variant],
                hidden_after_prefix=teacher_targets.hidden_after_prefix,
                attention_mask=batch["attention_mask"],
                train_entry_projector=bool(config.training.stage_b.train_entry_projector),
            )
            totals[f"{variant}_hidden_mse"] += float(
                masked_hidden_mse(prediction["hidden_after_removed"], teacher_targets.teacher_hidden, batch["attention_mask"]).cpu()
            )
            totals[f"{variant}_cosine_loss"] += float(
                masked_hidden_cosine_loss(prediction["hidden_after_removed"], teacher_targets.teacher_hidden, batch["attention_mask"]).cpu()
            )
            delta_stats = _delta_norm_stats(prediction["delta_large"], batch["attention_mask"])
            totals[f"{variant}_delta_norm_mean"] += delta_stats["delta_norm_mean"]
            totals[f"{variant}_delta_norm_max"] += delta_stats["delta_norm_max"]

            if prediction["static_weights"] is not None:
                totals[f"{variant}_weight_path_b"] += prediction["static_weights"][0]
                totals[f"{variant}_weight_path_a"] += prediction["static_weights"][1]
            if prediction["tokenwise_weights"] is not None:
                gate_stats = _tokenwise_gate_stats(
                    prediction["tokenwise_weights"],
                    prediction["path_outputs"],
                    batch["attention_mask"],
                    gate_settings["collapse_threshold"],
                )
                for key, value in gate_stats.items():
                    totals[f"{variant}_{key}"] += value
            gate_key = f"{variant}_gate_value"
            if prediction["gate_value"] is not None and gate_key in totals:
                totals[gate_key] += float(prediction["gate_value"])
        batches += 1

    for key, value in totals.items():
        metrics[key] = value / max(1, batches)
    for variant in EVAL_VARIANTS:
        metrics[f"{variant}_cosine"] = 1.0 - metrics.pop(f"{variant}_cosine_loss")
    return metrics


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _aggregate_results(seed_results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {"per_variant": {}}
    rows: list[dict[str, Any]] = []
    per_variant_metrics = {
        "skip_only": ["hidden_mse", "cosine", "delta_norm_mean", "delta_norm_max"],
        "bridge_only": ["hidden_mse", "cosine", "delta_norm_mean", "delta_norm_max", "gate_value"],
        "static_mixture_no_small": ["hidden_mse", "cosine", "delta_norm_mean", "delta_norm_max", "weight_path_b", "weight_path_a"],
        "static_mixture": ["hidden_mse", "cosine", "delta_norm_mean", "delta_norm_max", "weight_path_b", "weight_path_a"],
        "bridge_only_param_matched": ["hidden_mse", "cosine", "delta_norm_mean", "delta_norm_max", "gate_value"],
        "tokenwise_mixture_no_small": [
            "hidden_mse",
            "cosine",
            "delta_norm_mean",
            "delta_norm_max",
            "weight_path_b",
            "weight_path_a",
            "gate_entropy",
            "path_b_usage_var",
            "path_a_usage_var",
            "collapse_score",
            "path_b_weighted_delta_norm_mean",
            "path_a_weighted_delta_norm_mean",
        ],
        "tokenwise_mixture": [
            "hidden_mse",
            "cosine",
            "delta_norm_mean",
            "delta_norm_max",
            "weight_path_b",
            "weight_path_a",
            "gate_entropy",
            "path_b_usage_var",
            "path_a_usage_var",
            "collapse_score",
            "path_b_weighted_delta_norm_mean",
            "path_a_weighted_delta_norm_mean",
        ],
    }

    for variant, metric_names in per_variant_metrics.items():
        variant_summary: dict[str, float] = {}
        for metric_name in metric_names:
            values = [seed_result["metrics"][f"{variant}_{metric_name}"] for seed_result in seed_results]
            variant_summary[f"{metric_name}_mean"] = _mean(values)
            variant_summary[f"{metric_name}_std"] = _std(values)
        summary["per_variant"][variant] = variant_summary
        rows.append({"row_type": "variant", "label": variant, **variant_summary})
    return summary, rows


def _write_report(report_path: str | Path, results_payload: dict[str, Any]) -> None:
    summary = results_payload["summary"]["per_variant"]
    budget = results_payload["parameter_budget"]
    gate_settings = results_payload["tokenwise_gate_settings"]
    lines = [
        "# Idea 4 Token-Wise Report",
        "",
        "## setup",
        "",
        f"- Config: {results_payload['config_path']}",
        f"- static_stage_dir: {results_payload['static_stage_dir']}",
        f"- seq_len: {results_payload['seq_len']}",
        f"- max_train_steps: {results_payload['max_train_steps']}",
        f"- Seeds: {', '.join(str(seed) for seed in results_payload['seeds'])}",
        "- Token-wise gate input: large-prefix hidden state at the splice boundary only.",
        f"- Token-wise gate hidden_dim: {gate_settings['hidden_dim']}",
        f"- Gate regularization: entropy={gate_settings['entropy_reg_weight']}, prior_kl={gate_settings['prior_kl_weight']}, smoothness={gate_settings['smoothness_weight']}",
        "",
        "## fairness audit",
        "",
        f"- static_mixture trainable params: {budget['static_mixture_trainable_params']}",
        f"- static_mixture_no_small trainable params: {budget['static_mixture_no_small_trainable_params']}",
        f"- tokenwise_mixture trainable params: {budget['tokenwise_mixture_trainable_params']}",
        f"- tokenwise_mixture_no_small trainable params: {budget['tokenwise_mixture_no_small_trainable_params']}",
        f"- bridge_only trainable params: {budget['bridge_only_trainable_params']}",
        f"- updated parameter-matched bridge rank: {budget['bridge_only_param_matched_rank']}",
        f"- updated parameter-matched bridge trainable params: {budget['bridge_only_param_matched_trainable_params']}",
        "",
        "## hidden diagnostics",
        "",
        f"- tokenwise_mixture: hidden_mse_mean={summary['tokenwise_mixture']['hidden_mse_mean']:.6f}, cosine_mean={summary['tokenwise_mixture']['cosine_mean']:.6f}, weight_path_b_mean={summary['tokenwise_mixture']['weight_path_b_mean']:.6f}, weight_path_a_mean={summary['tokenwise_mixture']['weight_path_a_mean']:.6f}, gate_entropy_mean={summary['tokenwise_mixture']['gate_entropy_mean']:.6f}, collapse_score_mean={summary['tokenwise_mixture']['collapse_score_mean']:.6f}",
        f"- tokenwise_mixture_no_small: hidden_mse_mean={summary['tokenwise_mixture_no_small']['hidden_mse_mean']:.6f}, cosine_mean={summary['tokenwise_mixture_no_small']['cosine_mean']:.6f}, weight_path_b_mean={summary['tokenwise_mixture_no_small']['weight_path_b_mean']:.6f}, weight_path_a_mean={summary['tokenwise_mixture_no_small']['weight_path_a_mean']:.6f}, gate_entropy_mean={summary['tokenwise_mixture_no_small']['gate_entropy_mean']:.6f}, collapse_score_mean={summary['tokenwise_mixture_no_small']['collapse_score_mean']:.6f}",
        f"- static_mixture reference: hidden_mse_mean={summary['static_mixture']['hidden_mse_mean']:.6f}, cosine_mean={summary['static_mixture']['cosine_mean']:.6f}",
        f"- bridge_only reference: hidden_mse_mean={summary['bridge_only']['hidden_mse_mean']:.6f}, cosine_mean={summary['bridge_only']['cosine_mean']:.6f}",
        f"- updated parameter-matched bridge: hidden_mse_mean={summary['bridge_only_param_matched']['hidden_mse_mean']:.6f}, cosine_mean={summary['bridge_only_param_matched']['cosine_mean']:.6f}",
        "",
        "- Output-level results remain primary; see the paired output-probe note for the actual continuation decision.",
        "",
    ]
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    path_specs = load_mixture_path_specs(config)
    gate_settings = tokenwise_gate_settings(config)
    output_dir = ensure_dir(args.output_dir)
    save_config_snapshot(output_dir / "config_snapshot.yaml", config)
    export_run_metadata(
        output_dir / "metadata.json",
        config,
        {
            "stage": "idea4_tokenwise",
            "seeds": args.seeds,
            "variants": args.variants,
            "static_stage_dir": args.static_stage_dir,
            "path_specs": [spec.to_dict() for spec in path_specs],
            "tokenwise_gate_settings": gate_settings,
        },
    )

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    parameter_budget = _parameter_budget_summary(config, backbones, path_specs)
    train_dataloader, _ = build_dataloader(config, backbones.tokenizer, stage_name="stage_b", split_name="train")
    val_dataloader, _ = build_dataloader(config, backbones.tokenizer, stage_name="stage_b", split_name="validation")

    seed_results: list[dict[str, Any]] = []
    all_history_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "parameter_budget": parameter_budget,
        "path_specs": [spec.to_dict() for spec in path_specs],
        "static_stage_dir": args.static_stage_dir,
        "tokenwise_gate_settings": gate_settings,
        "seed_warm_starts": {},
    }

    for seed in args.seeds:
        LOGGER.info("idea4_tokenwise seed=%s", seed)
        seed_config = _clone_config_with_seed(config, seed)
        models, warm_start_metadata = _build_variant_models(
            seed_config,
            backbones,
            path_specs,
            seed,
            parameter_budget,
            args.static_stage_dir,
            args.variants,
        )
        diagnostics["seed_warm_starts"][str(seed)] = warm_start_metadata

        checkpoint_paths: dict[str, str] = {}
        for variant in args.variants:
            model, history_rows, checkpoint_payload = _train_variant(
                variant,
                models[variant],
                seed_config,
                backbones,
                train_dataloader,
                seed,
            )
            models[variant] = model
            all_history_rows.extend(history_rows)
            checkpoint_path = output_dir / f"seed_{seed}" / f"{variant}_checkpoint.pt"
            ensure_dir(checkpoint_path.parent)
            save_checkpoint(checkpoint_path, checkpoint_payload)
            checkpoint_paths[variant] = str(checkpoint_path)
            torch.cuda.empty_cache()

        metrics = _evaluate_models(models, seed_config, backbones, val_dataloader)
        seed_result = {
            "seed": seed,
            "metrics": metrics,
            "checkpoint_paths": checkpoint_paths,
            "warm_start": warm_start_metadata,
        }
        seed_results.append(seed_result)
        save_json(output_dir / f"seed_{seed}" / "metrics.json", seed_result)
        torch.cuda.empty_cache()

    summary, summary_rows = _aggregate_results(seed_results)
    results_payload = {
        "config_path": args.config,
        "static_stage_dir": args.static_stage_dir,
        "seq_len": config.training.seq_len,
        "max_train_steps": config.training.stage_b.max_steps,
        "seed_count": len(args.seeds),
        "seeds": args.seeds,
        "variants": args.variants,
        "parameter_budget": parameter_budget,
        "path_specs": [spec.to_dict() for spec in path_specs],
        "tokenwise_gate_settings": gate_settings,
        "stage_b_loss_weights": {
            "kl_weight": config.training.stage_b.kl_weight,
            "ce_weight": config.training.stage_b.ce_weight,
            "delta_reg_weight": config.training.stage_b.delta_reg_weight,
        },
        "seed_results": seed_results,
        "summary": summary,
    }
    save_json(output_dir / "results.json", results_payload)
    save_csv(output_dir / "history.csv", all_history_rows)
    save_csv(output_dir / "summary.csv", summary_rows)
    ensure_dir(Path(args.results_path).parent)
    ensure_dir(Path(args.summary_path).parent)
    ensure_dir(Path(args.diagnostics_path).parent)
    save_json(args.results_path, results_payload)
    save_csv(args.summary_path, summary_rows)
    save_json(args.diagnostics_path, diagnostics)
    _write_report(args.report_path, results_payload)


if __name__ == "__main__":
    main()
