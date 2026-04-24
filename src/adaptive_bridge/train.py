"""Train the first adaptive-bridge bridge-aware residual MoE variants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

from src.adaptive_bridge.common import (
    ADAPTIVE_BRIDGE_TRAINED_VARIANTS,
    adaptive_bridge_gate_settings,
    adaptive_bridge_settings,
    adaptive_bridge_trainable_prefixes,
    checkpoint_path_from_template,
    clone_config_with_seed,
    matched_bridge_rank,
    maybe_load_checkpoint,
    require_warm_start_checkpoints,
)
from src.adaptive_bridge.models import BridgeAwareResidualMoE, BridgeAwareResidualMoENoSmall
from src.eval.metrics import masked_hidden_cosine_loss, masked_hidden_mse, shifted_cross_entropy, shifted_kl_divergence
from src.models.backbone_loader import LoadedBackbones, load_backbones
from src.models.baselines import BridgeOnlyParamMatchedModel, SkipOnlyLargeModel
from src.models.hooks import count_parameters
from src.models.hybrid_gemma import _resolve_gate_value
from src.train.stage_b_objective import compute_stage_b_loss_breakdown, prepare_stage_b_teacher_targets
from src.train.trainer_utils import (
    build_dataloader,
    move_batch_to_device,
    save_checkpoint,
    zero_requires_grad,
)
from src.utils.io import ensure_dir, export_run_metadata, load_config, save_config_snapshot, save_csv, save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything
from src.v0_6.idea4_common import load_mixture_path_specs
from src.v0_6.idea4_tokenwise_models import TwoPathTokenwiseMixtureHybrid, tokenwise_mixture_trainable_prefixes


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for adaptive-bridge Stage B training."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/adaptive_bridge/train")
    parser.add_argument("--results-path", default="outputs/adaptive_bridge/train/results.json")
    parser.add_argument("--summary-path", default="outputs/adaptive_bridge/train/summary.csv")
    parser.add_argument("--diagnostics-path", default="outputs/adaptive_bridge/train/diagnostics.json")
    parser.add_argument("--variants", nargs="+", choices=ADAPTIVE_BRIDGE_TRAINED_VARIANTS, default=ADAPTIVE_BRIDGE_TRAINED_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    return parser.parse_args()


def _build_stage_b_optimizer(module: nn.Module, config: Any) -> AdamW:
    """Build a Stage B optimizer with module-specific learning rates."""

    stage_b = config.training.stage_b
    base_lr = config.training.learning_rate
    entry_lr = stage_b.entry_lr or base_lr
    return_lr = stage_b.return_lr or base_lr
    gate_lr = stage_b.gate_lr or base_lr

    grouped_parameters: dict[float, list[nn.Parameter]] = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = base_lr
        if name.startswith("entry_projector"):
            lr = entry_lr
        elif name.startswith(("return_adapter", "bridge_expert", "bridge")):
            lr = return_lr
        elif name.startswith(("gate_network", "gate")):
            lr = gate_lr
        grouped_parameters.setdefault(lr, []).append(parameter)

    return AdamW(
        [{"params": params, "lr": lr} for lr, params in grouped_parameters.items()],
        lr=base_lr,
        weight_decay=config.training.weight_decay,
    )


def _masked_mean(values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _gate_regularizers(
    model: BridgeAwareResidualMoE,
    gate_weights: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    settings = adaptive_bridge_gate_settings(model.config)
    eps = 1.0e-8
    gate_weights = gate_weights.clamp_min(eps)
    entropy = -(gate_weights * gate_weights.log()).sum(dim=-1)
    entropy_loss = _masked_mean(entropy, attention_mask)

    prior = model.expert_prior_weights().to(device=gate_weights.device, dtype=gate_weights.dtype).clamp_min(eps)
    prior_log = prior.log().view(1, 1, -1)
    prior_kl = (gate_weights * (gate_weights.log() - prior_log)).sum(dim=-1)
    prior_kl_loss = _masked_mean(prior_kl, attention_mask)

    smoothness = gate_weights.new_zeros(())
    if settings.smoothness_weight > 0.0 and gate_weights.shape[1] > 1:
        pair_mask = (attention_mask[:, 1:] * attention_mask[:, :-1]).to(gate_weights.dtype)
        diffs = (gate_weights[:, 1:, :] - gate_weights[:, :-1, :]).abs().sum(dim=-1)
        smoothness = (diffs * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    return {
        "entropy_loss": entropy_loss,
        "prior_kl_loss": prior_kl_loss,
        "smoothness_loss": smoothness,
    }


def _adaptive_gate_stats(
    gate_weights: torch.Tensor,
    attention_mask: torch.Tensor,
    collapse_threshold: float,
) -> dict[str, float]:
    weights = gate_weights.detach().float()
    mask = attention_mask.bool()
    valid_weights = weights[mask]
    if valid_weights.numel() == 0:
        return {
            "weight_bridge": 0.0,
            "weight_path_b": 0.0,
            "weight_path_a": 0.0,
            "gate_entropy": 0.0,
            "bridge_usage_var": 0.0,
            "path_b_usage_var": 0.0,
            "path_a_usage_var": 0.0,
            "collapse_score": 0.0,
        }
    entropy = -(weights.clamp_min(1.0e-8) * weights.clamp_min(1.0e-8).log()).sum(dim=-1)
    collapse = (weights.max(dim=-1).values > collapse_threshold).to(weights.dtype)
    return {
        "weight_bridge": float(valid_weights[:, 0].mean().cpu()),
        "weight_path_b": float(valid_weights[:, 1].mean().cpu()),
        "weight_path_a": float(valid_weights[:, 2].mean().cpu()),
        "gate_entropy": float(entropy[mask].mean().cpu()),
        "bridge_usage_var": float(valid_weights[:, 0].var(unbiased=False).cpu()),
        "path_b_usage_var": float(valid_weights[:, 1].var(unbiased=False).cpu()),
        "path_a_usage_var": float(valid_weights[:, 2].var(unbiased=False).cpu()),
        "collapse_score": float(collapse[mask].mean().cpu()),
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


def _parameter_budget_summary(config: Any, backbones: LoadedBackbones, path_specs: list[Any]) -> dict[str, Any]:
    tokenwise = TwoPathTokenwiseMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
    zero_requires_grad(tokenwise, except_prefixes=tokenwise_mixture_trainable_prefixes(config))

    adaptive = BridgeAwareResidualMoE(config, backbones.large_model, backbones.small_model, path_specs)
    zero_requires_grad(adaptive, except_prefixes=adaptive_bridge_trainable_prefixes(config))
    adaptive_no_small = BridgeAwareResidualMoENoSmall(config, backbones.large_model, backbones.small_model, path_specs)
    zero_requires_grad(adaptive_no_small, except_prefixes=adaptive_bridge_trainable_prefixes(config))

    strong_bridge_rank = adaptive_bridge_settings(config).bridge_rank
    bridge_only_strong = BridgeOnlyParamMatchedModel(config, backbones.large_model, rank=strong_bridge_rank)
    zero_requires_grad(bridge_only_strong, except_prefixes=["bridge", "gate"])

    target_trainable_params = count_parameters(adaptive).trainable_params
    param_matched_rank = matched_bridge_rank(adaptive.large_runner.hidden_size, target_trainable_params)
    bridge_only_param = BridgeOnlyParamMatchedModel(config, backbones.large_model, rank=param_matched_rank)
    zero_requires_grad(bridge_only_param, except_prefixes=["bridge", "gate"])

    return {
        "frozen_v060_tokenwise_trainable_params": count_parameters(tokenwise).trainable_params,
        "adaptive_bridge_moe_trainable_params": target_trainable_params,
        "adaptive_bridge_no_small_trainable_params": count_parameters(adaptive_no_small).trainable_params,
        "bridge_only_strong_rank": strong_bridge_rank,
        "bridge_only_strong_trainable_params": count_parameters(bridge_only_strong).trainable_params,
        "bridge_only_param_matched_rank": param_matched_rank,
        "bridge_only_param_matched_trainable_params": count_parameters(bridge_only_param).trainable_params,
    }


def _warm_start_adaptive_model(
    model: BridgeAwareResidualMoE,
    config: Any,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    settings = adaptive_bridge_settings(config)
    warm_start_path = None
    if settings.warm_start_from_v060:
        warm_start_path = checkpoint_path_from_template(settings.warm_start_template, seed)
        payload = maybe_load_checkpoint(warm_start_path, device)
        if payload is not None:
            model.warm_start_from_tokenwise_payload(payload)
            return {
                "enabled": True,
                "status": "loaded",
                "checkpoint_path": str(warm_start_path),
            }
        return {
            "enabled": True,
            "status": "missing_checkpoint",
            "checkpoint_path": None if warm_start_path is None else str(warm_start_path),
        }
    return {"enabled": False, "status": "disabled", "checkpoint_path": None}


def _build_models(
    config: Any,
    backbones: LoadedBackbones,
    path_specs: list[Any],
    seed: int,
    parameter_budget: dict[str, Any],
    variants: list[str],
) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    models: dict[str, nn.Module] = {
        "skip_only": SkipOnlyLargeModel(config, backbones.large_model),
    }
    warm_starts: dict[str, Any] = {}

    if "bridge_only_strong" in variants:
        model = BridgeOnlyParamMatchedModel(
            config,
            backbones.large_model,
            rank=parameter_budget["bridge_only_strong_rank"],
        )
        zero_requires_grad(model, except_prefixes=["bridge", "gate"])
        models["bridge_only_strong"] = model

    if "bridge_only_param_matched" in variants:
        model = BridgeOnlyParamMatchedModel(
            config,
            backbones.large_model,
            rank=parameter_budget["bridge_only_param_matched_rank"],
        )
        zero_requires_grad(model, except_prefixes=["bridge", "gate"])
        models["bridge_only_param_matched"] = model

    if "adaptive_bridge_no_small" in variants:
        model = BridgeAwareResidualMoENoSmall(config, backbones.large_model, backbones.small_model, path_specs)
        warm_starts["adaptive_bridge_no_small"] = _warm_start_adaptive_model(model, config, seed, backbones.device)
        zero_requires_grad(model, except_prefixes=adaptive_bridge_trainable_prefixes(config))
        models["adaptive_bridge_no_small"] = model

    if "adaptive_bridge_moe" in variants:
        model = BridgeAwareResidualMoE(config, backbones.large_model, backbones.small_model, path_specs)
        warm_starts["adaptive_bridge_moe"] = _warm_start_adaptive_model(model, config, seed, backbones.device)
        zero_requires_grad(model, except_prefixes=adaptive_bridge_trainable_prefixes(config))
        models["adaptive_bridge_moe"] = model

    return models, warm_starts


def _variant_prediction(
    variant: str,
    model: nn.Module,
    hidden_after_prefix: torch.Tensor,
    attention_mask: torch.Tensor,
    train_entry_projector: bool,
) -> dict[str, Any]:
    if variant == "skip_only":
        return {
            "hidden_after_removed": hidden_after_prefix,
            "delta_large": None,
            "gate_value": None,
            "gate_weights": None,
            "expert_outputs": None,
        }
    if variant in {"bridge_only_strong", "bridge_only_param_matched"}:
        delta_large = model.bridge(hidden_after_prefix)
        gated_delta_large, gate_value = _resolve_gate_value(model.gate, delta_large)
        return {
            "hidden_after_removed": hidden_after_prefix + gated_delta_large,
            "delta_large": delta_large,
            "gate_value": gate_value,
            "gate_weights": None,
            "expert_outputs": None,
        }
    if variant in {"adaptive_bridge_no_small", "adaptive_bridge_moe"}:
        expert_outputs, delta_large, gate_weights, gate_logits = model.compute_mixed_delta(
            hidden_after_prefix,
            attention_mask,
            train_entry_projector=train_entry_projector,
        )
        return {
            "hidden_after_removed": hidden_after_prefix + delta_large,
            "delta_large": delta_large,
            "gate_value": None,
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "expert_outputs": expert_outputs,
        }
    raise ValueError(f"Unsupported adaptive-bridge variant: {variant}")


def _train_variant(
    variant: str,
    model: nn.Module,
    config: Any,
    backbones: LoadedBackbones,
    train_dataloader: Any,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, Any]]:
    seed_everything(seed)
    optimizer = _build_stage_b_optimizer(model, config)
    history: list[dict[str, float]] = []
    gate_settings = adaptive_bridge_gate_settings(config)
    train_entry_projector = bool(config.training.stage_b.train_entry_projector)
    train_iterator = iter(train_dataloader)
    model.train()

    for step in range(1, config.training.stage_b.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_snapshot: dict[str, float] | None = None
        for _ in range(config.training.grad_accum_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
            batch = move_batch_to_device(batch, backbones.device)
            teacher_targets = prepare_stage_b_teacher_targets(model.large_runner, batch, config)
            prediction = _variant_prediction(
                variant,
                model,
                hidden_after_prefix=teacher_targets.hidden_after_prefix,
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
            if prediction["gate_weights"] is not None:
                regularizers = _gate_regularizers(model, prediction["gate_weights"], batch["attention_mask"])
                entropy_loss = regularizers["entropy_loss"]
                prior_kl_loss = regularizers["prior_kl_loss"]
                smoothness_loss = regularizers["smoothness_loss"]
                total_loss = total_loss + gate_settings.entropy_reg_weight * entropy_loss
                total_loss = total_loss + gate_settings.prior_kl_weight * prior_kl_loss
                total_loss = total_loss + gate_settings.smoothness_weight * smoothness_loss

            (total_loss / config.training.grad_accum_steps).backward()

            delta_stats = _delta_norm_stats(prediction["delta_large"], batch["attention_mask"])
            batch_snapshot = {
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
            if prediction["gate_weights"] is not None:
                gate_stats = _adaptive_gate_stats(
                    prediction["gate_weights"],
                    batch["attention_mask"],
                    gate_settings.collapse_threshold,
                )
                batch_snapshot.update(gate_stats)
            elif prediction["gate_value"] is not None:
                batch_snapshot["gate_value"] = float(prediction["gate_value"])

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()
        if batch_snapshot is None:
            raise RuntimeError("adaptive-bridge training did not record a batch snapshot.")
        history.append(batch_snapshot)
        if step % config.training.log_every == 0 or step == config.training.stage_b.max_steps:
            LOGGER.info(
                "adaptive_bridge seed=%s variant=%s step=%s loss=%.6f mse=%.6f cosine=%.6f kl=%.6f ce=%.6f",
                seed,
                variant,
                step,
                batch_snapshot["loss"],
                batch_snapshot["mse_loss"],
                batch_snapshot["cosine_loss"],
                batch_snapshot["kl_loss"],
                batch_snapshot["ce_loss"],
            )

    checkpoint_payload: dict[str, Any] = {
        "seed": seed,
        "variant": variant,
        "step": config.training.stage_b.max_steps,
    }
    if variant in {"bridge_only_strong", "bridge_only_param_matched"}:
        checkpoint_payload["bridge"] = model.bridge.state_dict()
        checkpoint_payload["gate"] = model.gate.state_dict()
    else:
        checkpoint_payload["entry_projector_b"] = model.entry_projector_b.state_dict()
        checkpoint_payload["entry_projector_a"] = model.entry_projector_a.state_dict()
        checkpoint_payload["return_adapter_b"] = model.return_adapter_b.state_dict()
        checkpoint_payload["return_adapter_a"] = model.return_adapter_a.state_dict()
        checkpoint_payload["bridge_expert"] = model.bridge_expert.state_dict()
        checkpoint_payload["gate_network"] = model.gate_network.state_dict()
        checkpoint_payload["expert_prior_logits"] = model.expert_prior_logits.detach().cpu()
    return model, history, checkpoint_payload


@torch.no_grad()
def _evaluate_models(models: dict[str, nn.Module], config: Any, backbones: LoadedBackbones, val_dataloader: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    totals: dict[str, float] = {}
    eval_variants = [name for name in ["skip_only"] + ADAPTIVE_BRIDGE_TRAINED_VARIANTS if name in models]
    for model in models.values():
        model.eval()
    for variant in eval_variants:
        totals[f"{variant}_hidden_mse"] = 0.0
        totals[f"{variant}_cosine_loss"] = 0.0
        totals[f"{variant}_nll"] = 0.0
        totals[f"{variant}_logit_kl_to_teacher"] = 0.0
        totals[f"{variant}_delta_norm_mean"] = 0.0
        totals[f"{variant}_delta_norm_max"] = 0.0
        if variant in {"bridge_only_strong", "bridge_only_param_matched"}:
            totals[f"{variant}_gate_value"] = 0.0
        if variant in {"adaptive_bridge_no_small", "adaptive_bridge_moe"}:
            totals[f"{variant}_weight_bridge"] = 0.0
            totals[f"{variant}_weight_path_b"] = 0.0
            totals[f"{variant}_weight_path_a"] = 0.0
            totals[f"{variant}_gate_entropy"] = 0.0
            totals[f"{variant}_bridge_usage_var"] = 0.0
            totals[f"{variant}_path_b_usage_var"] = 0.0
            totals[f"{variant}_path_a_usage_var"] = 0.0
            totals[f"{variant}_collapse_score"] = 0.0

    batches = 0
    for batch in val_dataloader:
        batch = move_batch_to_device(batch, backbones.device)
        teacher_targets = prepare_stage_b_teacher_targets(
            next(iter(models.values())).large_runner,
            batch,
            config,
            include_teacher_logits=True,
        )
        for variant in eval_variants:
            prediction = _variant_prediction(
                variant,
                models[variant],
                hidden_after_prefix=teacher_targets.hidden_after_prefix,
                attention_mask=batch["attention_mask"],
                train_entry_projector=bool(config.training.stage_b.train_entry_projector),
            )
            student_state = teacher_targets.prefix_state.with_hidden(prediction["hidden_after_removed"])
            student_state = models[variant].large_runner.run_layers(
                student_state,
                config.split.large_suffix_start,
                models[variant].large_runner.num_layers - 1,
            )
            student_logits = models[variant].large_runner.logits_from_hidden(student_state.hidden_states)
            totals[f"{variant}_hidden_mse"] += float(
                masked_hidden_mse(prediction["hidden_after_removed"], teacher_targets.teacher_hidden, batch["attention_mask"]).cpu()
            )
            totals[f"{variant}_cosine_loss"] += float(
                masked_hidden_cosine_loss(prediction["hidden_after_removed"], teacher_targets.teacher_hidden, batch["attention_mask"]).cpu()
            )
            totals[f"{variant}_nll"] += float(shifted_cross_entropy(student_logits, batch["labels"]).cpu())
            totals[f"{variant}_logit_kl_to_teacher"] += float(
                shifted_kl_divergence(student_logits, teacher_targets.teacher_logits, batch["labels"]).cpu()
            )
            delta_stats = _delta_norm_stats(prediction["delta_large"], batch["attention_mask"])
            totals[f"{variant}_delta_norm_mean"] += delta_stats["delta_norm_mean"]
            totals[f"{variant}_delta_norm_max"] += delta_stats["delta_norm_max"]
            if prediction["gate_value"] is not None:
                totals[f"{variant}_gate_value"] += float(prediction["gate_value"])
            if prediction["gate_weights"] is not None:
                gate_stats = _adaptive_gate_stats(
                    prediction["gate_weights"],
                    batch["attention_mask"],
                    adaptive_bridge_gate_settings(config).collapse_threshold,
                )
                for key, value in gate_stats.items():
                    totals[f"{variant}_{key}"] += value
        batches += 1

    for key, value in totals.items():
        metrics[key] = value / max(1, batches)
    for variant in eval_variants:
        metrics[f"{variant}_cosine"] = 1.0 - metrics.pop(f"{variant}_cosine_loss")
    return metrics


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    seed_values = args.seeds or [config.training.seed]
    path_specs = load_mixture_path_specs(config)

    output_dir = ensure_dir(args.output_dir)
    save_config_snapshot(output_dir / "config_snapshot.yaml", config)
    export_run_metadata(
        output_dir / "metadata.json",
        config,
        {
            "stage": "adaptive_bridge_train",
            "seeds": seed_values,
            "variants": args.variants,
            "path_specs": [spec.to_dict() for spec in path_specs],
            "adaptive_bridge_settings": adaptive_bridge_settings(config).__dict__,
            "gate_settings": adaptive_bridge_gate_settings(config).__dict__,
        },
    )

    missing_warm_starts = require_warm_start_checkpoints(config, seed_values)
    if missing_warm_starts:
        raise FileNotFoundError(
            "Required warm-start checkpoints are missing for this real run: "
            + ", ".join(missing_warm_starts)
        )

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    parameter_budget = _parameter_budget_summary(config, backbones, path_specs)
    train_dataloader, _ = build_dataloader(config, backbones.tokenizer, stage_name="adaptive_bridge", split_name="train")
    val_dataloader, _ = build_dataloader(config, backbones.tokenizer, stage_name="adaptive_bridge", split_name="validation")

    seed_results: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "parameter_budget": parameter_budget,
        "warm_starts": {},
        "path_specs": [spec.to_dict() for spec in path_specs],
    }

    for seed in seed_values:
        LOGGER.info("adaptive_bridge seed=%s", seed)
        seed_config = clone_config_with_seed(config, seed)
        models, warm_starts = _build_models(
            seed_config,
            backbones,
            path_specs,
            seed,
            parameter_budget,
            args.variants,
        )
        diagnostics["warm_starts"][str(seed)] = warm_starts

        checkpoint_paths: dict[str, str] = {}
        for variant in args.variants:
            model, variant_history, checkpoint_payload = _train_variant(
                variant,
                models[variant],
                seed_config,
                backbones,
                train_dataloader,
                seed,
            )
            models[variant] = model
            history_rows.extend(variant_history)
            checkpoint_path = output_dir / f"seed_{seed}" / f"{variant}_checkpoint.pt"
            ensure_dir(checkpoint_path.parent)
            save_checkpoint(checkpoint_path, checkpoint_payload)
            checkpoint_paths[variant] = str(checkpoint_path)
            torch.cuda.empty_cache()

        metrics = _evaluate_models(models, seed_config, backbones, val_dataloader)
        seed_payload = {
            "seed": seed,
            "metrics": metrics,
            "checkpoint_paths": checkpoint_paths,
            "warm_starts": warm_starts,
        }
        seed_results.append(seed_payload)
        save_json(output_dir / f"seed_{seed}" / "metrics.json", seed_payload)

    summary_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    eval_variants = ["skip_only"] + args.variants
    metric_names = [
        "hidden_mse",
        "cosine",
        "nll",
        "logit_kl_to_teacher",
        "delta_norm_mean",
        "delta_norm_max",
        "gate_value",
        "weight_bridge",
        "weight_path_b",
        "weight_path_a",
        "gate_entropy",
        "bridge_usage_var",
        "path_b_usage_var",
        "path_a_usage_var",
        "collapse_score",
    ]
    for variant in eval_variants:
        variant_summary: dict[str, float] = {}
        for metric_name in metric_names:
            key = f"{variant}_{metric_name}"
            values = [result["metrics"][key] for result in seed_results if key in result["metrics"]]
            if not values:
                continue
            variant_summary[f"{metric_name}_mean"] = float(sum(values) / len(values))
        summary[variant] = variant_summary
        summary_rows.append({"variant": variant, **variant_summary})

    results_payload = {
        "config_path": args.config,
        "seeds": seed_values,
        "variants": args.variants,
        "parameter_budget": parameter_budget,
        "path_specs": [spec.to_dict() for spec in path_specs],
        "seed_results": seed_results,
        "summary": summary,
    }
    save_json(output_dir / "results.json", results_payload)
    save_csv(output_dir / "history.csv", history_rows)
    save_csv(output_dir / "summary.csv", summary_rows)
    ensure_dir(Path(args.results_path).parent)
    ensure_dir(Path(args.summary_path).parent)
    ensure_dir(Path(args.diagnostics_path).parent)
    save_json(args.results_path, results_payload)
    save_csv(args.summary_path, summary_rows)
    save_json(args.diagnostics_path, diagnostics)


if __name__ == "__main__":
    main()
