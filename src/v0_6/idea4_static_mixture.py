"""Train and summarize the minimal Idea 4 static two-path mixture."""

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
from src.v0_6.idea4_common import gated_return_adapter_state_dict, load_mixture_path_specs
from src.v0_6.idea4_models import (
    TwoPathStaticMixtureHybrid,
    TwoPathStaticMixtureNoSmallModel,
    static_mixture_trainable_prefixes,
)


LOGGER = get_logger(__name__)
TRAINED_VARIANTS = [
    "bridge_only",
    "bridge_only_param_matched",
    "static_mixture_no_small",
    "static_mixture",
]
EVAL_VARIANTS = ["skip_only"] + TRAINED_VARIANTS


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments for the Idea 4 Stage B runner."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="artifacts/v0_6/idea4_static_mixture/stage_b")
    parser.add_argument("--results-path", default="artifacts/v0_6/idea4_static_mixture/results.json")
    parser.add_argument("--summary-path", default="artifacts/v0_6/idea4_static_mixture/summary.csv")
    parser.add_argument("--diagnostics-path", default="artifacts/v0_6/idea4_static_mixture/diagnostics.json")
    parser.add_argument("--report-path", default="notes/v0_6/idea4_static_mixture_report.md")
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
            "delta_norm_min": 0.0,
        }
    token_norms = delta_large.detach().float().norm(dim=-1)
    mask = attention_mask.to(token_norms.dtype)
    valid = token_norms[attention_mask.bool()]
    return {
        "delta_norm_mean": float(((token_norms * mask).sum() / mask.sum().clamp_min(1.0)).cpu()),
        "delta_norm_max": float(valid.max().cpu()) if valid.numel() else 0.0,
        "delta_norm_min": float(valid.min().cpu()) if valid.numel() else 0.0,
    }


def _parameter_budget_summary(
    config: Any,
    backbones: LoadedBackbones,
    path_specs: list[Any],
) -> dict[str, Any]:
    mixture = TwoPathStaticMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
    zero_requires_grad(mixture, except_prefixes=static_mixture_trainable_prefixes(config))
    mixture_trainable = count_parameters(mixture).trainable_params
    param_matched_rank = _matched_bridge_rank(mixture.large_runner.hidden_size, mixture_trainable)

    bridge_only = BridgeOnlyLargeModel(config, backbones.large_model)
    zero_requires_grad(bridge_only, except_prefixes=["bridge", "gate"])

    bridge_param = BridgeOnlyParamMatchedModel(config, backbones.large_model, rank=param_matched_rank)
    zero_requires_grad(bridge_param, except_prefixes=["bridge", "gate"])

    mixture_no_small = TwoPathStaticMixtureNoSmallModel(config, backbones.large_model, backbones.small_model, path_specs)
    zero_requires_grad(mixture_no_small, except_prefixes=static_mixture_trainable_prefixes(config))

    return {
        "target_static_mixture_trainable_params": mixture_trainable,
        "bridge_only_trainable_params": count_parameters(bridge_only).trainable_params,
        "bridge_only_param_matched_rank": param_matched_rank,
        "bridge_only_param_matched_trainable_params": count_parameters(bridge_param).trainable_params,
        "static_mixture_no_small_trainable_params": count_parameters(mixture_no_small).trainable_params,
    }


def _warm_start_mixture_model(
    model: Any,
    path_specs: list[Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"seed": seed, "paths": []}
    for path_spec in path_specs:
        payload = load_checkpoint(path_spec.checkpoint_path(seed, variant="hybrid"), device)
        suffix = path_spec.name.split("_")[-1]
        getattr(model, f"entry_projector_{suffix}").load_state_dict(payload["entry_projector"])
        return_state, absorbed_gate = gated_return_adapter_state_dict(payload)
        getattr(model, f"return_adapter_{suffix}").load_state_dict(return_state)
        metadata["paths"].append(
            {
                "name": path_spec.name,
                "label": path_spec.label,
                "checkpoint_path": str(path_spec.checkpoint_path(seed, variant="hybrid")),
                "absorbed_phase1_gate": absorbed_gate,
            }
        )
    with torch.no_grad():
        model.alpha.zero_()
    metadata["initial_mixture_weights"] = [0.5, 0.5]
    return metadata


def _build_variant_models(
    config: Any,
    backbones: LoadedBackbones,
    path_specs: list[Any],
    parameter_budget: dict[str, Any],
    seed: int,
    trained_variants: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    models: dict[str, Any] = {"skip_only": SkipOnlyLargeModel(config, backbones.large_model)}
    warm_start_metadata: dict[str, Any] = {}

    if "bridge_only" in trained_variants:
        models["bridge_only"] = BridgeOnlyLargeModel(config, backbones.large_model)
        zero_requires_grad(models["bridge_only"], except_prefixes=["bridge", "gate"])

    if "bridge_only_param_matched" in trained_variants:
        models["bridge_only_param_matched"] = BridgeOnlyParamMatchedModel(
            config,
            backbones.large_model,
            rank=parameter_budget["bridge_only_param_matched_rank"],
        )
        zero_requires_grad(models["bridge_only_param_matched"], except_prefixes=["bridge", "gate"])

    if "static_mixture_no_small" in trained_variants:
        model = TwoPathStaticMixtureNoSmallModel(config, backbones.large_model, backbones.small_model, path_specs)
        warm_start_metadata["static_mixture_no_small"] = _warm_start_mixture_model(model, path_specs, seed, backbones.device)
        zero_requires_grad(model, except_prefixes=static_mixture_trainable_prefixes(config))
        models["static_mixture_no_small"] = model

    if "static_mixture" in trained_variants:
        model = TwoPathStaticMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
        warm_start_metadata["static_mixture"] = _warm_start_mixture_model(model, path_specs, seed, backbones.device)
        zero_requires_grad(model, except_prefixes=static_mixture_trainable_prefixes(config))
        models["static_mixture"] = model

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
            "mixture_weights": None,
            "path_outputs": None,
            "gate_value": 0.0,
        }
    if variant in {"bridge_only", "bridge_only_param_matched"}:
        delta_large = model.bridge(hidden_after_prefix)
        gated_delta_large, gate_value = _resolve_gate_value(model.gate, delta_large)
        return {
            "hidden_after_removed": hidden_after_prefix + gated_delta_large,
            "delta_large": delta_large,
            "mixture_weights": None,
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
            "mixture_weights": [float(weights[0].detach().cpu()), float(weights[1].detach().cpu())],
            "path_outputs": path_outputs,
            "gate_value": None,
        }
    raise ValueError(f"Unsupported Idea 4 variant: {variant}")


def _path_delta_mean(path_outputs: dict[str, Any] | None, path_name: str, attention_mask: torch.Tensor) -> float:
    if path_outputs is None:
        return 0.0
    delta = path_outputs[path_name].delta_large
    return _delta_norm_stats(delta, attention_mask)["delta_norm_mean"]


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

    for step in range(1, config.training.stage_b.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
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
            (loss_terms.total_loss / config.training.grad_accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()

        delta_stats = _delta_norm_stats(prediction["delta_large"], batch["attention_mask"])
        row = {
            "seed": float(seed),
            "variant": variant,
            "step": float(step),
            "loss": float(loss_terms.total_loss.detach().cpu()),
            "mse_loss": float(loss_terms.mse_loss.detach().cpu()),
            "cosine_loss": float(loss_terms.cosine_loss.detach().cpu()),
            "kl_loss": float(loss_terms.kl_loss.detach().cpu()),
            "ce_loss": float(loss_terms.ce_loss.detach().cpu()),
            "delta_reg": float(loss_terms.delta_reg.detach().cpu()),
            "delta_norm_mean": delta_stats["delta_norm_mean"],
            "delta_norm_max": delta_stats["delta_norm_max"],
        }
        if prediction["mixture_weights"] is not None:
            row["weight_path_b"] = prediction["mixture_weights"][0]
            row["weight_path_a"] = prediction["mixture_weights"][1]
            row["path_b_delta_norm_mean"] = _path_delta_mean(prediction["path_outputs"], "path_b", batch["attention_mask"])
            row["path_a_delta_norm_mean"] = _path_delta_mean(prediction["path_outputs"], "path_a", batch["attention_mask"])
        else:
            row["gate_value"] = float(prediction["gate_value"] or 0.0)
        history.append(row)

        if step % config.training.log_every == 0 or step == config.training.stage_b.max_steps:
            LOGGER.info(
                "idea4_static_mixture seed=%s variant=%s step=%s loss=%.6f mse=%.6f cosine=%.6f kl=%.6f ce=%.6f delta=%.6f",
                seed,
                variant,
                step,
                row["loss"],
                row["mse_loss"],
                row["cosine_loss"],
                row["kl_loss"],
                row["ce_loss"],
                row["delta_reg"],
            )

    checkpoint_payload = {
        "seed": seed,
        "variant": variant,
        "step": config.training.stage_b.max_steps,
    }
    if variant in {"bridge_only", "bridge_only_param_matched"}:
        checkpoint_payload["bridge"] = model.bridge.state_dict()
        checkpoint_payload["gate"] = model.gate.state_dict()
    else:
        checkpoint_payload["entry_projector_b"] = model.entry_projector_b.state_dict()
        checkpoint_payload["entry_projector_a"] = model.entry_projector_a.state_dict()
        checkpoint_payload["return_adapter_b"] = model.return_adapter_b.state_dict()
        checkpoint_payload["return_adapter_a"] = model.return_adapter_a.state_dict()
        checkpoint_payload["alpha"] = model.alpha.detach().cpu()
    return model, history, checkpoint_payload


@torch.no_grad()
def _evaluate_models(
    models: dict[str, Any],
    config: Any,
    backbones: LoadedBackbones,
    val_dataloader: Any,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for model in models.values():
        model.eval()

    totals: dict[str, float] = {}
    batches = 0
    for variant in EVAL_VARIANTS:
        totals[f"{variant}_hidden_mse"] = 0.0
        totals[f"{variant}_cosine_loss"] = 0.0
        totals[f"{variant}_delta_norm_mean"] = 0.0
        totals[f"{variant}_delta_norm_max"] = 0.0
        if variant in {"static_mixture", "static_mixture_no_small"}:
            totals[f"{variant}_weight_path_b"] = 0.0
            totals[f"{variant}_weight_path_a"] = 0.0
            totals[f"{variant}_path_b_delta_norm_mean"] = 0.0
            totals[f"{variant}_path_a_delta_norm_mean"] = 0.0
        if variant in {"bridge_only", "bridge_only_param_matched"}:
            totals[f"{variant}_gate_value"] = 0.0

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

            if prediction["mixture_weights"] is not None:
                totals[f"{variant}_weight_path_b"] += prediction["mixture_weights"][0]
                totals[f"{variant}_weight_path_a"] += prediction["mixture_weights"][1]
                totals[f"{variant}_path_b_delta_norm_mean"] += _path_delta_mean(
                    prediction["path_outputs"],
                    "path_b",
                    batch["attention_mask"],
                )
                totals[f"{variant}_path_a_delta_norm_mean"] += _path_delta_mean(
                    prediction["path_outputs"],
                    "path_a",
                    batch["attention_mask"],
                )
            gate_key = f"{variant}_gate_value"
            if prediction["gate_value"] is not None and gate_key in totals:
                totals[f"{variant}_gate_value"] += float(prediction["gate_value"])
        batches += 1

    for key, value in totals.items():
        metrics[key] = value / max(1, batches)
    for variant in EVAL_VARIANTS:
        metrics[f"{variant}_cosine"] = 1.0 - metrics.pop(f"{variant}_cosine_loss")
    metrics["static_mixture_beats_skip_only"] = (
        metrics["static_mixture_hidden_mse"] < metrics["skip_only_hidden_mse"]
        and metrics["static_mixture_cosine"] > metrics["skip_only_cosine"]
    )
    metrics["static_mixture_beats_static_mixture_no_small"] = (
        metrics["static_mixture_hidden_mse"] < metrics["static_mixture_no_small_hidden_mse"]
        and metrics["static_mixture_cosine"] > metrics["static_mixture_no_small_cosine"]
    )
    metrics["static_mixture_beats_bridge_only"] = (
        metrics["static_mixture_hidden_mse"] < metrics["bridge_only_hidden_mse"]
        and metrics["static_mixture_cosine"] > metrics["bridge_only_cosine"]
    )
    metrics["static_mixture_beats_bridge_only_param_matched"] = (
        metrics["static_mixture_hidden_mse"] < metrics["bridge_only_param_matched_hidden_mse"]
        and metrics["static_mixture_cosine"] > metrics["bridge_only_param_matched_cosine"]
    )
    return metrics


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _aggregate_results(seed_results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {"per_variant": {}, "pairwise_wins": {}}
    rows: list[dict[str, Any]] = []

    for variant in EVAL_VARIANTS:
        metric_names = [
            "hidden_mse",
            "cosine",
            "delta_norm_mean",
            "delta_norm_max",
        ]
        if variant in {"static_mixture", "static_mixture_no_small"}:
            metric_names.extend(["weight_path_b", "weight_path_a", "path_b_delta_norm_mean", "path_a_delta_norm_mean"])
        if variant in {"bridge_only", "bridge_only_param_matched"}:
            metric_names.append("gate_value")

        variant_summary: dict[str, float] = {}
        for metric_name in metric_names:
            values = [result["metrics"][f"{variant}_{metric_name}"] for result in seed_results]
            variant_summary[f"{metric_name}_mean"] = _mean(values)
            variant_summary[f"{metric_name}_std"] = _std(values)
        summary["per_variant"][variant] = variant_summary
        rows.append({"row_type": "variant", "label": variant, **variant_summary})

    comparisons = {
        "skip_only": ("static_mixture_hidden_mse", "skip_only_hidden_mse", "static_mixture_cosine", "skip_only_cosine"),
        "static_mixture_no_small": (
            "static_mixture_hidden_mse",
            "static_mixture_no_small_hidden_mse",
            "static_mixture_cosine",
            "static_mixture_no_small_cosine",
        ),
        "bridge_only": ("static_mixture_hidden_mse", "bridge_only_hidden_mse", "static_mixture_cosine", "bridge_only_cosine"),
        "bridge_only_param_matched": (
            "static_mixture_hidden_mse",
            "bridge_only_param_matched_hidden_mse",
            "static_mixture_cosine",
            "bridge_only_param_matched_cosine",
        ),
    }
    for label, (mix_mse_key, other_mse_key, mix_cosine_key, other_cosine_key) in comparisons.items():
        wins = 0
        for result in seed_results:
            metrics = result["metrics"]
            if metrics[mix_mse_key] < metrics[other_mse_key] and metrics[mix_cosine_key] > metrics[other_cosine_key]:
                wins += 1
        summary["pairwise_wins"][label] = {"static_mixture_wins_on_both_metrics": wins, "seeds": len(seed_results)}
        rows.append(
            {
                "row_type": "pairwise",
                "label": label,
                "static_mixture_wins_on_both_metrics": wins,
                "seeds": len(seed_results),
            }
        )

    return summary, rows


def _write_static_mixture_report(report_path: str | Path, results_payload: dict[str, Any]) -> None:
    summary = results_payload["summary"]
    mixture = summary["per_variant"]["static_mixture"]
    no_small = summary["per_variant"]["static_mixture_no_small"]
    bridge = summary["per_variant"]["bridge_only"]
    bridge_param = summary["per_variant"]["bridge_only_param_matched"]
    pairwise_wins = summary["pairwise_wins"]

    lines = [
        "# Idea 4 Static Mixture Report",
        "",
        "## setup",
        "",
        f"- Config: {results_payload['config_path']}",
        f"- seq_len: {results_payload['seq_len']}",
        f"- max_train_steps: {results_payload['max_train_steps']}",
        f"- Seeds: {', '.join(str(seed) for seed in results_payload['seeds'])}",
        "- Static mixture policy: exactly two shortlisted delegated paths with one global 2-logit softmax.",
        "- Warm start policy: each path reuses the matching confirmed Phase 1 checkpoint, with the single-path scalar gate absorbed into the return adapter so the static mixture starts from actual confirmed path behavior.",
        "- Output-level metrics remain primary; this report is hidden-space and routing diagnostics only.",
        f"- Static mixture trainable params: {results_payload['parameter_budget']['target_static_mixture_trainable_params']}",
        f"- bridge_only trainable params: {results_payload['parameter_budget']['bridge_only_trainable_params']}",
        f"- parameter-matched bridge rank: {results_payload['parameter_budget']['bridge_only_param_matched_rank']}",
        f"- parameter-matched bridge trainable params: {results_payload['parameter_budget']['bridge_only_param_matched_trainable_params']}",
        "",
        "## aggregate summary",
        "",
        f"- static_mixture: hidden_mse_mean={mixture['hidden_mse_mean']:.6f}, cosine_mean={mixture['cosine_mean']:.6f}, delta_norm_mean={mixture['delta_norm_mean_mean']:.6f}, weight_path_b_mean={mixture['weight_path_b_mean']:.6f}, weight_path_a_mean={mixture['weight_path_a_mean']:.6f}",
        f"- static_mixture_no_small: hidden_mse_mean={no_small['hidden_mse_mean']:.6f}, cosine_mean={no_small['cosine_mean']:.6f}, delta_norm_mean={no_small['delta_norm_mean_mean']:.6f}, weight_path_b_mean={no_small['weight_path_b_mean']:.6f}, weight_path_a_mean={no_small['weight_path_a_mean']:.6f}",
        f"- bridge_only: hidden_mse_mean={bridge['hidden_mse_mean']:.6f}, cosine_mean={bridge['cosine_mean']:.6f}, delta_norm_mean={bridge['delta_norm_mean_mean']:.6f}, gate_value_mean={bridge['gate_value_mean']:.6f}",
        f"- bridge_only_param_matched: hidden_mse_mean={bridge_param['hidden_mse_mean']:.6f}, cosine_mean={bridge_param['cosine_mean']:.6f}, delta_norm_mean={bridge_param['delta_norm_mean_mean']:.6f}, gate_value_mean={bridge_param['gate_value_mean']:.6f}",
        "",
        "## hidden-space interpretation",
        "",
        f"- static_mixture vs no_small hidden wins: {pairwise_wins['static_mixture_no_small']['static_mixture_wins_on_both_metrics']}/{results_payload['seed_count']}",
        f"- static_mixture vs bridge_only hidden wins: {pairwise_wins['bridge_only']['static_mixture_wins_on_both_metrics']}/{results_payload['seed_count']}",
        f"- static_mixture vs parameter-matched bridge hidden wins: {pairwise_wins['bridge_only_param_matched']['static_mixture_wins_on_both_metrics']}/{results_payload['seed_count']}",
        "- Mixture weights should be read as routing diagnostics, not as the final decision rule. The output probe is the primary evidence for whether Idea 4 is worth continuing.",
        "",
    ]
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    path_specs = load_mixture_path_specs(config)
    output_dir = ensure_dir(args.output_dir)
    save_config_snapshot(output_dir / "config_snapshot.yaml", config)
    export_run_metadata(
        output_dir / "metadata.json",
        config,
        {
            "stage": "idea4_static_mixture",
            "seeds": args.seeds,
            "variants": args.variants,
            "path_specs": [spec.to_dict() for spec in path_specs],
        },
    )

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    parameter_budget = _parameter_budget_summary(config, backbones, path_specs)
    train_dataloader, _ = build_dataloader(config, backbones.tokenizer, stage_name="stage_b", split_name="train")
    val_dataloader, _ = build_dataloader(config, backbones.tokenizer, stage_name="stage_b", split_name="validation")

    seed_results: list[dict[str, Any]] = []
    all_history_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "path_specs": [spec.to_dict() for spec in path_specs],
        "parameter_budget": parameter_budget,
        "seed_warm_starts": {},
    }

    for seed in args.seeds:
        LOGGER.info("idea4_static_mixture seed=%s", seed)
        seed_everything(seed)
        seed_config = _clone_config_with_seed(config, seed)
        models, warm_start_metadata = _build_variant_models(
            seed_config,
            backbones,
            path_specs,
            parameter_budget,
            seed,
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
        "seq_len": config.training.seq_len,
        "max_train_steps": config.training.stage_b.max_steps,
        "seed_count": len(args.seeds),
        "seeds": args.seeds,
        "variants": args.variants,
        "parameter_budget": parameter_budget,
        "path_specs": [spec.to_dict() for spec in path_specs],
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
    _write_static_mixture_report(args.report_path, results_payload)


if __name__ == "__main__":
    main()
