"""Focused Stage B ablation runner for bridge-capacity disambiguation."""

from __future__ import annotations

import argparse
import copy
import itertools
import statistics
from pathlib import Path
from typing import Any

import torch

from src.eval.metrics import masked_hidden_cosine_loss, masked_hidden_mse
from src.models.baselines import BridgeOnlyLargeModel, BridgeOnlyParamMatchedModel, SkipOnlyLargeModel
from src.models.hooks import count_parameters
from src.models.hybrid_gemma import HybridDelegationModel, HybridNoSmallModel, _resolve_gate_value
from src.train.trainer_utils import (
    build_dataloader,
    build_optimizer,
    load_checkpoint,
    move_batch_to_device,
    save_checkpoint,
    zero_requires_grad,
)
from src.utils.io import (
    ensure_dir,
    export_run_metadata,
    load_config,
    save_config_snapshot,
    save_csv,
    save_json,
)
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.reporting import write_real_hardware_report
from src.utils.seed import seed_everything
from src.models.backbone_loader import LoadedBackbones, load_backbones


LOGGER = get_logger(__name__)
TRAINED_VARIANTS = ["bridge_only", "bridge_only_param_matched", "hybrid_no_small", "hybrid"]
ALL_VARIANTS = ["skip_only"] + TRAINED_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gemma2_conservative_pilot_256.yaml")
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/stage_b_ablation")
    parser.add_argument("--results-path", default="artifacts/stage_b_ablation_results.json")
    parser.add_argument("--summary-path", default="artifacts/stage_b_ablation_summary.csv")
    parser.add_argument("--diagnostics-path", default="artifacts/stage_b_diagnostics.json")
    parser.add_argument("--report-path", default="notes/stage_b_ablation_report.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    return parser.parse_args()


def _clone_config_with_seed(config: Any, seed: int) -> Any:
    cloned = copy.deepcopy(config)
    cloned.training.seed = seed
    cloned.raw["training"]["seed"] = seed
    return cloned


def _delta_norm_stats(delta_large: torch.Tensor | None, attention_mask: torch.Tensor) -> dict[str, float | None]:
    if delta_large is None:
        return {
            "delta_norm_mean": None,
            "delta_norm_max": None,
            "delta_norm_min": None,
        }
    token_norms = delta_large.detach().float().norm(dim=-1)
    mask = attention_mask.to(token_norms.dtype)
    valid = token_norms[attention_mask.bool()]
    return {
        "delta_norm_mean": float(((token_norms * mask).sum() / mask.sum().clamp_min(1.0)).cpu()),
        "delta_norm_max": float(valid.max().cpu()) if valid.numel() else 0.0,
        "delta_norm_min": float(valid.min().cpu()) if valid.numel() else 0.0,
    }


def _teacher_hidden(
    large_runner: Any,
    batch: dict[str, torch.Tensor],
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return hidden_after_prefix, teacher_hidden


def _variant_prediction(
    variant: str,
    model: Any,
    hidden_after_prefix: torch.Tensor,
    attention_mask: torch.Tensor,
    gate_override: float | None = None,
) -> dict[str, Any]:
    if variant == "skip_only":
        return {
            "hidden_after_removed": hidden_after_prefix,
            "delta_large": None,
            "gate_value": 0.0,
        }
    if variant in {"bridge_only", "bridge_only_param_matched"}:
        delta_large = model.bridge(hidden_after_prefix)
        gated_delta_large, gate_value = _resolve_gate_value(model.gate, delta_large, gate_override=gate_override)
        return {
            "hidden_after_removed": hidden_after_prefix + gated_delta_large,
            "delta_large": delta_large,
            "gate_value": gate_value,
        }
    if variant in {"hybrid_no_small", "hybrid"}:
        with torch.no_grad():
            projected_small = model.entry_projector(hidden_after_prefix)
            delegated_small_hidden = model.run_delegated_small_block(projected_small, attention_mask)
        delta_large = model.return_adapter(delegated_small_hidden.detach())
        gated_delta_large, gate_value = _resolve_gate_value(model.gate, delta_large, gate_override=gate_override)
        return {
            "hidden_after_removed": hidden_after_prefix + gated_delta_large,
            "delta_large": delta_large,
            "gate_value": gate_value,
        }
    raise ValueError(f"Unsupported variant: {variant}")


def _build_variant_models(
    config: Any,
    backbones: LoadedBackbones,
    stage_a_payload: dict[str, Any],
    param_matched_rank: int,
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "skip_only": SkipOnlyLargeModel(config, backbones.large_model),
        "bridge_only": BridgeOnlyLargeModel(config, backbones.large_model),
        "bridge_only_param_matched": BridgeOnlyParamMatchedModel(config, backbones.large_model, rank=param_matched_rank),
        "hybrid_no_small": HybridNoSmallModel(config, backbones.large_model, backbones.small_model),
        "hybrid": HybridDelegationModel(config, backbones.large_model, backbones.small_model),
    }
    zero_requires_grad(models["bridge_only"], except_prefixes=["bridge", "gate"])
    zero_requires_grad(models["bridge_only_param_matched"], except_prefixes=["bridge", "gate"])
    for hybrid_variant in ("hybrid_no_small", "hybrid"):
        models[hybrid_variant].entry_projector.load_state_dict(stage_a_payload["entry_projector"])
        zero_requires_grad(models[hybrid_variant], except_prefixes=["return_adapter", "gate"])
    return models


def _matched_bridge_rank(large_hidden_size: int, target_trainable_params: int) -> int:
    base = max(1, int((target_trainable_params - 1) / max(1, 2 * large_hidden_size)))
    candidates = sorted({max(1, base), max(1, base + 1)})
    return min(candidates, key=lambda rank: abs((2 * large_hidden_size * rank + 1) - target_trainable_params))


def _parameter_budget_summary(
    config: Any,
    backbones: LoadedBackbones,
    stage_a_payload: dict[str, Any],
) -> dict[str, Any]:
    hybrid = HybridDelegationModel(config, backbones.large_model, backbones.small_model)
    hybrid.entry_projector.load_state_dict(stage_a_payload["entry_projector"])
    zero_requires_grad(hybrid, except_prefixes=["return_adapter", "gate"])
    hybrid_trainable = count_parameters(hybrid).trainable_params
    param_matched_rank = _matched_bridge_rank(hybrid.large_runner.hidden_size, hybrid_trainable)

    bridge_only = BridgeOnlyLargeModel(config, backbones.large_model)
    zero_requires_grad(bridge_only, except_prefixes=["bridge", "gate"])
    param_matched = BridgeOnlyParamMatchedModel(config, backbones.large_model, rank=param_matched_rank)
    zero_requires_grad(param_matched, except_prefixes=["bridge", "gate"])
    hybrid_no_small = HybridNoSmallModel(config, backbones.large_model, backbones.small_model)
    hybrid_no_small.entry_projector.load_state_dict(stage_a_payload["entry_projector"])
    zero_requires_grad(hybrid_no_small, except_prefixes=["return_adapter", "gate"])

    return {
        "target_hybrid_trainable_params": hybrid_trainable,
        "bridge_only_trainable_params": count_parameters(bridge_only).trainable_params,
        "bridge_only_param_matched_trainable_params": count_parameters(param_matched).trainable_params,
        "bridge_only_param_matched_rank": param_matched_rank,
        "hybrid_no_small_trainable_params": count_parameters(hybrid_no_small).trainable_params,
    }


def _train_variant(
    variant: str,
    model: Any,
    config: Any,
    backbones: LoadedBackbones,
    train_dataloader: Any,
    seed: int,
) -> tuple[Any, list[dict[str, float]], dict[str, Any]]:
    seed_everything(seed)
    optimizer = build_optimizer(model, config)
    batch_iterator = itertools.cycle(train_dataloader)
    history: list[dict[str, float]] = []
    model.train()

    for step in range(1, config.training.stage_b.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.training.grad_accum_steps):
            batch = move_batch_to_device(next(batch_iterator), backbones.device)
            hidden_after_prefix, teacher_hidden = _teacher_hidden(model.large_runner, batch, config)
            prediction = _variant_prediction(
                variant,
                model,
                hidden_after_prefix=hidden_after_prefix,
                attention_mask=batch["attention_mask"],
            )
            predicted_hidden = prediction["hidden_after_removed"]
            mse_loss = masked_hidden_mse(predicted_hidden, teacher_hidden, batch["attention_mask"])
            cosine_loss = masked_hidden_cosine_loss(predicted_hidden, teacher_hidden, batch["attention_mask"])
            total_loss = (mse_loss + cosine_loss) / config.training.grad_accum_steps
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()
        delta_stats = _delta_norm_stats(prediction["delta_large"], batch["attention_mask"])
        row = {
            "seed": float(seed),
            "variant": variant,
            "step": float(step),
            "loss": float((mse_loss + cosine_loss).detach().cpu()),
            "mse_loss": float(mse_loss.detach().cpu()),
            "cosine_loss": float(cosine_loss.detach().cpu()),
            "gate_value": float(model.gate.value().detach().cpu()),
            "delta_norm_mean": float(delta_stats["delta_norm_mean"] or 0.0),
            "delta_norm_max": float(delta_stats["delta_norm_max"] or 0.0),
        }
        history.append(row)
        if step % config.training.log_every == 0 or step == config.training.stage_b.max_steps:
            LOGGER.info(
                "stage_b_ablation seed=%s variant=%s step=%s loss=%.6f mse=%.6f cosine=%.6f gate=%.6f delta_mean=%.6f",
                seed,
                variant,
                step,
                row["loss"],
                row["mse_loss"],
                row["cosine_loss"],
                row["gate_value"],
                row["delta_norm_mean"],
            )

    checkpoint_payload = {
        "seed": seed,
        "variant": variant,
        "step": config.training.stage_b.max_steps,
        "gate": model.gate.state_dict(),
    }
    if variant in {"bridge_only", "bridge_only_param_matched"}:
        checkpoint_payload["bridge"] = model.bridge.state_dict()
    else:
        checkpoint_payload["entry_projector"] = model.entry_projector.state_dict()
        checkpoint_payload["return_adapter"] = model.return_adapter.state_dict()
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
    for variant in ALL_VARIANTS:
        totals[f"{variant}_hidden_mse"] = 0.0
        totals[f"{variant}_cosine_loss"] = 0.0
        totals[f"{variant}_gate_value"] = 0.0
        totals[f"{variant}_delta_norm_mean"] = 0.0
        totals[f"{variant}_delta_norm_max"] = 0.0
    totals["hybrid_gate_zero_hidden_mse"] = 0.0
    totals["hybrid_gate_zero_cosine_loss"] = 0.0

    for batch in val_dataloader:
        batch = move_batch_to_device(batch, backbones.device)
        hidden_after_prefix, teacher_hidden = _teacher_hidden(models["hybrid"].large_runner, batch, config)

        for variant in ALL_VARIANTS:
            prediction = _variant_prediction(
                variant,
                models[variant],
                hidden_after_prefix=hidden_after_prefix,
                attention_mask=batch["attention_mask"],
            )
            totals[f"{variant}_hidden_mse"] += float(
                masked_hidden_mse(prediction["hidden_after_removed"], teacher_hidden, batch["attention_mask"]).cpu()
            )
            totals[f"{variant}_cosine_loss"] += float(
                masked_hidden_cosine_loss(prediction["hidden_after_removed"], teacher_hidden, batch["attention_mask"]).cpu()
            )
            totals[f"{variant}_gate_value"] += float(prediction["gate_value"])
            delta_stats = _delta_norm_stats(prediction["delta_large"], batch["attention_mask"])
            totals[f"{variant}_delta_norm_mean"] += float(delta_stats["delta_norm_mean"] or 0.0)
            totals[f"{variant}_delta_norm_max"] += float(delta_stats["delta_norm_max"] or 0.0)

        hybrid_gate_zero = _variant_prediction(
            "hybrid",
            models["hybrid"],
            hidden_after_prefix=hidden_after_prefix,
            attention_mask=batch["attention_mask"],
            gate_override=0.0,
        )
        totals["hybrid_gate_zero_hidden_mse"] += float(
            masked_hidden_mse(hybrid_gate_zero["hidden_after_removed"], teacher_hidden, batch["attention_mask"]).cpu()
        )
        totals["hybrid_gate_zero_cosine_loss"] += float(
            masked_hidden_cosine_loss(hybrid_gate_zero["hidden_after_removed"], teacher_hidden, batch["attention_mask"]).cpu()
        )
        batches += 1

    for key, value in totals.items():
        metrics[key] = value / max(1, batches)
    for variant in ALL_VARIANTS:
        metrics[f"{variant}_cosine"] = 1.0 - metrics.pop(f"{variant}_cosine_loss")
    metrics["hybrid_gate_zero_cosine"] = 1.0 - metrics.pop("hybrid_gate_zero_cosine_loss")
    metrics["hybrid_vs_gate_zero_hidden_mse_gain"] = (
        metrics["hybrid_gate_zero_hidden_mse"] - metrics["hybrid_hidden_mse"]
    )
    metrics["hybrid_vs_gate_zero_cosine_gain"] = metrics["hybrid_cosine"] - metrics["hybrid_gate_zero_cosine"]
    metrics["hybrid_beats_skip_only"] = (
        metrics["hybrid_hidden_mse"] < metrics["skip_only_hidden_mse"]
        and metrics["hybrid_cosine"] > metrics["skip_only_cosine"]
    )
    metrics["hybrid_beats_hybrid_no_small"] = (
        metrics["hybrid_hidden_mse"] < metrics["hybrid_no_small_hidden_mse"]
        and metrics["hybrid_cosine"] > metrics["hybrid_no_small_cosine"]
    )
    metrics["hybrid_beats_bridge_only"] = (
        metrics["hybrid_hidden_mse"] < metrics["bridge_only_hidden_mse"]
        and metrics["hybrid_cosine"] > metrics["bridge_only_cosine"]
    )
    metrics["hybrid_beats_bridge_only_param_matched"] = (
        metrics["hybrid_hidden_mse"] < metrics["bridge_only_param_matched_hidden_mse"]
        and metrics["hybrid_cosine"] > metrics["bridge_only_param_matched_cosine"]
    )
    metrics["delegated_path_used"] = (
        metrics["hybrid_gate_value"] > 0.0
        and metrics["hybrid_delta_norm_mean"] > 0.0
        and metrics["hybrid_vs_gate_zero_hidden_mse_gain"] > 0.0
        and metrics["hybrid_vs_gate_zero_cosine_gain"] > 0.0
    )
    return metrics


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _aggregate_results(seed_results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {"per_variant": {}, "pairwise_wins": {}}
    summary_rows: list[dict[str, Any]] = []
    for variant in ALL_VARIANTS:
        mse_values = [result["metrics"][f"{variant}_hidden_mse"] for result in seed_results]
        cosine_values = [result["metrics"][f"{variant}_cosine"] for result in seed_results]
        gate_values = [result["metrics"].get(f"{variant}_gate_value", 0.0) for result in seed_results]
        delta_values = [result["metrics"].get(f"{variant}_delta_norm_mean", 0.0) for result in seed_results]
        variant_summary = {
            "hidden_mse_mean": _mean(mse_values),
            "hidden_mse_std": _std(mse_values),
            "cosine_mean": _mean(cosine_values),
            "cosine_std": _std(cosine_values),
            "gate_value_mean": _mean(gate_values),
            "gate_value_std": _std(gate_values),
            "delta_norm_mean": _mean(delta_values),
            "delta_norm_std": _std(delta_values),
        }
        summary["per_variant"][variant] = variant_summary
        summary_rows.append({"variant": variant, **variant_summary})

    hybrid_gate_zero_mse = [result["metrics"]["hybrid_gate_zero_hidden_mse"] for result in seed_results]
    hybrid_gate_zero_cosine = [result["metrics"]["hybrid_gate_zero_cosine"] for result in seed_results]
    summary["per_variant"]["hybrid_gate_zero"] = {
        "hidden_mse_mean": _mean(hybrid_gate_zero_mse),
        "hidden_mse_std": _std(hybrid_gate_zero_mse),
        "cosine_mean": _mean(hybrid_gate_zero_cosine),
        "cosine_std": _std(hybrid_gate_zero_cosine),
    }
    summary_rows.append(
        {
            "variant": "hybrid_gate_zero",
            "hidden_mse_mean": summary["per_variant"]["hybrid_gate_zero"]["hidden_mse_mean"],
            "hidden_mse_std": summary["per_variant"]["hybrid_gate_zero"]["hidden_mse_std"],
            "cosine_mean": summary["per_variant"]["hybrid_gate_zero"]["cosine_mean"],
            "cosine_std": summary["per_variant"]["hybrid_gate_zero"]["cosine_std"],
            "gate_value_mean": 0.0,
            "gate_value_std": 0.0,
            "delta_norm_mean": 0.0,
            "delta_norm_std": 0.0,
        }
    )

    comparisons = {
        "skip_only": ("hybrid_hidden_mse", "skip_only_hidden_mse", "hybrid_cosine", "skip_only_cosine"),
        "hybrid_no_small": (
            "hybrid_hidden_mse",
            "hybrid_no_small_hidden_mse",
            "hybrid_cosine",
            "hybrid_no_small_cosine",
        ),
        "bridge_only": ("hybrid_hidden_mse", "bridge_only_hidden_mse", "hybrid_cosine", "bridge_only_cosine"),
        "bridge_only_param_matched": (
            "hybrid_hidden_mse",
            "bridge_only_param_matched_hidden_mse",
            "hybrid_cosine",
            "bridge_only_param_matched_cosine",
        ),
    }
    for label, (hybrid_mse_key, other_mse_key, hybrid_cosine_key, other_cosine_key) in comparisons.items():
        wins = 0
        for result in seed_results:
            metrics = result["metrics"]
            if metrics[hybrid_mse_key] < metrics[other_mse_key] and metrics[hybrid_cosine_key] > metrics[other_cosine_key]:
                wins += 1
        summary["pairwise_wins"][label] = {"hybrid_wins_on_both_metrics": wins, "seeds": len(seed_results)}

    return summary, summary_rows


def _stronger_bridge_label(summary: dict[str, Any]) -> str:
    bridge = summary["per_variant"]["bridge_only"]
    param_matched = summary["per_variant"]["bridge_only_param_matched"]
    if bridge["hidden_mse_mean"] < param_matched["hidden_mse_mean"]:
        return "bridge_only"
    if bridge["hidden_mse_mean"] > param_matched["hidden_mse_mean"]:
        return "bridge_only_param_matched"
    return "bridge_only" if bridge["cosine_mean"] >= param_matched["cosine_mean"] else "bridge_only_param_matched"


def _is_clear_reproducible_win(summary: dict[str, Any], label: str) -> bool:
    wins = summary["pairwise_wins"][label]["hybrid_wins_on_both_metrics"]
    hybrid = summary["per_variant"]["hybrid"]
    other = summary["per_variant"][label]
    return (
        wins >= 2
        and hybrid["hidden_mse_mean"] < other["hidden_mse_mean"]
        and hybrid["cosine_mean"] > other["cosine_mean"]
    )


def _write_ablation_report(
    report_path: str | Path,
    results_payload: dict[str, Any],
) -> None:
    summary = results_payload["summary"]
    stronger_bridge = _stronger_bridge_label(summary)
    hybrid_vs_skip = summary["pairwise_wins"]["skip_only"]["hybrid_wins_on_both_metrics"]
    hybrid_vs_no_small = summary["pairwise_wins"]["hybrid_no_small"]["hybrid_wins_on_both_metrics"]
    hybrid_vs_bridge = summary["pairwise_wins"]["bridge_only"]["hybrid_wins_on_both_metrics"]
    hybrid_vs_param = summary["pairwise_wins"]["bridge_only_param_matched"]["hybrid_wins_on_both_metrics"]
    clear_vs_no_small = _is_clear_reproducible_win(summary, "hybrid_no_small")
    clear_vs_bridge = _is_clear_reproducible_win(summary, "bridge_only")
    clear_vs_param = _is_clear_reproducible_win(summary, "bridge_only_param_matched")
    reproducible_controls = [
        label
        for label in ["skip_only", "hybrid_no_small", "bridge_only", "bridge_only_param_matched"]
        if _is_clear_reproducible_win(summary, label)
    ]
    matches_stronger_bridge = (
        summary["per_variant"]["hybrid"]["hidden_mse_mean"] <= summary["per_variant"][stronger_bridge]["hidden_mse_mean"]
        and summary["per_variant"]["hybrid"]["cosine_mean"] >= summary["per_variant"][stronger_bridge]["cosine_mean"]
    )
    proceed = (clear_vs_bridge and clear_vs_param) or (clear_vs_no_small and matches_stronger_bridge)

    lines = [
        "# Stage B Ablation Report",
        "",
        "## setup",
        "",
        f"- Config: {results_payload['config_path']}",
        f"- seq_len: {results_payload['seq_len']}",
        f"- max_train_steps: {results_payload['max_train_steps']}",
        f"- Seeds: {', '.join(str(seed) for seed in results_payload['seeds'])}",
        "- Stage A checkpoint policy: reused one fixed Stage A checkpoint across all seeds so the Stage B comparison isolates the delegated-vs-bridge question instead of reintroducing Stage A variation.",
        f"- Stage A checkpoint path: {results_payload['stage_a_checkpoint']}",
        f"- Hybrid Stage B trainable params: {results_payload['parameter_budget']['target_hybrid_trainable_params']}",
        f"- Original bridge-only Stage B trainable params: {results_payload['parameter_budget']['bridge_only_trainable_params']}",
        f"- Parameter-matched bridge rank: {results_payload['parameter_budget']['bridge_only_param_matched_rank']}",
        f"- Parameter-matched bridge Stage B trainable params: {results_payload['parameter_budget']['bridge_only_param_matched_trainable_params']}",
        "",
        "## aggregate summary",
        "",
    ]
    for variant in ["skip_only", "bridge_only", "bridge_only_param_matched", "hybrid_no_small", "hybrid", "hybrid_gate_zero"]:
        summary_row = summary["per_variant"][variant]
        line = (
            f"- {variant}: "
            f"hidden_mse_mean={summary_row['hidden_mse_mean']:.6f}, "
            f"hidden_mse_std={summary_row['hidden_mse_std']:.6f}, "
            f"cosine_mean={summary_row['cosine_mean']:.6f}, "
            f"cosine_std={summary_row['cosine_std']:.6f}"
        )
        if "gate_value_mean" in summary_row:
            line += (
                f", gate_value_mean={summary_row['gate_value_mean']:.6f}, "
                f"delta_norm_mean={summary_row['delta_norm_mean']:.6f}"
            )
        lines.append(line)

    lines.extend(
        [
            "",
            "## interpretation rule",
            "",
            "- A per-seed win means hybrid has lower hidden-state MSE and higher cosine on the same seed.",
            "- A result counts as clear and reproducible here only if hybrid wins on both metrics in at least 2 of 3 seeds and the aggregate means point in the same direction.",
            "",
            "## answers",
            "",
            f"1. Does hybrid consistently beat skip-only? {'Yes' if hybrid_vs_skip == len(results_payload['seeds']) else 'No'}. Hybrid wins on both metrics in {hybrid_vs_skip}/{len(results_payload['seeds'])} seeds.",
            f"2. Does hybrid beat hybrid_no_small? {'Yes' if clear_vs_no_small else 'No'}. Hybrid wins on both metrics in {hybrid_vs_no_small}/{len(results_payload['seeds'])} seeds.",
            f"3. Does hybrid beat the original bridge-only? {'Yes' if clear_vs_bridge else 'No'}. Hybrid wins on both metrics in {hybrid_vs_bridge}/{len(results_payload['seeds'])} seeds.",
            f"4. Does hybrid beat the parameter-matched bridge-only? {'Yes' if clear_vs_param else 'No'}. Hybrid wins on both metrics in {hybrid_vs_param}/{len(results_payload['seeds'])} seeds.",
            f"5. Are any wins consistent across seeds? {'Yes' if reproducible_controls else 'No'}. The only controls that meet the 2-of-3 reproducibility rule are: {', '.join(reproducible_controls) if reproducible_controls else 'none'}.",
            f"6. Is the delegated path actually used, based on gate and delta diagnostics? {'Yes' if results_payload['diagnostics']['delegated_path_used_across_seeds'] else 'No'}. Hybrid gate mean={summary['per_variant']['hybrid']['gate_value_mean']:.6f}, hybrid delta_norm_mean={summary['per_variant']['hybrid']['delta_norm_mean']:.6f}, and hybrid vs gate-zero gains are "
            f"mse={results_payload['diagnostics']['hybrid_vs_gate_zero_hidden_mse_gain_mean']:.6f}, cosine={results_payload['diagnostics']['hybrid_vs_gate_zero_cosine_gain_mean']:.6f}.",
            "7. What is the most defensible current claim? "
            + (
                "The delegated small-model path is active and helps relative to skip-only, but the current evidence does not show a reproducible advantage over the stronger bridge controls."
                if not (clear_vs_bridge and clear_vs_param)
                else "The delegated small-model path appears to add reproducible value beyond both bridge controls on this pilot."
            ),
            "",
            "## recommendation",
            "",
            ("Proceed to Stage C" if proceed else "Do not proceed to Stage C yet"),
            "",
            "## next minimal action",
            "",
            (
                "- Recommendation basis: hybrid clearly beats both bridge controls on the main hidden-recovery metrics, so Stage C is justified."
                if proceed
                else f"- Recommendation basis: the stronger bridge control on this run is `{stronger_bridge}`, and hybrid does not yet show a clear reproducible win over the stronger controls under the stated rule."
            ),
        ]
    )
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_logging()
    args = parse_args()
    base_config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    save_config_snapshot(output_dir / "config_snapshot.yaml", base_config)
    export_run_metadata(
        output_dir / "metadata.json",
        base_config,
        {
            "stage": "stage_b_ablation",
            "stage_a_checkpoint": args.stage_a_checkpoint,
            "seeds": args.seeds,
        },
    )

    backbones = load_backbones(base_config, load_large=True, load_small=True, load_tokenizer=True)
    stage_a_payload = load_checkpoint(args.stage_a_checkpoint, backbones.device)
    parameter_budget = _parameter_budget_summary(base_config, backbones, stage_a_payload)

    all_histories: list[dict[str, float]] = []
    seed_results: list[dict[str, Any]] = []
    diagnostics_per_seed: dict[str, Any] = {}

    for seed in args.seeds:
        config = _clone_config_with_seed(base_config, seed)
        seed_everything(seed)
        seed_dir = ensure_dir(output_dir / f"seed_{seed}")
        train_dataloader, train_corpus = build_dataloader(config, backbones.tokenizer, stage_name="stage_b", split_name="train")
        val_dataloader, val_corpus = build_dataloader(config, backbones.tokenizer, stage_name="stage_b", split_name="validation")
        save_json(seed_dir / "train_sample_ids.json", train_corpus.sample_metadata)
        save_json(seed_dir / "validation_sample_ids.json", val_corpus.sample_metadata)

        models = _build_variant_models(
            config,
            backbones,
            stage_a_payload,
            param_matched_rank=parameter_budget["bridge_only_param_matched_rank"],
        )

        histories_by_variant: dict[str, list[dict[str, float]]] = {}
        for variant in TRAINED_VARIANTS:
            model, history, checkpoint_payload = _train_variant(
                variant,
                models[variant],
                config,
                backbones,
                train_dataloader,
                seed=seed,
            )
            models[variant] = model
            histories_by_variant[variant] = history
            all_histories.extend(history)
            save_checkpoint(seed_dir / f"{variant}_checkpoint.pt", checkpoint_payload)

        metrics = _evaluate_models(models, config, backbones, val_dataloader)
        seed_payload = {
            "seed": seed,
            "metrics": metrics,
        }
        seed_results.append(seed_payload)
        save_json(seed_dir / "metrics.json", seed_payload)
        seed_history_rows: list[dict[str, float]] = []
        for variant in TRAINED_VARIANTS:
            seed_history_rows.extend(histories_by_variant[variant])
        save_csv(seed_dir / "history.csv", seed_history_rows)

        diagnostics_per_seed[str(seed)] = {
            "gate_traces": {
                variant: [row["gate_value"] for row in histories_by_variant[variant]]
                for variant in TRAINED_VARIANTS
            },
            "final_gate_values": {
                variant: histories_by_variant[variant][-1]["gate_value"] for variant in TRAINED_VARIANTS
            },
            "final_delta_norm_means": {
                variant: histories_by_variant[variant][-1]["delta_norm_mean"] for variant in TRAINED_VARIANTS
            },
            "metrics": metrics,
        }
        torch.cuda.empty_cache()

    summary, summary_rows = _aggregate_results(seed_results)
    diagnostics_payload = {
        "config_path": args.config,
        "stage_a_checkpoint": args.stage_a_checkpoint,
        "reused_fixed_stage_a_checkpoint": True,
        "parameter_budget": parameter_budget,
        "per_seed": diagnostics_per_seed,
        "delegated_path_used_across_seeds": all(
            diagnostics_per_seed[str(seed)]["metrics"]["delegated_path_used"] for seed in args.seeds
        ),
        "hybrid_vs_gate_zero_hidden_mse_gain_mean": _mean(
            [result["metrics"]["hybrid_vs_gate_zero_hidden_mse_gain"] for result in seed_results]
        ),
        "hybrid_vs_gate_zero_cosine_gain_mean": _mean(
            [result["metrics"]["hybrid_vs_gate_zero_cosine_gain"] for result in seed_results]
        ),
    }
    results_payload = {
        "config_path": args.config,
        "seq_len": base_config.training.seq_len,
        "max_train_steps": base_config.training.stage_b.max_steps,
        "stage_a_checkpoint": args.stage_a_checkpoint,
        "seeds": args.seeds,
        "parameter_budget": parameter_budget,
        "seed_results": seed_results,
        "summary": summary,
    }

    save_csv(output_dir / "history.csv", all_histories)
    save_json(output_dir / "results.json", results_payload)
    save_json(output_dir / "diagnostics.json", diagnostics_payload)
    ensure_dir(Path(args.results_path).parent)
    ensure_dir(Path(args.summary_path).parent)
    ensure_dir(Path(args.diagnostics_path).parent)
    save_json(args.results_path, results_payload)
    save_csv(args.summary_path, summary_rows)
    save_json(args.diagnostics_path, diagnostics_payload)
    _write_ablation_report(args.report_path, {**results_payload, "diagnostics": diagnostics_payload})
    write_real_hardware_report("notes/real_hardware_report.md")


if __name__ == "__main__":
    main()
