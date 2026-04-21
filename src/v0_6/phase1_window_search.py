"""Phase 1A output-aware asymmetric window search."""

from __future__ import annotations

import argparse
import itertools
import json
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
from src.models.baselines import SkipOnlyLargeModel
from src.models.hybrid_gemma import HybridDelegationModel, HybridNoSmallModel
from src.train.stage_b_objective import compute_stage_b_loss_breakdown, prepare_stage_b_teacher_targets
from src.train.stage_b_train_utils import stage_b_train_entry_projector, stage_b_trainable_prefixes
from src.train.trainer_utils import (
    build_dataloader,
    build_optimizer,
    build_stage_b_optimizer,
    move_batch_to_device,
    zero_requires_grad,
)
from src.utils.io import ensure_dir, load_config, save_csv, save_json, save_text
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything
from src.v0_6.common import WindowCandidate, clone_config
from src.v0_6.window_search import (
    distinct_small_windows,
    enumerate_window_candidates,
    load_shortlist,
    load_window_search_settings,
    ranking_key,
    shortlist_rows,
    window_row,
)


LOGGER = get_logger(__name__)
METRIC_NAMES = [
    "logit_kl_to_teacher",
    "nll",
    "perplexity",
    "hidden_mse",
    "hidden_cosine",
    "top1_agreement",
    "top5_overlap",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["pilot", "confirm"], default="pilot")
    parser.add_argument("--output-dir", default="artifacts/v0_6/phase1_window_search")
    parser.add_argument("--report-path", default="notes/v0_6/phase1_window_search_report.md")
    parser.add_argument("--shortlist-path", default=None)
    return parser.parse_args()


def _valid_next_token_count(labels: torch.Tensor) -> int:
    return int((labels[:, 1:] != -100).sum().item())


def _empty_metric_totals() -> dict[str, float]:
    return {
        "valid_tokens": 0.0,
        "hidden_positions": 0.0,
        "logit_kl_to_teacher_sum": 0.0,
        "nll_sum": 0.0,
        "hidden_mse_sum": 0.0,
        "hidden_cosine_sum": 0.0,
        "top1_agreement_sum": 0.0,
        "top5_overlap_sum": 0.0,
    }


def _finalize_metric_totals(totals: dict[str, float]) -> dict[str, float]:
    valid_tokens = max(1.0, totals["valid_tokens"])
    hidden_positions = max(1.0, totals["hidden_positions"])
    nll = totals["nll_sum"] / valid_tokens
    return {
        "logit_kl_to_teacher": totals["logit_kl_to_teacher_sum"] / valid_tokens,
        "nll": nll,
        "perplexity": perplexity_from_loss(nll),
        "hidden_mse": totals["hidden_mse_sum"] / hidden_positions,
        "hidden_cosine": totals["hidden_cosine_sum"] / hidden_positions,
        "top1_agreement": totals["top1_agreement_sum"] / valid_tokens,
        "top5_overlap": totals["top5_overlap_sum"] / valid_tokens,
        "valid_tokens": totals["valid_tokens"],
        "hidden_positions": totals["hidden_positions"],
    }


def _topk_overlap_sum(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, top_k: int) -> float:
    mask = labels[:, 1:] != -100
    student_topk = student_logits[:, :-1, :].topk(top_k, dim=-1).indices
    teacher_topk = teacher_logits[:, :-1, :].topk(top_k, dim=-1).indices
    overlap_fraction = (
        (student_topk.unsqueeze(-1) == teacher_topk.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float() / float(top_k)
    )
    return float(overlap_fraction[mask].sum().cpu())


def _compute_batch_metric_sums(
    outputs: Any,
    teacher_hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    top_k: int,
) -> dict[str, float]:
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    valid_tokens = _valid_next_token_count(labels)
    hidden_positions = int(attention_mask.sum().item())
    if valid_tokens == 0:
        return _empty_metric_totals()

    student_logits = outputs.logits
    hidden_mse = masked_hidden_mse(outputs.hidden_after_removed, teacher_hidden, attention_mask)
    hidden_cosine = 1.0 - float(masked_hidden_cosine_loss(outputs.hidden_after_removed, teacher_hidden, attention_mask).detach().cpu())
    nll_sum = float((shifted_cross_entropy(student_logits, labels) * valid_tokens).detach().cpu())
    kl_sum = float((shifted_kl_divergence(student_logits, teacher_logits, labels) * valid_tokens).detach().cpu())

    mask = labels[:, 1:] != -100
    student_top1 = student_logits[:, :-1, :].argmax(dim=-1)
    teacher_top1 = teacher_logits[:, :-1, :].argmax(dim=-1)
    top1_agreement_sum = float(((student_top1 == teacher_top1) & mask).sum().detach().cpu())
    top5_overlap_sum = _topk_overlap_sum(student_logits, teacher_logits, labels, top_k)

    return {
        "valid_tokens": float(valid_tokens),
        "hidden_positions": float(hidden_positions),
        "logit_kl_to_teacher_sum": kl_sum,
        "nll_sum": nll_sum,
        "hidden_mse_sum": float(hidden_mse.detach().cpu()) * float(hidden_positions),
        "hidden_cosine_sum": hidden_cosine * float(hidden_positions),
        "top1_agreement_sum": top1_agreement_sum,
        "top5_overlap_sum": top5_overlap_sum,
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _summarize_metrics(seed_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for model_name in ["hybrid", "skip_only", "hybrid_no_small"]:
        model_summary: dict[str, float] = {}
        for metric_name in METRIC_NAMES:
            values = [seed_result["metrics_by_model"][model_name][metric_name] for seed_result in seed_results]
            model_summary[f"{metric_name}_mean"] = _mean(values)
            model_summary[f"{metric_name}_std"] = _std(values)
        summary[model_name] = model_summary

    stage_a_losses = [seed_result["stage_a"]["final_loss"] for seed_result in seed_results]
    stage_b_losses = [seed_result["stage_b"]["final_loss"] for seed_result in seed_results]
    summary["training"] = {
        "stage_a_final_loss_mean": _mean(stage_a_losses),
        "stage_a_final_loss_std": _std(stage_a_losses),
        "stage_b_final_loss_mean": _mean(stage_b_losses),
        "stage_b_final_loss_std": _std(stage_b_losses),
    }
    return summary


def _build_seed_loaders(config: Any, backbones: LoadedBackbones, seed: int) -> tuple[Any, Any]:
    seed_config = clone_config(config, seed=seed)
    train_loader, _ = build_dataloader(seed_config, backbones.tokenizer, stage_name="phase1_window_search", split_name="train")
    val_loader, _ = build_dataloader(seed_config, backbones.tokenizer, stage_name="phase1_window_search", split_name="validation")
    return train_loader, val_loader


def _train_stage_a_candidate(candidate_config: Any, backbones: LoadedBackbones, train_loader: Any) -> tuple[dict[str, Any], dict[str, float]]:
    model = HybridDelegationModel(candidate_config, backbones.large_model, backbones.small_model)
    zero_requires_grad(model, except_prefixes=["entry_projector"])
    optimizer = build_optimizer(model, candidate_config)
    batch_iter = itertools.cycle(train_loader)
    model.train()

    final_loss = 0.0
    final_mse = 0.0
    final_cosine = 0.0
    for _step in range(1, candidate_config.training.stage_a.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(candidate_config.training.grad_accum_steps):
            batch = move_batch_to_device(next(batch_iter), backbones.device)
            with torch.no_grad():
                large_state = model.large_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
                large_state = model.large_runner.run_layers(large_state, 0, candidate_config.split.large_prefix_end)
                large_hidden = large_state.hidden_states.detach()

                small_state = model.small_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
                small_state = model.small_runner.run_layers(small_state, 0, candidate_config.split.small_entry_target_layer)
                small_hidden = small_state.hidden_states.detach()

            projected_hidden = model.entry_projector(large_hidden)
            mse_loss = masked_hidden_mse(projected_hidden, small_hidden, batch["attention_mask"])
            cosine_loss = masked_hidden_cosine_loss(projected_hidden, small_hidden, batch["attention_mask"])
            total_loss = (mse_loss + cosine_loss) / candidate_config.training.grad_accum_steps
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), candidate_config.training.max_grad_norm)
        optimizer.step()
        final_loss = float((mse_loss + cosine_loss).detach().cpu())
        final_mse = float(mse_loss.detach().cpu())
        final_cosine = float(1.0 - cosine_loss.detach().cpu())

    return (
        {"entry_projector": model.entry_projector.state_dict()},
        {
            "final_loss": final_loss,
            "final_mse": final_mse,
            "final_cosine": final_cosine,
        },
    )


def _train_stage_b_candidate(
    candidate_config: Any,
    backbones: LoadedBackbones,
    train_loader: Any,
    stage_a_payload: dict[str, Any],
) -> tuple[HybridDelegationModel, dict[str, float]]:
    model = HybridDelegationModel(candidate_config, backbones.large_model, backbones.small_model)
    model.entry_projector.load_state_dict(stage_a_payload["entry_projector"])
    zero_requires_grad(model, except_prefixes=stage_b_trainable_prefixes("hybrid", candidate_config))
    optimizer = build_stage_b_optimizer(model, candidate_config)
    batch_iter = itertools.cycle(train_loader)
    model.train()

    train_entry = stage_b_train_entry_projector(candidate_config)
    final_loss = 0.0
    final_kl = 0.0
    final_ce = 0.0
    final_gate = 0.0
    for _step in range(1, candidate_config.training.stage_b.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(candidate_config.training.grad_accum_steps):
            batch = move_batch_to_device(next(batch_iter), backbones.device)
            teacher_targets = prepare_stage_b_teacher_targets(model.large_runner, batch, candidate_config)
            large_hidden = teacher_targets.hidden_after_prefix

            if train_entry:
                projected_hidden = model.entry_projector(large_hidden)
                delegated_small_hidden = model.run_delegated_small_block(projected_hidden, batch["attention_mask"])
            else:
                with torch.no_grad():
                    projected_hidden = model.entry_projector(large_hidden)
                    delegated_small_hidden = model.run_delegated_small_block(projected_hidden, batch["attention_mask"])
                delegated_small_hidden = delegated_small_hidden.detach()

            delta_large = model.return_adapter(delegated_small_hidden)
            predicted_hidden = large_hidden + model.gate(delta_large)

            loss_terms = compute_stage_b_loss_breakdown(
                model.large_runner,
                candidate_config,
                teacher_targets,
                predicted_hidden,
                batch["attention_mask"],
                batch["labels"],
                delta_large,
            )
            (loss_terms.total_loss / candidate_config.training.grad_accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), candidate_config.training.max_grad_norm)
        optimizer.step()
        final_loss = float(loss_terms.total_loss.detach().cpu())
        final_kl = float(loss_terms.kl_loss.detach().cpu())
        final_ce = float(loss_terms.ce_loss.detach().cpu())
        final_gate = float(model.gate.value().detach().cpu())

    model.eval()
    return model, {
        "final_loss": final_loss,
        "final_kl": final_kl,
        "final_ce": final_ce,
        "final_gate": final_gate,
    }


def _build_hybrid_no_small_control(candidate_config: Any, backbones: LoadedBackbones, trained_model: HybridDelegationModel) -> HybridNoSmallModel:
    control = HybridNoSmallModel(candidate_config, backbones.large_model, backbones.small_model)
    control.entry_projector.load_state_dict(trained_model.entry_projector.state_dict())
    control.return_adapter.load_state_dict(trained_model.return_adapter.state_dict())
    control.gate.load_state_dict(trained_model.gate.state_dict())
    control.eval()
    return control


def _evaluate_candidate_models(
    candidate_config: Any,
    backbones: LoadedBackbones,
    val_loader: Any,
    hybrid_model: HybridDelegationModel,
    *,
    top_k: int,
    max_validation_batches: int | None,
) -> dict[str, dict[str, float]]:
    skip_only = SkipOnlyLargeModel(candidate_config, backbones.large_model)
    skip_only.eval()
    hybrid_no_small = _build_hybrid_no_small_control(candidate_config, backbones, hybrid_model)
    models = {
        "skip_only": skip_only,
        "hybrid_no_small": hybrid_no_small,
        "hybrid": hybrid_model,
    }
    totals = {name: _empty_metric_totals() for name in models}

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            if max_validation_batches is not None and batch_index >= max_validation_batches:
                break
            batch = move_batch_to_device(batch, backbones.device)
            teacher_targets = prepare_stage_b_teacher_targets(
                hybrid_model.large_runner,
                batch,
                candidate_config,
                include_teacher_logits=True,
            )
            if teacher_targets.teacher_logits is None:
                raise RuntimeError("Teacher logits were not computed for the window-search evaluation.")
            for model_name, model in models.items():
                outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
                batch_totals = _compute_batch_metric_sums(
                    outputs,
                    teacher_targets.teacher_hidden,
                    teacher_targets.teacher_logits,
                    batch,
                    top_k=top_k,
                )
                for key, value in batch_totals.items():
                    totals[model_name][key] += value

    return {model_name: _finalize_metric_totals(metric_totals) for model_name, metric_totals in totals.items()}


def _evaluate_candidate_for_seed(
    config: Any,
    candidate: WindowCandidate,
    seed: int,
    backbones: LoadedBackbones,
    train_loader: Any,
    val_loader: Any,
    *,
    stage_a_steps: int | None,
    stage_b_steps: int | None,
    top_k: int,
    max_validation_batches: int | None,
) -> dict[str, Any]:
    seed_everything(seed)
    candidate_config = clone_config(
        config,
        candidate=candidate,
        seed=seed,
        experiment_name=f"{config.experiment.name}_{candidate.label}_seed{seed}",
        stage_a_steps=stage_a_steps,
        stage_b_steps=stage_b_steps,
    )

    stage_a_payload, stage_a_metrics = _train_stage_a_candidate(candidate_config, backbones, train_loader)
    hybrid_model, stage_b_metrics = _train_stage_b_candidate(candidate_config, backbones, train_loader, stage_a_payload)
    metrics_by_model = _evaluate_candidate_models(
        candidate_config,
        backbones,
        val_loader,
        hybrid_model,
        top_k=top_k,
        max_validation_batches=max_validation_batches,
    )
    torch.cuda.empty_cache()
    return {
        "seed": seed,
        "candidate": candidate.to_dict(),
        "stage_a": stage_a_metrics,
        "stage_b": stage_b_metrics,
        "metrics_by_model": metrics_by_model,
    }


def _load_candidates(
    config: Any,
    backbones: LoadedBackbones,
    mode: str,
    shortlist_path: str | None,
) -> list[WindowCandidate]:
    settings = load_window_search_settings(config)
    if mode == "confirm":
        path = shortlist_path or "artifacts/v0_6/phase1_window_search/pilot_shortlist.json"
        return load_shortlist(path)
    return enumerate_window_candidates(
        config,
        settings,
        large_num_layers=int(backbones.large_model.config.num_hidden_layers),
        small_num_layers=int(backbones.small_model.config.num_hidden_layers),
    )


def _write_report(
    report_path: str | Path,
    *,
    config_path: str,
    mode: str,
    rows: list[dict[str, Any]],
    settings: Any,
    candidate_count: int,
    is_debug: bool,
) -> None:
    ranked = sorted(rows, key=ranking_key)
    top_rows = ranked[: min(5, len(ranked))]
    lines = [
        "# Phase 1A Window Search Report",
        "",
        f"- Config: `{config_path}`",
        f"- Mode: `{mode}`",
        f"- Candidate count: `{candidate_count}`",
        f"- Seeds: `{', '.join(str(seed) for seed in (settings.pilot_seeds if mode == 'pilot' else settings.confirm_seeds))}`",
        f"- Backbones: `{'debug_tiny' if is_debug else 'real_model'}`",
        "- Ranking rule: KL, then NLL, then PPL, then hidden MSE, then hidden cosine.",
        "",
        "## Top Candidates",
        "",
    ]
    for index, row in enumerate(top_rows, start=1):
        lines.append(
            f"{index}. `{row['label']}` | "
            f"KL={row['hybrid_logit_kl_to_teacher_mean']:.6f} | "
            f"NLL={row['hybrid_nll_mean']:.6f} | "
            f"PPL={row['hybrid_perplexity_mean']:.6f} | "
            f"hidden_mse={row['hybrid_hidden_mse_mean']:.6f} | "
            f"hidden_cos={row['hybrid_hidden_cosine_mean']:.6f} | "
            f"vs_skip_nll={row['hybrid_minus_skip_nll_mean']:.6f} | "
            f"vs_no_small_nll={row['hybrid_minus_no_small_nll_mean']:.6f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This artifact is a candidate-ranking harness, not a final scientific claim by itself.",
            "- Negative KL/NLL deltas against `skip_only` or `hybrid_no_small` indicate the delegated path recovered output behavior beyond those controls for that candidate.",
        ]
    )
    if is_debug:
        lines.extend(
            [
                "- This run used the debug-tiny path only. It validates the continuation pipeline and artifact flow, but it is not evidence about Gemma phase ordering.",
                "- Real Gemma pilot execution remains blocked by the existing environment notes in `notes/blockers.md`.",
            ]
        )
    save_text(report_path, "\n".join(lines))


def main() -> None:
    """Run the Phase 1A candidate search."""

    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    settings = load_window_search_settings(config)
    output_dir = ensure_dir(args.output_dir)

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    candidates = _load_candidates(config, backbones, args.mode, args.shortlist_path)
    seeds = settings.pilot_seeds if args.mode == "pilot" else settings.confirm_seeds
    if not candidates:
        raise RuntimeError("No window-search candidates were generated.")

    all_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    per_seed_loaders = {seed: _build_seed_loaders(config, backbones, seed) for seed in seeds}

    for candidate in candidates:
        LOGGER.info("phase1_window_search mode=%s candidate=%s", args.mode, candidate.label)
        seed_results = []
        for seed in seeds:
            train_loader, val_loader = per_seed_loaders[seed]
            seed_result = _evaluate_candidate_for_seed(
                config,
                candidate,
                seed,
                backbones,
                train_loader,
                val_loader,
                stage_a_steps=settings.stage_a_steps,
                stage_b_steps=settings.stage_b_steps,
                top_k=settings.top_k,
                max_validation_batches=settings.max_validation_batches,
            )
            seed_results.append(seed_result)

        summary = _summarize_metrics(seed_results)
        row = window_row(
            candidate,
            hybrid_metrics=summary["hybrid"],
            skip_metrics=summary["skip_only"],
            hybrid_no_small_metrics=summary["hybrid_no_small"],
            extra=summary["training"],
        )
        all_results.append(
            {
                "candidate": candidate.to_dict(),
                "seed_results": seed_results,
                "summary": summary,
            }
        )
        summary_rows.append(row)

    ranked_rows = sorted(summary_rows, key=ranking_key)
    for index, row in enumerate(ranked_rows, start=1):
        row["rank"] = index

    save_json(
        output_dir / f"{args.mode}_results.json",
        {
            "mode": args.mode,
            "config_path": args.config,
            "candidate_count": len(candidates),
            "is_debug": backbones.is_debug,
            "seeds": seeds,
            "results": all_results,
        },
    )
    save_csv(output_dir / f"{args.mode}_summary.csv", ranked_rows)

    shortlist = shortlist_rows(ranked_rows, settings.shortlist_size)
    save_json(output_dir / f"{args.mode}_shortlist.json", shortlist)

    if args.mode == "pilot":
        top_candidates = distinct_small_windows(
            [WindowCandidate(row["large_start"], row["large_end"], row["small_start"], row["small_end"]) for row in shortlist],
            settings.shortlist_size,
        )
        save_json(output_dir / "pilot_distinct_small_windows.json", [candidate.to_dict() for candidate in top_candidates])

    _write_report(
        args.report_path,
        config_path=args.config,
        mode=args.mode,
        rows=ranked_rows,
        settings=settings,
        candidate_count=len(candidates),
        is_debug=backbones.is_debug,
    )


if __name__ == "__main__":
    main()
