"""Output-level probe for existing Stage B checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from src.eval.metrics import perplexity_from_loss, shifted_cross_entropy, shifted_kl_divergence
from src.models.backbone_loader import LoadedBackbones, load_backbones
from src.models.baselines import (
    BridgeOnlyLargeModel,
    BridgeOnlyParamMatchedModel,
    FullLargeModel,
    SkipOnlyLargeModel,
)
from src.models.hybrid_gemma import HybridDelegationModel, HybridNoSmallModel
from src.train.trainer_utils import build_dataloader, load_checkpoint, move_batch_to_device
from src.utils.io import ensure_dir, export_run_metadata, load_config, save_config_snapshot, save_csv, save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything


LOGGER = get_logger(__name__)
MODEL_ORDER = [
    "skip_only",
    "hybrid_no_small",
    "bridge_only",
    "bridge_only_param_matched",
    "hybrid",
    "full_large",
]
PAIRWISE_BASELINES = ["skip_only", "hybrid_no_small", "bridge_only", "bridge_only_param_matched"]
PRIMARY_METRICS = ["logit_kl_to_teacher", "nll"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gemma2_conservative_pilot_256.yaml")
    parser.add_argument("--ablation-dir", default="artifacts/stage_b_ablation")
    parser.add_argument("--ablation-results", default="artifacts/stage_b_ablation_results.json")
    parser.add_argument("--output-dir", default="artifacts/stage_b_output_probe")
    parser.add_argument("--results-path", default="artifacts/stage_b_output_probe_results.json")
    parser.add_argument("--summary-path", default="artifacts/stage_b_output_probe_summary.csv")
    parser.add_argument("--report-path", default="notes/stage_b_output_probe_report.md")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


def _clone_config_with_seed(config: Any, seed: int) -> Any:
    cloned = copy.deepcopy(config)
    cloned.training.seed = seed
    cloned.raw["training"]["seed"] = seed
    return cloned


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _seed_checkpoint_paths(ablation_dir: str | Path, seed: int) -> dict[str, str]:
    seed_dir = Path(ablation_dir) / f"seed_{seed}"
    return {
        "bridge_only": str(seed_dir / "bridge_only_checkpoint.pt"),
        "bridge_only_param_matched": str(seed_dir / "bridge_only_param_matched_checkpoint.pt"),
        "hybrid_no_small": str(seed_dir / "hybrid_no_small_checkpoint.pt"),
        "hybrid": str(seed_dir / "hybrid_checkpoint.pt"),
    }


def _load_models_for_seed(
    config: Any,
    backbones: LoadedBackbones,
    checkpoint_paths: dict[str, str],
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {
        "full_large": FullLargeModel(config, backbones.large_model),
        "skip_only": SkipOnlyLargeModel(config, backbones.large_model),
        "bridge_only": BridgeOnlyLargeModel(config, backbones.large_model),
    }

    bridge_only_payload = load_checkpoint(checkpoint_paths["bridge_only"], device)
    models["bridge_only"].bridge.load_state_dict(bridge_only_payload["bridge"])
    models["bridge_only"].gate.load_state_dict(bridge_only_payload["gate"])

    bridge_param_payload = load_checkpoint(checkpoint_paths["bridge_only_param_matched"], device)
    bridge_param_rank = int(bridge_param_payload["bridge"]["down.weight"].shape[0])
    models["bridge_only_param_matched"] = BridgeOnlyParamMatchedModel(
        config,
        backbones.large_model,
        rank=bridge_param_rank,
    )
    models["bridge_only_param_matched"].bridge.load_state_dict(bridge_param_payload["bridge"])
    models["bridge_only_param_matched"].gate.load_state_dict(bridge_param_payload["gate"])

    for variant, model_cls in {
        "hybrid_no_small": HybridNoSmallModel,
        "hybrid": HybridDelegationModel,
    }.items():
        payload = load_checkpoint(checkpoint_paths[variant], device)
        model = model_cls(config, backbones.large_model, backbones.small_model)
        model.entry_projector.load_state_dict(payload["entry_projector"])
        model.return_adapter.load_state_dict(payload["return_adapter"])
        model.gate.load_state_dict(payload["gate"])
        models[variant] = model

    for model in models.values():
        model.eval()
    return models


def _valid_next_token_count(labels: torch.Tensor) -> int:
    return int((labels[:, 1:] != -100).sum().item())


def compute_topk_overlap(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    top_k: int,
) -> tuple[float, int]:
    """Return the summed top-k overlap fraction and valid-token count."""

    mask = labels[:, 1:] != -100
    student_topk = student_logits[:, :-1, :].topk(top_k, dim=-1).indices
    teacher_topk = teacher_logits[:, :-1, :].topk(top_k, dim=-1).indices
    overlap_fraction = (
        (student_topk.unsqueeze(-1) == teacher_topk.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float() / float(top_k)
    )
    return float(overlap_fraction[mask].sum().cpu()), int(mask.sum().item())


def compute_batch_output_sums(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    top_k: int,
) -> dict[str, float]:
    """Return token-summed output metrics for one batch."""

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
    top1_agreement_sum = float(((student_top1 == teacher_top1) & mask).sum().detach().cpu())
    top5_overlap_sum, _ = compute_topk_overlap(student_logits, teacher_logits, labels, top_k)
    return {
        "valid_tokens": float(valid_tokens),
        "logit_kl_to_teacher_sum": kl_sum,
        "nll_sum": nll_sum,
        "top1_agreement_sum": top1_agreement_sum,
        "top5_overlap_sum": top5_overlap_sum,
    }


def _teacher_reference_sums(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
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
    nll_sum = float((shifted_cross_entropy(teacher_logits, labels) * valid_tokens).detach().cpu())
    return {
        "valid_tokens": float(valid_tokens),
        "logit_kl_to_teacher_sum": 0.0,
        "nll_sum": nll_sum,
        "top1_agreement_sum": float(valid_tokens),
        "top5_overlap_sum": float(valid_tokens),
    }


def _empty_totals() -> dict[str, float]:
    return {
        "valid_tokens": 0.0,
        "logit_kl_to_teacher_sum": 0.0,
        "nll_sum": 0.0,
        "top1_agreement_sum": 0.0,
        "top5_overlap_sum": 0.0,
    }


def _finalize_totals(totals: dict[str, float]) -> dict[str, float]:
    valid_tokens = max(1.0, totals["valid_tokens"])
    nll = totals["nll_sum"] / valid_tokens
    return {
        "logit_kl_to_teacher": totals["logit_kl_to_teacher_sum"] / valid_tokens,
        "nll": nll,
        "perplexity": perplexity_from_loss(nll),
        "top1_agreement": totals["top1_agreement_sum"] / valid_tokens,
        "top5_overlap": totals["top5_overlap_sum"] / valid_tokens,
        "valid_tokens": totals["valid_tokens"],
    }


def _evaluate_seed(
    config: Any,
    backbones: LoadedBackbones,
    seed: int,
    ablation_dir: str | Path,
    top_k: int,
) -> dict[str, Any]:
    seed_config = _clone_config_with_seed(config, seed)
    checkpoint_paths = _seed_checkpoint_paths(ablation_dir, seed)
    models = _load_models_for_seed(seed_config, backbones, checkpoint_paths, backbones.device)
    dataloader, corpus = build_dataloader(seed_config, backbones.tokenizer, stage_name="stage_b", split_name="validation")

    totals = {variant: _empty_totals() for variant in MODEL_ORDER}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, backbones.device)
            teacher_logits = models["full_large"](batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            batch_totals = {
                "full_large": _teacher_reference_sums(teacher_logits, batch["labels"], top_k),
            }
            for variant in MODEL_ORDER:
                if variant == "full_large":
                    continue
                student_logits = models[variant](batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                batch_totals[variant] = compute_batch_output_sums(student_logits, teacher_logits, batch["labels"], top_k)

            for variant, variant_totals in batch_totals.items():
                for key, value in variant_totals.items():
                    totals[variant][key] += value
            num_batches += 1

    metrics_by_model = {variant: _finalize_totals(totals[variant]) for variant in MODEL_ORDER}
    paired_deltas = {}
    for baseline in PAIRWISE_BASELINES:
        paired_deltas[f"hybrid_minus_{baseline}"] = {
            "logit_kl_to_teacher": (
                metrics_by_model["hybrid"]["logit_kl_to_teacher"] - metrics_by_model[baseline]["logit_kl_to_teacher"]
            ),
            "nll": metrics_by_model["hybrid"]["nll"] - metrics_by_model[baseline]["nll"],
            "perplexity": metrics_by_model["hybrid"]["perplexity"] - metrics_by_model[baseline]["perplexity"],
            "top1_agreement": metrics_by_model["hybrid"]["top1_agreement"] - metrics_by_model[baseline]["top1_agreement"],
            "top5_overlap": metrics_by_model["hybrid"]["top5_overlap"] - metrics_by_model[baseline]["top5_overlap"],
        }

    hybrid_primary_wins = {
        baseline: (
            metrics_by_model["hybrid"]["logit_kl_to_teacher"] < metrics_by_model[baseline]["logit_kl_to_teacher"]
            and metrics_by_model["hybrid"]["nll"] < metrics_by_model[baseline]["nll"]
        )
        for baseline in PAIRWISE_BASELINES
    }

    return {
        "seed": seed,
        "num_batches": num_batches,
        "sample_count": len(corpus.sample_metadata),
        "heldout_policy": "stage_b validation split reused for the matching seed",
        "checkpoint_paths": checkpoint_paths,
        "sample_metadata": corpus.sample_metadata,
        "metrics_by_model": metrics_by_model,
        "paired_deltas": paired_deltas,
        "hybrid_primary_wins": hybrid_primary_wins,
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _aggregate_results(seed_results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {"per_model": {}, "pairwise_deltas": {}, "pairwise_wins": {}}
    rows: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        model_summary: dict[str, float] = {}
        for metric in ["logit_kl_to_teacher", "nll", "perplexity", "top1_agreement", "top5_overlap"]:
            values = [seed_result["metrics_by_model"][model_name][metric] for seed_result in seed_results]
            model_summary[f"{metric}_mean"] = _mean(values)
            model_summary[f"{metric}_std"] = _std(values)
        summary["per_model"][model_name] = model_summary
        rows.append({"row_type": "model", "label": model_name, "hybrid_wins_on_primary_metrics": "", **model_summary})

    for baseline in PAIRWISE_BASELINES:
        delta_key = f"hybrid_minus_{baseline}"
        delta_summary: dict[str, float] = {}
        for metric in ["logit_kl_to_teacher", "nll", "perplexity", "top1_agreement", "top5_overlap"]:
            values = [seed_result["paired_deltas"][delta_key][metric] for seed_result in seed_results]
            delta_summary[f"{metric}_mean"] = _mean(values)
            delta_summary[f"{metric}_std"] = _std(values)
        win_count = sum(1 for seed_result in seed_results if seed_result["hybrid_primary_wins"][baseline])
        summary["pairwise_deltas"][baseline] = delta_summary
        summary["pairwise_wins"][baseline] = {
            "hybrid_wins_on_primary_metrics": win_count,
            "seeds": len(seed_results),
        }
        rows.append(
            {
                "row_type": "pairwise_delta",
                "label": f"hybrid_minus_{baseline}",
                "hybrid_wins_on_primary_metrics": win_count,
                **delta_summary,
            }
        )

    return summary, rows


def _is_reproducible_output_win(summary: dict[str, Any], baseline: str) -> bool:
    wins = summary["pairwise_wins"][baseline]["hybrid_wins_on_primary_metrics"]
    deltas = summary["pairwise_deltas"][baseline]
    return (
        wins >= 2
        and deltas["logit_kl_to_teacher_mean"] < 0.0
        and deltas["nll_mean"] < 0.0
    )


def _format_metric(value: float) -> str:
    return f"{value:.6f}"


def _write_output_probe_report(report_path: str | Path, results_payload: dict[str, Any]) -> None:
    summary = results_payload["summary"]
    reproducible_baselines = [baseline for baseline in PAIRWISE_BASELINES if _is_reproducible_output_win(summary, baseline)]
    proceed = _is_reproducible_output_win(summary, "hybrid_no_small") and (
        _is_reproducible_output_win(summary, "bridge_only")
        or _is_reproducible_output_win(summary, "bridge_only_param_matched")
    )

    lines = [
        "# Stage B Output Probe Report",
        "",
        "## setup",
        "",
        f"- Config: {results_payload['config_path']}",
        f"- seq_len: {results_payload['seq_len']}",
        f"- Seeds: {', '.join(str(seed) for seed in results_payload['seeds'])}",
        "- Held-out policy: reused the seed-matched Stage B validation text slice. No training was run.",
        "- Primary output-level decision metrics: logit KL to the full-large teacher and held-out next-token NLL.",
        "- Supporting metrics: perplexity, teacher top-1 agreement, and teacher top-5 overlap.",
        "- Metric note: perplexity is an exponential transform of NLL, so it should not be treated as independent evidence.",
        "",
        "## aggregate summary",
        "",
    ]

    for model_name in MODEL_ORDER:
        row = summary["per_model"][model_name]
        lines.append(
            f"- {model_name}: "
            f"kl_mean={_format_metric(row['logit_kl_to_teacher_mean'])}, "
            f"nll_mean={_format_metric(row['nll_mean'])}, "
            f"ppl_mean={_format_metric(row['perplexity_mean'])}, "
            f"top1_mean={_format_metric(row['top1_agreement_mean'])}, "
            f"top5_mean={_format_metric(row['top5_overlap_mean'])}"
        )

    lines.extend(
        [
            "",
            "## paired hybrid deltas",
            "",
            "- Delta sign convention: negative is better for KL/NLL/PPL, positive is better for top-1/top-5 agreement.",
        ]
    )
    for baseline in PAIRWISE_BASELINES:
        row = summary["pairwise_deltas"][baseline]
        wins = summary["pairwise_wins"][baseline]["hybrid_wins_on_primary_metrics"]
        lines.append(
            f"- hybrid_minus_{baseline}: "
            f"kl_delta_mean={_format_metric(row['logit_kl_to_teacher_mean'])}, "
            f"nll_delta_mean={_format_metric(row['nll_mean'])}, "
            f"ppl_delta_mean={_format_metric(row['perplexity_mean'])}, "
            f"top1_delta_mean={_format_metric(row['top1_agreement_mean'])}, "
            f"top5_delta_mean={_format_metric(row['top5_overlap_mean'])}, "
            f"primary_wins={wins}/{results_payload['seed_count']}"
        )

    lines.extend(
        [
            "",
            "## interpretation rule",
            "",
            "- A seed-level output win means hybrid has lower KL and lower NLL than the comparator on the same held-out slice.",
            "- A reproducible output-level win means hybrid wins on the primary metrics in at least 2 of 3 seeds and the aggregate KL/NLL deltas point in the same direction.",
            "",
            "## answers",
            "",
            f"1. Does hybrid beat skip_only at the output level? {'Yes' if _is_reproducible_output_win(summary, 'skip_only') else 'No'}. Hybrid wins on the primary metrics in {summary['pairwise_wins']['skip_only']['hybrid_wins_on_primary_metrics']}/{results_payload['seed_count']} seeds.",
            f"2. Does hybrid beat hybrid_no_small at the output level? {'Yes' if _is_reproducible_output_win(summary, 'hybrid_no_small') else 'No'}. Hybrid wins on the primary metrics in {summary['pairwise_wins']['hybrid_no_small']['hybrid_wins_on_primary_metrics']}/{results_payload['seed_count']} seeds.",
            f"3. Does hybrid beat bridge_only at the output level? {'Yes' if _is_reproducible_output_win(summary, 'bridge_only') else 'No'}. Hybrid wins on the primary metrics in {summary['pairwise_wins']['bridge_only']['hybrid_wins_on_primary_metrics']}/{results_payload['seed_count']} seeds.",
            f"4. Does hybrid beat bridge_only_param_matched at the output level? {'Yes' if _is_reproducible_output_win(summary, 'bridge_only_param_matched') else 'No'}. Hybrid wins on the primary metrics in {summary['pairwise_wins']['bridge_only_param_matched']['hybrid_wins_on_primary_metrics']}/{results_payload['seed_count']} seeds.",
            f"5. Are any wins consistent across the 3 seeds? {'Yes' if reproducible_baselines else 'No'}. The reproducible output-level wins are: {', '.join(reproducible_baselines) if reproducible_baselines else 'none'}.",
            "6. Which metric is most trustworthy here: KL, CE/NLL, or PPL? CE/NLL. It is the direct held-out text likelihood objective, while PPL is just its exponential transform and KL is primarily a teacher-matching measure.",
            "7. What is the best current research claim after adding output-level evidence? "
            + (
                "The delegated small-model path is active and clearly better than skip-only, but the current Stage B checkpoints do not translate the hidden-space advantage over no-small or bridge controls into better output-level language-model behavior."
                if not proceed
                else "The delegated small-model path shows output-level value beyond passthrough/no-small controls and at least one bridge control, which is enough to justify a tiny Stage C diagnostic."
            ),
            "",
            "## decision",
            "",
            ("Proceed to tiny Stage C diagnostic" if proceed else "Do not proceed to Stage C"),
            "",
        ]
    )

    if proceed:
        lines.extend(
            [
                "## tiny Stage C diagnostic plan",
                "",
                "- seq_len: 256",
                "- max_train_steps: 100 to 200",
                "- data: same lightweight held-out text policy only",
                "- variants: hybrid and bridge_only_param_matched only",
                "- objective: test whether a minimal output-distillation step widens the current output-level gap without changing the benchmark scope",
                "- do not run GSM8K or StrategyQA in this diagnostic",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## framing",
                "",
                "- Recommendation basis: the project remains a qualified positive result relative to skip-only, but it is a negative result on the stronger claims `hybrid > hybrid_no_small` and `hybrid > strong bridge` at this milestone.",
                "",
            ]
        )

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    ablation_payload = _load_json(args.ablation_results)
    seeds = args.seeds if args.seeds else list(ablation_payload.get("seeds", []))
    if not seeds:
        raise ValueError("No seeds were provided and none could be read from the Stage B ablation results.")
    seed_everything(config.training.seed)

    output_dir = ensure_dir(args.output_dir)
    save_config_snapshot(output_dir / "config_snapshot.yaml", config)
    export_run_metadata(
        output_dir / "metadata.json",
        config,
        {
            "stage": "stage_b_output_probe",
            "ablation_dir": args.ablation_dir,
            "ablation_results": args.ablation_results,
            "seeds": seeds,
            "top_k": args.top_k,
        },
    )

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        LOGGER.info("stage_b_output_probe evaluating seed=%s", seed)
        seed_result = _evaluate_seed(config, backbones, seed, args.ablation_dir, args.top_k)
        seed_results.append(seed_result)
        seed_dir = ensure_dir(output_dir / f"seed_{seed}")
        save_json(seed_dir / "metrics.json", seed_result)
        save_json(seed_dir / "sample_ids.json", seed_result["sample_metadata"])
        torch.cuda.empty_cache()

    summary, summary_rows = _aggregate_results(seed_results)
    results_payload = {
        "config_path": args.config,
        "seq_len": config.training.seq_len,
        "top_k": args.top_k,
        "seed_count": len(seeds),
        "seeds": seeds,
        "heldout_policy": "stage_b validation split reused for the matching seed",
        "ablation_dir": args.ablation_dir,
        "ablation_results": args.ablation_results,
        "seed_results": seed_results,
        "summary": summary,
    }

    save_json(output_dir / "results.json", results_payload)
    save_csv(output_dir / "summary.csv", summary_rows)
    ensure_dir(Path(args.results_path).parent)
    ensure_dir(Path(args.summary_path).parent)
    save_json(args.results_path, results_payload)
    save_csv(args.summary_path, summary_rows)
    _write_output_probe_report(args.report_path, results_payload)


if __name__ == "__main__":
    main()
