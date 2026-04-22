"""Output-level probe for the Idea 4 token-wise mixture."""

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
from src.models.baselines import BridgeOnlyLargeModel, BridgeOnlyParamMatchedModel, FullLargeModel, SkipOnlyLargeModel
from src.train.trainer_utils import load_checkpoint, move_batch_to_device
from src.utils.io import ensure_dir, export_run_metadata, load_config, save_config_snapshot, save_csv, save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything
from src.v0_6.idea4_common import load_mixture_path_specs
from src.v0_6.idea4_holdout import build_probe_holdout_slice
from src.v0_6.idea4_models import TwoPathStaticMixtureHybrid, TwoPathStaticMixtureNoSmallModel
from src.v0_6.idea4_tokenwise_models import TwoPathTokenwiseMixtureHybrid, TwoPathTokenwiseMixtureNoSmallModel


LOGGER = get_logger(__name__)
MODEL_ORDER = [
    "skip_only",
    "static_mixture_no_small",
    "bridge_only",
    "bridge_only_param_matched",
    "tokenwise_mixture_no_small",
    "static_mixture",
    "tokenwise_mixture",
    "full_large",
]
PAIRWISE_BASELINES = [
    "skip_only",
    "static_mixture_no_small",
    "bridge_only",
    "bridge_only_param_matched",
    "tokenwise_mixture_no_small",
    "static_mixture",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Idea 4 token-wise output probe."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--static-stage-dir", default="artifacts/v0_6/idea4_static_mixture/confirm/stage_b")
    parser.add_argument("--tokenwise-stage-dir", default="artifacts/v0_6/idea4_tokenwise/confirm/stage_b")
    parser.add_argument("--output-dir", default="artifacts/v0_6/idea4_tokenwise/output_probe")
    parser.add_argument("--results-path", default="artifacts/v0_6/idea4_tokenwise/output_probe_results.json")
    parser.add_argument("--summary-path", default="artifacts/v0_6/idea4_tokenwise/output_probe_summary.csv")
    parser.add_argument("--report-path", default="notes/v0_6/idea4_tokenwise_output_probe.md")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--holdout-policy", choices=["main_validation", "fresh_untouched"], default="main_validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    return parser.parse_args()


def _clone_config_with_seed(config: Any, seed: int) -> Any:
    cloned = copy.deepcopy(config)
    cloned.training.seed = seed
    cloned.raw["training"]["seed"] = seed
    return cloned


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_static_payload(model: Any, payload: dict[str, Any], device: torch.device) -> None:
    model.entry_projector_b.load_state_dict(payload["entry_projector_b"])
    model.entry_projector_a.load_state_dict(payload["entry_projector_a"])
    model.return_adapter_b.load_state_dict(payload["return_adapter_b"])
    model.return_adapter_a.load_state_dict(payload["return_adapter_a"])
    with torch.no_grad():
        model.alpha.copy_(payload["alpha"].to(device=device, dtype=model.alpha.dtype))


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
    path_specs: list[Any],
    seed: int,
    static_stage_dir: str | Path,
    tokenwise_stage_dir: str | Path,
) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {
        "full_large": FullLargeModel(config, backbones.large_model),
        "skip_only": SkipOnlyLargeModel(config, backbones.large_model),
    }

    bridge_payload = load_checkpoint(Path(static_stage_dir) / f"seed_{seed}" / "bridge_only_checkpoint.pt", backbones.device)
    bridge_only = BridgeOnlyLargeModel(config, backbones.large_model)
    bridge_only.bridge.load_state_dict(bridge_payload["bridge"])
    bridge_only.gate.load_state_dict(bridge_payload["gate"])
    models["bridge_only"] = bridge_only

    static_no_small_payload = load_checkpoint(
        Path(static_stage_dir) / f"seed_{seed}" / "static_mixture_no_small_checkpoint.pt",
        backbones.device,
    )
    static_no_small = TwoPathStaticMixtureNoSmallModel(config, backbones.large_model, backbones.small_model, path_specs)
    _load_static_payload(static_no_small, static_no_small_payload, backbones.device)
    models["static_mixture_no_small"] = static_no_small

    static_payload = load_checkpoint(Path(static_stage_dir) / f"seed_{seed}" / "static_mixture_checkpoint.pt", backbones.device)
    static_mixture = TwoPathStaticMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
    _load_static_payload(static_mixture, static_payload, backbones.device)
    models["static_mixture"] = static_mixture

    tokenwise_no_small_payload = load_checkpoint(
        Path(tokenwise_stage_dir) / f"seed_{seed}" / "tokenwise_mixture_no_small_checkpoint.pt",
        backbones.device,
    )
    tokenwise_no_small = TwoPathTokenwiseMixtureNoSmallModel(config, backbones.large_model, backbones.small_model, path_specs)
    _load_tokenwise_payload(tokenwise_no_small, tokenwise_no_small_payload, backbones.device)
    models["tokenwise_mixture_no_small"] = tokenwise_no_small

    tokenwise_payload = load_checkpoint(
        Path(tokenwise_stage_dir) / f"seed_{seed}" / "tokenwise_mixture_checkpoint.pt",
        backbones.device,
    )
    tokenwise_mixture = TwoPathTokenwiseMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
    _load_tokenwise_payload(tokenwise_mixture, tokenwise_payload, backbones.device)
    models["tokenwise_mixture"] = tokenwise_mixture

    bridge_param_payload = load_checkpoint(
        Path(tokenwise_stage_dir) / f"seed_{seed}" / "bridge_only_param_matched_checkpoint.pt",
        backbones.device,
    )
    bridge_param_rank = int(bridge_param_payload["bridge"]["down.weight"].shape[0])
    bridge_param = BridgeOnlyParamMatchedModel(config, backbones.large_model, rank=bridge_param_rank)
    bridge_param.bridge.load_state_dict(bridge_param_payload["bridge"])
    bridge_param.gate.load_state_dict(bridge_param_payload["gate"])
    models["bridge_only_param_matched"] = bridge_param

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


def _teacher_reference_sums(teacher_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
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
    static_stage_dir: str | Path,
    tokenwise_stage_dir: str | Path,
    path_specs: list[Any],
    top_k: int,
    holdout_policy: str,
) -> dict[str, Any]:
    seed_config = _clone_config_with_seed(config, seed)
    models = _load_models_for_seed(seed_config, backbones, path_specs, seed, static_stage_dir, tokenwise_stage_dir)
    holdout_slice = build_probe_holdout_slice(
        seed_config,
        backbones.tokenizer,
        holdout_policy=holdout_policy,
        seed=seed,
    )
    dataloader = holdout_slice.dataloader

    totals = {variant: _empty_totals() for variant in MODEL_ORDER}
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, backbones.device)
            teacher_logits = models["full_large"](batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            batch_totals = {
                "full_large": _teacher_reference_sums(teacher_logits, batch["labels"]),
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
    tokenwise_primary_wins = {}
    for baseline in PAIRWISE_BASELINES:
        paired_deltas[f"tokenwise_mixture_minus_{baseline}"] = {
            "logit_kl_to_teacher": (
                metrics_by_model["tokenwise_mixture"]["logit_kl_to_teacher"]
                - metrics_by_model[baseline]["logit_kl_to_teacher"]
            ),
            "nll": metrics_by_model["tokenwise_mixture"]["nll"] - metrics_by_model[baseline]["nll"],
            "perplexity": metrics_by_model["tokenwise_mixture"]["perplexity"] - metrics_by_model[baseline]["perplexity"],
            "top1_agreement": metrics_by_model["tokenwise_mixture"]["top1_agreement"] - metrics_by_model[baseline]["top1_agreement"],
            "top5_overlap": metrics_by_model["tokenwise_mixture"]["top5_overlap"] - metrics_by_model[baseline]["top5_overlap"],
        }
        tokenwise_primary_wins[baseline] = (
            metrics_by_model["tokenwise_mixture"]["logit_kl_to_teacher"] < metrics_by_model[baseline]["logit_kl_to_teacher"]
            and metrics_by_model["tokenwise_mixture"]["nll"] < metrics_by_model[baseline]["nll"]
        )

    return {
        "seed": seed,
        "num_batches": num_batches,
        "sample_count": len(holdout_slice.sample_metadata),
        "heldout_policy": holdout_slice.heldout_policy,
        "sample_metadata": holdout_slice.sample_metadata,
        "slice_definition": holdout_slice.slice_definition,
        "metrics_by_model": metrics_by_model,
        "paired_deltas": paired_deltas,
        "tokenwise_primary_wins": tokenwise_primary_wins,
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
        rows.append({"row_type": "model", "label": model_name, **model_summary})

    for baseline in PAIRWISE_BASELINES:
        delta_key = f"tokenwise_mixture_minus_{baseline}"
        delta_summary: dict[str, float] = {}
        for metric in ["logit_kl_to_teacher", "nll", "perplexity", "top1_agreement", "top5_overlap"]:
            values = [seed_result["paired_deltas"][delta_key][metric] for seed_result in seed_results]
            delta_summary[f"{metric}_mean"] = _mean(values)
            delta_summary[f"{metric}_std"] = _std(values)
        win_count = sum(1 for seed_result in seed_results if seed_result["tokenwise_primary_wins"][baseline])
        summary["pairwise_deltas"][baseline] = delta_summary
        summary["pairwise_wins"][baseline] = {
            "tokenwise_mixture_wins_on_primary_metrics": win_count,
            "seeds": len(seed_results),
        }
        rows.append(
            {
                "row_type": "pairwise_delta",
                "label": f"tokenwise_mixture_minus_{baseline}",
                "tokenwise_mixture_wins_on_primary_metrics": win_count,
                **delta_summary,
            }
        )

    return summary, rows


def _write_output_probe_report(report_path: str | Path, results_payload: dict[str, Any]) -> None:
    summary = results_payload["summary"]
    slice_definition = results_payload["slice_definition"]
    lines = [
        "# Idea 4 Token-Wise Output Probe",
        "",
        "## setup",
        "",
        f"- Config: {results_payload['config_path']}",
        f"- static_stage_dir: {results_payload['static_stage_dir']}",
        f"- tokenwise_stage_dir: {results_payload['tokenwise_stage_dir']}",
        f"- seq_len: {results_payload['seq_len']}",
        f"- Seeds: {', '.join(str(seed) for seed in results_payload['seeds'])}",
        f"- Held-out policy: {results_payload['heldout_policy']}.",
        f"- Holdout definition: dataset={slice_definition['dataset_name']}, split={slice_definition['split']}, sample_count={slice_definition['sample_count']}, sampling_seed={slice_definition['sampling_seed']}.",
        "",
        "## aggregate summary",
        "",
    ]
    for model_name in MODEL_ORDER:
        row = summary["per_model"][model_name]
        lines.append(
            f"- {model_name}: "
            f"kl_mean={row['logit_kl_to_teacher_mean']:.6f}, "
            f"nll_mean={row['nll_mean']:.6f}, "
            f"ppl_mean={row['perplexity_mean']:.6f}, "
            f"top1_mean={row['top1_agreement_mean']:.6f}, "
            f"top5_mean={row['top5_overlap_mean']:.6f}"
        )
    lines.extend(
        [
            "",
            "## paired token-wise deltas",
            "",
            "- Delta sign convention: negative is better for KL/NLL/PPL, positive is better for top-1/top-5 agreement.",
        ]
    )
    for baseline in PAIRWISE_BASELINES:
        row = summary["pairwise_deltas"][baseline]
        wins = summary["pairwise_wins"][baseline]["tokenwise_mixture_wins_on_primary_metrics"]
        lines.append(
            f"- tokenwise_mixture_minus_{baseline}: "
            f"kl_delta_mean={row['logit_kl_to_teacher_mean']:.6f}, "
            f"nll_delta_mean={row['nll_mean']:.6f}, "
            f"ppl_delta_mean={row['perplexity_mean']:.6f}, "
            f"top1_delta_mean={row['top1_agreement_mean']:.6f}, "
            f"top5_delta_mean={row['top5_overlap_mean']:.6f}, "
            f"primary_wins={wins}/{results_payload['seed_count']}"
        )
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    path_specs = load_mixture_path_specs(config)
    seed_everything(config.training.seed)

    output_dir = ensure_dir(args.output_dir)
    save_config_snapshot(output_dir / "config_snapshot.yaml", config)
    export_run_metadata(
        output_dir / "metadata.json",
        config,
        {
            "stage": "idea4_tokenwise_output_probe",
            "static_stage_dir": args.static_stage_dir,
            "tokenwise_stage_dir": args.tokenwise_stage_dir,
            "seeds": args.seeds,
            "top_k": args.top_k,
            "holdout_policy": args.holdout_policy,
        },
    )

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        LOGGER.info("idea4_tokenwise_output_probe evaluating seed=%s", seed)
        seed_result = _evaluate_seed(
            config,
            backbones,
            seed,
            args.static_stage_dir,
            args.tokenwise_stage_dir,
            path_specs,
            args.top_k,
            args.holdout_policy,
        )
        seed_results.append(seed_result)
        seed_dir = ensure_dir(output_dir / f"seed_{seed}")
        save_json(seed_dir / "metrics.json", seed_result)
        save_json(seed_dir / "sample_ids.json", seed_result["sample_metadata"])
        save_json(seed_dir / "slice_definition.json", seed_result["slice_definition"])
        torch.cuda.empty_cache()

    summary, summary_rows = _aggregate_results(seed_results)
    slice_definition = seed_results[0]["slice_definition"]
    results_payload = {
        "config_path": args.config,
        "static_stage_dir": args.static_stage_dir,
        "tokenwise_stage_dir": args.tokenwise_stage_dir,
        "seq_len": config.training.seq_len,
        "top_k": args.top_k,
        "seed_count": len(args.seeds),
        "seeds": args.seeds,
        "heldout_policy": seed_results[0]["heldout_policy"],
        "slice_definition": slice_definition,
        "seed_results": seed_results,
        "summary": summary,
    }
    save_json(output_dir / "results.json", results_payload)
    save_csv(output_dir / "summary.csv", summary_rows)
    save_json(output_dir / "slice_definition.json", slice_definition)
    if args.holdout_policy == "fresh_untouched":
        save_json(output_dir / "sample_ids.json", seed_results[0]["sample_metadata"])
    ensure_dir(Path(args.results_path).parent)
    ensure_dir(Path(args.summary_path).parent)
    save_json(args.results_path, results_payload)
    save_csv(args.summary_path, summary_rows)
    _write_output_probe_report(args.report_path, results_payload)


if __name__ == "__main__":
    main()
