"""Compare frozen-entry and entry-tuned output-probe results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, save_csv, save_json


PRIMARY_METRICS = ["logit_kl_to_teacher", "nll"]
ALL_OUTPUT_METRICS = ["logit_kl_to_teacher", "nll", "perplexity", "top1_agreement", "top5_overlap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-ablation-results", default="artifacts/stage_b_ablation_output_aware_results.json")
    parser.add_argument("--tuned-ablation-results", default="artifacts/stage_b_ablation_output_aware_train_entry_results.json")
    parser.add_argument("--tuned-train-diagnostics", default="artifacts/stage_b_ablation_output_aware_train_entry_diagnostics.json")
    parser.add_argument("--frozen-probe-results", default="artifacts/stage_b_output_probe_output_aware_results.json")
    parser.add_argument("--tuned-probe-results", default="artifacts/stage_b_output_probe_output_aware_train_entry_results.json")
    parser.add_argument("--diagnostics-path", default="artifacts/stage_b_entry_tune_diagnostics.json")
    parser.add_argument("--results-path", default="artifacts/stage_b_entry_tune_output_probe_results.json")
    parser.add_argument("--summary-path", default="artifacts/stage_b_entry_tune_output_probe_summary.csv")
    parser.add_argument("--report-path", default="notes/stage_b_entry_tune_output_probe_report.md")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _probe_seed_metrics(payload: dict[str, Any], variant: str) -> dict[int, dict[str, float]]:
    metrics: dict[int, dict[str, float]] = {}
    for seed_result in payload["seed_results"]:
        metrics[int(seed_result["seed"])] = {
            metric: float(seed_result["metrics_by_model"][variant][metric]) for metric in ALL_OUTPUT_METRICS
        }
    return metrics


def _hidden_seed_metrics(payload: dict[str, Any], variant: str) -> dict[int, dict[str, float]]:
    metrics: dict[int, dict[str, float]] = {}
    for seed_result in payload["seed_results"]:
        metrics[int(seed_result["seed"])] = {
            "hidden_mse": float(seed_result["metrics"][f"{variant}_hidden_mse"]),
            "cosine": float(seed_result["metrics"][f"{variant}_cosine"]),
            "gate_value": float(seed_result["metrics"].get(f"{variant}_gate_value", 0.0)),
            "delta_norm_mean": float(seed_result["metrics"].get(f"{variant}_delta_norm_mean", 0.0)),
        }
    return metrics


def _primary_output_win(candidate: dict[str, float], reference: dict[str, float]) -> bool:
    return (
        candidate["logit_kl_to_teacher"] < reference["logit_kl_to_teacher"]
        and candidate["nll"] < reference["nll"]
    )


def _delta_summary(
    left: dict[int, dict[str, float]],
    right: dict[int, dict[str, float]],
    *,
    primary_lower_is_better: bool = True,
) -> dict[str, float]:
    seeds = sorted(set(left) & set(right))
    summary: dict[str, float] = {"seed_count": len(seeds)}
    win_count = 0
    for metric in ALL_OUTPUT_METRICS:
        deltas = [left[seed][metric] - right[seed][metric] for seed in seeds]
        summary[f"{metric}_delta_mean"] = _mean(deltas)
        summary[f"{metric}_delta_std"] = _std(deltas)
    for seed in seeds:
        if _primary_output_win(left[seed], right[seed]):
            win_count += 1
    summary["wins_on_primary_metrics"] = win_count
    return summary


def _is_clear_output_improvement(delta_summary: dict[str, float]) -> bool:
    return (
        delta_summary["wins_on_primary_metrics"] >= 2
        and delta_summary["logit_kl_to_teacher_delta_mean"] < 0.0
        and delta_summary["nll_delta_mean"] < 0.0
    )


def _gap_to_bridge_summary(
    frozen_hybrid: dict[int, dict[str, float]],
    tuned_hybrid: dict[int, dict[str, float]],
    bridge: dict[int, dict[str, float]],
) -> dict[str, float]:
    seeds = sorted(set(frozen_hybrid) & set(tuned_hybrid) & set(bridge))
    frozen_gap_kl = [frozen_hybrid[seed]["logit_kl_to_teacher"] - bridge[seed]["logit_kl_to_teacher"] for seed in seeds]
    tuned_gap_kl = [tuned_hybrid[seed]["logit_kl_to_teacher"] - bridge[seed]["logit_kl_to_teacher"] for seed in seeds]
    frozen_gap_nll = [frozen_hybrid[seed]["nll"] - bridge[seed]["nll"] for seed in seeds]
    tuned_gap_nll = [tuned_hybrid[seed]["nll"] - bridge[seed]["nll"] for seed in seeds]
    reduced_wins = sum(
        1
        for seed in seeds
        if (tuned_hybrid[seed]["logit_kl_to_teacher"] - bridge[seed]["logit_kl_to_teacher"])
        < (frozen_hybrid[seed]["logit_kl_to_teacher"] - bridge[seed]["logit_kl_to_teacher"])
        and (tuned_hybrid[seed]["nll"] - bridge[seed]["nll"])
        < (frozen_hybrid[seed]["nll"] - bridge[seed]["nll"])
    )
    return {
        "seed_count": len(seeds),
        "frozen_gap_kl_mean": _mean(frozen_gap_kl),
        "tuned_gap_kl_mean": _mean(tuned_gap_kl),
        "frozen_gap_nll_mean": _mean(frozen_gap_nll),
        "tuned_gap_nll_mean": _mean(tuned_gap_nll),
        "gap_reduced_on_both_metrics": reduced_wins,
    }


def _materially_reduces_gap(gap_summary: dict[str, float]) -> bool:
    frozen_gap_kl = gap_summary["frozen_gap_kl_mean"]
    tuned_gap_kl = gap_summary["tuned_gap_kl_mean"]
    frozen_gap_nll = gap_summary["frozen_gap_nll_mean"]
    tuned_gap_nll = gap_summary["tuned_gap_nll_mean"]
    if tuned_gap_kl >= frozen_gap_kl or tuned_gap_nll >= frozen_gap_nll:
        return False
    kl_reduction = (frozen_gap_kl - tuned_gap_kl) / max(abs(frozen_gap_kl), 1.0e-8)
    nll_reduction = (frozen_gap_nll - tuned_gap_nll) / max(abs(frozen_gap_nll), 1.0e-8)
    return gap_summary["gap_reduced_on_both_metrics"] >= 2 and kl_reduction >= 0.2 and nll_reduction >= 0.2


def _aggregate(
    frozen_ablation: dict[str, Any],
    tuned_ablation: dict[str, Any],
    tuned_train_diagnostics: dict[str, Any],
    frozen_probe: dict[str, Any],
    tuned_probe: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    frozen_probe_metrics = {variant: _probe_seed_metrics(frozen_probe, variant) for variant in ["skip_only", "hybrid_no_small", "bridge_only", "bridge_only_param_matched", "hybrid", "full_large"]}
    tuned_probe_metrics = {variant: _probe_seed_metrics(tuned_probe, variant) for variant in ["skip_only", "hybrid_no_small", "bridge_only", "bridge_only_param_matched", "hybrid", "full_large"]}
    frozen_hidden_metrics = {variant: _hidden_seed_metrics(frozen_ablation, variant) for variant in ["skip_only", "hybrid_no_small", "bridge_only", "bridge_only_param_matched", "hybrid"]}
    tuned_hidden_metrics = {variant: _hidden_seed_metrics(tuned_ablation, variant) for variant in ["skip_only", "hybrid_no_small", "hybrid"]}

    results: dict[str, Any] = {
        "seeds": tuned_probe["seeds"],
        "per_model": {
            "skip_only_reference": frozen_probe["summary"]["per_model"]["skip_only"],
            "bridge_only_reference": frozen_probe["summary"]["per_model"]["bridge_only"],
            "bridge_only_param_matched_reference": frozen_probe["summary"]["per_model"]["bridge_only_param_matched"],
            "hybrid_frozen_entry": frozen_probe["summary"]["per_model"]["hybrid"],
            "hybrid_train_entry": tuned_probe["summary"]["per_model"]["hybrid"],
            "hybrid_no_small_frozen_entry": frozen_probe["summary"]["per_model"]["hybrid_no_small"],
            "hybrid_no_small_train_entry": tuned_probe["summary"]["per_model"]["hybrid_no_small"],
            "full_large_reference": frozen_probe["summary"]["per_model"]["full_large"],
        },
        "pairwise_deltas": {},
        "bridge_gap": {},
    }
    rows: list[dict[str, Any]] = []
    for label, summary in results["per_model"].items():
        rows.append({"row_type": "model", "label": label, **summary})

    results["pairwise_deltas"]["hybrid_train_entry_minus_hybrid_frozen_entry"] = _delta_summary(
        tuned_probe_metrics["hybrid"],
        frozen_probe_metrics["hybrid"],
    )
    results["pairwise_deltas"]["hybrid_no_small_train_entry_minus_hybrid_no_small_frozen_entry"] = _delta_summary(
        tuned_probe_metrics["hybrid_no_small"],
        frozen_probe_metrics["hybrid_no_small"],
    )
    results["pairwise_deltas"]["hybrid_train_entry_minus_hybrid_no_small_train_entry"] = _delta_summary(
        tuned_probe_metrics["hybrid"],
        tuned_probe_metrics["hybrid_no_small"],
    )
    results["pairwise_deltas"]["hybrid_train_entry_minus_bridge_only"] = _delta_summary(
        tuned_probe_metrics["hybrid"],
        frozen_probe_metrics["bridge_only"],
    )
    results["pairwise_deltas"]["hybrid_train_entry_minus_bridge_only_param_matched"] = _delta_summary(
        tuned_probe_metrics["hybrid"],
        frozen_probe_metrics["bridge_only_param_matched"],
    )
    results["pairwise_deltas"]["hybrid_train_entry_minus_skip_only"] = _delta_summary(
        tuned_probe_metrics["hybrid"],
        frozen_probe_metrics["skip_only"],
    )

    for label, summary in results["pairwise_deltas"].items():
        rows.append({"row_type": "pairwise_delta", "label": label, **summary})

    results["bridge_gap"]["bridge_only"] = _gap_to_bridge_summary(
        frozen_probe_metrics["hybrid"],
        tuned_probe_metrics["hybrid"],
        frozen_probe_metrics["bridge_only"],
    )
    results["bridge_gap"]["bridge_only_param_matched"] = _gap_to_bridge_summary(
        frozen_probe_metrics["hybrid"],
        tuned_probe_metrics["hybrid"],
        frozen_probe_metrics["bridge_only_param_matched"],
    )
    for label, summary in results["bridge_gap"].items():
        rows.append({"row_type": "bridge_gap", "label": label, **summary})

    diagnostics_payload = {
        "seeds": tuned_probe["seeds"],
        "stage_b_train_entry_projector": tuned_train_diagnostics.get("train_entry_projector"),
        "stage_b_lrs": tuned_train_diagnostics.get("stage_b_lrs", {}),
        "stage_b_loss_weights": tuned_train_diagnostics.get("stage_b_loss_weights", {}),
        "entry_grad_norm_stats": {
            variant: {
                "mean_of_means": _mean(
                    [tuned_train_diagnostics["per_seed"][str(seed)]["entry_grad_norm_stats"][variant]["mean"] for seed in tuned_probe["seeds"]]
                ),
                "max_across_seeds": max(
                    tuned_train_diagnostics["per_seed"][str(seed)]["entry_grad_norm_stats"][variant]["max"] for seed in tuned_probe["seeds"]
                ),
                "final_mean": _mean(
                    [tuned_train_diagnostics["per_seed"][str(seed)]["entry_grad_norm_stats"][variant]["final"] for seed in tuned_probe["seeds"]]
                ),
            }
            for variant in ("hybrid", "hybrid_no_small")
        },
        "entry_update_norm_stats": {
            variant: {
                "mean_of_means": _mean(
                    [tuned_train_diagnostics["per_seed"][str(seed)]["entry_update_norm_stats"][variant]["mean"] for seed in tuned_probe["seeds"]]
                ),
                "max_across_seeds": max(
                    tuned_train_diagnostics["per_seed"][str(seed)]["entry_update_norm_stats"][variant]["max"] for seed in tuned_probe["seeds"]
                ),
                "final_mean": _mean(
                    [tuned_train_diagnostics["per_seed"][str(seed)]["entry_update_norm_stats"][variant]["final"] for seed in tuned_probe["seeds"]]
                ),
            }
            for variant in ("hybrid", "hybrid_no_small")
        },
        "hidden_metrics": {
            "hybrid_frozen_entry": frozen_ablation["summary"]["per_variant"]["hybrid"],
            "hybrid_train_entry": tuned_ablation["summary"]["per_variant"]["hybrid"],
            "hybrid_no_small_frozen_entry": frozen_ablation["summary"]["per_variant"]["hybrid_no_small"],
            "hybrid_no_small_train_entry": tuned_ablation["summary"]["per_variant"]["hybrid_no_small"],
            "bridge_only_reference": frozen_ablation["summary"]["per_variant"]["bridge_only"],
            "bridge_only_param_matched_reference": frozen_ablation["summary"]["per_variant"]["bridge_only_param_matched"],
        },
        "output_metrics": results["per_model"],
        "pairwise_output_deltas": results["pairwise_deltas"],
        "bridge_gap": results["bridge_gap"],
        "tuned_train_per_seed": tuned_train_diagnostics.get("per_seed", {}),
        "frozen_hidden_per_seed": frozen_hidden_metrics,
        "tuned_hidden_per_seed": tuned_hidden_metrics,
    }
    return results, rows, diagnostics_payload


def _write_report(report_path: str | Path, payload: dict[str, Any], diagnostics_payload: dict[str, Any]) -> None:
    pairwise = payload["pairwise_deltas"]
    bridge_gap = payload["bridge_gap"]

    hybrid_improves = _is_clear_output_improvement(pairwise["hybrid_train_entry_minus_hybrid_frozen_entry"])
    no_small_improves = _is_clear_output_improvement(pairwise["hybrid_no_small_train_entry_minus_hybrid_no_small_frozen_entry"])
    hybrid_beats_tuned_no_small = _is_clear_output_improvement(pairwise["hybrid_train_entry_minus_hybrid_no_small_train_entry"])
    bridge_gap_reduced = _materially_reduces_gap(bridge_gap["bridge_only"]) or _materially_reduces_gap(bridge_gap["bridge_only_param_matched"])
    proceed = hybrid_improves and hybrid_beats_tuned_no_small

    lines = [
        "# Stage B Entry-Tune Output Probe Report",
        "",
        "## setup",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in payload['seeds'])}",
        f"- Train entry projector: {diagnostics_payload['stage_b_train_entry_projector']}",
        f"- Stage B learning rates: base={diagnostics_payload['stage_b_lrs'].get('base_lr')}, entry={diagnostics_payload['stage_b_lrs'].get('entry_lr')}, return={diagnostics_payload['stage_b_lrs'].get('return_lr')}, gate={diagnostics_payload['stage_b_lrs'].get('gate_lr')}",
        "- Bridge controls were reused from the frozen-entry output-aware Stage B reference run.",
        "- Primary metrics: teacher-logit KL and held-out NLL.",
        "",
        "## aggregate output summary",
        "",
    ]

    for label, summary in payload["per_model"].items():
        lines.append(
            f"- {label}: kl_mean={summary['logit_kl_to_teacher_mean']:.6f}, nll_mean={summary['nll_mean']:.6f}, ppl_mean={summary['perplexity_mean']:.6f}, top1_mean={summary['top1_agreement_mean']:.6f}, top5_mean={summary['top5_overlap_mean']:.6f}"
        )

    lines.extend(
        [
            "",
            "## paired deltas",
            "",
            "- Delta sign convention: negative is better for KL/NLL/PPL, positive is better for top-1/top-5 agreement.",
        ]
    )
    for label, summary in pairwise.items():
        lines.append(
            f"- {label}: kl_delta_mean={summary['logit_kl_to_teacher_delta_mean']:.6f}, nll_delta_mean={summary['nll_delta_mean']:.6f}, ppl_delta_mean={summary['perplexity_delta_mean']:.6f}, top1_delta_mean={summary['top1_agreement_delta_mean']:.6f}, top5_delta_mean={summary['top5_overlap_delta_mean']:.6f}, primary_wins={summary['wins_on_primary_metrics']}/{summary['seed_count']}"
        )

    lines.extend(
        [
            "",
            "## questions",
            "",
            f"1. Does training the entry projector improve hybrid output metrics? {'Yes' if hybrid_improves else 'No'}.",
            f"2. Does training the entry projector improve hybrid_no_small output metrics? {'Yes' if no_small_improves else 'No'}.",
            f"3. When both are allowed to tune the entry projector, does hybrid still beat hybrid_no_small? {'Yes' if hybrid_beats_tuned_no_small else 'No'}.",
            f"4. Does entry tuning materially reduce the gap to bridge_only or bridge_only_param_matched? {'Yes' if bridge_gap_reduced else 'No'}. bridge_only gap changed from kl={bridge_gap['bridge_only']['frozen_gap_kl_mean']:.6f}/nll={bridge_gap['bridge_only']['frozen_gap_nll_mean']:.6f} to kl={bridge_gap['bridge_only']['tuned_gap_kl_mean']:.6f}/nll={bridge_gap['bridge_only']['tuned_gap_nll_mean']:.6f}; parameter-matched gap changed from kl={bridge_gap['bridge_only_param_matched']['frozen_gap_kl_mean']:.6f}/nll={bridge_gap['bridge_only_param_matched']['frozen_gap_nll_mean']:.6f} to kl={bridge_gap['bridge_only_param_matched']['tuned_gap_kl_mean']:.6f}/nll={bridge_gap['bridge_only_param_matched']['tuned_gap_nll_mean']:.6f}.",
            "5. Is the remaining gap now small enough to justify one tiny architecture sweep, or is the result already converging to a qualified negative against strong bridges? "
            + (
                "The remaining gap is small enough to justify one tiny architecture sweep."
                if proceed
                else "The result is still converging toward a qualified negative against strong bridges."
            ),
            "",
            "## recommendation",
            "",
            ("Proceed to one tiny architecture sweep" if proceed else "Do not proceed; write up the qualified result"),
            "",
        ]
    )

    if proceed:
        lines.extend(
            [
                "## single next architectural variable",
                "",
                "- delegated window placement",
                "",
                "- Rationale: entry tuning would have validated the interface, so the next highest-value question is whether the current delegated window is simply in the wrong large-model region rather than under-parameterized at the adapter level.",
                "",
            ]
        )

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frozen_ablation = _load_json(args.frozen_ablation_results)
    tuned_ablation = _load_json(args.tuned_ablation_results)
    tuned_train_diagnostics = _load_json(args.tuned_train_diagnostics)
    frozen_probe = _load_json(args.frozen_probe_results)
    tuned_probe = _load_json(args.tuned_probe_results)
    results_payload, rows, diagnostics_payload = _aggregate(
        frozen_ablation,
        tuned_ablation,
        tuned_train_diagnostics,
        frozen_probe,
        tuned_probe,
    )
    ensure_dir(Path(args.results_path).parent)
    ensure_dir(Path(args.summary_path).parent)
    ensure_dir(Path(args.report_path).parent)
    ensure_dir(Path(args.diagnostics_path).parent)
    save_json(args.results_path, results_payload)
    save_csv(args.summary_path, rows)
    save_json(args.diagnostics_path, diagnostics_payload)
    _write_report(args.report_path, results_payload, diagnostics_payload)


if __name__ == "__main__":
    main()
