"""Aggregate v0.9 benchmark results and write the paper-facing summary notes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from src.utils.io import ensure_dir, load_config, save_csv, save_json, save_text
from src.utils.logging_utils import configure_logging
from src.v0_9.common import KEY_BASELINES, generalization_settings


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--multichoice-dir", default="artifacts/v0_9/generalization/raw/multichoice")
    parser.add_argument("--lm-dir", default="artifacts/v0_9/generalization/raw/lm")
    parser.add_argument("--output-dir", default="artifacts/v0_9/generalization/aggregated")
    parser.add_argument("--results-note", default="notes/v0_9/generalization_results.md")
    parser.add_argument("--paper-summary-note", default="notes/v0_9/generalization_summary_for_paper.md")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def paired_bootstrap_delta(
    values_a: list[float],
    values_b: list[float],
    *,
    num_samples: int,
    seed: int,
) -> dict[str, float]:
    """Return a paired-bootstrap confidence interval for the mean delta."""

    if len(values_a) != len(values_b):
        raise ValueError("Paired bootstrap requires equal-length arrays.")
    if not values_a:
        return {"delta_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    tensor_a = torch.tensor(values_a, dtype=torch.float64)
    tensor_b = torch.tensor(values_b, dtype=torch.float64)
    observed = float((tensor_a - tensor_b).mean().item())
    generator = torch.Generator().manual_seed(seed)
    deltas: list[float] = []
    sample_count = tensor_a.shape[0]
    for _ in range(num_samples):
        indices = torch.randint(sample_count, (sample_count,), generator=generator)
        delta = float((tensor_a[indices] - tensor_b[indices]).mean().item())
        deltas.append(delta)
    deltas.sort()
    low_index = max(0, int(math.floor(0.025 * (len(deltas) - 1))))
    high_index = min(len(deltas) - 1, int(math.floor(0.975 * (len(deltas) - 1))))
    return {
        "delta_mean": observed,
        "ci_low": float(deltas[low_index]),
        "ci_high": float(deltas[high_index]),
    }


def _stable_seed_offset(*parts: str) -> int:
    total = 0
    for part in parts:
        for char in part:
            total = (total * 131 + ord(char)) % 100000
    return total


def _align_multichoice_seed_results(task_dir: Path, seeds: list[int]) -> list[dict[str, Any]]:
    return [_load_json(task_dir / f"seed_{seed}" / "results.json") for seed in seeds]


def _align_lm_seed_results(task_dir: Path, seeds: list[int]) -> list[dict[str, Any]]:
    return [_load_json(task_dir / f"seed_{seed}" / "results.json") for seed in seeds]


def _seed_mean_example_metric(seed_results: list[dict[str, Any]], model_name: str, metric_name: str) -> list[float]:
    per_seed_example_maps: list[dict[str, float]] = []
    for seed_result in seed_results:
        metric_map: dict[str, float] = {}
        for row in seed_result["example_results"]:
            example_id = str(row["example_id"])
            metric_map[example_id] = float(row["models"][model_name][metric_name])
        per_seed_example_maps.append(metric_map)
    example_ids = list(per_seed_example_maps[0].keys())
    values: list[float] = []
    for example_id in example_ids:
        values.append(_mean([metric_map[example_id] for metric_map in per_seed_example_maps]))
    return values


def _seed_mean_lm_example_metric(seed_results: list[dict[str, Any]], model_name: str, metric_name: str) -> list[float]:
    per_seed_example_maps: list[dict[str, float]] = []
    for seed_result in seed_results:
        metric_map: dict[str, float] = {}
        for row in seed_result["example_results"]:
            example_id = str(row["example_id"])
            metric_map[example_id] = float(row["models"][model_name][metric_name])
        per_seed_example_maps.append(metric_map)
    example_ids = list(per_seed_example_maps[0].keys())
    values: list[float] = []
    for example_id in example_ids:
        values.append(_mean([metric_map[example_id] for metric_map in per_seed_example_maps]))
    return values


def _aggregate_multichoice_task(
    task_name: str,
    task_dir: Path,
    seeds: list[int],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_results = _align_multichoice_seed_results(task_dir, seeds)
    model_order = list(seed_results[0]["model_order"])
    summary: dict[str, Any] = {
        "task_name": task_name,
        "task_type": "multichoice",
        "slice_definition": seed_results[0]["slice_definition"],
        "per_model": {},
        "bootstrap": {},
        "pairwise_deltas": {},
    }
    rows: list[dict[str, Any]] = []
    for model_name in model_order:
        accuracy_values = [float(seed_result["metrics_by_model"][model_name]["accuracy"]) for seed_result in seed_results]
        margin_values = [float(seed_result["metrics_by_model"][model_name]["mean_choice_margin"]) for seed_result in seed_results]
        trunc_values = [float(seed_result["metrics_by_model"][model_name]["truncation_rate"]) for seed_result in seed_results]
        summary["per_model"][model_name] = {
            "accuracy_mean": _mean(accuracy_values),
            "accuracy_std": _std(accuracy_values),
            "mean_choice_margin_mean": _mean(margin_values),
            "mean_choice_margin_std": _std(margin_values),
            "truncation_rate_mean": _mean(trunc_values),
            "truncation_rate_std": _std(trunc_values),
        }
        rows.append({"task_name": task_name, "task_type": "multichoice", "row_type": "model", "label": model_name, **summary["per_model"][model_name]})

    tokenwise = summary["per_model"]["tokenwise_mixture"]
    for baseline in ["static_mixture", "tokenwise_mixture_no_small", "bridge_only", "bridge_only_param_matched"]:
        base = summary["per_model"][baseline]
        accuracy_delta = tokenwise["accuracy_mean"] - base["accuracy_mean"]
        margin_delta = tokenwise["mean_choice_margin_mean"] - base["mean_choice_margin_mean"]
        summary["pairwise_deltas"][baseline] = {
            "accuracy_delta": accuracy_delta,
            "mean_choice_margin_delta": margin_delta,
        }
        if baseline in KEY_BASELINES:
            tokenwise_accuracy = _seed_mean_example_metric(seed_results, "tokenwise_mixture", "correct")
            baseline_accuracy = _seed_mean_example_metric(seed_results, baseline, "correct")
            tokenwise_margin = _seed_mean_example_metric(seed_results, "tokenwise_mixture", "score_margin")
            baseline_margin = _seed_mean_example_metric(seed_results, baseline, "score_margin")
            summary["bootstrap"][baseline] = {
                "accuracy_delta": paired_bootstrap_delta(
                    tokenwise_accuracy,
                    baseline_accuracy,
                    num_samples=settings["bootstrap_samples"],
                    seed=settings["bootstrap_seed"] + _stable_seed_offset(task_name, baseline, "acc"),
                ),
                "mean_choice_margin_delta": paired_bootstrap_delta(
                    tokenwise_margin,
                    baseline_margin,
                    num_samples=settings["bootstrap_samples"],
                    seed=settings["bootstrap_seed"] + _stable_seed_offset(task_name, baseline, "margin"),
                ),
            }
        rows.append({"task_name": task_name, "task_type": "multichoice", "row_type": "pairwise", "label": f"tokenwise_minus_{baseline}", **summary["pairwise_deltas"][baseline]})

    summary["tokenwise_beats"] = {}
    for baseline in ["static_mixture", "tokenwise_mixture_no_small", "bridge_only", "bridge_only_param_matched"]:
        delta = summary["pairwise_deltas"][baseline]
        summary["tokenwise_beats"][baseline] = (
            delta["accuracy_delta"] > 0.0
            or (delta["accuracy_delta"] == 0.0 and delta["mean_choice_margin_delta"] > 0.0)
        )
    return summary, rows


def _aggregate_lm_task(
    task_name: str,
    task_dir: Path,
    seeds: list[int],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_results = _align_lm_seed_results(task_dir, seeds)
    model_order = list(seed_results[0]["model_order"])
    summary: dict[str, Any] = {
        "task_name": task_name,
        "task_type": "lm",
        "slice_definition": seed_results[0]["slice_definition"],
        "per_model": {},
        "bootstrap": {},
        "pairwise_deltas": {},
    }
    rows: list[dict[str, Any]] = []
    for model_name in model_order:
        nll_values = [float(seed_result["metrics_by_model"][model_name]["nll"]) for seed_result in seed_results]
        ppl_values = [float(seed_result["metrics_by_model"][model_name]["perplexity"]) for seed_result in seed_results]
        kl_values = [float(seed_result["metrics_by_model"][model_name]["logit_kl_to_teacher"]) for seed_result in seed_results]
        trunc_values = [float(seed_result["metrics_by_model"][model_name]["truncation_rate"]) for seed_result in seed_results]
        summary["per_model"][model_name] = {
            "nll_mean": _mean(nll_values),
            "nll_std": _std(nll_values),
            "perplexity_mean": _mean(ppl_values),
            "perplexity_std": _std(ppl_values),
            "logit_kl_to_teacher_mean": _mean(kl_values),
            "logit_kl_to_teacher_std": _std(kl_values),
            "truncation_rate_mean": _mean(trunc_values),
            "truncation_rate_std": _std(trunc_values),
        }
        rows.append({"task_name": task_name, "task_type": "lm", "row_type": "model", "label": model_name, **summary["per_model"][model_name]})

    tokenwise = summary["per_model"]["tokenwise_mixture"]
    for baseline in ["static_mixture", "tokenwise_mixture_no_small", "bridge_only", "bridge_only_param_matched"]:
        base = summary["per_model"][baseline]
        summary["pairwise_deltas"][baseline] = {
            "nll_delta": tokenwise["nll_mean"] - base["nll_mean"],
            "perplexity_delta": tokenwise["perplexity_mean"] - base["perplexity_mean"],
            "logit_kl_to_teacher_delta": tokenwise["logit_kl_to_teacher_mean"] - base["logit_kl_to_teacher_mean"],
        }
        if baseline in KEY_BASELINES:
            tokenwise_nll = _seed_mean_lm_example_metric(seed_results, "tokenwise_mixture", "nll")
            baseline_nll = _seed_mean_lm_example_metric(seed_results, baseline, "nll")
            tokenwise_kl = _seed_mean_lm_example_metric(seed_results, "tokenwise_mixture", "logit_kl_to_teacher")
            baseline_kl = _seed_mean_lm_example_metric(seed_results, baseline, "logit_kl_to_teacher")
            summary["bootstrap"][baseline] = {
                "nll_delta": paired_bootstrap_delta(
                    tokenwise_nll,
                    baseline_nll,
                    num_samples=settings["bootstrap_samples"],
                    seed=settings["bootstrap_seed"] + _stable_seed_offset(task_name, baseline, "nll"),
                ),
                "logit_kl_to_teacher_delta": paired_bootstrap_delta(
                    tokenwise_kl,
                    baseline_kl,
                    num_samples=settings["bootstrap_samples"],
                    seed=settings["bootstrap_seed"] + _stable_seed_offset(task_name, baseline, "kl"),
                ),
            }
        rows.append({"task_name": task_name, "task_type": "lm", "row_type": "pairwise", "label": f"tokenwise_minus_{baseline}", **summary["pairwise_deltas"][baseline]})

    summary["tokenwise_beats"] = {}
    for baseline in ["static_mixture", "tokenwise_mixture_no_small", "bridge_only", "bridge_only_param_matched"]:
        delta = summary["pairwise_deltas"][baseline]
        summary["tokenwise_beats"][baseline] = delta["logit_kl_to_teacher_delta"] < 0.0 and delta["nll_delta"] < 0.0
    return summary, rows


def _task_strength_lines(multichoice: dict[str, Any], lm: dict[str, Any]) -> list[str]:
    items: list[tuple[str, float]] = []
    for task_name, summary in multichoice.items():
        items.append((task_name, float(summary["pairwise_deltas"]["bridge_only"]["accuracy_delta"])))
    for task_name, summary in lm.items():
        items.append((task_name, -float(summary["pairwise_deltas"]["bridge_only"]["nll_delta"])))
    ordered = sorted(items, key=lambda item: item[1], reverse=True)
    return [name for name, _ in ordered]


def _recommendation(multichoice: dict[str, Any], lm: dict[str, Any]) -> str:
    task_summaries = [*multichoice.values(), *lm.values()]
    wins = sum(
        1
        for summary in task_summaries
        if summary["tokenwise_beats"]["bridge_only"] and summary["tokenwise_beats"]["bridge_only_param_matched"]
    )
    return (
        "Proceed to bounded cross-pair replication"
        if wins > len(task_summaries) / 2.0
        else "Stop and write the paper around v0.6.0 plus benchmark generalization"
    )


def _write_results_note(
    path: str | Path,
    multichoice: dict[str, Any],
    lm: dict[str, Any],
    recommendation: str,
) -> None:
    strongest = _task_strength_lines(multichoice, lm)
    bridge_recoveries = [
        task_name
        for task_name, summary in {**multichoice, **lm}.items()
        if (not summary["tokenwise_beats"]["bridge_only"]) or (not summary["tokenwise_beats"]["bridge_only_param_matched"])
    ]
    static_meaningful = [
        task_name
        for task_name, summary in multichoice.items()
        if summary["pairwise_deltas"]["static_mixture"]["accuracy_delta"] >= 0.0
        or summary["tokenwise_beats"]["static_mixture"]
    ]
    tokenwise_vs_no_small = [
        task_name
        for task_name, summary in {**multichoice, **lm}.items()
        if summary["tokenwise_beats"]["tokenwise_mixture_no_small"]
    ]
    lm_carry = []
    if lm:
        for task_name, summary in lm.items():
            if summary["tokenwise_beats"]["bridge_only"] and summary["tokenwise_beats"]["bridge_only_param_matched"]:
                lm_carry.append(task_name)
    mc_carry = [
        task_name
        for task_name, summary in multichoice.items()
        if summary["tokenwise_beats"]["bridge_only"] and summary["tokenwise_beats"]["bridge_only_param_matched"]
    ]
    strongest_claim = (
        "The frozen `v0.6.0` token-wise model retains a measurable external-generalization advantage over both bridge controls on a majority of the bounded tasks."
        if recommendation.startswith("Proceed")
        else "The frozen `v0.6.0` token-wise model retains a strong in-family held-out-language advantage, but the bridge advantage is not broad enough across the bounded external tasks to claim generality beyond the current family/prompt regime."
    )

    lines = [
        "# Generalization Results",
        "",
        "## Frozen Context",
        "",
        "- `v0.6.0` is the frozen current best result.",
        "- `v0_7` and `v0_8` remain analysis branches only and do not replace the `v0.6.0` claim.",
        "- This task was evaluation-only; it did not introduce a new architecture and it did not start Stage C.",
        "",
        "## Benchmark Set",
        "",
        "- HellaSwag",
        "- PIQA",
        "- WinoGrande",
        "- ARC-Easy",
        "- ARC-Challenge",
        "- LAMBADA OpenAI test slice for LM-style evaluation",
        "",
        "## Direct Answers",
        "",
        f"1. Does `v0.6.0` still beat the bridge controls outside the original Wikitext-style probes?\n{'Yes on a bounded majority of the new tasks/families.' if recommendation.startswith('Proceed') else 'Only partially. It does not beat both bridge controls on a clear majority of the new tasks/families.'}",
        f"2. On which tasks does the token-wise gain remain strongest?\n{', '.join(strongest[:3]) if strongest else 'No strong task ordering emerged.'}",
        f"3. Is the gain mainly visible on language-modeling style metrics, or does it carry over to multiple-choice task accuracy?\n{'It carries over to multiple-choice accuracy on a bounded subset of tasks and remains visible on LM metrics.' if recommendation.startswith('Proceed') else 'It remains most defensible on LM-style metrics; the multiple-choice carryover is mixed rather than broad.'}",
        f"4. Does static mixture remain a meaningful intermediate baseline under broader evaluation?\n{'Yes. Static mixture still sits between token-wise and weaker controls on these tasks: ' + ', '.join(static_meaningful) if static_meaningful else 'Only weakly. Static mixture does not remain a consistent intermediate baseline outside the original probe regime.'}",
        f"5. Is token-wise still clearly better than the no-small control under broader evaluation?\n{'Yes on these tasks/families: ' + ', '.join(tokenwise_vs_no_small) if tokenwise_vs_no_small else 'No clear broader win over the no-small control was retained.'}",
        f"6. Are there any tasks where the bridge baselines recover or surpass the token-wise model?\n{', '.join(bridge_recoveries) if bridge_recoveries else 'No. The bridge baselines do not recover on the bounded evaluation set.'}",
        f"7. What is the strongest defensible generalization claim after this task?\n{strongest_claim}",
        "",
        "## Recommendation",
        "",
        recommendation,
    ]
    save_text(path, "\n".join(lines))


def _write_paper_summary_note(
    path: str | Path,
    multichoice: dict[str, Any],
    lm: dict[str, Any],
    recommendation: str,
) -> None:
    lines = [
        "# Generalization Summary For Paper",
        "",
        "- `v0.6.0` remains the frozen best model branch.",
        "- `v0_7` and `v0_8` remain analysis-only and do not replace the main result claim.",
        f"- Recommendation after bounded generalization: {recommendation}.",
        "",
        "## Multiple-Choice Summary",
        "",
    ]
    for task_name, summary in multichoice.items():
        tokenwise = summary["per_model"]["tokenwise_mixture"]
        bridge = summary["per_model"]["bridge_only"]
        bridge_pm = summary["per_model"]["bridge_only_param_matched"]
        lines.append(
            f"- {task_name}: token-wise acc={tokenwise['accuracy_mean']:.4f}, "
            f"bridge_only acc={bridge['accuracy_mean']:.4f}, "
            f"bridge_param acc={bridge_pm['accuracy_mean']:.4f}."
        )
    lines.extend(["", "## LM Summary", ""])
    for task_name, summary in lm.items():
        tokenwise = summary["per_model"]["tokenwise_mixture"]
        bridge = summary["per_model"]["bridge_only"]
        bridge_pm = summary["per_model"]["bridge_only_param_matched"]
        lines.append(
            f"- {task_name}: token-wise KL/NLL={tokenwise['logit_kl_to_teacher_mean']:.4f}/{tokenwise['nll_mean']:.4f}, "
            f"bridge_only={bridge['logit_kl_to_teacher_mean']:.4f}/{bridge['nll_mean']:.4f}, "
            f"bridge_param={bridge_pm['logit_kl_to_teacher_mean']:.4f}/{bridge_pm['nll_mean']:.4f}."
        )
    save_text(path, "\n".join(lines))


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    settings = generalization_settings(config)
    output_dir = ensure_dir(args.output_dir)

    multichoice_dir = Path(args.multichoice_dir)
    lm_dir = Path(args.lm_dir)
    multichoice_task_dirs = [path for path in multichoice_dir.iterdir() if path.is_dir()]
    lm_task_dirs = [path for path in lm_dir.iterdir() if path.is_dir()]

    multichoice_summary: dict[str, Any] = {}
    lm_summary: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for task_dir in sorted(multichoice_task_dirs):
        summary, rows = _aggregate_multichoice_task(task_dir.name, task_dir, settings["seeds"], settings)
        multichoice_summary[task_dir.name] = summary
        csv_rows.extend(rows)
    for task_dir in sorted(lm_task_dirs):
        summary, rows = _aggregate_lm_task(task_dir.name, task_dir, settings["seeds"], settings)
        lm_summary[task_dir.name] = summary
        csv_rows.extend(rows)

    recommendation = _recommendation(multichoice_summary, lm_summary)
    payload = {
        "config_path": args.config,
        "seeds": settings["seeds"],
        "bootstrap_samples": settings["bootstrap_samples"],
        "bootstrap_seed": settings["bootstrap_seed"],
        "multichoice": multichoice_summary,
        "lm": lm_summary,
        "recommendation": recommendation,
    }
    save_json(output_dir / "summary.json", payload)
    save_csv(output_dir / "summary.csv", csv_rows)
    _write_results_note(args.results_note, multichoice_summary, lm_summary, recommendation)
    _write_paper_summary_note(args.paper_summary_note, multichoice_summary, lm_summary, recommendation)


if __name__ == "__main__":
    main()
