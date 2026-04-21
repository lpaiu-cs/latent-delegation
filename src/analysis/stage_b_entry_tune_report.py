"""Compare frozen-entry and entry-tuned Stage B runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, save_csv, save_json


REFERENCE_VARIANTS = ["skip_only", "bridge_only", "bridge_only_param_matched"]
COMPARE_VARIANTS = ["hybrid", "hybrid_no_small"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-results", default="artifacts/stage_b_ablation_output_aware_results.json")
    parser.add_argument("--tuned-results", default="artifacts/stage_b_ablation_output_aware_train_entry_results.json")
    parser.add_argument("--tuned-diagnostics", default="artifacts/stage_b_ablation_output_aware_train_entry_diagnostics.json")
    parser.add_argument("--results-path", default="artifacts/stage_b_entry_tune_results.json")
    parser.add_argument("--summary-path", default="artifacts/stage_b_entry_tune_summary.csv")
    parser.add_argument("--report-path", default="notes/stage_b_entry_tune_report.md")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _seed_metric_by_variant(payload: dict[str, Any], variant: str, metric_key: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for seed_result in payload["seed_results"]:
        values[int(seed_result["seed"])] = float(seed_result["metrics"][f"{variant}_{metric_key}"])
    return values


def _hidden_delta_summary(
    frozen_payload: dict[str, Any],
    tuned_payload: dict[str, Any],
    variant: str,
) -> dict[str, float]:
    frozen_mse = _seed_metric_by_variant(frozen_payload, variant, "hidden_mse")
    frozen_cosine = _seed_metric_by_variant(frozen_payload, variant, "cosine")
    tuned_mse = _seed_metric_by_variant(tuned_payload, variant, "hidden_mse")
    tuned_cosine = _seed_metric_by_variant(tuned_payload, variant, "cosine")
    seeds = sorted(set(frozen_mse) & set(tuned_mse))
    mse_deltas = [tuned_mse[seed] - frozen_mse[seed] for seed in seeds]
    cosine_deltas = [tuned_cosine[seed] - frozen_cosine[seed] for seed in seeds]
    wins = sum(1 for seed in seeds if tuned_mse[seed] < frozen_mse[seed] and tuned_cosine[seed] > frozen_cosine[seed])
    return {
        "mse_delta_mean": _mean(mse_deltas),
        "mse_delta_std": _std(mse_deltas),
        "cosine_delta_mean": _mean(cosine_deltas),
        "cosine_delta_std": _std(cosine_deltas),
        "wins_on_both_metrics": wins,
        "seed_count": len(seeds),
    }


def _bridge_gap_summary(
    frozen_payload: dict[str, Any],
    tuned_payload: dict[str, Any],
    bridge_variant: str,
) -> dict[str, float]:
    frozen_hybrid_mse = _seed_metric_by_variant(frozen_payload, "hybrid", "hidden_mse")
    frozen_hybrid_cosine = _seed_metric_by_variant(frozen_payload, "hybrid", "cosine")
    tuned_hybrid_mse = _seed_metric_by_variant(tuned_payload, "hybrid", "hidden_mse")
    tuned_hybrid_cosine = _seed_metric_by_variant(tuned_payload, "hybrid", "cosine")
    bridge_mse = _seed_metric_by_variant(frozen_payload, bridge_variant, "hidden_mse")
    bridge_cosine = _seed_metric_by_variant(frozen_payload, bridge_variant, "cosine")
    seeds = sorted(set(frozen_hybrid_mse) & set(tuned_hybrid_mse) & set(bridge_mse))

    frozen_gap_mse = [frozen_hybrid_mse[seed] - bridge_mse[seed] for seed in seeds]
    tuned_gap_mse = [tuned_hybrid_mse[seed] - bridge_mse[seed] for seed in seeds]
    frozen_gap_cosine = [bridge_cosine[seed] - frozen_hybrid_cosine[seed] for seed in seeds]
    tuned_gap_cosine = [bridge_cosine[seed] - tuned_hybrid_cosine[seed] for seed in seeds]
    reduced_wins = sum(
        1
        for seed in seeds
        if (tuned_hybrid_mse[seed] - bridge_mse[seed]) < (frozen_hybrid_mse[seed] - bridge_mse[seed])
        and (bridge_cosine[seed] - tuned_hybrid_cosine[seed]) < (bridge_cosine[seed] - frozen_hybrid_cosine[seed])
    )
    return {
        "frozen_gap_mse_mean": _mean(frozen_gap_mse),
        "tuned_gap_mse_mean": _mean(tuned_gap_mse),
        "frozen_gap_cosine_mean": _mean(frozen_gap_cosine),
        "tuned_gap_cosine_mean": _mean(tuned_gap_cosine),
        "gap_reduced_on_both_metrics": reduced_wins,
        "seed_count": len(seeds),
    }


def _aggregate_payload(
    frozen_payload: dict[str, Any],
    tuned_payload: dict[str, Any],
    tuned_diagnostics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    results = {
        "reference_paths": {
            "frozen_results": frozen_payload["config_path"],
            "tuned_results": tuned_payload["config_path"],
        },
        "seeds": tuned_payload["seeds"],
        "per_variant": {},
        "pairwise_deltas": {},
        "bridge_gap": {},
        "entry_diagnostics": tuned_diagnostics,
    }

    for variant in REFERENCE_VARIANTS:
        summary = frozen_payload["summary"]["per_variant"][variant]
        label = f"{variant}_reference"
        results["per_variant"][label] = summary
        rows.append({"row_type": "variant", "label": label, **summary})

    for variant in COMPARE_VARIANTS:
        frozen_summary = frozen_payload["summary"]["per_variant"][variant]
        tuned_summary = tuned_payload["summary"]["per_variant"][variant]
        results["per_variant"][f"{variant}_frozen_entry"] = frozen_summary
        results["per_variant"][f"{variant}_train_entry"] = tuned_summary
        rows.append({"row_type": "variant", "label": f"{variant}_frozen_entry", **frozen_summary})
        rows.append({"row_type": "variant", "label": f"{variant}_train_entry", **tuned_summary})

        delta_summary = _hidden_delta_summary(frozen_payload, tuned_payload, variant)
        results["pairwise_deltas"][f"{variant}_train_entry_minus_frozen_entry"] = delta_summary
        rows.append({"row_type": "pairwise_delta", "label": f"{variant}_train_entry_minus_frozen_entry", **delta_summary})

    for bridge_variant in ("bridge_only", "bridge_only_param_matched"):
        gap_summary = _bridge_gap_summary(frozen_payload, tuned_payload, bridge_variant)
        results["bridge_gap"][bridge_variant] = gap_summary
        rows.append({"row_type": "bridge_gap", "label": bridge_variant, **gap_summary})

    return results, rows


def _write_report(report_path: str | Path, payload: dict[str, Any]) -> None:
    pairwise = payload["pairwise_deltas"]
    bridge_gap = payload["bridge_gap"]
    entry_diag = payload["entry_diagnostics"]
    lrs = entry_diag.get("stage_b_lrs", {})
    variants = entry_diag.get("variants", [])

    lines = [
        "# Stage B Entry-Tune Report",
        "",
        "## setup",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in payload['seeds'])}",
        f"- Tuned variants: {', '.join(variants)}",
        f"- Train entry projector: {entry_diag.get('train_entry_projector')}",
        f"- Stage B learning rates: base={lrs.get('base_lr')}, entry={lrs.get('entry_lr')}, return={lrs.get('return_lr')}, gate={lrs.get('gate_lr')}",
        "- Reference bridge results were reused from the frozen-entry output-aware Stage B run.",
        "",
        "## aggregate hidden summary",
        "",
    ]

    for label, summary in payload["per_variant"].items():
        lines.append(
            f"- {label}: hidden_mse_mean={summary['hidden_mse_mean']:.6f}, cosine_mean={summary['cosine_mean']:.6f}, "
            f"gate_value_mean={summary.get('gate_value_mean', 0.0):.6f}, delta_norm_mean={summary.get('delta_norm_mean', 0.0):.6f}"
        )

    lines.extend(
        [
            "",
            "## entry diagnostics",
            "",
        ]
    )
    for variant in ("hybrid", "hybrid_no_small"):
        stats = entry_diag["per_seed"][str(payload["seeds"][0])]["entry_grad_norm_stats"].get(variant) if entry_diag.get("per_seed") else None
        if stats is None:
            continue
        grad_means = [entry_diag["per_seed"][str(seed)]["entry_grad_norm_stats"][variant]["mean"] for seed in payload["seeds"]]
        update_finals = [entry_diag["per_seed"][str(seed)]["entry_update_norm_stats"][variant]["final"] for seed in payload["seeds"]]
        lines.append(
            f"- {variant}: entry_grad_norm_mean={_mean(grad_means):.6f}, entry_grad_norm_std={_std(grad_means):.6f}, "
            f"final_entry_update_norm_mean={_mean(update_finals):.6f}, final_entry_update_norm_std={_std(update_finals):.6f}"
        )

    lines.extend(
        [
            "",
            "## interpretation",
            "",
            f"- hybrid train-entry vs frozen-entry: mse_delta_mean={pairwise['hybrid_train_entry_minus_frozen_entry']['mse_delta_mean']:.6f}, cosine_delta_mean={pairwise['hybrid_train_entry_minus_frozen_entry']['cosine_delta_mean']:.6f}, wins_on_both={pairwise['hybrid_train_entry_minus_frozen_entry']['wins_on_both_metrics']}/{pairwise['hybrid_train_entry_minus_frozen_entry']['seed_count']}",
            f"- hybrid_no_small train-entry vs frozen-entry: mse_delta_mean={pairwise['hybrid_no_small_train_entry_minus_frozen_entry']['mse_delta_mean']:.6f}, cosine_delta_mean={pairwise['hybrid_no_small_train_entry_minus_frozen_entry']['cosine_delta_mean']:.6f}, wins_on_both={pairwise['hybrid_no_small_train_entry_minus_frozen_entry']['wins_on_both_metrics']}/{pairwise['hybrid_no_small_train_entry_minus_frozen_entry']['seed_count']}",
            f"- hybrid gap to bridge_only: frozen_gap_mse_mean={bridge_gap['bridge_only']['frozen_gap_mse_mean']:.6f}, tuned_gap_mse_mean={bridge_gap['bridge_only']['tuned_gap_mse_mean']:.6f}, frozen_gap_cosine_mean={bridge_gap['bridge_only']['frozen_gap_cosine_mean']:.6f}, tuned_gap_cosine_mean={bridge_gap['bridge_only']['tuned_gap_cosine_mean']:.6f}",
            f"- hybrid gap to bridge_only_param_matched: frozen_gap_mse_mean={bridge_gap['bridge_only_param_matched']['frozen_gap_mse_mean']:.6f}, tuned_gap_mse_mean={bridge_gap['bridge_only_param_matched']['tuned_gap_mse_mean']:.6f}, frozen_gap_cosine_mean={bridge_gap['bridge_only_param_matched']['frozen_gap_cosine_mean']:.6f}, tuned_gap_cosine_mean={bridge_gap['bridge_only_param_matched']['tuned_gap_cosine_mean']:.6f}",
            "- Output-level interpretation is deferred to the dedicated output-probe comparison report.",
            "",
        ]
    )

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frozen_payload = _load_json(args.frozen_results)
    tuned_payload = _load_json(args.tuned_results)
    tuned_diagnostics = _load_json(args.tuned_diagnostics)
    results_payload, rows = _aggregate_payload(frozen_payload, tuned_payload, tuned_diagnostics)
    ensure_dir(Path(args.results_path).parent)
    ensure_dir(Path(args.summary_path).parent)
    ensure_dir(Path(args.report_path).parent)
    save_json(args.results_path, results_payload)
    save_csv(args.summary_path, rows)
    _write_report(args.report_path, results_payload)


if __name__ == "__main__":
    main()
