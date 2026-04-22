"""Build paper-facing tables and figure specs from frozen repo artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.utils.io import ensure_dir, git_commit_hash, save_csv, save_json, save_text


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TableArtifact:
    """A machine-readable paper table plus its markdown rendering metadata."""

    slug: str
    title: str
    columns: list[str]
    rows: list[dict[str, Any]]
    notes: list[str]
    source_paths: list[str]


@dataclass(frozen=True)
class FigureArtifact:
    """A figure-ready specification and its markdown-facing description."""

    slug: str
    title: str
    spec: dict[str, Any]
    notes: list[str]
    source_paths: list[str]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tables-dir", default="artifacts/paper_tables")
    parser.add_argument("--figures-dir", default="artifacts/paper_figures")
    parser.add_argument("--tables-note", default="notes/paper/tables.md")
    parser.add_argument("--figures-note", default="notes/paper/figures.md")
    return parser.parse_args()


def _resolve(root: Path, relative_path: str) -> Path:
    return root / relative_path.replace("/", "\\")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_json(root: Path, relative_path: str) -> dict[str, Any]:
    return json.loads(_resolve(root, relative_path).read_text(encoding="utf-8"))


def _load_yaml(root: Path, relative_path: str) -> dict[str, Any]:
    return yaml.safe_load(_resolve(root, relative_path).read_text(encoding="utf-8"))


def _config_seed(root: Path, config_path: str) -> int | None:
    config = _load_yaml(root, config_path)
    training = config.get("training", {})
    seed = training.get("seed")
    return int(seed) if seed is not None else None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        text = f"{value:.6f}"
        return text.rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _render_markdown_table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def _table_note(table: TableArtifact) -> str:
    lines = [
        f"## {table.title}",
        "",
        *[f"- {note}" for note in table.notes],
        f"- Machine-readable files: `{table.slug}.csv`, `{table.slug}.json`.",
        f"- Source artifacts: {', '.join(f'`{path}`' for path in table.source_paths)}.",
        "",
        _render_markdown_table(table.columns, table.rows),
        "",
    ]
    return "\n".join(lines)


def _figure_note(figure: FigureArtifact) -> str:
    lines = [
        f"## {figure.title}",
        "",
        *[f"- {note}" for note in figure.notes],
        f"- Spec file: `{figure.slug}.json`.",
        f"- Source artifacts: {', '.join(f'`{path}`' for path in figure.source_paths)}.",
        "",
    ]
    preview_rows = figure.spec.get("preview_rows")
    if preview_rows:
        columns = list(preview_rows[0].keys())
        lines.extend(
            [
                _render_markdown_table(columns, preview_rows),
                "",
            ]
        )
    return "\n".join(lines)


def build_bring_up_smoke_table(root: Path) -> TableArtifact:
    """Build the bring-up and smoke summary table."""

    env = _load_json(root, "artifacts/env_sanity.json")
    smoke = _load_json(root, "artifacts/real_gemma_smoke.json")
    stage_a = _load_json(root, "artifacts/stage_a_pilot_metrics.json")

    rows: list[dict[str, Any]] = [
        {
            "section": "environment",
            "compared_model": "all",
            "seed_policy": "not_applicable",
            "holdout_policy": "not_applicable",
            "overall_pass": env["overall_pass"],
            "cuda_available": env["cuda_available"],
            "hf_auth_token_present": env["hf_auth"]["token_present"],
            "gemma_access_success": env["gemma_access"]["success"],
            "bitsandbytes_available": env["bitsandbytes"]["available"],
            "device_name": env["device_name"],
            "total_vram_gb": env["total_vram_gb"],
            "python_version": env["python_version"],
            "torch_version": env["torch_version"],
        },
        {
            "section": "smoke_matrix",
            "compared_model": "all",
            "seed_policy": "not_applicable",
            "holdout_policy": "not_applicable",
            "overall_success": smoke["overall_success"],
            "completed_cases": smoke["completed_cases"],
            "expected_cases": smoke["expected_cases"],
        },
        {
            "section": "stage_a_pilot",
            "compared_model": "entry_projector",
            "seed_policy": _config_seed(root, stage_a["config_path"]),
            "holdout_policy": "alignment_validation_subset",
            "seq_len": stage_a["seq_len"],
            "train_loss_start": stage_a["train_loss_start"],
            "train_loss_end": stage_a["train_loss_end"],
            "heldout_mse_before": stage_a["heldout_mse_before"],
            "heldout_mse_after": stage_a["heldout_mse_after"],
            "heldout_cosine_before": stage_a["heldout_cosine_before"],
            "heldout_cosine_after": stage_a["heldout_cosine_after"],
            "trainable_parameters": stage_a["trainable_parameters"],
        },
    ]

    case_to_model = {
        "load_small_only": "small_only_load",
        "load_large_only": "large_only_load",
        "full_large_forward": "full_large",
        "skip_only_forward": "skip_only",
        "bridge_only_forward": "bridge_only",
        "hybrid_forward": "hybrid",
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in smoke["results"]:
        grouped.setdefault(result["case"], []).append(result)
    for case_name, model_name in case_to_model.items():
        case_rows = grouped.get(case_name, [])
        if not case_rows:
            continue
        best_row = max(case_rows, key=lambda row: -1 if row["seq_len"] is None else int(row["seq_len"]))
        rows.append(
            {
                "section": "smoke_case",
                "compared_model": model_name,
                "seed_policy": "not_applicable",
                "holdout_policy": "synthetic_forward_smoke",
                "success": best_row["success"],
                "largest_successful_seq_len": best_row["seq_len"],
                "peak_vram_mb": best_row["peak_vram_mb"],
                "wall_time_sec": best_row["wall_time_sec"],
                "device": best_row["device"],
            }
        )

    columns = [
        "section",
        "compared_model",
        "seed_policy",
        "holdout_policy",
        "overall_pass",
        "overall_success",
        "completed_cases",
        "expected_cases",
        "success",
        "largest_successful_seq_len",
        "peak_vram_mb",
        "wall_time_sec",
        "cuda_available",
        "hf_auth_token_present",
        "gemma_access_success",
        "bitsandbytes_available",
        "device_name",
        "total_vram_gb",
        "python_version",
        "torch_version",
        "seq_len",
        "train_loss_start",
        "train_loss_end",
        "heldout_mse_before",
        "heldout_mse_after",
        "heldout_cosine_before",
        "heldout_cosine_after",
        "trainable_parameters",
        "device",
    ]
    return TableArtifact(
        slug="table_01_bring_up_smoke_summary",
        title="Table 1. Bring-Up / Smoke Summary",
        columns=columns,
        rows=rows,
        notes=[
            "Exact metric names are preserved in the machine-readable CSV/JSON.",
            "The Stage A row is included here because it was part of the initial bring-up feasibility path.",
        ],
        source_paths=[
            "artifacts/env_sanity.json",
            "artifacts/real_gemma_smoke.json",
            "artifacts/stage_a_pilot_metrics.json",
        ],
    )


def _merge_stage_b_rows(
    *,
    subphase: str,
    hidden_summary: dict[str, Any],
    output_summary: dict[str, Any],
    seeds: list[int],
    holdout_policy: str,
    source_path_hidden: str,
    source_path_output: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, output_metrics in output_summary["per_model"].items():
        hidden_metrics = hidden_summary.get(model_name, {})
        rows.append(
            {
                "subphase": subphase,
                "compared_model": model_name,
                "seed_policy": ",".join(str(seed) for seed in seeds),
                "holdout_policy": holdout_policy,
                "hidden_mse_mean": hidden_metrics.get("hidden_mse_mean"),
                "hidden_cosine_mean": hidden_metrics.get("cosine_mean"),
                "delta_norm_mean": hidden_metrics.get("delta_norm_mean"),
                "gate_value_mean": hidden_metrics.get("gate_value_mean"),
                "logit_kl_to_teacher_mean": output_metrics.get("logit_kl_to_teacher_mean"),
                "nll_mean": output_metrics.get("nll_mean"),
                "perplexity_mean": output_metrics.get("perplexity_mean"),
                "top1_agreement_mean": output_metrics.get("top1_agreement_mean"),
                "top5_overlap_mean": output_metrics.get("top5_overlap_mean"),
                "source_hidden": source_path_hidden,
                "source_output": source_path_output,
            }
        )
    return rows


def build_v05_key_ablation_table(root: Path) -> TableArtifact:
    """Build the v0.5.x milestone summary table."""

    hidden_only_hidden = _load_json(root, "artifacts/stage_b_ablation_results.json")
    hidden_only_output = _load_json(root, "artifacts/stage_b_output_probe_results.json")
    output_aware_hidden = _load_json(root, "artifacts/stage_b_ablation_output_aware_results.json")
    output_aware_output = _load_json(root, "artifacts/stage_b_output_probe_output_aware_results.json")
    entry_tune_hidden = _load_json(root, "artifacts/stage_b_entry_tune_results.json")
    entry_tune_output = _load_json(root, "artifacts/stage_b_entry_tune_output_probe_results.json")

    rows = []
    rows.extend(
        _merge_stage_b_rows(
            subphase="stage_b_hidden_only",
            hidden_summary=hidden_only_hidden["summary"]["per_variant"],
            output_summary=hidden_only_output["summary"],
            seeds=hidden_only_output["seeds"],
            holdout_policy=hidden_only_output["heldout_policy"],
            source_path_hidden="artifacts/stage_b_ablation_results.json",
            source_path_output="artifacts/stage_b_output_probe_results.json",
        )
    )
    rows.extend(
        _merge_stage_b_rows(
            subphase="stage_b_output_aware",
            hidden_summary=output_aware_hidden["summary"]["per_variant"],
            output_summary=output_aware_output["summary"],
            seeds=output_aware_output["seeds"],
            holdout_policy=output_aware_output["heldout_policy"],
            source_path_hidden="artifacts/stage_b_ablation_output_aware_results.json",
            source_path_output="artifacts/stage_b_output_probe_output_aware_results.json",
        )
    )
    entry_hidden_map = entry_tune_hidden["per_variant"]
    entry_output_map = entry_tune_output["per_model"]
    entry_name_map = {
        "hybrid_frozen_entry": "hybrid_frozen_entry",
        "hybrid_no_small_frozen_entry": "hybrid_no_small_frozen_entry",
        "hybrid_train_entry": "hybrid_train_entry",
        "hybrid_no_small_train_entry": "hybrid_no_small_train_entry",
        "bridge_only_reference": "bridge_only_reference",
        "bridge_only_param_matched_reference": "bridge_only_param_matched_reference",
        "skip_only_reference": "skip_only_reference",
        "full_large_reference": "full_large_reference",
    }
    for output_name, output_metrics in entry_output_map.items():
        hidden_metrics = entry_hidden_map.get(entry_name_map[output_name], {})
        rows.append(
            {
                "subphase": "stage_b_entry_tune_follow_up",
                "compared_model": output_name,
                "seed_policy": ",".join(str(seed) for seed in entry_tune_output["seeds"]),
                "holdout_policy": "teacher_logit_output_probe",
                "hidden_mse_mean": hidden_metrics.get("hidden_mse_mean"),
                "hidden_cosine_mean": hidden_metrics.get("cosine_mean"),
                "delta_norm_mean": hidden_metrics.get("delta_norm_mean"),
                "gate_value_mean": hidden_metrics.get("gate_value_mean"),
                "logit_kl_to_teacher_mean": output_metrics.get("logit_kl_to_teacher_mean"),
                "nll_mean": output_metrics.get("nll_mean"),
                "perplexity_mean": output_metrics.get("perplexity_mean"),
                "top1_agreement_mean": output_metrics.get("top1_agreement_mean"),
                "top5_overlap_mean": output_metrics.get("top5_overlap_mean"),
                "source_hidden": "artifacts/stage_b_entry_tune_results.json",
                "source_output": "artifacts/stage_b_entry_tune_output_probe_results.json",
            }
        )

    columns = [
        "subphase",
        "compared_model",
        "seed_policy",
        "holdout_policy",
        "hidden_mse_mean",
        "hidden_cosine_mean",
        "delta_norm_mean",
        "gate_value_mean",
        "logit_kl_to_teacher_mean",
        "nll_mean",
        "perplexity_mean",
        "top1_agreement_mean",
        "top5_overlap_mean",
        "source_hidden",
        "source_output",
    ]
    return TableArtifact(
        slug="table_02_v05_key_ablation_summary",
        title="Table 2. v0.5.x Key Ablation Summary",
        columns=columns,
        rows=rows,
        notes=[
            "This table keeps the v0.5.x pilot progression explicit rather than rewriting history around the later v0.6.0 branch.",
            "The output-aware Stage B rows are the last v0.5.x architecture before the Phase 1 continuation work.",
        ],
        source_paths=[
            "artifacts/stage_b_ablation_results.json",
            "artifacts/stage_b_output_probe_results.json",
            "artifacts/stage_b_ablation_output_aware_results.json",
            "artifacts/stage_b_output_probe_output_aware_results.json",
            "artifacts/stage_b_entry_tune_results.json",
            "artifacts/stage_b_entry_tune_output_probe_results.json",
        ],
    )


def build_phase1_shortlist_table(root: Path) -> TableArtifact:
    """Build the real Gemma Phase 1 shortlist table."""

    ranking = _load_json(root, "artifacts/v0_6/phase1_real/combined/ranking_summary.json")
    rows: list[dict[str, Any]] = []
    for row in ranking["coarse"]:
        rows.append(
            {
                "screening_stage": row["stage"],
                "candidate_id": row["candidate_id"],
                "mapping": row["mapping"],
                "seed_count": row["seed_count"],
                "holdout_policy": "phase1_real_coarse_output_probe",
                "logit_kl_to_teacher": row["kl"],
                "nll": row["nll"],
                "perplexity": row["ppl"],
                "top1_agreement": row["top1"],
                "top5_overlap": row["top5"],
                "delta_kl_vs_hybrid_no_small": row["delta_kl_vs_hybrid_no_small"],
                "delta_nll_vs_hybrid_no_small": row["delta_nll_vs_hybrid_no_small"],
                "delta_kl_vs_bridge_only": row["delta_kl_vs_bridge_only"],
                "delta_nll_vs_bridge_only": row["delta_nll_vs_bridge_only"],
            }
        )
    for row in ranking["confirmation"]:
        rows.append(
            {
                "screening_stage": row["stage"],
                "candidate_id": row["candidate_id"],
                "mapping": row["mapping"],
                "seed_count": row["seed_count"],
                "holdout_policy": "phase1_real_confirmation_output_probe",
                "logit_kl_to_teacher": row["kl_mean"],
                "nll": row["nll_mean"],
                "perplexity": row["ppl_mean"],
                "top1_agreement": row["top1_mean"],
                "top5_overlap": row["top5_mean"],
                "delta_kl_vs_hybrid_no_small": row["delta_kl_vs_hybrid_no_small_mean"],
                "delta_nll_vs_hybrid_no_small": row["delta_nll_vs_hybrid_no_small_mean"],
                "delta_kl_vs_bridge_only": row["delta_kl_vs_bridge_only_mean"],
                "delta_nll_vs_bridge_only": row["delta_nll_vs_bridge_only_mean"],
                "wins_vs_hybrid_no_small_all_seeds": row.get("wins_vs_hybrid_no_small_all_seeds"),
                "wins_vs_skip_only_all_seeds": row.get("wins_vs_skip_only_all_seeds"),
            }
        )

    columns = [
        "screening_stage",
        "candidate_id",
        "mapping",
        "seed_count",
        "holdout_policy",
        "logit_kl_to_teacher",
        "nll",
        "perplexity",
        "top1_agreement",
        "top5_overlap",
        "delta_kl_vs_hybrid_no_small",
        "delta_nll_vs_hybrid_no_small",
        "delta_kl_vs_bridge_only",
        "delta_nll_vs_bridge_only",
        "wins_vs_hybrid_no_small_all_seeds",
        "wins_vs_skip_only_all_seeds",
    ]
    return TableArtifact(
        slug="table_03_phase1_shortlist_summary",
        title="Table 3. Phase 1 Shortlist Summary",
        columns=columns,
        rows=rows,
        notes=[
            "The real Gemma Phase 1 decision rejected the legacy `24..29 -> 14..19` split as the best default.",
            "The confirmed shortlist carried forward into Idea 4 was exactly `24..27 -> 14..19` and `24..27 -> 16..18`.",
        ],
        source_paths=["artifacts/v0_6/phase1_real/combined/ranking_summary.json"],
    )


def _rows_from_per_model(
    *,
    phase_name: str,
    holdout_policy: str,
    seed_count: int,
    per_model: dict[str, Any],
    source_path: str,
) -> list[dict[str, Any]]:
    rows = []
    for model_name, metrics in per_model.items():
        rows.append(
            {
                "phase_name": phase_name,
                "holdout_policy": holdout_policy,
                "seed_count": seed_count,
                "compared_model": model_name,
                "hidden_mse_mean": metrics.get("hidden_mse_mean"),
                "hidden_cosine_mean": metrics.get("hidden_cosine_mean"),
                "logit_kl_to_teacher_mean": metrics.get("logit_kl_to_teacher_mean"),
                "nll_mean": metrics.get("nll_mean"),
                "perplexity_mean": metrics.get("perplexity_mean"),
                "top1_agreement_mean": metrics.get("top1_agreement_mean"),
                "top5_overlap_mean": metrics.get("top5_overlap_mean"),
                "source_path": source_path,
            }
        )
    return rows


def build_static_mixture_table(root: Path) -> TableArtifact:
    """Build the static mixture summary table."""

    confirm = _load_json(root, "artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json")
    fresh = _load_json(root, "artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json")

    rows: list[dict[str, Any]] = []
    rows.extend(
        _rows_from_per_model(
            phase_name="static_mixture_confirmation_main",
            holdout_policy=confirm["heldout_policy"],
            seed_count=confirm["seed_count"],
            per_model=confirm["summary"]["per_model"],
            source_path="artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json",
        )
    )
    rows.extend(
        _rows_from_per_model(
            phase_name="static_mixture_fresh_holdout_recheck",
            holdout_policy=fresh["heldout_policy"],
            seed_count=fresh["seed_count"],
            per_model=fresh["summary"]["per_model"],
            source_path="artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json",
        )
    )
    columns = [
        "phase_name",
        "holdout_policy",
        "seed_count",
        "compared_model",
        "hidden_mse_mean",
        "hidden_cosine_mean",
        "logit_kl_to_teacher_mean",
        "nll_mean",
        "perplexity_mean",
        "top1_agreement_mean",
        "top5_overlap_mean",
        "source_path",
    ]
    return TableArtifact(
        slug="table_04_static_mixture_summary",
        title="Table 4. Static Mixture Summary",
        columns=columns,
        rows=rows,
        notes=[
            "The fresh-holdout recheck is included because it was the required rigor step before token-wise gating.",
            "This table keeps the static-mixture main and fresh holdouts together so the bridge win can be read in one place.",
        ],
        source_paths=[
            "artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json",
            "artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json",
        ],
    )


def build_tokenwise_table(root: Path) -> TableArtifact:
    """Build the v0.6.0 token-wise summary table."""

    main = _load_json(root, "artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json")
    fresh = _load_json(root, "artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json")

    rows: list[dict[str, Any]] = []
    rows.extend(
        _rows_from_per_model(
            phase_name="tokenwise_confirmation_main",
            holdout_policy=main["heldout_policy"],
            seed_count=main["seed_count"],
            per_model=main["summary"]["per_model"],
            source_path="artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json",
        )
    )
    rows.extend(
        _rows_from_per_model(
            phase_name="tokenwise_confirmation_fresh_holdout",
            holdout_policy=fresh["heldout_policy"],
            seed_count=fresh["seed_count"],
            per_model=fresh["summary"]["per_model"],
            source_path="artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json",
        )
    )

    columns = [
        "phase_name",
        "holdout_policy",
        "seed_count",
        "compared_model",
        "hidden_mse_mean",
        "hidden_cosine_mean",
        "logit_kl_to_teacher_mean",
        "nll_mean",
        "perplexity_mean",
        "top1_agreement_mean",
        "top5_overlap_mean",
        "source_path",
    ]
    return TableArtifact(
        slug="table_05_tokenwise_summary",
        title="Table 5. Token-Wise Summary on Original and Fresh Holdouts",
        columns=columns,
        rows=rows,
        notes=[
            "This is the canonical `v0.6.0` table.",
            "The current best model/result claim remains tied to these token-wise rows, not to the later analysis branches.",
        ],
        source_paths=[
            "artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json",
            "artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json",
        ],
    )


def build_idea5_table(root: Path) -> TableArtifact:
    """Build the Idea 5 discovery summary table."""

    solver = _load_json(root, "artifacts/v0_7/idea5_discovery/solver/top_paths.json")
    empirical = _load_json(root, "artifacts/v0_7/idea5_discovery/empirical_check/pilot_results.json")

    rows: list[dict[str, Any]] = []
    for row in solver["window_diagnostics"]:
        rows.append(
            {
                "row_group": "proxy_window",
                "name": row["name"],
                "mapping": row["mapping"],
                "seed_policy": "not_applicable",
                "holdout_policy": "local_monotone_proxy_region",
                "combined_proxy_cost": row["combined_proxy_cost"],
                "stage_signature_distance": row["stage_signature_distance"],
                "hidden_alignment_proxy": row["hidden_alignment_proxy"],
                "logit_disruption_proxy": row["logit_disruption_proxy"],
                "output_anchor_proxy": row["output_anchor_proxy"],
            }
        )
    for segment_index, segment in enumerate(solver["top_paths"][0]["segments"], start=1):
        rows.append(
            {
                "row_group": "top_path_segment",
                "name": f"path_1_segment_{segment_index}",
                "mapping": f"{segment['large_start']}..{segment['large_end']} -> {segment['small_start']}..{segment['small_end']}",
                "seed_policy": "not_applicable",
                "holdout_policy": "local_monotone_proxy_region",
                "combined_proxy_cost": segment["combined_cost"],
                "stage_signature_distance": segment["stage_signature_distance"],
                "hidden_alignment_proxy": segment["hidden_alignment_proxy"],
                "logit_disruption_proxy": segment["logit_disruption_proxy"],
                "output_anchor_proxy": segment["output_anchor_proxy"],
            }
        )
    empirical_result = empirical["results"][0]
    for model_name, metrics in empirical_result["summary"].items():
        if model_name == "training":
            continue
        rows.append(
            {
                "row_group": "bounded_empirical_check",
                "name": model_name,
                "mapping": empirical_result["candidate"]["label"],
                "seed_policy": ",".join(str(seed) for seed in empirical["seeds"]),
                "holdout_policy": "idea5_bounded_pilot",
                "logit_kl_to_teacher": metrics.get("logit_kl_to_teacher_mean"),
                "nll": metrics.get("nll_mean"),
                "perplexity": metrics.get("perplexity_mean"),
                "top1_agreement": metrics.get("top1_agreement_mean"),
                "top5_overlap": metrics.get("top5_overlap_mean"),
            }
        )

    columns = [
        "row_group",
        "name",
        "mapping",
        "seed_policy",
        "holdout_policy",
        "combined_proxy_cost",
        "stage_signature_distance",
        "hidden_alignment_proxy",
        "logit_disruption_proxy",
        "output_anchor_proxy",
        "logit_kl_to_teacher",
        "nll",
        "perplexity",
        "top1_agreement",
        "top5_overlap",
    ]
    return TableArtifact(
        slug="table_06_idea5_bounded_discovery_summary",
        title="Table 6. Idea 5 Bounded Discovery Summary",
        columns=columns,
        rows=rows,
        notes=[
            "The discovery branch succeeded analytically by recovering a local corridor, but its bounded empirical candidate did not beat `v0.6.0`.",
            "This table keeps both the proxy ranking and the single bounded pilot in the same canonical view.",
        ],
        source_paths=[
            "artifacts/v0_7/idea5_discovery/solver/top_paths.json",
            "artifacts/v0_7/idea5_discovery/empirical_check/pilot_results.json",
        ],
    )


def build_idea2_table(root: Path) -> TableArtifact:
    """Build the Idea 2 attribution summary table."""

    attribution = _load_json(root, "artifacts/v0_8/idea2_attribution/results.json")
    rows: list[dict[str, Any]] = []
    for holdout_name, holdout_payload in attribution["holdouts"].items():
        summary = holdout_payload["summary"]
        for variant_name, delta_metrics in summary["pairwise_deltas_from_full"].items():
            rows.append(
                {
                    "row_group": "overall_suppression",
                    "holdout_policy": holdout_payload["holdout_policy"],
                    "variant_name": variant_name,
                    "seed_policy": ",".join(str(seed) for seed in holdout_payload["seeds"]),
                    "logit_kl_to_teacher_delta": delta_metrics["logit_kl_to_teacher_mean"],
                    "nll_delta": delta_metrics["nll_mean"],
                    "perplexity_delta": delta_metrics["perplexity_mean"],
                    "top1_agreement_delta": delta_metrics["top1_agreement_mean"],
                    "top5_overlap_delta": delta_metrics["top5_overlap_mean"],
                }
            )
        for variant_name, delta_metrics in summary["path_specific_deltas_from_full"].items():
            rows.append(
                {
                    "row_group": "path_specific_suppression",
                    "holdout_policy": holdout_payload["holdout_policy"],
                    "variant_name": variant_name,
                    "seed_policy": ",".join(str(seed) for seed in holdout_payload["seeds"]),
                    "logit_kl_to_teacher_delta": delta_metrics["logit_kl_to_teacher_mean"],
                    "nll_delta": delta_metrics["nll_mean"],
                    "perplexity_delta": delta_metrics["perplexity_mean"],
                    "top1_agreement_delta": delta_metrics["top1_agreement_mean"],
                    "top5_overlap_delta": delta_metrics["top5_overlap_mean"],
                }
            )

    columns = [
        "row_group",
        "holdout_policy",
        "variant_name",
        "seed_policy",
        "logit_kl_to_teacher_delta",
        "nll_delta",
        "perplexity_delta",
        "top1_agreement_delta",
        "top5_overlap_delta",
    ]
    return TableArtifact(
        slug="table_07_idea2_attribution_summary",
        title="Table 7. Idea 2 Attribution Summary",
        columns=columns,
        rows=rows,
        notes=[
            "All deltas are relative to the full `v0.6.0` token-wise model on the named holdout policy.",
            "The path-specific rows are included because the analysis branch found path A to be more sensitive than path B.",
        ],
        source_paths=["artifacts/v0_8/idea2_attribution/results.json"],
    )


def build_generalization_table(root: Path) -> TableArtifact:
    """Build the bounded generalization summary table."""

    summary = _load_json(root, "artifacts/v0_9/generalization/aggregated/summary.json")
    rows: list[dict[str, Any]] = []
    seeds = ",".join(str(seed) for seed in summary["seeds"])
    for task_name, task_payload in summary["multichoice"].items():
        sample_seed = task_payload["slice_definition"]["sampling_seed"]
        for model_name, metrics in task_payload["per_model"].items():
            rows.append(
                {
                    "task_name": task_name,
                    "task_type": "multichoice",
                    "holdout_policy": f"deterministic_{task_payload['slice_definition']['split']}_slice",
                    "sampling_seed": sample_seed,
                    "seed_policy": seeds,
                    "compared_model": model_name,
                    "accuracy_mean": metrics["accuracy_mean"],
                    "mean_choice_margin_mean": metrics["mean_choice_margin_mean"],
                    "truncation_rate_mean": metrics["truncation_rate_mean"],
                }
            )
        for baseline_name, bootstrap in task_payload["bootstrap"].items():
            rows.append(
                {
                    "task_name": task_name,
                    "task_type": "multichoice_bootstrap",
                    "holdout_policy": f"deterministic_{task_payload['slice_definition']['split']}_slice",
                    "sampling_seed": sample_seed,
                    "seed_policy": seeds,
                    "compared_model": f"tokenwise_minus_{baseline_name}",
                    "accuracy_delta_mean": bootstrap["accuracy_delta"]["delta_mean"],
                    "accuracy_delta_ci_low": bootstrap["accuracy_delta"]["ci_low"],
                    "accuracy_delta_ci_high": bootstrap["accuracy_delta"]["ci_high"],
                    "mean_choice_margin_delta_mean": bootstrap["mean_choice_margin_delta"]["delta_mean"],
                    "mean_choice_margin_delta_ci_low": bootstrap["mean_choice_margin_delta"]["ci_low"],
                    "mean_choice_margin_delta_ci_high": bootstrap["mean_choice_margin_delta"]["ci_high"],
                }
            )
    for task_name, task_payload in summary["lm"].items():
        sample_seed = task_payload["slice_definition"]["sampling_seed"]
        for model_name, metrics in task_payload["per_model"].items():
            rows.append(
                {
                    "task_name": task_name,
                    "task_type": "lm",
                    "holdout_policy": f"deterministic_{task_payload['slice_definition']['split']}_slice",
                    "sampling_seed": sample_seed,
                    "seed_policy": seeds,
                    "compared_model": model_name,
                    "logit_kl_to_teacher_mean": metrics["logit_kl_to_teacher_mean"],
                    "nll_mean": metrics["nll_mean"],
                    "perplexity_mean": metrics["perplexity_mean"],
                    "truncation_rate_mean": metrics["truncation_rate_mean"],
                }
            )
        for baseline_name, bootstrap in task_payload["bootstrap"].items():
            rows.append(
                {
                    "task_name": task_name,
                    "task_type": "lm_bootstrap",
                    "holdout_policy": f"deterministic_{task_payload['slice_definition']['split']}_slice",
                    "sampling_seed": sample_seed,
                    "seed_policy": seeds,
                    "compared_model": f"tokenwise_minus_{baseline_name}",
                    "logit_kl_delta_mean": bootstrap["logit_kl_to_teacher_delta"]["delta_mean"],
                    "logit_kl_delta_ci_low": bootstrap["logit_kl_to_teacher_delta"]["ci_low"],
                    "logit_kl_delta_ci_high": bootstrap["logit_kl_to_teacher_delta"]["ci_high"],
                    "nll_delta_mean": bootstrap["nll_delta"]["delta_mean"],
                    "nll_delta_ci_low": bootstrap["nll_delta"]["ci_low"],
                    "nll_delta_ci_high": bootstrap["nll_delta"]["ci_high"],
                }
            )

    columns = [
        "task_name",
        "task_type",
        "holdout_policy",
        "sampling_seed",
        "seed_policy",
        "compared_model",
        "accuracy_mean",
        "mean_choice_margin_mean",
        "accuracy_delta_mean",
        "accuracy_delta_ci_low",
        "accuracy_delta_ci_high",
        "mean_choice_margin_delta_mean",
        "mean_choice_margin_delta_ci_low",
        "mean_choice_margin_delta_ci_high",
        "logit_kl_to_teacher_mean",
        "nll_mean",
        "perplexity_mean",
        "logit_kl_delta_mean",
        "logit_kl_delta_ci_low",
        "logit_kl_delta_ci_high",
        "nll_delta_mean",
        "nll_delta_ci_low",
        "nll_delta_ci_high",
        "truncation_rate_mean",
    ]
    return TableArtifact(
        slug="table_08_v09_generalization_summary",
        title="Table 8. v0_9 Generalization Summary",
        columns=columns,
        rows=rows,
        notes=[
            "The bounded generalization branch is evaluation-only; it does not alter the frozen `v0.6.0` best-model claim.",
            "Bootstrap rows are included only for token-wise versus static and bridge baselines, matching the saved analysis.",
        ],
        source_paths=["artifacts/v0_9/generalization/aggregated/summary.json"],
    )


def build_figure_specs(root: Path) -> list[FigureArtifact]:
    """Build the figure-ready specs."""

    phase1 = _load_json(root, "artifacts/v0_6/phase1_real/combined/ranking_summary.json")
    static_main = _load_json(root, "artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json")
    static_fresh = _load_json(root, "artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json")
    token_main = _load_json(root, "artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json")
    token_fresh = _load_json(root, "artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json")
    idea5 = _load_json(root, "artifacts/v0_7/idea5_discovery/solver/top_paths.json")
    idea2 = _load_json(root, "artifacts/v0_8/idea2_attribution/results.json")
    generalization = _load_json(root, "artifacts/v0_9/generalization/aggregated/summary.json")

    timeline_rows = [
        {
            "milestone": "v0.4.0",
            "status": "bring_up_complete",
            "claim_status": "same_family_path_runs_on_single_gpu",
        },
        {
            "milestone": "v0.5.0",
            "status": "output_aware_stage_b_added",
            "claim_status": "hybrid_beats_skip_and_no_small_but_not_bridges",
        },
        {
            "milestone": "v0.5.1",
            "status": "entry_tune_follow_up_complete",
            "claim_status": "qualified_feasibility_result",
        },
        {
            "milestone": "v0.6.0",
            "status": "tokenwise_idea4_frozen_best",
            "claim_status": "beats_bridges_on_original_and_fresh_holdouts",
        },
        {
            "milestone": "v0_7",
            "status": "analysis_only",
            "claim_status": "corridor_hypothesis_strengthened_not_best_model",
        },
        {
            "milestone": "v0_8",
            "status": "analysis_only",
            "claim_status": "attention_and_mlp_both_needed",
        },
        {
            "milestone": "v0_9",
            "status": "bounded_generalization",
            "claim_status": "mixed_external_validity",
        },
    ]

    progression_rows = [
        {
            "stage_name": "legacy_split_coarse",
            "mapping": "24..29 -> 14..19",
            "seed_count": 1,
            "holdout_policy": "phase1_real_coarse_output_probe",
            "logit_kl_to_teacher": next(row["kl"] for row in phase1["coarse"] if row["candidate_id"] == "legacy"),
            "nll": next(row["nll"] for row in phase1["coarse"] if row["candidate_id"] == "legacy"),
        },
        {
            "stage_name": "best_single_path_confirm",
            "mapping": "24..27 -> 14..19",
            "seed_count": 3,
            "holdout_policy": "phase1_real_confirmation_output_probe",
            "logit_kl_to_teacher": next(row["kl_mean"] for row in phase1["confirmation"] if row["mapping"] == "24..27 -> 14..19"),
            "nll": next(row["nll_mean"] for row in phase1["confirmation"] if row["mapping"] == "24..27 -> 14..19"),
        },
        {
            "stage_name": "static_mixture_main",
            "mapping": "softmix(path_b,path_a)",
            "seed_count": 3,
            "holdout_policy": static_main["heldout_policy"],
            "logit_kl_to_teacher": static_main["summary"]["per_model"]["static_mixture"]["logit_kl_to_teacher_mean"],
            "nll": static_main["summary"]["per_model"]["static_mixture"]["nll_mean"],
        },
        {
            "stage_name": "tokenwise_main",
            "mapping": "tokenwise(path_b,path_a)",
            "seed_count": 3,
            "holdout_policy": token_main["heldout_policy"],
            "logit_kl_to_teacher": token_main["summary"]["per_model"]["tokenwise_mixture"]["logit_kl_to_teacher_mean"],
            "nll": token_main["summary"]["per_model"]["tokenwise_mixture"]["nll_mean"],
        },
    ]

    holdout_rows = []
    for holdout_name, payload in {
        "original_main": token_main,
        "fresh_untouched": token_fresh,
    }.items():
        for model_name in [
            "tokenwise_mixture",
            "static_mixture",
            "tokenwise_mixture_no_small",
            "bridge_only",
            "bridge_only_param_matched",
        ]:
            metrics = payload["summary"]["per_model"][model_name]
            holdout_rows.append(
                {
                    "holdout_name": holdout_name,
                    "model_name": model_name,
                    "seed_count": payload["seed_count"],
                    "logit_kl_to_teacher_mean": metrics["logit_kl_to_teacher_mean"],
                    "nll_mean": metrics["nll_mean"],
                }
            )

    corridor_rows = [
        {
            "name": row["name"],
            "mapping": row["mapping"],
            "combined_proxy_cost": row["combined_proxy_cost"],
        }
        for row in idea5["window_diagnostics"]
    ]
    top_path_preview = [
        {
            "large_segment": f"{segment['large_start']}..{segment['large_end']}",
            "small_segment": f"{segment['small_start']}..{segment['small_end']}",
            "combined_cost": segment["combined_cost"],
        }
        for segment in idea5["top_paths"][0]["segments"]
    ]

    attribution_rows = []
    for holdout_name, payload in idea2["holdouts"].items():
        summary = payload["summary"]["pairwise_deltas_from_full"]
        for variant_name in [
            "tokenwise_attn_suppressed",
            "tokenwise_mlp_suppressed",
            "tokenwise_both_suppressed",
        ]:
            metrics = summary[variant_name]
            attribution_rows.append(
                {
                    "holdout_name": holdout_name,
                    "variant_name": variant_name,
                    "logit_kl_to_teacher_delta": metrics["logit_kl_to_teacher_mean"],
                    "nll_delta": metrics["nll_mean"],
                }
            )

    generalization_rows = []
    for task_name, payload in generalization["multichoice"].items():
        for model_name in [
            "tokenwise_mixture",
            "static_mixture",
            "bridge_only",
            "bridge_only_param_matched",
        ]:
            metrics = payload["per_model"][model_name]
            generalization_rows.append(
                {
                    "task_name": task_name,
                    "task_type": "multichoice",
                    "model_name": model_name,
                    "primary_metric_name": "accuracy_mean",
                    "primary_metric_value": metrics["accuracy_mean"],
                }
            )
    for task_name, payload in generalization["lm"].items():
        for model_name in [
            "tokenwise_mixture",
            "static_mixture",
            "bridge_only",
            "bridge_only_param_matched",
        ]:
            metrics = payload["per_model"][model_name]
            generalization_rows.append(
                {
                    "task_name": task_name,
                    "task_type": "lm",
                    "model_name": model_name,
                    "primary_metric_name": "nll_mean",
                    "primary_metric_value": metrics["nll_mean"],
                }
            )

    return [
        FigureArtifact(
            slug="figure_01_timeline_milestones",
            title="Figure 1. Timeline / Milestone Progression",
            spec={
                "figure_type": "timeline",
                "x_axis": "milestone",
                "rows": timeline_rows,
                "preview_rows": timeline_rows,
            },
            notes=[
                "Spec-only figure. Use a milestone timeline or ladder chart.",
                "The figure should show why `v0.6.0` stays frozen as the best branch while later branches remain analytical or evaluative.",
            ],
            source_paths=[
                "notes/final_report.md",
                "notes/v0_6/idea4_tokenwise_combined_decision.md",
                "notes/v0_7/idea5_combined_decision.md",
                "notes/v0_8/idea2_combined_decision.md",
                "notes/v0_9/generalization_results.md",
            ],
        ),
        FigureArtifact(
            slug="figure_02_structural_progression",
            title="Figure 2. Legacy Split to Token-Wise Progression",
            spec={
                "figure_type": "line_or_bar",
                "x_axis": "stage_name",
                "y_axes": ["logit_kl_to_teacher", "nll"],
                "rows": progression_rows,
                "preview_rows": progression_rows,
            },
            notes=[
                "Spec-only figure. Plot lower-is-better KL and NLL across the structural progression.",
                "This figure is the clearest one-panel summary of why the repo stops at `v0.6.0` rather than at `v0.5.1`.",
            ],
            source_paths=[
                "artifacts/v0_6/phase1_real/combined/ranking_summary.json",
                "artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json",
                "artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json",
            ],
        ),
        FigureArtifact(
            slug="figure_03_original_vs_fresh_holdout",
            title="Figure 3. Original vs Fresh Holdout KL/NLL",
            spec={
                "figure_type": "grouped_bar",
                "group_axis": "holdout_name",
                "series_axis": "model_name",
                "y_axes": ["logit_kl_to_teacher_mean", "nll_mean"],
                "rows": holdout_rows,
                "preview_rows": holdout_rows,
            },
            notes=[
                "Spec-only figure. The same model set should be plotted on both holdouts.",
                "The key reading is whether the token-wise bridge win survives the fresh untouched slice.",
            ],
            source_paths=[
                "artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json",
                "artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json",
                "artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json",
                "artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json",
            ],
        ),
        FigureArtifact(
            slug="figure_04_idea5_corridor",
            title="Figure 4. Idea 5 Corridor Visualization Summary",
            spec={
                "figure_type": "ranked_corridor",
                "ranked_windows": corridor_rows,
                "top_path_segments": top_path_preview,
                "preview_rows": corridor_rows,
            },
            notes=[
                "Spec-only figure. A corridor heatmap or ranked strip chart both fit.",
                "The figure should make it visually obvious that the successful Idea 4 shortlist lives in a broader low-cost corridor.",
            ],
            source_paths=["artifacts/v0_7/idea5_discovery/solver/top_paths.json"],
        ),
        FigureArtifact(
            slug="figure_05_idea2_attribution",
            title="Figure 5. Idea 2 Attribution Deltas",
            spec={
                "figure_type": "paired_delta_bar",
                "group_axis": "holdout_name",
                "series_axis": "variant_name",
                "y_axes": ["logit_kl_to_teacher_delta", "nll_delta"],
                "rows": attribution_rows,
                "preview_rows": attribution_rows,
            },
            notes=[
                "Spec-only figure. Lower deltas are better because all rows are degradations relative to the full token-wise baseline.",
                "The intended reading is that both attention and MLP matter, with larger degradation when MLP is suppressed.",
            ],
            source_paths=["artifacts/v0_8/idea2_attribution/results.json"],
        ),
        FigureArtifact(
            slug="figure_06_generalization_summary",
            title="Figure 6. Generalization Summary Across Benchmarks",
            spec={
                "figure_type": "faceted_bar",
                "facet_axis": "task_type",
                "x_axis": "task_name",
                "series_axis": "model_name",
                "y_axis": "primary_metric_value",
                "rows": generalization_rows,
                "preview_rows": generalization_rows,
            },
            notes=[
                "Spec-only figure. Use separate facets for multiple-choice accuracy and LM NLL.",
                "This is the figure that should visually enforce the mixed-generalization claim boundary.",
            ],
            source_paths=["artifacts/v0_9/generalization/aggregated/summary.json"],
        ),
    ]


def build_tables(root: Path) -> list[TableArtifact]:
    """Build the full paper table set."""

    return [
        build_bring_up_smoke_table(root),
        build_v05_key_ablation_table(root),
        build_phase1_shortlist_table(root),
        build_static_mixture_table(root),
        build_tokenwise_table(root),
        build_idea5_table(root),
        build_idea2_table(root),
        build_generalization_table(root),
    ]


def write_tables_note(path: Path, tables: list[TableArtifact]) -> None:
    """Write the markdown note for the generated tables."""

    lines = [
        "# Canonical Tables",
        "",
        "These tables are generated directly from frozen artifacts. CSV and JSON copies live under `artifacts/paper_tables/`.",
        "",
    ]
    for table in tables:
        lines.append(_table_note(table))
    save_text(path, "\n".join(lines).rstrip() + "\n")


def write_figures_note(path: Path, figures: list[FigureArtifact]) -> None:
    """Write the markdown note for the figure specs."""

    lines = [
        "# Figure Specs",
        "",
        "These are figure-ready specs, not new results. The specs live under `artifacts/paper_figures/` and are derived only from frozen artifacts and notes.",
        "",
    ]
    for figure in figures:
        lines.append(_figure_note(figure))
    save_text(path, "\n".join(lines).rstrip() + "\n")


def write_outputs(
    *,
    root: Path,
    tables_dir: Path,
    figures_dir: Path,
    tables_note: Path,
    figures_note: Path,
) -> None:
    """Generate all paper tables and figure specs."""

    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(tables_note.parent)

    tables = build_tables(root)
    for table in tables:
        save_csv(tables_dir / f"{table.slug}.csv", table.rows)
        save_json(
            tables_dir / f"{table.slug}.json",
            {
                "slug": table.slug,
                "title": table.title,
                "columns": table.columns,
                "notes": table.notes,
                "source_paths": table.source_paths,
                "rows": table.rows,
            },
        )
    save_json(
        tables_dir / "manifest.json",
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "git_commit_hash": git_commit_hash(),
            "tables": [
                {
                    "slug": table.slug,
                    "title": table.title,
                    "source_paths": table.source_paths,
                }
                for table in tables
            ],
        },
    )

    figures = build_figure_specs(root)
    for figure in figures:
        save_json(
            figures_dir / f"{figure.slug}.json",
            {
                "slug": figure.slug,
                "title": figure.title,
                "notes": figure.notes,
                "source_paths": figure.source_paths,
                **figure.spec,
            },
        )
    save_json(
        figures_dir / "manifest.json",
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "git_commit_hash": git_commit_hash(),
            "figures": [
                {
                    "slug": figure.slug,
                    "title": figure.title,
                    "source_paths": figure.source_paths,
                }
                for figure in figures
            ],
        },
    )

    write_tables_note(tables_note, tables)
    write_figures_note(figures_note, figures)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    write_outputs(
        root=repo_root,
        tables_dir=repo_root / args.tables_dir,
        figures_dir=repo_root / args.figures_dir,
        tables_note=repo_root / args.tables_note,
        figures_note=repo_root / args.figures_note,
    )


if __name__ == "__main__":
    main()
