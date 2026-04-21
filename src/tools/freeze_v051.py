"""Generate v0.5.1 release artifacts from existing experiment outputs only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.utils.io import ensure_dir, save_json


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
NOTES_DIR = ROOT / "notes"
FIGURES_DIR = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default="artifacts/manifest_v0.5.1.json")
    parser.add_argument("--summary-table-path", default="artifacts/final_summary_table.csv")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / Path(path)).read_text(encoding="utf-8"))


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in sorted(path.rglob("*")) if child.is_file())


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()

    for child in sorted(file for file in path.rglob("*") if file.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _float_list(values: list[float]) -> list[float]:
    return [float(value) for value in values]


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric_bars(
    ax: Any,
    labels: list[str],
    means: list[float],
    stds: list[float],
    title: str,
    ylabel: str,
) -> None:
    positions = list(range(len(labels)))
    ax.bar(positions, means, yerr=stds, capsize=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def _generate_hidden_metrics_figure(path: Path, output_aware_ablation: dict[str, Any]) -> None:
    summary = output_aware_ablation["summary"]["per_variant"]
    labels = ["skip_only", "hybrid_no_small", "hybrid", "bridge_only", "bridge_only_param_matched"]
    mse_means = [summary[label]["hidden_mse_mean"] for label in labels]
    mse_stds = [summary[label]["hidden_mse_std"] for label in labels]
    cosine_means = [summary[label]["cosine_mean"] for label in labels]
    cosine_stds = [summary[label]["cosine_std"] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    _plot_metric_bars(axes[0], labels, _float_list(mse_means), _float_list(mse_stds), "Stage B Hidden MSE (output-aware)", "hidden_mse")
    _plot_metric_bars(axes[1], labels, _float_list(cosine_means), _float_list(cosine_stds), "Stage B Hidden Cosine (output-aware)", "cosine")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _generate_output_metrics_figure(path: Path, output_probe: dict[str, Any]) -> None:
    summary = output_probe["summary"]["per_model"]
    labels = ["skip_only", "hybrid_no_small", "hybrid", "bridge_only", "bridge_only_param_matched"]
    metrics = [
        ("logit_kl_to_teacher", "KL"),
        ("nll", "NLL"),
        ("perplexity", "PPL"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for index, (metric_key, metric_label) in enumerate(metrics):
        means = [summary[label][f"{metric_key}_mean"] for label in labels]
        stds = [summary[label][f"{metric_key}_std"] for label in labels]
        _plot_metric_bars(
            axes[index],
            labels,
            _float_list(means),
            _float_list(stds),
            f"Output Metric: {metric_label} (output-aware)",
            metric_label,
        )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _generate_progression_figure(
    path: Path,
    hidden_only_probe: dict[str, Any],
    output_aware_probe: dict[str, Any],
    entry_tune_probe: dict[str, Any],
) -> None:
    labels = ["hidden_only", "output_aware", "entry_tune"]
    progression = {
        "KL": [
            hidden_only_probe["summary"]["per_model"]["hybrid"]["logit_kl_to_teacher_mean"],
            output_aware_probe["summary"]["per_model"]["hybrid"]["logit_kl_to_teacher_mean"],
            entry_tune_probe["per_model"]["hybrid_train_entry"]["logit_kl_to_teacher_mean"],
        ],
        "NLL": [
            hidden_only_probe["summary"]["per_model"]["hybrid"]["nll_mean"],
            output_aware_probe["summary"]["per_model"]["hybrid"]["nll_mean"],
            entry_tune_probe["per_model"]["hybrid_train_entry"]["nll_mean"],
        ],
        "PPL": [
            hidden_only_probe["summary"]["per_model"]["hybrid"]["perplexity_mean"],
            output_aware_probe["summary"]["per_model"]["hybrid"]["perplexity_mean"],
            entry_tune_probe["per_model"]["hybrid_train_entry"]["perplexity_mean"],
        ],
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), constrained_layout=True)
    for index, (metric_label, values) in enumerate(progression.items()):
        axes[index].plot(labels, _float_list(values), marker="o")
        axes[index].set_title(f"Hybrid {metric_label} Progression")
        axes[index].set_ylabel(metric_label)
        axes[index].grid(True, alpha=0.3)
    fig.suptitle("Hybrid Output Metrics Across Milestones")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _generate_entry_tune_figure(
    path: Path,
    entry_tune_hidden: dict[str, Any],
    entry_tune_output: dict[str, Any],
) -> None:
    hidden = entry_tune_hidden["per_variant"]
    output = entry_tune_output["per_model"]
    labels = ["hybrid", "hybrid_no_small"]
    width = 0.35
    positions = [0, 1]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    frozen_hidden_mse = [hidden["hybrid_frozen_entry"]["hidden_mse_mean"], hidden["hybrid_no_small_frozen_entry"]["hidden_mse_mean"]]
    tuned_hidden_mse = [hidden["hybrid_train_entry"]["hidden_mse_mean"], hidden["hybrid_no_small_train_entry"]["hidden_mse_mean"]]
    axes[0, 0].bar([p - width / 2 for p in positions], frozen_hidden_mse, width=width, label="frozen_entry")
    axes[0, 0].bar([p + width / 2 for p in positions], tuned_hidden_mse, width=width, label="train_entry")
    axes[0, 0].set_xticks(positions)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_title("Hidden MSE")
    axes[0, 0].set_ylabel("hidden_mse")
    axes[0, 0].legend()

    frozen_hidden_cosine = [hidden["hybrid_frozen_entry"]["cosine_mean"], hidden["hybrid_no_small_frozen_entry"]["cosine_mean"]]
    tuned_hidden_cosine = [hidden["hybrid_train_entry"]["cosine_mean"], hidden["hybrid_no_small_train_entry"]["cosine_mean"]]
    axes[0, 1].bar([p - width / 2 for p in positions], frozen_hidden_cosine, width=width, label="frozen_entry")
    axes[0, 1].bar([p + width / 2 for p in positions], tuned_hidden_cosine, width=width, label="train_entry")
    axes[0, 1].set_xticks(positions)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_title("Hidden Cosine")
    axes[0, 1].set_ylabel("cosine")

    frozen_output_kl = [output["hybrid_frozen_entry"]["logit_kl_to_teacher_mean"], output["hybrid_no_small_frozen_entry"]["logit_kl_to_teacher_mean"]]
    tuned_output_kl = [output["hybrid_train_entry"]["logit_kl_to_teacher_mean"], output["hybrid_no_small_train_entry"]["logit_kl_to_teacher_mean"]]
    axes[1, 0].bar([p - width / 2 for p in positions], frozen_output_kl, width=width, label="frozen_entry")
    axes[1, 0].bar([p + width / 2 for p in positions], tuned_output_kl, width=width, label="train_entry")
    axes[1, 0].set_xticks(positions)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_title("Teacher-Logit KL")
    axes[1, 0].set_ylabel("logit_kl_to_teacher")

    frozen_output_nll = [output["hybrid_frozen_entry"]["nll_mean"], output["hybrid_no_small_frozen_entry"]["nll_mean"]]
    tuned_output_nll = [output["hybrid_train_entry"]["nll_mean"], output["hybrid_no_small_train_entry"]["nll_mean"]]
    axes[1, 1].bar([p - width / 2 for p in positions], frozen_output_nll, width=width, label="frozen_entry")
    axes[1, 1].bar([p + width / 2 for p in positions], tuned_output_nll, width=width, label="train_entry")
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_title("Held-out NLL")
    axes[1, 1].set_ylabel("nll")

    fig.suptitle("Entry Projector Finetuning Effect")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _build_summary_rows(
    smoke: dict[str, Any],
    stage_a: dict[str, Any],
    stage_b_pilot: dict[str, Any],
    hidden_only_ablation: dict[str, Any],
    hidden_only_probe: dict[str, Any],
    output_aware_ablation: dict[str, Any],
    output_aware_probe: dict[str, Any],
    entry_tune_hidden: dict[str, Any],
    entry_tune_output: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        [
            {"section": "smoke", "phase": "bring_up", "model": "all", "metric": "overall_success", "value": smoke["overall_success"], "config_used": "configs/gemma2_conservative.yaml", "seeds_used": "", "notes": "14/14 cases passed"},
            {"section": "smoke", "phase": "bring_up", "model": "hybrid", "metric": "largest_successful_seq_len", "value": 256, "config_used": "configs/gemma2_conservative.yaml", "seeds_used": "", "notes": "smoke matrix"},
            {"section": "stage_a", "phase": "pilot", "model": "entry_projector", "metric": "train_loss_start", "value": stage_a["train_loss_start"], "config_used": stage_a["config_path"], "seeds_used": "42", "notes": ""},
            {"section": "stage_a", "phase": "pilot", "model": "entry_projector", "metric": "train_loss_end", "value": stage_a["train_loss_end"], "config_used": stage_a["config_path"], "seeds_used": "42", "notes": ""},
            {"section": "stage_a", "phase": "pilot", "model": "entry_projector", "metric": "heldout_mse_after", "value": stage_a["heldout_mse_after"], "config_used": stage_a["config_path"], "seeds_used": "42", "notes": ""},
            {"section": "stage_a", "phase": "pilot", "model": "entry_projector", "metric": "heldout_cosine_after", "value": stage_a["heldout_cosine_after"], "config_used": stage_a["config_path"], "seeds_used": "42", "notes": ""},
        ]
    )

    for model in ("skip_only", "bridge_only", "hybrid"):
        rows.append(
            {
                "section": "stage_b_hidden_only_pilot",
                "phase": "pilot",
                "model": model,
                "metric": "hidden_mse",
                "value": stage_b_pilot[f"{model}_hidden_mse"],
                "config_used": stage_b_pilot["config_path"],
                "seeds_used": "42",
                "notes": "",
            }
        )
        rows.append(
            {
                "section": "stage_b_hidden_only_pilot",
                "phase": "pilot",
                "model": model,
                "metric": "cosine",
                "value": stage_b_pilot[f"{model}_cosine"],
                "config_used": stage_b_pilot["config_path"],
                "seeds_used": "42",
                "notes": "",
            }
        )

    for phase_name, ablation_payload, probe_payload in (
        ("stage_b_hidden_only", hidden_only_ablation, hidden_only_probe),
        ("stage_b_output_aware", output_aware_ablation, output_aware_probe),
    ):
        for model, summary in ablation_payload["summary"]["per_variant"].items():
            if model == "hybrid_gate_zero":
                continue
            rows.append(
                {
                    "section": phase_name,
                    "phase": "ablation_hidden",
                    "model": model,
                    "metric": "hidden_mse_mean",
                    "value": summary["hidden_mse_mean"],
                    "config_used": ablation_payload["config_path"],
                    "seeds_used": "42,43,44",
                    "notes": "",
                }
            )
            rows.append(
                {
                    "section": phase_name,
                    "phase": "ablation_hidden",
                    "model": model,
                    "metric": "cosine_mean",
                    "value": summary["cosine_mean"],
                    "config_used": ablation_payload["config_path"],
                    "seeds_used": "42,43,44",
                    "notes": "",
                }
            )
        for model, summary in probe_payload["summary"]["per_model"].items():
            if model == "full_large":
                continue
            for metric in ("logit_kl_to_teacher_mean", "nll_mean", "perplexity_mean"):
                rows.append(
                    {
                        "section": phase_name,
                        "phase": "output_probe",
                        "model": model,
                        "metric": metric,
                        "value": summary[metric],
                        "config_used": probe_payload["config_path"],
                        "seeds_used": "42,43,44",
                        "notes": "",
                    }
                )

    for model in ("hybrid_frozen_entry", "hybrid_train_entry", "hybrid_no_small_frozen_entry", "hybrid_no_small_train_entry"):
        hidden_summary = entry_tune_hidden["per_variant"][model]
        rows.append(
            {
                "section": "stage_b_entry_tune",
                "phase": "hidden_comparison",
                "model": model,
                "metric": "hidden_mse_mean",
                "value": hidden_summary["hidden_mse_mean"],
                "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml",
                "seeds_used": "42,43,44",
                "notes": "",
            }
        )
        rows.append(
            {
                "section": "stage_b_entry_tune",
                "phase": "hidden_comparison",
                "model": model,
                "metric": "cosine_mean",
                "value": hidden_summary["cosine_mean"],
                "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml",
                "seeds_used": "42,43,44",
                "notes": "",
            }
        )

    for model in ("hybrid_frozen_entry", "hybrid_train_entry", "hybrid_no_small_frozen_entry", "hybrid_no_small_train_entry", "bridge_only_reference", "bridge_only_param_matched_reference"):
        output_summary = entry_tune_output["per_model"][model]
        for metric in ("logit_kl_to_teacher_mean", "nll_mean", "perplexity_mean"):
            rows.append(
                {
                    "section": "stage_b_entry_tune",
                    "phase": "output_comparison",
                    "model": model,
                    "metric": metric,
                    "value": output_summary[metric],
                    "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml",
                    "seeds_used": "42,43,44",
                    "notes": "",
                }
            )
    return rows


def _manifest_specs() -> list[dict[str, Any]]:
    return [
        {"path": "artifacts/env_sanity.json", "purpose": "Environment and Gemma auth sanity result", "config_used": "configs/gemma2_conservative.yaml", "seeds_used": [], "lineage": "reference"},
        {"path": "artifacts/real_gemma_smoke.json", "purpose": "Real-hardware smoke matrix", "config_used": "configs/gemma2_conservative.yaml", "seeds_used": [], "lineage": "reference"},
        {"path": "artifacts/milestone_parameter_audit.json", "purpose": "Parameter audit for milestone baselines", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [], "lineage": "reference"},
        {"path": "artifacts/stage_a_pilot_ckpt", "purpose": "Stage A pilot checkpoint bundle", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42], "lineage": "reference"},
        {"path": "artifacts/stage_a_pilot_metrics.json", "purpose": "Stage A pilot metrics", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42], "lineage": "reference"},
        {"path": "artifacts/stage_a_pilot_history.csv", "purpose": "Stage A pilot training history", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42], "lineage": "reference"},
        {"path": "artifacts/stage_b_pilot_ckpt", "purpose": "Stage B hidden-only pilot checkpoint bundle", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42], "lineage": "reference"},
        {"path": "artifacts/stage_b_pilot_metrics.json", "purpose": "Stage B hidden-only pilot metrics", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42], "lineage": "reference"},
        {"path": "artifacts/stage_b_pilot_history.csv", "purpose": "Stage B hidden-only pilot history", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42], "lineage": "reference"},
        {"path": "artifacts/stage_b_ablation_results.json", "purpose": "Three-seed hidden-only Stage B ablation results", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "artifacts/stage_b_ablation_summary.csv", "purpose": "Three-seed hidden-only Stage B ablation summary", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "artifacts/stage_b_diagnostics.json", "purpose": "Hidden-only Stage B diagnostics", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "artifacts/stage_b_output_probe_results.json", "purpose": "Hidden-only Stage B output probe results", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "artifacts/stage_b_output_probe_summary.csv", "purpose": "Hidden-only Stage B output probe summary", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "artifacts/stage_b_ablation_output_aware", "purpose": "Output-aware Stage B checkpoint bundle", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "artifacts/stage_b_ablation_output_aware_results.json", "purpose": "Output-aware Stage B ablation results", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "artifacts/stage_b_ablation_output_aware_summary.csv", "purpose": "Output-aware Stage B ablation summary", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "artifacts/stage_b_output_aware_diagnostics.json", "purpose": "Output-aware Stage B diagnostics", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "artifacts/stage_b_output_probe_output_aware_results.json", "purpose": "Output-aware Stage B output probe results", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "artifacts/stage_b_output_probe_output_aware_summary.csv", "purpose": "Output-aware Stage B output probe summary", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "artifacts/stage_b_ablation_output_aware_train_entry_raw", "purpose": "Entry-tuned Stage B raw checkpoint bundle", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml", "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "artifacts/stage_b_ablation_output_aware_train_entry_results.json", "purpose": "Entry-tuned Stage B ablation results", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml", "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "artifacts/stage_b_ablation_output_aware_train_entry_summary.csv", "purpose": "Entry-tuned Stage B ablation summary", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml", "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "artifacts/stage_b_ablation_output_aware_train_entry_diagnostics.json", "purpose": "Entry-tuned Stage B training diagnostics", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml", "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "artifacts/stage_b_output_probe_output_aware_train_entry_results.json", "purpose": "Entry-tuned output probe raw results", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml", "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "artifacts/stage_b_output_probe_output_aware_train_entry_summary.csv", "purpose": "Entry-tuned output probe raw summary", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml", "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "artifacts/stage_b_entry_tune_results.json", "purpose": "Frozen-entry vs train-entry hidden comparison", "config_used": ["configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml"], "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "artifacts/stage_b_entry_tune_summary.csv", "purpose": "Frozen-entry vs train-entry hidden comparison summary", "config_used": ["configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml"], "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "artifacts/stage_b_entry_tune_diagnostics.json", "purpose": "Entry-tune comparison diagnostics", "config_used": ["configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml"], "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "artifacts/stage_b_entry_tune_output_probe_results.json", "purpose": "Frozen-entry vs train-entry output comparison", "config_used": ["configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml"], "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "artifacts/stage_b_entry_tune_output_probe_summary.csv", "purpose": "Frozen-entry vs train-entry output comparison summary", "config_used": ["configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml"], "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "notes/real_hardware_report.md", "purpose": "Hardware bring-up and milestone history report", "config_used": "multiple", "seeds_used": [42, 43, 44], "lineage": "release_doc"},
        {"path": "notes/final_report.md", "purpose": "Final qualified-result report", "config_used": "multiple", "seeds_used": [42, 43, 44], "lineage": "release_doc"},
        {"path": "notes/release_notes_v0.5.1.md", "purpose": "v0.5.1 release notes", "config_used": "multiple", "seeds_used": [42, 43, 44], "lineage": "release_doc"},
        {"path": "notes/abstract.md", "purpose": "Paper-style abstract", "config_used": "multiple", "seeds_used": [42, 43, 44], "lineage": "release_doc"},
        {"path": "notes/one_page_summary.md", "purpose": "Plain-language one-page summary", "config_used": "multiple", "seeds_used": [42, 43, 44], "lineage": "release_doc"},
        {"path": "notes/reproducibility.md", "purpose": "Reproducibility and provenance notes", "config_used": "multiple", "seeds_used": [42, 43, 44], "lineage": "release_doc"},
        {"path": "notes/stage_b_ablation_report.md", "purpose": "Hidden-only Stage B ablation report", "config_used": "configs/gemma2_conservative_pilot_256.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "notes/stage_b_output_probe_output_aware_report.md", "purpose": "Output-aware Stage B output probe report", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "seeds_used": [42, 43, 44], "lineage": "reference"},
        {"path": "notes/stage_b_entry_tune_report.md", "purpose": "Entry-tune hidden comparison report", "config_used": ["configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml"], "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "notes/stage_b_entry_tune_output_probe_report.md", "purpose": "Entry-tune output comparison report", "config_used": ["configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml"], "seeds_used": [42, 43, 44], "lineage": "follow_up"},
        {"path": "artifacts/final_summary_table.csv", "purpose": "Release summary table", "config_used": "multiple", "seeds_used": [42, 43, 44], "lineage": "release_doc"},
        {"path": "figures/hidden_metrics_stage_b.png", "purpose": "Publication figure for Stage B hidden metrics", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "seeds_used": [42, 43, 44], "lineage": "release_doc"},
        {"path": "figures/output_metrics_stage_b.png", "purpose": "Publication figure for Stage B output metrics", "config_used": "configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "seeds_used": [42, 43, 44], "lineage": "release_doc"},
        {"path": "figures/milestone_progression.png", "purpose": "Publication figure for hybrid milestone progression", "config_used": "multiple", "seeds_used": [42, 43, 44], "lineage": "release_doc"},
        {"path": "figures/entry_tune_effect.png", "purpose": "Publication figure for entry-tuning effect", "config_used": ["configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml", "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml"], "seeds_used": [42, 43, 44], "lineage": "release_doc"},
    ]


def _build_manifest() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for spec in _manifest_specs():
        path = ROOT / spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing release artifact: {spec['path']}")
        entries.append(
            {
                **spec,
                "path": spec["path"].replace("\\", "/"),
                "size_bytes": _path_size_bytes(path),
                "sha256": _path_sha256(path),
            }
        )
    return {
        "release": "v0.5.1",
        "description": "Qualified-result freeze for same-family Gemma-2 latent delegation",
        "entries": entries,
    }


def main() -> None:
    args = parse_args()
    figures_dir = ensure_dir(ROOT / args.figures_dir)
    manifest_path = ROOT / args.manifest_path
    summary_table_path = ROOT / args.summary_table_path

    smoke = _load_json("artifacts/real_gemma_smoke.json")
    stage_a = _load_json("artifacts/stage_a_pilot_metrics.json")
    stage_b_pilot = _load_json("artifacts/stage_b_pilot_metrics.json")
    hidden_only_ablation = _load_json("artifacts/stage_b_ablation_results.json")
    hidden_only_probe = _load_json("artifacts/stage_b_output_probe_results.json")
    output_aware_ablation = _load_json("artifacts/stage_b_ablation_output_aware_results.json")
    output_aware_probe = _load_json("artifacts/stage_b_output_probe_output_aware_results.json")
    entry_tune_hidden = _load_json("artifacts/stage_b_entry_tune_results.json")
    entry_tune_output = _load_json("artifacts/stage_b_entry_tune_output_probe_results.json")

    _generate_hidden_metrics_figure(figures_dir / "hidden_metrics_stage_b.png", output_aware_ablation)
    _generate_output_metrics_figure(figures_dir / "output_metrics_stage_b.png", output_aware_probe)
    _generate_progression_figure(figures_dir / "milestone_progression.png", hidden_only_probe, output_aware_probe, entry_tune_output)
    _generate_entry_tune_figure(figures_dir / "entry_tune_effect.png", entry_tune_hidden, entry_tune_output)

    summary_rows = _build_summary_rows(
        smoke,
        stage_a,
        stage_b_pilot,
        hidden_only_ablation,
        hidden_only_probe,
        output_aware_ablation,
        output_aware_probe,
        entry_tune_hidden,
        entry_tune_output,
    )
    ensure_dir(summary_table_path.parent)
    _save_csv(summary_table_path, summary_rows)

    manifest = _build_manifest()
    ensure_dir(manifest_path.parent)
    save_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
