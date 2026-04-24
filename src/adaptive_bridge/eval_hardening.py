"""Evaluation hardening for the adaptive-bridge bounded milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.adaptive_bridge.analysis_runtime import aggregate_stats, gate_stats_for_model_task, paired_bootstrap_summary, task_family
from src.adaptive_bridge.common import adaptive_bridge_settings, adaptive_eval_spec
from src.adaptive_bridge.evaluate import _build_lm_examples, _build_multichoice_examples, _load_models_for_seed
from src.models.backbone_loader import load_backbones
from src.utils.io import ensure_dir, load_config, save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.v0_6.idea4_common import load_mixture_path_specs


LOGGER = get_logger(__name__)


COMPARISONS = (
    ("adaptive_bridge_moe", "frozen_v060_tokenwise"),
    ("adaptive_bridge_moe", "bridge_only_strong"),
    ("adaptive_bridge_moe", "bridge_only_param_matched"),
    ("adaptive_bridge_moe", "adaptive_bridge_no_small"),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for adaptive-bridge eval hardening."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    return parser.parse_args()


def _train_results_candidates(train_dir: Path) -> list[Path]:
    root = train_dir.parents[1]
    return sorted(root.glob("*/train/results.json"))


def _warm_start_records_by_seed(train_dir: Path) -> dict[int, list[dict[str, Any]]]:
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for candidate in _train_results_candidates(train_dir):
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        for item in payload.get("seed_results", []):
            seed = int(item["seed"])
            records_by_seed.setdefault(seed, []).append(
                {
                    "source": str(candidate),
                    "warm_starts": item.get("warm_starts", {}),
                }
            )
    return records_by_seed


def _task_result_path(eval_dir: Path, task_name: str, seed: int) -> Path:
    return eval_dir / task_name / f"seed_{seed}" / "results.json"


def _load_eval_results(eval_dir: Path) -> dict[str, Any]:
    return json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))


def _summary_row_index(summary_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["task_name"], row["model_name"]): row for row in summary_rows}


def _stable_seed(*parts: str) -> int:
    seed = 17
    for part in parts:
        for char in part:
            seed = (seed * 31 + ord(char)) % 1_000_000_007
    return seed % 100_000


def _integrity_checks(config: Any, train_dir: Path, eval_dir: Path) -> dict[str, Any]:
    eval_spec = adaptive_eval_spec(config)
    eval_results = _load_eval_results(eval_dir)
    warm_start_records = _warm_start_records_by_seed(train_dir)
    settings = adaptive_bridge_settings(config)

    integrity: dict[str, Any] = {"seeds": {}}
    invalid: list[str] = []
    for seed in eval_spec.seeds:
        frozen_path = settings.frozen_tokenwise_template.format(seed=seed) if settings.frozen_tokenwise_template else None
        checkpoint_exists = bool(frozen_path and Path(frozen_path).exists())

        warm_start_loaded = False
        warm_start_sources: list[str] = []
        for record in warm_start_records.get(seed, []):
            warm_start_sources.append(record["source"])
            moe_status = record["warm_starts"].get("adaptive_bridge_moe", {}).get("status")
            no_small_status = record["warm_starts"].get("adaptive_bridge_no_small", {}).get("status")
            if moe_status == "loaded" and no_small_status == "loaded":
                warm_start_loaded = True
                break

        eval_diag = eval_results["model_diagnostics"].get(str(seed), {})
        frozen_eval_loaded = eval_diag.get("frozen_v060_tokenwise", {}).get("status") == "loaded"

        task_comparison_loaded: dict[str, bool] = {}
        for task in (
            [item.name for item in eval_spec.internal_tasks]
            + [item.name for item in eval_spec.lm_tasks]
            + [item.name for item in eval_spec.multichoice_tasks]
        ):
            task_payload = json.loads(_task_result_path(eval_dir, task, seed).read_text(encoding="utf-8"))
            example_rows = task_payload.get("example_results", [])
            task_comparison_loaded[task] = bool(example_rows) and all(
                "frozen_v060_tokenwise" in row.get("models", {}) for row in example_rows
            )

        seed_integrity = {
            "frozen_checkpoint_exists": checkpoint_exists,
            "warm_start_loaded": warm_start_loaded,
            "warm_start_sources": warm_start_sources,
            "eval_live_frozen_reference_loaded": frozen_eval_loaded,
            "task_level_frozen_reference_present": task_comparison_loaded,
        }
        integrity["seeds"][str(seed)] = seed_integrity

        if not checkpoint_exists:
            invalid.append(f"seed {seed}: frozen checkpoint missing")
        if not warm_start_loaded:
            invalid.append(f"seed {seed}: warm-start not verified as loaded")
        if not frozen_eval_loaded:
            invalid.append(f"seed {seed}: eval did not load frozen_v060_tokenwise")
        missing_tasks = [task for task, present in task_comparison_loaded.items() if not present]
        if missing_tasks:
            invalid.append(f"seed {seed}: frozen reference missing from tasks {missing_tasks}")

    integrity["valid"] = not invalid
    integrity["failures"] = invalid
    if invalid:
        raise RuntimeError(f"Frozen reference integrity failed: {'; '.join(invalid)}")
    return integrity


def _guardrail_checks(config: Any, eval_dir: Path) -> dict[str, Any]:
    eval_spec = adaptive_eval_spec(config)
    eval_results = _load_eval_results(eval_dir)
    index = _summary_row_index(eval_results["summary_rows"])
    output: dict[str, Any] = {}
    for task_name in ("development_holdout", "confirmation_holdout"):
        output[task_name] = {}
        frozen = index[(task_name, "frozen_v060_tokenwise")]
        for model_name in ("adaptive_bridge_moe", "adaptive_bridge_no_small"):
            row = index[(task_name, model_name)]
            output[task_name][model_name] = {
                "kl": float(row["logit_kl_to_teacher_mean"]),
                "nll": float(row["nll_mean"]),
                "frozen_kl": float(frozen["logit_kl_to_teacher_mean"]),
                "frozen_nll": float(frozen["nll_mean"]),
                "preserved_kl": float(row["logit_kl_to_teacher_mean"]) <= float(frozen["logit_kl_to_teacher_mean"]) + eval_spec.internal_kl_tolerance,
                "preserved_nll": float(row["nll_mean"]) <= float(frozen["nll_mean"]) + eval_spec.internal_nll_tolerance,
            }
    return output


def _no_small_separation(eval_dir: Path) -> dict[str, Any]:
    eval_results = _load_eval_results(eval_dir)
    index = _summary_row_index(eval_results["summary_rows"])
    output: dict[str, Any] = {}
    for task_name in ("development_holdout", "confirmation_holdout", "lambada_openai"):
        moe = index[(task_name, "adaptive_bridge_moe")]
        no_small = index[(task_name, "adaptive_bridge_no_small")]
        output[task_name] = {
            "same_run_comparison": "adaptive_bridge_moe vs adaptive_bridge_no_small",
            "beats_on_kl": float(moe["logit_kl_to_teacher_mean"]) < float(no_small["logit_kl_to_teacher_mean"]),
            "beats_on_nll": float(moe["nll_mean"]) < float(no_small["nll_mean"]),
            "delta_kl": float(moe["logit_kl_to_teacher_mean"]) - float(no_small["logit_kl_to_teacher_mean"]),
            "delta_nll": float(moe["nll_mean"]) - float(no_small["nll_mean"]),
        }
    for task_name in ("piqa", "arc_easy"):
        moe = index[(task_name, "adaptive_bridge_moe")]
        no_small = index[(task_name, "adaptive_bridge_no_small")]
        output[task_name] = {
            "same_run_comparison": "adaptive_bridge_moe vs adaptive_bridge_no_small",
            "beats_on_accuracy": float(moe["accuracy_mean"]) > float(no_small["accuracy_mean"]),
            "delta_accuracy": float(moe["accuracy_mean"]) - float(no_small["accuracy_mean"]),
        }
    return output


def _load_paired_observations(eval_dir: Path, task_name: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    task_payload = json.loads(_task_result_path(eval_dir, task_name, seed).read_text(encoding="utf-8"))
    return task_payload["example_results"]


def _paired_uncertainty(eval_dir: Path, eval_spec: Any, bootstrap_samples: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task in eval_spec.internal_tasks + eval_spec.lm_tasks:
        task_result: dict[str, Any] = {}
        all_rows: list[dict[str, Any]] = []
        for seed in eval_spec.seeds:
            for row in _load_paired_observations(eval_dir, task.name, seed):
                all_rows.append({"seed": seed, **row})
        for left, right in COMPARISONS:
            pair_result = {}
            for metric_name in ("logit_kl_to_teacher", "nll"):
                values_left = [float(row["models"][left][metric_name]) for row in all_rows]
                values_right = [float(row["models"][right][metric_name]) for row in all_rows]
                weights = [float(row["valid_tokens"]) for row in all_rows]
                pair_result[metric_name] = paired_bootstrap_summary(
                    values_left,
                    values_right,
                    weights=weights,
                    higher_is_better=False,
                    num_samples=bootstrap_samples,
                    seed=1200 + _stable_seed(task.name, left, right, metric_name),
                )
            task_result[f"{left}__vs__{right}"] = pair_result
        output[task.name] = {"category": task.category, "comparisons": task_result}

    for task in eval_spec.multichoice_tasks:
        task_result = {}
        all_rows = []
        for seed in eval_spec.seeds:
            for row in _load_paired_observations(eval_dir, task.name, seed):
                all_rows.append({"seed": seed, **row})
        for left, right in COMPARISONS:
            values_left = [1.0 if row["models"][left]["correct"] else 0.0 for row in all_rows]
            values_right = [1.0 if row["models"][right]["correct"] else 0.0 for row in all_rows]
            task_result[f"{left}__vs__{right}"] = {
                "accuracy": paired_bootstrap_summary(
                    values_left,
                    values_right,
                    weights=None,
                    higher_is_better=True,
                    num_samples=bootstrap_samples,
                    seed=3200 + _stable_seed(task.name, left, right, "accuracy"),
                )
            }
        output[task.name] = {"category": task.category, "comparisons": task_result}
    return output


def _task_examples(config: Any) -> dict[str, dict[str, Any]]:
    eval_spec = adaptive_eval_spec(config)
    tasks: dict[str, dict[str, Any]] = {}
    for task in eval_spec.internal_tasks + eval_spec.lm_tasks:
        examples, slice_definition, sample_metadata = _build_lm_examples(task, debug_mode=False)
        tasks[task.name] = {
            "category": task.category,
            "examples": examples,
            "slice_definition": slice_definition,
            "sample_metadata": sample_metadata,
        }
    for task in eval_spec.multichoice_tasks:
        examples, slice_definition, sample_metadata = _build_multichoice_examples(task, debug_mode=False)
        tasks[task.name] = {
            "category": task.category,
            "examples": examples,
            "slice_definition": slice_definition,
            "sample_metadata": sample_metadata,
        }
    return tasks


def _expert_usage(config: Any, train_dir: Path) -> dict[str, Any]:
    eval_spec = adaptive_eval_spec(config)
    tasks = _task_examples(config)
    path_specs = load_mixture_path_specs(config)
    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    output: dict[str, Any] = {"adaptive_bridge_moe": {}, "adaptive_bridge_no_small": {}}
    try:
        for seed in eval_spec.seeds:
            LOGGER.info("expert-usage seed=%s", seed)
            seed_config = config
            models, _ = _load_models_for_seed(seed_config, backbones, path_specs, train_dir, seed)
            for model_name in ("adaptive_bridge_moe", "adaptive_bridge_no_small"):
                model = models[model_name]
                for task_name, task_payload in tasks.items():
                    stats = gate_stats_for_model_task(
                        model,
                        backbones.tokenizer,
                        task_payload["examples"],
                        task_category=task_payload["category"],
                        max_seq_len=eval_spec.max_seq_len,
                        length_normalize_choices=eval_spec.length_normalize_choices,
                        device=backbones.device,
                    )
                    output[model_name].setdefault(task_name, {"family": task_family(task_name), "per_seed": {}})
                    output[model_name][task_name]["per_seed"][str(seed)] = stats.__dict__

        for model_name, task_rows in output.items():
            for task_name, payload in task_rows.items():
                per_seed_stats = [payload["per_seed"][str(seed)] for seed in eval_spec.seeds]
                items = [SimpleNamespace(**row) for row in per_seed_stats]
                payload["aggregate"] = aggregate_stats(items)
    finally:
        del backbones
    return output


def _task_conditional_pattern(expert_usage: dict[str, Any]) -> dict[str, Any]:
    moe_usage = expert_usage["adaptive_bridge_moe"]
    lm_style_tasks = [task for task, payload in moe_usage.items() if payload["family"] == "lm_style"]
    multichoice_tasks = [task for task, payload in moe_usage.items() if payload["family"] == "multichoice"]

    def _family_average(task_names: list[str], key: str) -> float:
        values = [float(moe_usage[task]["aggregate"][f"{key}_mean"]) for task in task_names]
        return sum(values) / max(1, len(values))

    per_task = {}
    for task_name, payload in moe_usage.items():
        agg = payload["aggregate"]
        per_task[task_name] = {
            "family": payload["family"],
            "weight_bridge_mean": agg["weight_bridge_mean"],
            "weight_path_b_mean": agg["weight_path_b_mean"],
            "weight_path_a_mean": agg["weight_path_a_mean"],
            "delegated_total_mean": agg["weight_path_b_mean"] + agg["weight_path_a_mean"],
            "path_a_minus_path_b_mean": agg["weight_path_a_mean"] - agg["weight_path_b_mean"],
        }
    return {
        "family_means": {
            "lm_style": {
                "bridge_weight_mean": _family_average(lm_style_tasks, "weight_bridge"),
                "delegated_total_mean": _family_average(lm_style_tasks, "weight_path_b") + _family_average(lm_style_tasks, "weight_path_a"),
                "path_b_weight_mean": _family_average(lm_style_tasks, "weight_path_b"),
                "path_a_weight_mean": _family_average(lm_style_tasks, "weight_path_a"),
            },
            "multichoice": {
                "bridge_weight_mean": _family_average(multichoice_tasks, "weight_bridge"),
                "delegated_total_mean": _family_average(multichoice_tasks, "weight_path_b") + _family_average(multichoice_tasks, "weight_path_a"),
                "path_b_weight_mean": _family_average(multichoice_tasks, "weight_path_b"),
                "path_a_weight_mean": _family_average(multichoice_tasks, "weight_path_a"),
            },
        },
        "per_task": per_task,
    }


def main() -> None:
    """Run the bounded hardening checks and paired uncertainty analysis."""

    args = parse_args()
    configure_logging()
    config = load_config(args.config)
    train_dir = Path(args.train_dir)
    eval_dir = Path(args.eval_dir)
    ensure_dir(Path(args.output_path).parent)

    integrity = _integrity_checks(config, train_dir, eval_dir)
    eval_spec = adaptive_eval_spec(config)
    paired_uncertainty = _paired_uncertainty(eval_dir, eval_spec, bootstrap_samples=args.bootstrap_samples)
    guardrail = _guardrail_checks(config, eval_dir)
    no_small = _no_small_separation(eval_dir)
    eval_results = _load_eval_results(eval_dir)
    summary_index = _summary_row_index(eval_results["summary_rows"])
    expert_usage = _expert_usage(config, train_dir)
    task_pattern = _task_conditional_pattern(expert_usage)

    output = {
        "config_path": args.config,
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "same_run_vs_frozen_reference_distinction": {
            "same_run": "Use same-run comparisons only for adaptive_bridge_moe vs adaptive_bridge_no_small and bridge baselines.",
            "frozen_reference": "Use frozen-reference comparisons only for adaptive_bridge_moe/adaptive_bridge_no_small vs frozen_v060_tokenwise.",
        },
        "reference_integrity": integrity,
        "internal_guardrail": guardrail,
        "no_small_separation": no_small,
        "paired_uncertainty": paired_uncertainty,
        "expert_usage": expert_usage,
        "task_conditional_pattern": task_pattern,
        "unresolved_weakness": {
            "arc_easy_remains_unresolved": not (
                float(summary_index[("arc_easy", "adaptive_bridge_moe")]["accuracy_mean"])
                > float(summary_index[("arc_easy", "bridge_only_strong")]["accuracy_mean"])
                and float(summary_index[("arc_easy", "adaptive_bridge_moe")]["accuracy_mean"])
                > float(summary_index[("arc_easy", "bridge_only_param_matched")]["accuracy_mean"])
            ),
        },
    }
    save_json(Path(args.output_path), output)


if __name__ == "__main__":
    main()
