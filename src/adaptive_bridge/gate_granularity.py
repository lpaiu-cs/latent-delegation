"""Inference-only gate granularity audit for the adaptive-bridge milestone."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.adaptive_bridge.analysis_runtime import (
    RouteAblatedAdaptiveModel,
    collect_task_gate_weight_batches,
    gate_granularity_policies,
    lm_example_metrics,
    multichoice_metrics_from_logits,
    task_global_mean_gate_weights,
)
from src.adaptive_bridge.common import adaptive_eval_spec
from src.adaptive_bridge.evaluate import _build_lm_examples, _build_multichoice_examples, _load_models_for_seed
from src.models.backbone_loader import load_backbones
from src.utils.io import ensure_dir, load_config, save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.v0_6.idea4_common import load_mixture_path_specs
from src.v0_9.task_scoring import build_choice_batch


LOGGER = get_logger(__name__)


TASK_NAMES = ("confirmation_holdout", "lambada_openai", "piqa", "arc_easy")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the gate granularity audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-path", required=True)
    return parser.parse_args()


def _selected_tasks(config: Any) -> dict[str, dict[str, Any]]:
    eval_spec = adaptive_eval_spec(config)
    tasks: dict[str, dict[str, Any]] = {}
    for task in eval_spec.internal_tasks + eval_spec.lm_tasks:
        if task.name not in TASK_NAMES:
            continue
        examples, slice_definition, sample_metadata = _build_lm_examples(task, debug_mode=False)
        tasks[task.name] = {
            "category": task.category,
            "examples": examples,
            "slice_definition": slice_definition,
            "sample_metadata": sample_metadata,
        }
    for task in eval_spec.multichoice_tasks:
        if task.name not in TASK_NAMES:
            continue
        examples, slice_definition, sample_metadata = _build_multichoice_examples(task, debug_mode=False)
        tasks[task.name] = {
            "category": task.category,
            "examples": examples,
            "slice_definition": slice_definition,
            "sample_metadata": sample_metadata,
        }
    return tasks


def _aggregate_lm_example_rows(example_rows: list[dict[str, Any]], variant_names: list[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for variant in variant_names:
        valid_tokens = sum(float(row["models"][variant]["valid_tokens"]) for row in example_rows)
        nll_sum = sum(float(row["models"][variant]["nll"]) * float(row["models"][variant]["valid_tokens"]) for row in example_rows)
        kl_sum = sum(
            float(row["models"][variant]["logit_kl_to_teacher"]) * float(row["models"][variant]["valid_tokens"])
            for row in example_rows
        )
        truncated = sum(1.0 if row["models"][variant]["truncated"] else 0.0 for row in example_rows)
        denom = max(1.0, valid_tokens)
        metrics[variant] = {
            "nll": nll_sum / denom,
            "logit_kl_to_teacher": kl_sum / denom,
            "valid_tokens": valid_tokens,
            "example_count": float(len(example_rows)),
            "truncation_rate": truncated / max(1, len(example_rows)),
        }
    return metrics


def _aggregate_multichoice_example_rows(example_rows: list[dict[str, Any]], variant_names: list[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for variant in variant_names:
        accuracy = sum(1.0 if row["models"][variant]["correct"] else 0.0 for row in example_rows) / max(1, len(example_rows))
        mean_margin = sum(float(row["models"][variant]["score_margin"]) for row in example_rows) / max(1, len(example_rows))
        truncation_count = 0.0
        total_flags = 0
        for row in example_rows:
            flags = row["models"][variant]["truncated_flags"]
            truncation_count += sum(1.0 for flag in flags if flag)
            total_flags += len(flags)
        metrics[variant] = {
            "accuracy": accuracy,
            "mean_choice_margin": mean_margin,
            "example_count": float(len(example_rows)),
            "truncation_rate": truncation_count / max(1, total_flags),
        }
    return metrics


def _summary_rows(seed_payloads: dict[str, dict[str, Any]], variant_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_name, task_payload in seed_payloads.items():
        category = task_payload["category"]
        for variant in variant_names:
            entries = [seed_entry["metrics"][variant] for seed_entry in task_payload["per_seed"].values()]
            summary = {
                "task_name": task_name,
                "task_category": category,
                "variant_name": variant,
            }
            for key in entries[0]:
                summary[f"{key}_mean"] = sum(float(entry[key]) for entry in entries) / len(entries)
            rows.append(summary)
    return rows


def _variant_deltas(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task_name in TASK_NAMES:
        task_rows = {row["variant_name"]: row for row in summary_rows if row["task_name"] == task_name}
        full = task_rows["full_tokenwise_gate"]
        output[task_name] = {}
        for variant_name, row in task_rows.items():
            if variant_name == "full_tokenwise_gate":
                continue
            if "accuracy_mean" in row:
                output[task_name][variant_name] = {
                    "delta_accuracy_vs_full": float(row["accuracy_mean"]) - float(full["accuracy_mean"]),
                    "delta_margin_vs_full": float(row["mean_choice_margin_mean"]) - float(full["mean_choice_margin_mean"]),
                }
            else:
                output[task_name][variant_name] = {
                    "delta_kl_vs_full": float(row["logit_kl_to_teacher_mean"]) - float(full["logit_kl_to_teacher_mean"]),
                    "delta_nll_vs_full": float(row["nll_mean"]) - float(full["nll_mean"]),
                }
    return output


def _global_mean_weights_for_task(
    adaptive_moe: Any,
    tokenizer: Any,
    task_payload: dict[str, Any],
    *,
    max_seq_len: int,
    device: Any,
) -> Any:
    gate_batches, mask_batches = collect_task_gate_weight_batches(
        adaptive_moe,
        tokenizer,
        task_payload["examples"],
        task_category=task_payload["category"],
        max_seq_len=max_seq_len,
        device=device,
    )
    return task_global_mean_gate_weights(gate_batches, mask_batches)


def main() -> None:
    """Run inference-only gate granularity analyses for the trained checkpoints."""

    args = parse_args()
    configure_logging()
    config = load_config(args.config)
    train_dir = Path(args.train_dir)
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(Path(args.results_path).parent)
    eval_spec = adaptive_eval_spec(config)
    tasks = _selected_tasks(config)
    path_specs = load_mixture_path_specs(config)
    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    policies = gate_granularity_policies()
    variant_names = [policy.name for policy in policies] + ["adaptive_bridge_no_small"]
    seed_payloads: dict[str, dict[str, Any]] = {
        task_name: {"category": payload["category"], "per_seed": {}}
        for task_name, payload in tasks.items()
    }
    try:
        for seed in eval_spec.seeds:
            LOGGER.info("gate_granularity seed=%s", seed)
            models, diagnostics = _load_models_for_seed(config, backbones, path_specs, train_dir, seed)
            teacher = models["full_large"]
            adaptive_moe = models["adaptive_bridge_moe"]
            no_small = models["adaptive_bridge_no_small"]

            for task_name, task_payload in tasks.items():
                global_mean_weights = _global_mean_weights_for_task(
                    adaptive_moe,
                    backbones.tokenizer,
                    task_payload,
                    max_seq_len=eval_spec.max_seq_len,
                    device=backbones.device,
                )
                policy_models = {}
                for policy in policies:
                    kwargs = {}
                    if policy.gate_granularity == "global_mean":
                        kwargs["global_mean_weights"] = global_mean_weights
                    policy_models[policy.name] = RouteAblatedAdaptiveModel(adaptive_moe, policy, **kwargs)

                example_rows: list[dict[str, Any]] = []
                if task_payload["category"] == "multichoice":
                    for example in task_payload["examples"]:
                        batch = build_choice_batch(
                            backbones.tokenizer,
                            example.prompt,
                            example.choices,
                            max_seq_len=eval_spec.max_seq_len,
                            device=backbones.device,
                        )
                        row = {
                            "example_id": example.example_id,
                            "label_index": example.label_index,
                            "metadata": example.metadata,
                            "models": {},
                        }
                        for variant_name, model in policy_models.items():
                            outputs = model(batch.input_ids, attention_mask=batch.attention_mask)
                            row["models"][variant_name] = multichoice_metrics_from_logits(
                                outputs.logits,
                                batch,
                                label_index=example.label_index,
                                length_normalize=eval_spec.length_normalize_choices,
                            )
                        outputs = no_small(batch.input_ids, attention_mask=batch.attention_mask)
                        row["models"]["adaptive_bridge_no_small"] = multichoice_metrics_from_logits(
                            outputs.logits,
                            batch,
                            label_index=example.label_index,
                            length_normalize=eval_spec.length_normalize_choices,
                        )
                        example_rows.append(row)
                    metrics = _aggregate_multichoice_example_rows(example_rows, variant_names)
                else:
                    for example in task_payload["examples"]:
                        row = {
                            "example_id": example.example_id,
                            "metadata": example.metadata,
                            "models": {},
                        }
                        for variant_name, model in policy_models.items():
                            metrics_row = lm_example_metrics(
                                model,
                                teacher,
                                backbones.tokenizer,
                                example.text,
                                max_seq_len=eval_spec.max_seq_len,
                                device=backbones.device,
                            )
                            if metrics_row is not None:
                                row["models"][variant_name] = metrics_row
                        no_small_row = lm_example_metrics(
                            no_small,
                            teacher,
                            backbones.tokenizer,
                            example.text,
                            max_seq_len=eval_spec.max_seq_len,
                            device=backbones.device,
                        )
                        if no_small_row is None:
                            continue
                        row["models"]["adaptive_bridge_no_small"] = no_small_row
                        example_rows.append(row)
                    metrics = _aggregate_lm_example_rows(example_rows, variant_names)

                task_dir = ensure_dir(output_dir / task_name / f"seed_{seed}")
                save_json(
                    task_dir / "results.json",
                    {
                        "task_name": task_name,
                        "seed": seed,
                        "category": task_payload["category"],
                        "slice_definition": task_payload["slice_definition"],
                        "sample_metadata": task_payload["sample_metadata"],
                        "model_diagnostics": diagnostics,
                        "policies": {
                            policy.name: {
                                "allowed_experts": None if policy.allowed_experts is None else list(policy.allowed_experts),
                                "gate_granularity": policy.gate_granularity,
                            }
                            for policy in policies
                        },
                        "global_mean_weights": [float(value) for value in global_mean_weights.tolist()],
                        "metrics": metrics,
                        "example_results": example_rows,
                    },
                )
                seed_payloads[task_name]["per_seed"][str(seed)] = {
                    "metrics": metrics,
                    "example_count": len(example_rows),
                    "global_mean_weights": [float(value) for value in global_mean_weights.tolist()],
                }
    finally:
        del backbones

    summary_rows = _summary_rows(seed_payloads, variant_names)
    output = {
        "config_path": args.config,
        "train_dir": str(train_dir),
        "output_dir": str(output_dir),
        "seeds": adaptive_eval_spec(config).seeds,
        "same_run_vs_frozen_reference_distinction": {
            "same_run": "All gate granularity variants are inference-only same-run counterfactuals on adaptive_bridge_moe or adaptive_bridge_no_small.",
            "frozen_reference": "Frozen v0.6.0 comparisons are intentionally excluded from this artifact and remain in the main eval artifacts.",
        },
        "variant_order": variant_names,
        "policies": {
            policy.name: {
                "allowed_experts": None if policy.allowed_experts is None else list(policy.allowed_experts),
                "gate_granularity": policy.gate_granularity,
            }
            for policy in policies
        },
        "tasks": seed_payloads,
        "summary_rows": summary_rows,
        "variant_deltas_vs_full": _variant_deltas(summary_rows),
    }
    save_json(Path(args.results_path), output)


if __name__ == "__main__":
    main()
