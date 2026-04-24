"""Bounded evaluation for the adaptive-bridge first milestone."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from src.adaptive_bridge.common import (
    ADAPTIVE_BRIDGE_MODEL_ORDER,
    adaptive_bridge_gate_settings,
    adaptive_bridge_settings,
    adaptive_eval_spec,
    checkpoint_path_from_template,
    clone_config_with_seed,
    maybe_load_checkpoint,
)
from src.adaptive_bridge.models import BridgeAwareResidualMoE, BridgeAwareResidualMoENoSmall
from src.eval.metrics import perplexity_from_loss, shifted_cross_entropy, shifted_kl_divergence
from src.models.backbone_loader import LoadedBackbones, load_backbones
from src.models.baselines import BridgeOnlyParamMatchedModel, FullLargeModel, SkipOnlyLargeModel
from src.train.trainer_utils import load_checkpoint
from src.utils.io import ensure_dir, export_run_metadata, load_config, save_config_snapshot, save_csv, save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything
from src.v0_6.idea4_common import load_mixture_path_specs
from src.v0_6.idea4_tokenwise_models import TwoPathTokenwiseMixtureHybrid
from src.v0_9.task_scoring import LMExample, TaskExample, sample_indices, score_multichoice_example


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse adaptive-bridge evaluation CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-dir", default="outputs/adaptive_bridge/train")
    parser.add_argument("--output-dir", default="outputs/adaptive_bridge/eval")
    parser.add_argument("--results-path", default="outputs/adaptive_bridge/eval/results.json")
    parser.add_argument("--summary-path", default="outputs/adaptive_bridge/eval/summary.csv")
    parser.add_argument("--report-path", default="outputs/adaptive_bridge/eval/summary_note.md")
    return parser.parse_args()


def _clean_text(text: str) -> str:
    return " ".join(text.replace("[title]", " ").split()).strip()


def _synthetic_lm_examples(name: str, sample_count: int) -> list[LMExample]:
    texts = {
        "development_holdout": [
            "The latent bridge keeps the large model in charge of the final logits.",
            "A frozen delegated path can still provide a useful residual correction.",
        ],
        "confirmation_holdout": [
            "Token wise routing lets the model vary expert usage across positions.",
            "A strong bridge expert may recover generic language modeling behavior.",
        ],
        "lambada_openai": [
            "The boy went to school and forgot his lunch at home so he felt very hungry by noon.",
            "The mechanic tightened the final bolt and then closed the heavy metal hood.",
        ],
    }
    rows = texts[name][:sample_count]
    return [
        LMExample(
            task_name=name,
            example_id=f"{name}_{index}",
            text=row,
            metadata={"dataset_index": index, "synthetic": True},
        )
        for index, row in enumerate(rows)
    ]


def _synthetic_multichoice_examples(name: str, sample_count: int) -> list[TaskExample]:
    examples_by_name = {
        "piqa": [
            TaskExample(
                task_name="piqa",
                example_id="piqa_0",
                prompt="Goal: dry a wet counter.\nSolution:",
                choices=[" wipe it with a towel", " pour more water on it"],
                label_index=0,
                metadata={"synthetic": True},
            ),
            TaskExample(
                task_name="piqa",
                example_id="piqa_1",
                prompt="Goal: open a locked door with the key you have.\nSolution:",
                choices=[" insert the key and turn it", " hide the key under the mat"],
                label_index=0,
                metadata={"synthetic": True},
            ),
        ],
        "arc_easy": [
            TaskExample(
                task_name="arc_easy",
                example_id="arc_easy_0",
                prompt="Question: Which object gives off light?\nAnswer:",
                choices=[" a lamp", " a pillow", " a blanket", " a spoon"],
                label_index=0,
                metadata={"synthetic": True},
            ),
            TaskExample(
                task_name="arc_easy",
                example_id="arc_easy_1",
                prompt="Question: Which season is usually the coldest?\nAnswer:",
                choices=[" winter", " summer", " spring", " autumn"],
                label_index=0,
                metadata={"synthetic": True},
            ),
        ],
    }
    return examples_by_name[name][:sample_count]


def _build_lm_examples(task: Any, debug_mode: bool) -> tuple[list[LMExample], dict[str, Any], list[dict[str, Any]]]:
    if debug_mode:
        examples = _synthetic_lm_examples(task.name, task.sample_count)
        return examples, {
            "task_name": task.name,
            "dataset_name": "synthetic",
            "split": task.split,
            "sample_count_requested": task.sample_count,
            "sample_count_actual": len(examples),
            "sampling_seed": task.sampling_seed,
        }, [
            {"task_name": task.name, "example_id": example.example_id, "synthetic": True}
            for example in examples
        ]

    dataset = load_dataset(task.dataset_name, task.dataset_config_name, split=task.split)
    selected = sample_indices(len(dataset), task.sample_count, task.sampling_seed)
    examples: list[LMExample] = []
    sample_metadata: list[dict[str, Any]] = []
    for dataset_index in selected:
        record = dataset[dataset_index]
        if task.name == "lambada_openai":
            text = _clean_text(str(record["text"]))
        else:
            text = _clean_text(str(record["text"]))
        if not text:
            continue
        examples.append(
            LMExample(
                task_name=task.name,
                example_id=str(dataset_index),
                text=text,
                metadata={"dataset_index": dataset_index},
            )
        )
        sample_metadata.append(
            {
                "task_name": task.name,
                "dataset_name": task.dataset_name,
                "dataset_config_name": task.dataset_config_name,
                "split": task.split,
                "dataset_index": dataset_index,
                "example_id": str(dataset_index),
            }
        )
    return examples, {
        "task_name": task.name,
        "dataset_name": task.dataset_name,
        "dataset_config_name": task.dataset_config_name,
        "split": task.split,
        "sample_count_requested": task.sample_count,
        "sample_count_actual": len(examples),
        "sampling_seed": task.sampling_seed,
    }, sample_metadata


def _build_multichoice_examples(task: Any, debug_mode: bool) -> tuple[list[TaskExample], dict[str, Any], list[dict[str, Any]]]:
    if debug_mode:
        examples = _synthetic_multichoice_examples(task.name, task.sample_count)
        return examples, {
            "task_name": task.name,
            "dataset_name": "synthetic",
            "split": task.split,
            "sample_count_requested": task.sample_count,
            "sample_count_actual": len(examples),
            "sampling_seed": task.sampling_seed,
        }, [
            {"task_name": task.name, "example_id": example.example_id, "synthetic": True}
            for example in examples
        ]

    dataset = load_dataset(task.dataset_name, task.dataset_config_name, split=task.split)
    selected = sample_indices(len(dataset), task.sample_count, task.sampling_seed)
    examples: list[TaskExample] = []
    sample_metadata: list[dict[str, Any]] = []
    skipped = 0
    for dataset_index in selected:
        record = dataset[dataset_index]
        try:
            if task.name == "piqa":
                prompt = f"Goal: {_clean_text(record['goal'])}\nSolution:"
                choices = [f" {_clean_text(record['sol1'])}", f" {_clean_text(record['sol2'])}"]
                label_index = int(record["label"])
            elif task.name == "arc_easy":
                choice_labels = list(record["choices"]["label"])
                choice_texts = [f" {_clean_text(text)}" for text in record["choices"]["text"]]
                answer_key = str(record["answerKey"])
                label_index = choice_labels.index(answer_key)
                prompt = f"Question: {_clean_text(record['question'])}\nAnswer:"
                choices = choice_texts
            else:
                raise ValueError(f"Unsupported multichoice task: {task.name}")
        except Exception:
            skipped += 1
            continue
        example = TaskExample(
            task_name=task.name,
            example_id=str(record.get("id", dataset_index)),
            prompt=prompt,
            choices=choices,
            label_index=label_index,
            metadata={"dataset_index": dataset_index},
        )
        examples.append(example)
        sample_metadata.append(
            {
                "task_name": task.name,
                "dataset_name": task.dataset_name,
                "dataset_config_name": task.dataset_config_name,
                "split": task.split,
                "dataset_index": dataset_index,
                "example_id": example.example_id,
            }
        )
    return examples, {
        "task_name": task.name,
        "dataset_name": task.dataset_name,
        "dataset_config_name": task.dataset_config_name,
        "split": task.split,
        "sample_count_requested": task.sample_count,
        "sample_count_actual": len(examples),
        "sampling_seed": task.sampling_seed,
        "skipped_examples": skipped,
    }, sample_metadata


def _example_tensors(tokenizer: Any, text: str, max_seq_len: int, device: torch.device) -> dict[str, torch.Tensor | bool]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "truncated": bool(input_ids.shape[1] >= max_seq_len),
    }


def _load_tokenwise_payload(model: TwoPathTokenwiseMixtureHybrid, payload: dict[str, Any], device: torch.device) -> None:
    model.entry_projector_b.load_state_dict(payload["entry_projector_b"])
    model.entry_projector_a.load_state_dict(payload["entry_projector_a"])
    model.return_adapter_b.load_state_dict(payload["return_adapter_b"])
    model.return_adapter_a.load_state_dict(payload["return_adapter_a"])
    model.set_static_prior_logits(payload["static_prior_logits"].to(device=device, dtype=torch.float32))
    model.gate_network.load_state_dict(payload["gate_network"])


def _load_adaptive_payload(model: BridgeAwareResidualMoE, payload: dict[str, Any], device: torch.device) -> None:
    model.entry_projector_b.load_state_dict(payload["entry_projector_b"])
    model.entry_projector_a.load_state_dict(payload["entry_projector_a"])
    model.return_adapter_b.load_state_dict(payload["return_adapter_b"])
    model.return_adapter_a.load_state_dict(payload["return_adapter_a"])
    model.bridge_expert.load_state_dict(payload["bridge_expert"])
    model.gate_network.load_state_dict(payload["gate_network"])
    model.set_expert_prior_logits(payload["expert_prior_logits"].to(device=device, dtype=torch.float32))


def _load_models_for_seed(
    config: Any,
    backbones: LoadedBackbones,
    path_specs: list[Any],
    train_dir: Path,
    seed: int,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    models: dict[str, torch.nn.Module] = {
        "full_large": FullLargeModel(config, backbones.large_model),
        "skip_only": SkipOnlyLargeModel(config, backbones.large_model),
    }
    diagnostics: dict[str, Any] = {}

    for variant, model_cls in {
        "adaptive_bridge_moe": BridgeAwareResidualMoE,
        "adaptive_bridge_no_small": BridgeAwareResidualMoENoSmall,
    }.items():
        checkpoint_path = train_dir / f"seed_{seed}" / f"{variant}_checkpoint.pt"
        if checkpoint_path.exists():
            payload = load_checkpoint(checkpoint_path, backbones.device)
            model = model_cls(config, backbones.large_model, backbones.small_model, path_specs)
            _load_adaptive_payload(model, payload, backbones.device)
            models[variant] = model
            diagnostics[variant] = {"status": "loaded", "checkpoint_path": str(checkpoint_path)}
        else:
            diagnostics[variant] = {"status": "missing_checkpoint", "checkpoint_path": str(checkpoint_path)}

    for variant in ("bridge_only_strong", "bridge_only_param_matched"):
        checkpoint_path = train_dir / f"seed_{seed}" / f"{variant}_checkpoint.pt"
        if checkpoint_path.exists():
            payload = load_checkpoint(checkpoint_path, backbones.device)
            rank = int(payload["bridge"]["down.weight"].shape[0])
            model = BridgeOnlyParamMatchedModel(config, backbones.large_model, rank=rank)
            model.bridge.load_state_dict(payload["bridge"])
            model.gate.load_state_dict(payload["gate"])
            models[variant] = model
            diagnostics[variant] = {"status": "loaded", "checkpoint_path": str(checkpoint_path), "rank": rank}
        else:
            diagnostics[variant] = {"status": "missing_checkpoint", "checkpoint_path": str(checkpoint_path)}

    frozen_tokenwise_path = checkpoint_path_from_template(adaptive_bridge_settings(config).frozen_tokenwise_template, seed)
    payload = maybe_load_checkpoint(frozen_tokenwise_path, backbones.device)
    if payload is not None:
        model = TwoPathTokenwiseMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
        _load_tokenwise_payload(model, payload, backbones.device)
        models["frozen_v060_tokenwise"] = model
        diagnostics["frozen_v060_tokenwise"] = {"status": "loaded", "checkpoint_path": str(frozen_tokenwise_path)}
    else:
        diagnostics["frozen_v060_tokenwise"] = {
            "status": "missing_checkpoint",
            "checkpoint_path": None if frozen_tokenwise_path is None else str(frozen_tokenwise_path),
        }

    for model in models.values():
        model.eval()
    return models, diagnostics


def _lm_metrics_for_examples(
    models: dict[str, torch.nn.Module],
    tokenizer: Any,
    examples: list[LMExample],
    max_seq_len: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    model_names = [name for name in ADAPTIVE_BRIDGE_MODEL_ORDER if name in models]
    totals = {
        model_name: {
            "valid_tokens": 0.0,
            "nll_sum": 0.0,
            "kl_sum": 0.0,
            "truncated_examples": 0.0,
        }
        for model_name in model_names
    }
    example_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for example in examples:
            tensors = _example_tensors(tokenizer, example.text, max_seq_len=max_seq_len, device=device)
            labels = tensors["labels"]
            valid_tokens = int((labels[:, 1:] != -100).sum().item())
            if valid_tokens == 0:
                continue
            teacher_logits = models["full_large"](
                tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
            ).logits
            row = {
                "example_id": example.example_id,
                "metadata": example.metadata,
                "valid_tokens": valid_tokens,
                "truncated": tensors["truncated"],
                "models": {},
            }
            for model_name in model_names:
                student_logits = models[model_name](
                    tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                ).logits
                nll = float(shifted_cross_entropy(student_logits, labels).cpu())
                kl = 0.0 if model_name == "full_large" else float(shifted_kl_divergence(student_logits, teacher_logits, labels).cpu())
                row["models"][model_name] = {
                    "nll": nll,
                    "perplexity": perplexity_from_loss(nll),
                    "logit_kl_to_teacher": kl,
                }
                totals[model_name]["valid_tokens"] += float(valid_tokens)
                totals[model_name]["nll_sum"] += nll * valid_tokens
                totals[model_name]["kl_sum"] += kl * valid_tokens
                totals[model_name]["truncated_examples"] += 1.0 if tensors["truncated"] else 0.0
            example_rows.append(row)

    metrics: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        valid_tokens = max(1.0, totals[model_name]["valid_tokens"])
        mean_nll = totals[model_name]["nll_sum"] / valid_tokens
        mean_kl = totals[model_name]["kl_sum"] / valid_tokens
        metrics[model_name] = {
            "nll": mean_nll,
            "perplexity": perplexity_from_loss(mean_nll),
            "logit_kl_to_teacher": mean_kl,
            "truncation_rate": totals[model_name]["truncated_examples"] / max(1, len(example_rows)),
            "valid_tokens": totals[model_name]["valid_tokens"],
            "example_count": float(len(example_rows)),
        }
    return metrics, example_rows


def _multichoice_metrics_for_examples(
    models: dict[str, torch.nn.Module],
    tokenizer: Any,
    examples: list[TaskExample],
    max_seq_len: int,
    device: torch.device,
    *,
    length_normalize: bool,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    model_names = [name for name in ADAPTIVE_BRIDGE_MODEL_ORDER if name in models]
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for example in examples:
            row = {
                "example_id": example.example_id,
                "label_index": example.label_index,
                "metadata": example.metadata,
                "models": {},
            }
            for model_name in model_names:
                row["models"][model_name] = score_multichoice_example(
                    models[model_name],
                    tokenizer,
                    example,
                    max_seq_len=max_seq_len,
                    device=device,
                    length_normalize=length_normalize,
                )
            rows.append(row)

    metrics: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        accuracy = sum(1.0 for row in rows if row["models"][model_name]["correct"]) / max(1, len(rows))
        mean_margin = sum(row["models"][model_name]["score_margin"] for row in rows) / max(1, len(rows))
        truncation_rate = (
            sum(1.0 for row in rows if any(bool(flag) for flag in row["models"][model_name]["truncated_flags"]))
            / max(1, len(rows))
        )
        metrics[model_name] = {
            "accuracy": accuracy,
            "mean_choice_margin": mean_margin,
            "truncation_rate": truncation_rate,
            "example_count": float(len(rows)),
        }
    return metrics, rows


def _aggregate_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault((row["task_name"], row["model_name"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (task_name, model_name), rows in sorted(grouped.items()):
        metric_names = [key for key in rows[0].keys() if key not in {"task_name", "task_category", "seed", "model_name"}]
        summary_row = {
            "task_name": task_name,
            "task_category": rows[0]["task_category"],
            "model_name": model_name,
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in rows]
            summary_row[f"{metric_name}_mean"] = float(sum(values) / len(values))
        summary_rows.append(summary_row)
    return summary_rows


def _summary_lookup(summary_rows: list[dict[str, Any]], task_name: str, model_name: str, metric_name: str) -> float | None:
    key = f"{metric_name}_mean"
    for row in summary_rows:
        if row["task_name"] == task_name and row["model_name"] == model_name and key in row:
            return float(row[key])
    return None


def _recommendation(summary_rows: list[dict[str, Any]], config: Any) -> dict[str, Any]:
    spec = adaptive_eval_spec(config)
    target = "adaptive_bridge_moe"
    reference = "frozen_v060_tokenwise"
    if _summary_lookup(summary_rows, "development_holdout", reference, "nll") is None:
        return {
            "status": "blocked_missing_reference",
            "recommendation": None,
            "reason": "Frozen v0.6.0 token-wise checkpoint is required for the main adaptive-bridge decision.",
        }

    checks = {
        "internal_dev_kl": (_summary_lookup(summary_rows, "development_holdout", target, "logit_kl_to_teacher"), _summary_lookup(summary_rows, "development_holdout", reference, "logit_kl_to_teacher"), spec.internal_kl_tolerance),
        "internal_dev_nll": (_summary_lookup(summary_rows, "development_holdout", target, "nll"), _summary_lookup(summary_rows, "development_holdout", reference, "nll"), spec.internal_nll_tolerance),
        "internal_confirm_kl": (_summary_lookup(summary_rows, "confirmation_holdout", target, "logit_kl_to_teacher"), _summary_lookup(summary_rows, "confirmation_holdout", reference, "logit_kl_to_teacher"), spec.internal_kl_tolerance),
        "internal_confirm_nll": (_summary_lookup(summary_rows, "confirmation_holdout", target, "nll"), _summary_lookup(summary_rows, "confirmation_holdout", reference, "nll"), spec.internal_nll_tolerance),
        "lambada_kl": (_summary_lookup(summary_rows, "lambada_openai", target, "logit_kl_to_teacher"), _summary_lookup(summary_rows, "lambada_openai", reference, "logit_kl_to_teacher"), spec.lambada_kl_tolerance),
        "lambada_nll": (_summary_lookup(summary_rows, "lambada_openai", target, "nll"), _summary_lookup(summary_rows, "lambada_openai", reference, "nll"), spec.lambada_nll_tolerance),
    }
    pass_fail: dict[str, bool] = {}
    for name, (candidate, baseline, tolerance) in checks.items():
        if candidate is None or baseline is None:
            pass_fail[name] = False
        else:
            pass_fail[name] = candidate <= baseline + tolerance

    bridge_baselines = ["bridge_only_strong", "bridge_only_param_matched"]
    recovered_task = None
    for task_name in ("piqa", "arc_easy"):
        target_accuracy = _summary_lookup(summary_rows, task_name, target, "accuracy")
        if target_accuracy is None:
            continue
        bridge_best = max(
            (
                _summary_lookup(summary_rows, task_name, model_name, "accuracy")
                for model_name in bridge_baselines
            ),
            default=None,
        )
        if bridge_best is not None and target_accuracy >= bridge_best + spec.multichoice_min_delta:
            recovered_task = task_name
            break

    should_continue = all(pass_fail.values()) and recovered_task is not None
    return {
        "status": "ok",
        "recommendation": "continue_adaptive_bridge" if should_continue else "stop_this_fork",
        "preservation_checks": pass_fail,
        "recovered_task": recovered_task,
    }


def _write_report(report_path: str | Path, summary_rows: list[dict[str, Any]], recommendation: dict[str, Any], config_path: str) -> None:
    lines = [
        "# Adaptive Bridge Summary",
        "",
        f"- Config: {config_path}",
        f"- Recommendation status: {recommendation['status']}",
        f"- Recommendation: {recommendation.get('recommendation')}",
        "",
        "## Summary rows",
        "",
    ]
    for row in summary_rows:
        metric_parts = [f"{key}={value:.6f}" for key, value in row.items() if key.endswith("_mean")]
        lines.append(f"- {row['task_name']} | {row['model_name']} | " + ", ".join(metric_parts))
    if recommendation.get("preservation_checks"):
        lines.extend(["", "## Preservation checks", ""])
        for key, value in recommendation["preservation_checks"].items():
            lines.append(f"- {key}: {value}")
    if recommendation.get("recovered_task") is not None:
        lines.append(f"- recovered_multichoice_task: {recommendation['recovered_task']}")
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    eval_spec = adaptive_eval_spec(config)
    seed_everything(config.training.seed)
    path_specs = load_mixture_path_specs(config)

    output_dir = ensure_dir(args.output_dir)
    save_config_snapshot(output_dir / "config_snapshot.yaml", config)
    export_run_metadata(
        output_dir / "metadata.json",
        config,
        {
            "stage": "adaptive_bridge_eval",
            "seeds": eval_spec.seeds,
            "train_dir": args.train_dir,
            "adaptive_bridge_settings": adaptive_bridge_settings(config).__dict__,
            "gate_settings": adaptive_bridge_gate_settings(config).__dict__,
        },
    )

    debug_mode = bool(config.model.debug_random_init)
    internal_payloads: dict[str, Any] = {}
    for task in eval_spec.internal_tasks:
        examples, slice_definition, sample_metadata = _build_lm_examples(task, debug_mode)
        internal_payloads[task.name] = {
            "task": task,
            "examples": examples,
            "slice_definition": slice_definition,
            "sample_metadata": sample_metadata,
        }

    lm_payloads: dict[str, Any] = {}
    for task in eval_spec.lm_tasks:
        examples, slice_definition, sample_metadata = _build_lm_examples(task, debug_mode)
        lm_payloads[task.name] = {
            "task": task,
            "examples": examples,
            "slice_definition": slice_definition,
            "sample_metadata": sample_metadata,
        }

    multichoice_payloads: dict[str, Any] = {}
    for task in eval_spec.multichoice_tasks:
        examples, slice_definition, sample_metadata = _build_multichoice_examples(task, debug_mode)
        multichoice_payloads[task.name] = {
            "task": task,
            "examples": examples,
            "slice_definition": slice_definition,
            "sample_metadata": sample_metadata,
        }

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    train_dir = Path(args.train_dir)
    seed_rows: list[dict[str, Any]] = []
    raw_results: dict[str, Any] = {
        "config_path": args.config,
        "train_dir": str(train_dir),
        "seeds": eval_spec.seeds,
        "tasks": {},
        "model_diagnostics": {},
    }

    for seed in eval_spec.seeds:
        LOGGER.info("adaptive_bridge eval seed=%s", seed)
        seed_config = clone_config_with_seed(config, seed)
        models, model_diagnostics = _load_models_for_seed(seed_config, backbones, path_specs, train_dir, seed)
        raw_results["model_diagnostics"][str(seed)] = model_diagnostics

        for task_name, payload in internal_payloads.items():
            metrics_by_model, example_rows = _lm_metrics_for_examples(
                models,
                backbones.tokenizer,
                payload["examples"],
                eval_spec.max_seq_len,
                backbones.device,
            )
            task_dir = ensure_dir(output_dir / task_name / f"seed_{seed}")
            save_json(task_dir / "results.json", {
                "task_name": task_name,
                "seed": seed,
                "task_category": "internal",
                "metrics_by_model": metrics_by_model,
                "example_results": example_rows,
                "slice_definition": payload["slice_definition"],
                "sample_metadata": payload["sample_metadata"],
            })
            raw_results["tasks"].setdefault(task_name, {"category": "internal", "seeds": []})
            raw_results["tasks"][task_name]["seeds"].append(seed)
            for model_name, metrics in metrics_by_model.items():
                seed_rows.append({
                    "task_name": task_name,
                    "task_category": "internal",
                    "seed": seed,
                    "model_name": model_name,
                    **metrics,
                })

        for task_name, payload in lm_payloads.items():
            metrics_by_model, example_rows = _lm_metrics_for_examples(
                models,
                backbones.tokenizer,
                payload["examples"],
                eval_spec.max_seq_len,
                backbones.device,
            )
            task_dir = ensure_dir(output_dir / task_name / f"seed_{seed}")
            save_json(task_dir / "results.json", {
                "task_name": task_name,
                "seed": seed,
                "task_category": "lm",
                "metrics_by_model": metrics_by_model,
                "example_results": example_rows,
                "slice_definition": payload["slice_definition"],
                "sample_metadata": payload["sample_metadata"],
            })
            raw_results["tasks"].setdefault(task_name, {"category": "lm", "seeds": []})
            raw_results["tasks"][task_name]["seeds"].append(seed)
            for model_name, metrics in metrics_by_model.items():
                seed_rows.append({
                    "task_name": task_name,
                    "task_category": "lm",
                    "seed": seed,
                    "model_name": model_name,
                    **metrics,
                })

        for task_name, payload in multichoice_payloads.items():
            metrics_by_model, example_rows = _multichoice_metrics_for_examples(
                models,
                backbones.tokenizer,
                payload["examples"],
                eval_spec.max_seq_len,
                backbones.device,
                length_normalize=eval_spec.length_normalize_choices,
            )
            task_dir = ensure_dir(output_dir / task_name / f"seed_{seed}")
            save_json(task_dir / "results.json", {
                "task_name": task_name,
                "seed": seed,
                "task_category": "multichoice",
                "metrics_by_model": metrics_by_model,
                "example_results": example_rows,
                "slice_definition": payload["slice_definition"],
                "sample_metadata": payload["sample_metadata"],
            })
            raw_results["tasks"].setdefault(task_name, {"category": "multichoice", "seeds": []})
            raw_results["tasks"][task_name]["seeds"].append(seed)
            for model_name, metrics in metrics_by_model.items():
                seed_rows.append({
                    "task_name": task_name,
                    "task_category": "multichoice",
                    "seed": seed,
                    "model_name": model_name,
                    **metrics,
                })

        torch.cuda.empty_cache()

    summary_rows = _aggregate_rows(seed_rows)
    recommendation = _recommendation(summary_rows, config)
    raw_results["recommendation"] = recommendation
    raw_results["summary_rows"] = summary_rows
    save_json(output_dir / "results.json", raw_results)
    save_csv(output_dir / "summary.csv", summary_rows)
    ensure_dir(Path(args.results_path).parent)
    ensure_dir(Path(args.summary_path).parent)
    ensure_dir(Path(args.report_path).parent)
    save_json(args.results_path, raw_results)
    save_csv(args.summary_path, summary_rows)
    _write_report(args.report_path, summary_rows, recommendation, args.config)


if __name__ == "__main__":
    main()
