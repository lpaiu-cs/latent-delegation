"""Held-out LM-style generalization evaluation for the frozen v0.6.0 family."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.eval.metrics import perplexity_from_loss, shifted_cross_entropy, shifted_kl_divergence
from src.models.backbone_loader import load_backbones
from src.utils.io import ensure_dir, export_run_metadata, load_config, save_config_snapshot, save_csv, save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything
from src.v0_9.common import (
    FROZEN_MODEL_ORDER,
    clone_config_with_seed,
    generalization_settings,
    lm_task_specs,
    load_frozen_v060_models,
)
from src.v0_9.task_scoring import build_lm_examples


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="artifacts/v0_9/generalization/raw/lm")
    parser.add_argument("--results-path", default="artifacts/v0_9/generalization/raw/lm/results.json")
    parser.add_argument("--summary-path", default="artifacts/v0_9/generalization/raw/lm/summary.csv")
    return parser.parse_args()


def _model_order(include_skip_only: bool) -> list[str]:
    if include_skip_only:
        return list(FROZEN_MODEL_ORDER)
    return [name for name in FROZEN_MODEL_ORDER if name != "skip_only"]


def _valid_tokens(labels: torch.Tensor) -> int:
    return int((labels[:, 1:] != -100).sum().item())


def _example_tensors(tokenizer: Any, text: str, max_seq_len: int, device: torch.device) -> dict[str, torch.Tensor]:
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


def _empty_totals() -> dict[str, float]:
    return {
        "valid_tokens": 0.0,
        "nll_sum": 0.0,
        "kl_sum": 0.0,
        "truncated_examples": 0.0,
    }


def _finalize_totals(totals: dict[str, float], example_count: int) -> dict[str, float]:
    valid_tokens = max(1.0, totals["valid_tokens"])
    mean_nll = totals["nll_sum"] / valid_tokens
    mean_kl = totals["kl_sum"] / valid_tokens
    return {
        "nll": mean_nll,
        "perplexity": perplexity_from_loss(mean_nll),
        "logit_kl_to_teacher": mean_kl,
        "truncation_rate": totals["truncated_examples"] / max(1, example_count),
        "valid_tokens": totals["valid_tokens"],
        "example_count": float(example_count),
    }


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    settings = generalization_settings(config)
    seed_everything(config.training.seed)

    output_dir = ensure_dir(args.output_dir)
    save_config_snapshot(output_dir / "config_snapshot.yaml", config)
    export_run_metadata(
        output_dir / "metadata.json",
        config,
        {
            "stage": "v0_9_eval_lm_generalization",
            "seeds": settings["seeds"],
            "max_seq_len": settings["max_seq_len"],
            "include_skip_only": settings["include_skip_only"],
        },
    )

    task_payloads: dict[str, Any] = {}
    for spec in lm_task_specs(config):
        examples, slice_definition, sample_metadata = build_lm_examples(spec)
        task_payloads[spec.name] = {
            "examples": examples,
            "slice_definition": slice_definition,
            "sample_metadata": sample_metadata,
        }
        task_dir = ensure_dir(output_dir / spec.name)
        save_json(task_dir / "slice_definition.json", slice_definition)
        save_json(task_dir / "sample_ids.json", sample_metadata)

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    model_order = _model_order(settings["include_skip_only"])
    summary_rows: list[dict[str, Any]] = []
    raw_manifest: dict[str, Any] = {
        "config_path": args.config,
        "seeds": settings["seeds"],
        "max_seq_len": settings["max_seq_len"],
        "include_skip_only": settings["include_skip_only"],
        "tasks": {},
    }

    for seed in settings["seeds"]:
        LOGGER.info("v0_9 lm seed=%s", seed)
        seed_config = clone_config_with_seed(config, seed)
        models = load_frozen_v060_models(
            seed_config,
            backbones,
            seed=seed,
            include_skip_only=settings["include_skip_only"],
            include_full_large=True,
        )
        with torch.no_grad():
            for task_name, task_payload in task_payloads.items():
                LOGGER.info("v0_9 lm seed=%s task=%s", seed, task_name)
                totals = {model_name: _empty_totals() for model_name in model_order}
                example_rows: list[dict[str, Any]] = []
                for example in task_payload["examples"]:
                    tensors = _example_tensors(
                        backbones.tokenizer,
                        example.text,
                        max_seq_len=settings["max_seq_len"],
                        device=backbones.device,
                    )
                    valid_tokens = _valid_tokens(tensors["labels"])
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
                    for model_name in model_order:
                        student_logits = models[model_name](
                            tensors["input_ids"],
                            attention_mask=tensors["attention_mask"],
                        ).logits
                        nll = float(shifted_cross_entropy(student_logits, tensors["labels"]).detach().cpu())
                        kl = float(shifted_kl_divergence(student_logits, teacher_logits, tensors["labels"]).detach().cpu())
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

                metrics_by_model = {
                    model_name: _finalize_totals(model_totals, len(example_rows))
                    for model_name, model_totals in totals.items()
                }
                seed_payload = {
                    "task_name": task_name,
                    "seed": seed,
                    "model_order": model_order,
                    "slice_definition": task_payload["slice_definition"],
                    "sample_metadata": task_payload["sample_metadata"],
                    "metrics_by_model": metrics_by_model,
                    "example_results": example_rows,
                }
                seed_dir = ensure_dir(output_dir / task_name / f"seed_{seed}")
                save_json(seed_dir / "results.json", seed_payload)
                raw_manifest["tasks"].setdefault(task_name, {"slice_definition": task_payload["slice_definition"], "seeds": []})
                raw_manifest["tasks"][task_name]["seeds"].append(seed)
                for model_name, metrics in metrics_by_model.items():
                    summary_rows.append(
                        {
                            "task_name": task_name,
                            "seed": seed,
                            "model_name": model_name,
                            **metrics,
                        }
                    )
        torch.cuda.empty_cache()

    save_json(args.results_path, raw_manifest)
    save_csv(args.summary_path, summary_rows)


if __name__ == "__main__":
    main()
