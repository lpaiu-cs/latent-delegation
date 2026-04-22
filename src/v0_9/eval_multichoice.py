"""Bounded multichoice log-likelihood evaluation for the frozen v0.6.0 family."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.models.backbone_loader import load_backbones
from src.utils.io import ensure_dir, export_run_metadata, load_config, save_config_snapshot, save_csv, save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything
from src.v0_9.common import (
    FROZEN_MODEL_ORDER,
    clone_config_with_seed,
    generalization_settings,
    load_frozen_v060_models,
    multichoice_task_specs,
)
from src.v0_9.task_scoring import build_multichoice_examples, score_multichoice_example


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="artifacts/v0_9/generalization/raw/multichoice")
    parser.add_argument("--results-path", default="artifacts/v0_9/generalization/raw/multichoice/results.json")
    parser.add_argument("--summary-path", default="artifacts/v0_9/generalization/raw/multichoice/summary.csv")
    return parser.parse_args()


def _model_order(include_skip_only: bool) -> list[str]:
    if include_skip_only:
        return list(FROZEN_MODEL_ORDER)
    return [name for name in FROZEN_MODEL_ORDER if name != "skip_only"]


def _task_summary(example_rows: list[dict[str, Any]], model_order: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for model_name in model_order:
        accuracy = sum(1.0 for row in example_rows if row["models"][model_name]["correct"]) / max(1, len(example_rows))
        mean_margin = sum(row["models"][model_name]["score_margin"] for row in example_rows) / max(1, len(example_rows))
        truncation_rate = (
            sum(
                1.0
                for row in example_rows
                if any(bool(flag) for flag in row["models"][model_name]["truncated_flags"])
            )
            / max(1, len(example_rows))
        )
        summary[model_name] = {
            "accuracy": accuracy,
            "mean_choice_margin": mean_margin,
            "truncation_rate": truncation_rate,
            "example_count": float(len(example_rows)),
        }
    return summary


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
            "stage": "v0_9_eval_multichoice",
            "seeds": settings["seeds"],
            "max_seq_len": settings["max_seq_len"],
            "length_normalize_choices": settings["length_normalize_choices"],
            "include_skip_only": settings["include_skip_only"],
        },
    )

    task_payloads: dict[str, Any] = {}
    for spec in multichoice_task_specs(config):
        examples, slice_definition, sample_metadata = build_multichoice_examples(spec)
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
        "length_normalize_choices": settings["length_normalize_choices"],
        "include_skip_only": settings["include_skip_only"],
        "tasks": {},
    }

    for seed in settings["seeds"]:
        LOGGER.info("v0_9 multichoice seed=%s", seed)
        seed_config = clone_config_with_seed(config, seed)
        models = load_frozen_v060_models(
            seed_config,
            backbones,
            seed=seed,
            include_skip_only=settings["include_skip_only"],
            include_full_large=False,
        )
        with torch.no_grad():
            for task_name, task_payload in task_payloads.items():
                LOGGER.info("v0_9 multichoice seed=%s task=%s", seed, task_name)
                example_rows: list[dict[str, Any]] = []
                for example in task_payload["examples"]:
                    row = {
                        "example_id": example.example_id,
                        "label_index": example.label_index,
                        "metadata": example.metadata,
                        "models": {},
                    }
                    for model_name in model_order:
                        result = score_multichoice_example(
                            models[model_name],
                            backbones.tokenizer,
                            example,
                            max_seq_len=settings["max_seq_len"],
                            device=backbones.device,
                            length_normalize=settings["length_normalize_choices"],
                        )
                        row["models"][model_name] = result
                    example_rows.append(row)

                task_summary = _task_summary(example_rows, model_order)
                seed_payload = {
                    "task_name": task_name,
                    "seed": seed,
                    "model_order": model_order,
                    "slice_definition": task_payload["slice_definition"],
                    "sample_metadata": task_payload["sample_metadata"],
                    "metrics_by_model": task_summary,
                    "example_results": example_rows,
                }
                seed_dir = ensure_dir(output_dir / task_name / f"seed_{seed}")
                save_json(seed_dir / "results.json", seed_payload)
                raw_manifest["tasks"].setdefault(task_name, {"slice_definition": task_payload["slice_definition"], "seeds": []})
                raw_manifest["tasks"][task_name]["seeds"].append(seed)
                for model_name, metrics in task_summary.items():
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
