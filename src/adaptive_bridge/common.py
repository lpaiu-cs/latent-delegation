"""Shared config and checkpoint helpers for the adaptive-bridge fork."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.train.trainer_utils import load_checkpoint
from src.utils.io import ExperimentConfig


ADAPTIVE_BRIDGE_TRAINED_VARIANTS = [
    "bridge_only_strong",
    "bridge_only_param_matched",
    "adaptive_bridge_no_small",
    "adaptive_bridge_moe",
]
ADAPTIVE_BRIDGE_REFERENCE_VARIANTS = [
    "skip_only",
    "full_large",
    "frozen_v060_tokenwise",
]
ADAPTIVE_BRIDGE_MODEL_ORDER = ADAPTIVE_BRIDGE_REFERENCE_VARIANTS + ADAPTIVE_BRIDGE_TRAINED_VARIANTS


@dataclass(frozen=True)
class AdaptiveGateSettings:
    """Token-wise adaptive gate settings."""

    hidden_dim: int
    use_rmsnorm: bool
    entropy_reg_weight: float
    prior_kl_weight: float
    smoothness_weight: float
    collapse_threshold: float
    bridge_init_logit: float


@dataclass(frozen=True)
class AdaptiveBridgeSettings:
    """Fork-specific model, warm-start, and fairness settings."""

    bridge_rank: int
    warm_start_from_v060: bool
    warm_start_template: str | None
    frozen_tokenwise_template: str | None
    frozen_bridge_template: str | None
    frozen_bridge_param_template: str | None


@dataclass(frozen=True)
class AdaptiveEvalTask:
    """One bounded evaluation task or holdout slice."""

    name: str
    category: str
    dataset_name: str
    dataset_config_name: str | None
    split: str
    sample_count: int
    sampling_seed: int


@dataclass(frozen=True)
class AdaptiveEvalSpec:
    """Evaluation spec for the first adaptive-bridge milestone."""

    seeds: list[int]
    max_seq_len: int
    length_normalize_choices: bool
    internal_tasks: list[AdaptiveEvalTask]
    lm_tasks: list[AdaptiveEvalTask]
    multichoice_tasks: list[AdaptiveEvalTask]
    internal_kl_tolerance: float
    internal_nll_tolerance: float
    lambada_kl_tolerance: float
    lambada_nll_tolerance: float
    multichoice_min_delta: float


def adaptive_bridge_gate_settings(config: ExperimentConfig) -> AdaptiveGateSettings:
    """Return gate hyperparameters for the bridge-aware token-wise mixture."""

    raw = dict(config.raw.get("adaptive_bridge", {}).get("gate", {}))
    return AdaptiveGateSettings(
        hidden_dim=int(raw.get("hidden_dim", 0)),
        use_rmsnorm=bool(raw.get("use_rmsnorm", True)),
        entropy_reg_weight=float(raw.get("entropy_reg_weight", 0.0)),
        prior_kl_weight=float(raw.get("prior_kl_weight", 0.0)),
        smoothness_weight=float(raw.get("smoothness_weight", 0.0)),
        collapse_threshold=float(raw.get("collapse_threshold", 0.9)),
        bridge_init_logit=float(raw.get("bridge_init_logit", -0.5)),
    )


def adaptive_bridge_settings(config: ExperimentConfig) -> AdaptiveBridgeSettings:
    """Return fork-specific adaptive-bridge settings."""

    raw = dict(config.raw.get("adaptive_bridge", {}))
    checkpoints = dict(raw.get("checkpoints", {}))
    return AdaptiveBridgeSettings(
        bridge_rank=int(raw.get("bridge_rank", config.adapters.bridge_rank)),
        warm_start_from_v060=bool(raw.get("warm_start_from_v060", False)),
        warm_start_template=(
            None if checkpoints.get("warm_start_tokenwise_template") in {None, "", "null"} else str(checkpoints["warm_start_tokenwise_template"])
        ),
        frozen_tokenwise_template=(
            None if checkpoints.get("frozen_tokenwise_template") in {None, "", "null"} else str(checkpoints["frozen_tokenwise_template"])
        ),
        frozen_bridge_template=(
            None if checkpoints.get("frozen_bridge_template") in {None, "", "null"} else str(checkpoints["frozen_bridge_template"])
        ),
        frozen_bridge_param_template=(
            None
            if checkpoints.get("frozen_bridge_param_template") in {None, "", "null"}
            else str(checkpoints["frozen_bridge_param_template"])
        ),
    )


def adaptive_bridge_trainable_prefixes(config: ExperimentConfig) -> list[str]:
    """Return the trainable parameter prefixes for the adaptive-bridge MoE."""

    prefixes = [
        "return_adapter_b",
        "return_adapter_a",
        "bridge_expert",
        "gate_network",
    ]
    if config.training.stage_b.train_entry_projector:
        prefixes = ["entry_projector_b", "entry_projector_a"] + prefixes
    return prefixes


def clone_config_with_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    """Clone a config and override the training seed."""

    cloned = copy.deepcopy(config)
    cloned.training.seed = seed
    cloned.raw["training"]["seed"] = seed
    return cloned


def matched_bridge_rank(large_hidden_size: int, target_trainable_params: int) -> int:
    """Return the closest large-only bridge rank for a target trainable budget."""

    base = max(1, int((target_trainable_params - 1) / max(1, 2 * large_hidden_size)))
    candidates = sorted({max(1, base), max(1, base + 1)})
    return min(candidates, key=lambda rank: abs((2 * large_hidden_size * rank + 1) - target_trainable_params))


def _task_specs(entries: list[dict[str, Any]], category: str) -> list[AdaptiveEvalTask]:
    specs: list[AdaptiveEvalTask] = []
    for entry in entries:
        specs.append(
            AdaptiveEvalTask(
                name=str(entry["name"]),
                category=category,
                dataset_name=str(entry["dataset_name"]),
                dataset_config_name=(
                    None if entry.get("dataset_config_name") in {None, "", "null"} else str(entry["dataset_config_name"])
                ),
                split=str(entry["split"]),
                sample_count=int(entry["sample_count"]),
                sampling_seed=int(entry["sampling_seed"]),
            )
        )
    return specs


def adaptive_eval_spec(config: ExperimentConfig) -> AdaptiveEvalSpec:
    """Return the bounded evaluation spec for adaptive-bridge."""

    raw = dict(config.raw.get("adaptive_bridge", {}).get("evaluation", {}))
    internal_tasks = _task_specs(list(raw.get("internal_tasks", [])), category="internal")
    lm_tasks = _task_specs(list(raw.get("lm_tasks", [])), category="lm")
    multichoice_tasks = _task_specs(list(raw.get("multichoice_tasks", [])), category="multichoice")
    if not internal_tasks:
        raise ValueError("adaptive_bridge.evaluation.internal_tasks is required.")
    if not lm_tasks:
        raise ValueError("adaptive_bridge.evaluation.lm_tasks is required.")
    if not multichoice_tasks:
        raise ValueError("adaptive_bridge.evaluation.multichoice_tasks is required.")
    return AdaptiveEvalSpec(
        seeds=[int(value) for value in raw.get("seeds", [config.training.seed])],
        max_seq_len=int(raw.get("max_seq_len", config.training.seq_len)),
        length_normalize_choices=bool(raw.get("length_normalize_choices", True)),
        internal_tasks=internal_tasks,
        lm_tasks=lm_tasks,
        multichoice_tasks=multichoice_tasks,
        internal_kl_tolerance=float(raw.get("internal_kl_tolerance", 0.02)),
        internal_nll_tolerance=float(raw.get("internal_nll_tolerance", 0.05)),
        lambada_kl_tolerance=float(raw.get("lambada_kl_tolerance", 0.02)),
        lambada_nll_tolerance=float(raw.get("lambada_nll_tolerance", 0.05)),
        multichoice_min_delta=float(raw.get("multichoice_min_delta", 0.0)),
    )


def checkpoint_path_from_template(template: str | None, seed: int) -> Path | None:
    """Resolve a seed-templated checkpoint path."""

    if template is None:
        return None
    return Path(template.format(seed=seed))


def maybe_load_checkpoint(path: Path | None, device: torch.device) -> dict[str, Any] | None:
    """Load a checkpoint when a path is provided and exists."""

    if path is None or not path.exists():
        return None
    return load_checkpoint(path, device)
