"""Shared loaders and config helpers for v0.9 generalization evaluation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.models.backbone_loader import LoadedBackbones
from src.models.baselines import BridgeOnlyLargeModel, BridgeOnlyParamMatchedModel, FullLargeModel, SkipOnlyLargeModel
from src.train.trainer_utils import load_checkpoint
from src.utils.io import ExperimentConfig
from src.v0_6.idea4_common import load_mixture_path_specs
from src.v0_6.idea4_models import TwoPathStaticMixtureHybrid, TwoPathStaticMixtureNoSmallModel
from src.v0_6.idea4_tokenwise_models import TwoPathTokenwiseMixtureHybrid, TwoPathTokenwiseMixtureNoSmallModel


FROZEN_MODEL_ORDER = [
    "tokenwise_mixture",
    "static_mixture",
    "tokenwise_mixture_no_small",
    "bridge_only",
    "bridge_only_param_matched",
    "skip_only",
]
KEY_BASELINES = [
    "static_mixture",
    "bridge_only",
    "bridge_only_param_matched",
]


@dataclass(frozen=True)
class MultichoiceTaskSpec:
    """Config-backed description of one multiple-choice benchmark slice."""

    name: str
    dataset_name: str
    dataset_config_name: str | None
    split: str
    sample_count: int
    sampling_seed: int


@dataclass(frozen=True)
class LMTaskSpec:
    """Config-backed description of one LM benchmark slice."""

    name: str
    dataset_name: str
    dataset_config_name: str | None
    split: str
    sample_count: int
    sampling_seed: int


def clone_config_with_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    """Clone the base config and override the training seed."""

    cloned = copy.deepcopy(config)
    cloned.training.seed = seed
    cloned.raw["training"]["seed"] = seed
    return cloned


def generalization_settings(config: ExperimentConfig) -> dict[str, Any]:
    """Return the raw v0.9 settings with stable defaults."""

    raw = dict(config.raw.get("generalization", {}))
    return {
        "seeds": list(raw.get("seeds", [42, 43, 44])),
        "max_seq_len": int(raw.get("max_seq_len", config.training.seq_len)),
        "length_normalize_choices": bool(raw.get("length_normalize_choices", True)),
        "include_skip_only": bool(raw.get("include_skip_only", True)),
        "bootstrap_samples": int(raw.get("bootstrap_samples", 1000)),
        "bootstrap_seed": int(raw.get("bootstrap_seed", 9090)),
        "static_stage_dir": str(raw.get("static_stage_dir", "artifacts/v0_6/idea4_static_mixture/confirm/stage_b")),
        "tokenwise_stage_dir": str(raw.get("tokenwise_stage_dir", "artifacts/v0_6/idea4_tokenwise/confirm/stage_b")),
        "multichoice_tasks": list(raw.get("multichoice_tasks", [])),
        "lm_tasks": list(raw.get("lm_tasks", [])),
    }


def multichoice_task_specs(config: ExperimentConfig) -> list[MultichoiceTaskSpec]:
    """Return typed multichoice task specs from config."""

    specs: list[MultichoiceTaskSpec] = []
    for entry in generalization_settings(config)["multichoice_tasks"]:
        specs.append(
            MultichoiceTaskSpec(
                name=str(entry["name"]),
                dataset_name=str(entry["dataset_name"]),
                dataset_config_name=(
                    None if entry.get("dataset_config_name") in {None, "", "null"} else str(entry["dataset_config_name"])
                ),
                split=str(entry["split"]),
                sample_count=int(entry["sample_count"]),
                sampling_seed=int(entry["sampling_seed"]),
            )
        )
    if not specs:
        raise ValueError("v0.9 generalization config is missing multichoice_tasks.")
    return specs


def lm_task_specs(config: ExperimentConfig) -> list[LMTaskSpec]:
    """Return typed LM task specs from config."""

    specs: list[LMTaskSpec] = []
    for entry in generalization_settings(config)["lm_tasks"]:
        specs.append(
            LMTaskSpec(
                name=str(entry["name"]),
                dataset_name=str(entry["dataset_name"]),
                dataset_config_name=(
                    None if entry.get("dataset_config_name") in {None, "", "null"} else str(entry["dataset_config_name"])
                ),
                split=str(entry["split"]),
                sample_count=int(entry["sample_count"]),
                sampling_seed=int(entry["sampling_seed"]),
            )
        )
    if not specs:
        raise ValueError("v0.9 generalization config is missing lm_tasks.")
    return specs


def _load_static_payload(model: Any, payload: dict[str, Any], device: torch.device) -> None:
    model.entry_projector_b.load_state_dict(payload["entry_projector_b"])
    model.entry_projector_a.load_state_dict(payload["entry_projector_a"])
    model.return_adapter_b.load_state_dict(payload["return_adapter_b"])
    model.return_adapter_a.load_state_dict(payload["return_adapter_a"])
    with torch.no_grad():
        model.alpha.copy_(payload["alpha"].to(device=device, dtype=model.alpha.dtype))


def _load_tokenwise_payload(model: Any, payload: dict[str, Any], device: torch.device) -> None:
    model.entry_projector_b.load_state_dict(payload["entry_projector_b"])
    model.entry_projector_a.load_state_dict(payload["entry_projector_a"])
    model.return_adapter_b.load_state_dict(payload["return_adapter_b"])
    model.return_adapter_a.load_state_dict(payload["return_adapter_a"])
    model.set_static_prior_logits(payload["static_prior_logits"].to(device=device, dtype=torch.float32))
    model.gate_network.load_state_dict(payload["gate_network"])


def load_frozen_v060_models(
    config: ExperimentConfig,
    backbones: LoadedBackbones,
    *,
    seed: int,
    include_skip_only: bool,
    include_full_large: bool,
) -> dict[str, torch.nn.Module]:
    """Load the frozen v0.6.0 model family for one seed."""

    settings = generalization_settings(config)
    static_stage_dir = Path(settings["static_stage_dir"])
    tokenwise_stage_dir = Path(settings["tokenwise_stage_dir"])
    path_specs = load_mixture_path_specs(config)

    models: dict[str, torch.nn.Module] = {}
    if include_full_large:
        models["full_large"] = FullLargeModel(config, backbones.large_model)
    if include_skip_only:
        models["skip_only"] = SkipOnlyLargeModel(config, backbones.large_model)

    static_payload = load_checkpoint(static_stage_dir / f"seed_{seed}" / "static_mixture_checkpoint.pt", backbones.device)
    static_mixture = TwoPathStaticMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
    _load_static_payload(static_mixture, static_payload, backbones.device)
    models["static_mixture"] = static_mixture

    static_no_small_payload = load_checkpoint(
        static_stage_dir / f"seed_{seed}" / "static_mixture_no_small_checkpoint.pt",
        backbones.device,
    )
    static_no_small = TwoPathStaticMixtureNoSmallModel(config, backbones.large_model, backbones.small_model, path_specs)
    _load_static_payload(static_no_small, static_no_small_payload, backbones.device)

    tokenwise_payload = load_checkpoint(
        tokenwise_stage_dir / f"seed_{seed}" / "tokenwise_mixture_checkpoint.pt",
        backbones.device,
    )
    tokenwise_mixture = TwoPathTokenwiseMixtureHybrid(config, backbones.large_model, backbones.small_model, path_specs)
    _load_tokenwise_payload(tokenwise_mixture, tokenwise_payload, backbones.device)
    models["tokenwise_mixture"] = tokenwise_mixture

    tokenwise_no_small_payload = load_checkpoint(
        tokenwise_stage_dir / f"seed_{seed}" / "tokenwise_mixture_no_small_checkpoint.pt",
        backbones.device,
    )
    tokenwise_no_small = TwoPathTokenwiseMixtureNoSmallModel(config, backbones.large_model, backbones.small_model, path_specs)
    _load_tokenwise_payload(tokenwise_no_small, tokenwise_no_small_payload, backbones.device)
    models["tokenwise_mixture_no_small"] = tokenwise_no_small

    bridge_payload = load_checkpoint(static_stage_dir / f"seed_{seed}" / "bridge_only_checkpoint.pt", backbones.device)
    bridge_only = BridgeOnlyLargeModel(config, backbones.large_model)
    bridge_only.bridge.load_state_dict(bridge_payload["bridge"])
    bridge_only.gate.load_state_dict(bridge_payload["gate"])
    models["bridge_only"] = bridge_only

    bridge_param_payload = load_checkpoint(
        tokenwise_stage_dir / f"seed_{seed}" / "bridge_only_param_matched_checkpoint.pt",
        backbones.device,
    )
    bridge_param_rank = int(bridge_param_payload["bridge"]["down.weight"].shape[0])
    bridge_param = BridgeOnlyParamMatchedModel(config, backbones.large_model, rank=bridge_param_rank)
    bridge_param.bridge.load_state_dict(bridge_param_payload["bridge"])
    bridge_param.gate.load_state_dict(bridge_param_payload["gate"])
    models["bridge_only_param_matched"] = bridge_param

    models["static_mixture_no_small"] = static_no_small
    for model in models.values():
        model.eval()
    return models
