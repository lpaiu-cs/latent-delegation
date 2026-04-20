"""Training helpers shared across stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.build_corpus import CorpusBundle, build_corpus_bundle
from src.data.collators import CausalLMCollator
from src.models.backbone_loader import LoadedBackbones
from src.models.baselines import BridgeOnlyLargeModel, FullLargeModel, SkipOnlyLargeModel
from src.models.hybrid_gemma import HybridDelegationModel
from src.utils.io import ExperimentConfig, create_run_dir, export_run_metadata, save_config_snapshot, save_csv, save_json


def build_dataloader(config: ExperimentConfig, tokenizer: Any, stage_name: str, split_name: str) -> tuple[DataLoader, CorpusBundle]:
    """Build a tokenized dataloader."""

    corpus = build_corpus_bundle(config, tokenizer, stage_name=stage_name, split_name=split_name)
    dataloader = DataLoader(
        corpus.dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=(split_name == "train"),
        num_workers=config.training.num_workers,
        collate_fn=CausalLMCollator(),
    )
    return dataloader, corpus


def build_optimizer(module: nn.Module, config: ExperimentConfig) -> AdamW:
    """Create an AdamW optimizer over trainable parameters only."""

    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    return AdamW(parameters, lr=config.training.learning_rate, weight_decay=config.training.weight_decay)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move tensor batches to the model device."""

    return {key: value.to(device) for key, value in batch.items()}


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a PyTorch checkpoint."""

    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a PyTorch checkpoint."""

    return torch.load(path, map_location=device)


def initialize_run_dir(config: ExperimentConfig, stage_name: str) -> Path:
    """Create a run directory and save common metadata."""

    run_dir = create_run_dir(config, stage_name)
    save_config_snapshot(run_dir / "config_snapshot.yaml", config)
    export_run_metadata(run_dir / "metadata.json", config, {"stage": stage_name})
    return run_dir


def save_history(run_dir: Path, history_rows: list[dict[str, Any]], final_metrics: dict[str, Any]) -> None:
    """Persist CSV and JSON outputs for a stage."""

    save_csv(run_dir / "history.csv", history_rows)
    save_json(run_dir / "metrics.json", final_metrics)


def required_backbones_for_variant(variant: str) -> tuple[bool, bool]:
    """Return whether a variant requires the large and small backbones."""

    if variant in {"full_large", "skip_only", "bridge_only"}:
        return True, False
    if variant == "hybrid":
        return True, True
    raise ValueError(f"Unsupported variant: {variant}")


def require_large_model(backbones: LoadedBackbones) -> nn.Module:
    """Return the loaded large model or raise."""

    if backbones.large_model is None:
        raise RuntimeError("Large model was not loaded for this code path.")
    return backbones.large_model


def require_small_model(backbones: LoadedBackbones) -> nn.Module:
    """Return the loaded small model or raise."""

    if backbones.small_model is None:
        raise RuntimeError("Small model was not loaded for this code path.")
    return backbones.small_model


def require_tokenizer(backbones: LoadedBackbones) -> Any:
    """Return the loaded tokenizer or raise."""

    if backbones.tokenizer is None:
        raise RuntimeError("Tokenizer was not loaded for this code path.")
    return backbones.tokenizer


def build_model_variant(variant: str, config: ExperimentConfig, backbones: LoadedBackbones) -> nn.Module:
    """Instantiate one baseline or hybrid model without duplicating backbones."""

    large_model = require_large_model(backbones)
    if variant == "full_large":
        return FullLargeModel(config, large_model)
    if variant == "skip_only":
        return SkipOnlyLargeModel(config, large_model)
    if variant == "bridge_only":
        return BridgeOnlyLargeModel(config, large_model)
    if variant == "hybrid":
        small_model = require_small_model(backbones)
        return HybridDelegationModel(config, large_model, small_model)
    raise ValueError(f"Unsupported variant: {variant}")


def trainable_parameter_names(module: nn.Module) -> list[str]:
    """Return trainable parameter names."""

    return [name for name, parameter in module.named_parameters() if parameter.requires_grad]


def zero_requires_grad(module: nn.Module, except_prefixes: Iterable[str]) -> None:
    """Freeze every parameter except those whose names start with an allowed prefix."""

    allowed = tuple(except_prefixes)
    for name, parameter in module.named_parameters():
        parameter.requires_grad = name.startswith(allowed)
