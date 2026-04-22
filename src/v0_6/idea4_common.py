"""Shared helpers for the Idea 4 static-mixture experiments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.utils.io import ExperimentConfig
from src.v0_6.common import WindowCandidate, clone_config


@dataclass(frozen=True)
class MixturePathSpec:
    """One shortlisted delegated path for the static two-path mixture."""

    name: str
    candidate: WindowCandidate
    phase1_root: str

    @property
    def small_entry_target_layer(self) -> int:
        """Return the small-model pre-entry layer index."""

        return self.candidate.small_start - 1

    @property
    def label(self) -> str:
        """Return a stable human-readable path label."""

        return f"{self.candidate.large_start}..{self.candidate.large_end} -> {self.candidate.small_start}..{self.candidate.small_end}"

    def checkpoint_path(self, seed: int, variant: str = "hybrid") -> Path:
        """Return the Phase 1 checkpoint path for one seed and variant."""

        return Path(self.phase1_root) / f"seed_{seed}" / f"{variant}_checkpoint.pt"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the path spec for reports and artifacts."""

        payload = self.candidate.to_dict()
        payload["name"] = self.name
        payload["phase1_root"] = self.phase1_root
        payload["small_entry_target_layer"] = self.small_entry_target_layer
        payload["label"] = self.label
        return payload


def load_mixture_path_specs(config: ExperimentConfig) -> list[MixturePathSpec]:
    """Load the two shortlisted Idea 4 paths from the raw config payload."""

    idea4 = config.raw.get("idea4", {})
    entries = idea4.get("paths", [])
    if len(entries) != 2:
        raise ValueError("Idea 4 requires exactly two shortlisted paths.")

    specs: list[MixturePathSpec] = []
    expected_large_start = config.split.large_removed_start
    expected_large_end = config.split.large_removed_end
    for entry in entries:
        name = str(entry["name"])
        large_start = int(entry.get("large_start", expected_large_start))
        large_end = int(entry.get("large_end", expected_large_end))
        small_entry_target = int(entry["small_entry_target_layer"])
        small_start = int(entry["small_delegate_start"])
        small_end = int(entry["small_delegate_end"])
        if small_entry_target + 1 != small_start:
            raise ValueError(f"Idea 4 path {name} has inconsistent small entry and delegate start.")
        if large_start != expected_large_start or large_end != expected_large_end:
            raise ValueError("Idea 4 path definitions must keep the fixed large removed window.")
        specs.append(
            MixturePathSpec(
                name=name,
                candidate=WindowCandidate(
                    large_start=large_start,
                    large_end=large_end,
                    small_start=small_start,
                    small_end=small_end,
                ),
                phase1_root=str(entry["phase1_root"]),
            )
        )

    ordered_names = [spec.name for spec in specs]
    if ordered_names != ["path_b", "path_a"]:
        raise ValueError(f"Idea 4 path order must be ['path_b', 'path_a'], got {ordered_names}.")
    return specs


def path_spec_by_name(path_specs: list[MixturePathSpec], name: str) -> MixturePathSpec:
    """Return one path spec by name."""

    for spec in path_specs:
        if spec.name == name:
            return spec
    raise KeyError(name)


def clone_single_path_config(
    config: ExperimentConfig,
    path_spec: MixturePathSpec,
    *,
    seed: int | None = None,
    experiment_name: str | None = None,
) -> ExperimentConfig:
    """Clone the base config into the single-path split used by one shortlisted path."""

    return clone_config(
        config,
        candidate=path_spec.candidate,
        seed=seed,
        experiment_name=experiment_name,
    )


def gate_value_from_phase1_payload(payload: dict[str, Any]) -> float:
    """Return the bounded scalar gate value stored in a Phase 1 checkpoint."""

    gate_state = payload.get("gate")
    if gate_state is None:
        return 1.0
    raw_gate = gate_state["raw_gate"]
    if not isinstance(raw_gate, torch.Tensor):
        raw_gate = torch.tensor(raw_gate)
    return float(torch.tanh(raw_gate.detach().float()).cpu())


def gated_return_adapter_state_dict(payload: dict[str, Any]) -> tuple[dict[str, torch.Tensor], float]:
    """Absorb the Phase 1 scalar gate into the return adapter weights."""

    return_adapter = payload.get("return_adapter")
    if return_adapter is None:
        raise KeyError("Phase 1 hybrid payload is missing return_adapter state.")
    gate_value = gate_value_from_phase1_payload(payload)
    state = copy.deepcopy(return_adapter)
    state["up.weight"] = state["up.weight"] * gate_value
    return state, gate_value
