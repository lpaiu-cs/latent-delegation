"""Shared helpers for the v0.6 continuation track."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from src.utils.io import ExperimentConfig, SplitConfig


@dataclass(frozen=True)
class WindowCandidate:
    """One contiguous large-window / small-window pairing."""

    large_start: int
    large_end: int
    small_start: int
    small_end: int

    @property
    def large_length(self) -> int:
        """Return the large-window length."""

        return self.large_end - self.large_start + 1

    @property
    def small_length(self) -> int:
        """Return the small-window length."""

        return self.small_end - self.small_start + 1

    @property
    def label(self) -> str:
        """Return a deterministic candidate identifier."""

        return f"L{self.large_start}-{self.large_end}__S{self.small_start}-{self.small_end}"

    def to_dict(self) -> dict[str, int | str]:
        """Serialize the candidate for JSON or CSV outputs."""

        return {
            "label": self.label,
            "large_start": self.large_start,
            "large_end": self.large_end,
            "large_length": self.large_length,
            "small_start": self.small_start,
            "small_end": self.small_end,
            "small_length": self.small_length,
        }


def candidate_to_split(candidate: WindowCandidate) -> SplitConfig:
    """Translate one candidate into a typed split config."""

    return SplitConfig(
        large_prefix_end=candidate.large_start - 1,
        large_removed_start=candidate.large_start,
        large_removed_end=candidate.large_end,
        large_suffix_start=candidate.large_end + 1,
        small_entry_target_layer=candidate.small_start - 1,
        small_delegate_start=candidate.small_start,
        small_delegate_end=candidate.small_end,
    )


def clone_config(
    config: ExperimentConfig,
    *,
    candidate: WindowCandidate | None = None,
    seed: int | None = None,
    experiment_name: str | None = None,
    stage_a_steps: int | None = None,
    stage_b_steps: int | None = None,
) -> ExperimentConfig:
    """Return a deep-cloned config with optional split and training overrides."""

    raw = copy.deepcopy(config.raw)
    split = config.split
    if candidate is not None:
        split = candidate_to_split(candidate)
        raw["split"] = {
            "large_prefix_end": split.large_prefix_end,
            "large_removed_start": split.large_removed_start,
            "large_removed_end": split.large_removed_end,
            "large_suffix_start": split.large_suffix_start,
            "small_entry_target_layer": split.small_entry_target_layer,
            "small_delegate_start": split.small_delegate_start,
            "small_delegate_end": split.small_delegate_end,
        }

    training = config.training
    if seed is not None:
        training = replace(training, seed=seed)
        raw["training"]["seed"] = seed

    if stage_a_steps is not None:
        stage_a = replace(training.stage_a, max_steps=stage_a_steps)
        training = replace(training, stage_a=stage_a)
        raw["training"]["stage_a"]["max_steps"] = stage_a_steps

    if stage_b_steps is not None:
        stage_b = replace(training.stage_b, max_steps=stage_b_steps)
        training = replace(training, stage_b=stage_b)
        raw["training"]["stage_b"]["max_steps"] = stage_b_steps

    experiment = config.experiment
    if experiment_name is not None:
        experiment = replace(experiment, name=experiment_name)
        raw["experiment"]["name"] = experiment_name

    return ExperimentConfig(
        experiment=experiment,
        model=config.model,
        split=split,
        adapters=config.adapters,
        training=training,
        data=config.data,
        eval=config.eval,
        raw=raw,
    )
