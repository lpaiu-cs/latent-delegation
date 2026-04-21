"""Window-search utilities for the v0.6 continuation track."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils.io import ExperimentConfig
from src.v0_6.common import WindowCandidate


@dataclass(frozen=True)
class WindowSearchSettings:
    """Search-space and evaluation settings for Phase 1A."""

    pilot_seeds: list[int]
    confirm_seeds: list[int]
    large_window_lengths: list[int]
    small_window_lengths: list[int]
    large_start_offsets: list[int]
    small_start_offsets: list[int]
    shortlist_size: int
    stage_a_steps: int | None = None
    stage_b_steps: int | None = None
    top_k: int = 5
    max_validation_batches: int | None = None


def load_window_search_settings(config: ExperimentConfig) -> WindowSearchSettings:
    """Parse the optional `window_search` section from a config."""

    values = config.raw.get("window_search", {})
    return WindowSearchSettings(
        pilot_seeds=list(values.get("pilot_seeds", [config.training.seed])),
        confirm_seeds=list(values.get("confirm_seeds", [config.training.seed, config.training.seed + 1, config.training.seed + 2])),
        large_window_lengths=list(values.get("large_window_lengths", [4, 6, 8])),
        small_window_lengths=list(values.get("small_window_lengths", [2, 3, 4, 5, 6])),
        large_start_offsets=list(values.get("large_start_offsets", [-2, 0, 2])),
        small_start_offsets=list(values.get("small_start_offsets", [-2, 0, 2])),
        shortlist_size=int(values.get("shortlist_size", 3)),
        stage_a_steps=values.get("stage_a_steps"),
        stage_b_steps=values.get("stage_b_steps"),
        top_k=int(values.get("top_k", 5)),
        max_validation_batches=values.get("max_validation_batches"),
    )


def enumerate_window_candidates(
    config: ExperimentConfig,
    settings: WindowSearchSettings,
    *,
    large_num_layers: int,
    small_num_layers: int,
) -> list[WindowCandidate]:
    """Enumerate unique order-preserving contiguous window candidates."""

    base_large_start = config.split.large_removed_start
    base_small_start = config.split.small_delegate_start
    unique: dict[str, WindowCandidate] = {}

    for large_length in settings.large_window_lengths:
        for small_length in settings.small_window_lengths:
            for large_offset in settings.large_start_offsets:
                large_start = base_large_start + large_offset
                large_end = large_start + large_length - 1
                if large_start < 1 or large_end >= large_num_layers - 1:
                    continue
                for small_offset in settings.small_start_offsets:
                    small_start = base_small_start + small_offset
                    small_end = small_start + small_length - 1
                    if small_start < 1 or small_end >= small_num_layers:
                        continue
                    candidate = WindowCandidate(
                        large_start=large_start,
                        large_end=large_end,
                        small_start=small_start,
                        small_end=small_end,
                    )
                    unique[candidate.label] = candidate

    return sorted(
        unique.values(),
        key=lambda candidate: (
            candidate.large_start,
            candidate.large_end,
            candidate.small_start,
            candidate.small_end,
        ),
    )


def load_shortlist(path: str | Path) -> list[WindowCandidate]:
    """Load shortlisted candidates from a JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates: list[WindowCandidate] = []
    for row in payload:
        candidates.append(
            WindowCandidate(
                large_start=int(row["large_start"]),
                large_end=int(row["large_end"]),
                small_start=int(row["small_start"]),
                small_end=int(row["small_end"]),
            )
        )
    return candidates


def shortlist_rows(rows: list[dict[str, Any]], shortlist_size: int) -> list[dict[str, Any]]:
    """Return the top rows under the primary-then-secondary ranking."""

    ranked = sorted(rows, key=ranking_key)
    return ranked[:shortlist_size]


def ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Primary output-aware ranking, then hidden metrics as tie-breakers."""

    return (
        float(row["hybrid_logit_kl_to_teacher_mean"]),
        float(row["hybrid_nll_mean"]),
        float(row["hybrid_perplexity_mean"]),
        float(row["hybrid_hidden_mse_mean"]),
        -float(row["hybrid_hidden_cosine_mean"]),
    )


def window_row(
    candidate: WindowCandidate,
    *,
    hybrid_metrics: dict[str, float],
    skip_metrics: dict[str, float],
    hybrid_no_small_metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one flat CSV/JSON-ready row for a candidate summary."""

    row: dict[str, Any] = {
        **candidate.to_dict(),
        "hybrid_logit_kl_to_teacher_mean": hybrid_metrics["logit_kl_to_teacher_mean"],
        "hybrid_nll_mean": hybrid_metrics["nll_mean"],
        "hybrid_perplexity_mean": hybrid_metrics["perplexity_mean"],
        "hybrid_hidden_mse_mean": hybrid_metrics["hidden_mse_mean"],
        "hybrid_hidden_cosine_mean": hybrid_metrics["hidden_cosine_mean"],
        "hybrid_minus_skip_kl_mean": hybrid_metrics["logit_kl_to_teacher_mean"] - skip_metrics["logit_kl_to_teacher_mean"],
        "hybrid_minus_skip_nll_mean": hybrid_metrics["nll_mean"] - skip_metrics["nll_mean"],
        "hybrid_minus_skip_hidden_mse_mean": hybrid_metrics["hidden_mse_mean"] - skip_metrics["hidden_mse_mean"],
        "hybrid_minus_skip_hidden_cosine_mean": hybrid_metrics["hidden_cosine_mean"] - skip_metrics["hidden_cosine_mean"],
        "hybrid_minus_no_small_kl_mean": hybrid_metrics["logit_kl_to_teacher_mean"] - hybrid_no_small_metrics["logit_kl_to_teacher_mean"],
        "hybrid_minus_no_small_nll_mean": hybrid_metrics["nll_mean"] - hybrid_no_small_metrics["nll_mean"],
        "hybrid_minus_no_small_hidden_mse_mean": hybrid_metrics["hidden_mse_mean"] - hybrid_no_small_metrics["hidden_mse_mean"],
        "hybrid_minus_no_small_hidden_cosine_mean": hybrid_metrics["hidden_cosine_mean"] - hybrid_no_small_metrics["hidden_cosine_mean"],
    }
    if extra:
        row.update(extra)
    return row


def distinct_small_windows(candidates: Iterable[WindowCandidate], limit: int) -> list[WindowCandidate]:
    """Return candidates with distinct small windows up to a flat limit."""

    selected: list[WindowCandidate] = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        key = (candidate.small_start, candidate.small_end)
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected
