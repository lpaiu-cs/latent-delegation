"""Shared helpers for the v0.7 Idea 5 discovery track."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.io import ExperimentConfig


DISCOVERY_COMPONENTS = [
    "stage_signature_distance",
    "hidden_alignment_proxy",
    "logit_disruption_proxy",
    "output_anchor_proxy",
]

STAGE_SIGNATURE_COMPONENT_METRICS = [
    "hidden_norm_mean",
    "delta_norm_mean",
    "delta_cosine_mean",
    "logit_entropy_mean",
    "logit_kl_to_final_mean",
]
HIDDEN_ALIGNMENT_METRICS = [
    "hidden_norm_mean",
    "delta_norm_mean",
    "delta_cosine_mean",
]
LOGIT_DISRUPTION_METRICS = [
    "logit_entropy_mean",
    "logit_kl_to_final_mean",
]
MAPPING_PATTERN = re.compile(r"(?P<large_start>\d+)\.\.(?P<large_end>\d+)\s*->\s*(?P<small_start>\d+)\.\.(?P<small_end>\d+)")


@dataclass(frozen=True)
class AlignmentWindow:
    """One local contiguous window on either the large or small backbone."""

    family: str
    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the number of layers in the window."""

        return self.end - self.start + 1

    @property
    def label(self) -> str:
        """Return a deterministic label for JSON and CSV outputs."""

        return f"{self.family}:{self.start}..{self.end}"

    def to_dict(self) -> dict[str, int | str]:
        """Serialize the window."""

        return {
            "family": self.family,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }


@dataclass(frozen=True)
class MappingAnchor:
    """One empirically-supported mapping anchor from Phase 1."""

    name: str
    large_start: int
    large_end: int
    small_start: int
    small_end: int
    stage: str
    rank: int

    @property
    def label(self) -> str:
        """Return a human-readable mapping label."""

        return f"{self.large_start}..{self.large_end} -> {self.small_start}..{self.small_end}"

    def to_dict(self) -> dict[str, int | str]:
        """Serialize the anchor for JSON outputs."""

        return {
            "name": self.name,
            "label": self.label,
            "large_start": self.large_start,
            "large_end": self.large_end,
            "small_start": self.small_start,
            "small_end": self.small_end,
            "stage": self.stage,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class Idea5DiscoverySettings:
    """Configuration for the local monotone-alignment discovery run."""

    stage_signature_artifact: str
    phase1_ranking_artifact: str
    static_summary_artifact: str | None
    tokenwise_summary_artifact: str | None
    large_layer_min: int
    large_layer_max: int
    small_layer_min: int
    small_layer_max: int
    large_segment_lengths: list[int]
    small_segment_lengths: list[int]
    allowed_moves: list[tuple[int, int]]
    component_weights: dict[str, float]
    top_paths: int
    top_pairs: int


def load_idea5_discovery_settings(config: ExperimentConfig) -> Idea5DiscoverySettings:
    """Load the optional `idea5_discovery` section from a YAML config."""

    values = config.raw.get("idea5_discovery", {})
    raw_weights = values.get("component_weights", {})
    component_weights = {
        "stage_signature_distance": float(raw_weights.get("stage_signature_distance", 0.45)),
        "hidden_alignment_proxy": float(raw_weights.get("hidden_alignment_proxy", 0.30)),
        "logit_disruption_proxy": float(raw_weights.get("logit_disruption_proxy", 0.25)),
        "output_anchor_proxy": float(raw_weights.get("output_anchor_proxy", 0.0)),
    }
    raw_moves = values.get("allowed_moves", [[1, 1], [2, 1], [1, 2], [2, 2], [3, 2]])
    allowed_moves = [(int(move[0]), int(move[1])) for move in raw_moves]
    return Idea5DiscoverySettings(
        stage_signature_artifact=str(values.get("stage_signature_artifact", "artifacts/v0_6/phase1_real/stage_signature/signatures.json")),
        phase1_ranking_artifact=str(values.get("phase1_ranking_artifact", "artifacts/v0_6/phase1_real/combined/ranking_summary.json")),
        static_summary_artifact=values.get("static_summary_artifact", "artifacts/v0_6/idea4_static_mixture/combined/summary.json"),
        tokenwise_summary_artifact=values.get("tokenwise_summary_artifact", "artifacts/v0_6/idea4_tokenwise/combined/summary.json"),
        large_layer_min=int(values.get("large_layer_min", 22)),
        large_layer_max=int(values.get("large_layer_max", 30)),
        small_layer_min=int(values.get("small_layer_min", 13)),
        small_layer_max=int(values.get("small_layer_max", 20)),
        large_segment_lengths=[int(length) for length in values.get("large_segment_lengths", [1, 2, 3])],
        small_segment_lengths=[int(length) for length in values.get("small_segment_lengths", [1, 2, 3])],
        allowed_moves=allowed_moves,
        component_weights=component_weights,
        top_paths=int(values.get("top_paths", 5)),
        top_pairs=int(values.get("top_pairs", 12)),
    )


def enumerate_local_windows(family: str, *, layer_min: int, layer_max: int, lengths: list[int]) -> list[AlignmentWindow]:
    """Enumerate contiguous windows inside a local layer range."""

    windows: list[AlignmentWindow] = []
    for length in sorted({int(value) for value in lengths if int(value) > 0}):
        for start in range(layer_min, layer_max - length + 2):
            windows.append(AlignmentWindow(family=family, start=start, end=start + length - 1))
    return windows


def load_json(path: str | Path) -> Any:
    """Load a UTF-8 JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_mapping_label(mapping: str) -> tuple[int, int, int, int]:
    """Parse a `24..27 -> 14..19`-style mapping label."""

    match = MAPPING_PATTERN.fullmatch(mapping.strip())
    if match is None:
        raise ValueError(f"Unsupported mapping label: {mapping!r}")
    return (
        int(match.group("large_start")),
        int(match.group("large_end")),
        int(match.group("small_start")),
        int(match.group("small_end")),
    )


def load_phase1_shortlist_anchors(path: str | Path) -> list[MappingAnchor]:
    """Load the confirmed Phase 1 mappings as light empirical anchors."""

    payload = load_json(path)
    anchors: list[MappingAnchor] = []
    for rank, row in enumerate(payload.get("confirmation", []), start=1):
        large_start, large_end, small_start, small_end = parse_mapping_label(str(row["mapping"]))
        anchors.append(
            MappingAnchor(
                name=str(row["candidate_id"]),
                large_start=large_start,
                large_end=large_end,
                small_start=small_start,
                small_end=small_end,
                stage=str(row["stage"]),
                rank=rank,
            )
        )
    return anchors


def layer_overlap_score(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    """Return the IoU overlap score between two closed integer ranges."""

    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    intersection = max(0, intersection_end - intersection_start + 1)
    if intersection == 0:
        return 0.0
    union = (end_a - start_a + 1) + (end_b - start_b + 1) - intersection
    return float(intersection) / float(union)


def output_anchor_proxy(large_window: AlignmentWindow, small_window: AlignmentWindow, anchors: list[MappingAnchor | dict[str, Any]]) -> float:
    """Return the best overlap-distance to the confirmed Phase 1 shortlist."""

    if not anchors:
        return 0.0
    distances = []
    for anchor in anchors:
        if isinstance(anchor, MappingAnchor):
            anchor_large_start = anchor.large_start
            anchor_large_end = anchor.large_end
            anchor_small_start = anchor.small_start
            anchor_small_end = anchor.small_end
        else:
            anchor_large_start = int(anchor["large_start"])
            anchor_large_end = int(anchor["large_end"])
            anchor_small_start = int(anchor["small_start"])
            anchor_small_end = int(anchor["small_end"])
        large_overlap = layer_overlap_score(large_window.start, large_window.end, anchor_large_start, anchor_large_end)
        small_overlap = layer_overlap_score(small_window.start, small_window.end, anchor_small_start, anchor_small_end)
        distances.append(1.0 - (0.5 * large_overlap + 0.5 * small_overlap))
    return min(distances)


def build_metric_scales(
    large_window_signatures: list[dict[str, float]],
    small_window_signatures: list[dict[str, float]],
    metrics: list[str],
) -> dict[str, float]:
    """Build combined z-score scales for one metric subset."""

    scales: dict[str, float] = {}
    all_rows = [*large_window_signatures, *small_window_signatures]
    for metric in metrics:
        values = [float(row[metric]) for row in all_rows]
        mean = sum(values) / float(len(values))
        variance = sum((value - mean) ** 2 for value in values) / float(len(values))
        scales[metric] = math.sqrt(variance) or 1.0
    return scales


def z_scored_distance(
    large_signature: dict[str, float],
    small_signature: dict[str, float],
    *,
    metrics: list[str],
    scales: dict[str, float],
) -> float:
    """Return a z-scored Euclidean distance between two window signatures."""

    squared_distance = 0.0
    for metric in metrics:
        normalized = (float(large_signature[metric]) - float(small_signature[metric])) / float(scales[metric])
        squared_distance += normalized * normalized
    return math.sqrt(squared_distance)


def normalize_matrix(matrix: list[list[float]]) -> list[list[float]]:
    """Min-max normalize one dense matrix to the `[0, 1]` interval."""

    values = [float(value) for row in matrix for value in row]
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return [[0.0 for _ in row] for row in matrix]
    scale = maximum - minimum
    return [[(float(value) - minimum) / scale for value in row] for row in matrix]


def matrix_to_rows(
    large_windows: list[AlignmentWindow],
    small_windows: list[AlignmentWindow],
    matrix: list[list[float]],
    *,
    value_key: str,
) -> list[dict[str, float | int | str]]:
    """Flatten one matrix into CSV-ready rows."""

    rows: list[dict[str, float | int | str]] = []
    for large_index, large_window in enumerate(large_windows):
        for small_index, small_window in enumerate(small_windows):
            rows.append(
                {
                    "large_label": large_window.label,
                    "large_start": large_window.start,
                    "large_end": large_window.end,
                    "large_length": large_window.length,
                    "small_label": small_window.label,
                    "small_start": small_window.start,
                    "small_end": small_window.end,
                    "small_length": small_window.length,
                    value_key: float(matrix[large_index][small_index]),
                }
            )
    return rows
