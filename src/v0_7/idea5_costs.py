"""Pairwise local-window cost estimation for the v0.7 Idea 5 discovery track."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, load_config, save_csv, save_json
from src.v0_6.stage_signatures import build_window_signature
from src.v0_7.common import (
    DISCOVERY_COMPONENTS,
    HIDDEN_ALIGNMENT_METRICS,
    LOGIT_DISRUPTION_METRICS,
    STAGE_SIGNATURE_COMPONENT_METRICS,
    AlignmentWindow,
    build_metric_scales,
    enumerate_local_windows,
    load_idea5_discovery_settings,
    load_json,
    load_phase1_shortlist_anchors,
    matrix_to_rows,
    normalize_matrix,
    output_anchor_proxy,
    z_scored_distance,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="artifacts/v0_7/idea5_discovery/costs")
    return parser.parse_args()


def _window_signatures(
    layer_signatures: list[dict[str, float]],
    windows: list[AlignmentWindow],
) -> list[dict[str, float]]:
    signatures: list[dict[str, float]] = []
    for window in windows:
        signature = build_window_signature(layer_signatures, window.start, window.end)
        signature["label"] = window.label
        signatures.append(signature)
    return signatures


def _combine_component_matrices(
    normalized_matrices: dict[str, list[list[float]]],
    component_weights: dict[str, float],
) -> list[list[float]]:
    height = len(next(iter(normalized_matrices.values())))
    width = len(next(iter(normalized_matrices.values()))[0])
    combined = [[0.0 for _ in range(width)] for _ in range(height)]
    for component_name, matrix in normalized_matrices.items():
        weight = float(component_weights.get(component_name, 0.0))
        for row_index in range(height):
            for column_index in range(width):
                combined[row_index][column_index] += weight * float(matrix[row_index][column_index])
    return combined


def _normalize_value(value: float, matrix: list[list[float]]) -> float:
    values = [float(item) for row in matrix for item in row]
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return 0.0
    return (float(value) - minimum) / (maximum - minimum)


def _top_pairs(
    large_windows: list[AlignmentWindow],
    small_windows: list[AlignmentWindow],
    combined_matrix: list[list[float]],
    raw_matrices: dict[str, list[list[float]]],
    *,
    top_pairs: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for large_index, large_window in enumerate(large_windows):
        for small_index, small_window in enumerate(small_windows):
            row: dict[str, float | int | str] = {
                "large_label": large_window.label,
                "large_start": large_window.start,
                "large_end": large_window.end,
                "large_length": large_window.length,
                "small_label": small_window.label,
                "small_start": small_window.start,
                "small_end": small_window.end,
                "small_length": small_window.length,
                "combined_cost": float(combined_matrix[large_index][small_index]),
            }
            for component_name, matrix in raw_matrices.items():
                row[component_name] = float(matrix[large_index][small_index])
            rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["combined_cost"]),
            int(row["large_start"]),
            int(row["small_start"]),
            int(row["large_length"]),
            int(row["small_length"]),
        )
    )
    return rows[:top_pairs]


def _window_diagnostics(
    *,
    signature_payload: dict[str, Any],
    stage_scales: dict[str, float],
    hidden_scales: dict[str, float],
    logit_scales: dict[str, float],
    raw_matrices: dict[str, list[list[float]]],
    component_weights: dict[str, float],
    anchors: list[Any],
) -> list[dict[str, float | int | str]]:
    specs = [
        ("legacy_fixed_split", 24, 29, 14, 19),
        ("shortlist_path_b", 24, 27, 14, 19),
        ("shortlist_path_a", 24, 27, 16, 18),
        ("derived_midpoint", 24, 27, 15, 18),
        ("phase1_candidate_c", 25, 29, 15, 19),
    ]
    rows: list[dict[str, float | int | str]] = []
    for name, large_start, large_end, small_start, small_end in specs:
        large_signature = build_window_signature(signature_payload["layer_signatures"]["large"], large_start, large_end)
        small_signature = build_window_signature(signature_payload["layer_signatures"]["small"], small_start, small_end)
        stage_distance = z_scored_distance(
            large_signature,
            small_signature,
            metrics=STAGE_SIGNATURE_COMPONENT_METRICS,
            scales=stage_scales,
        )
        hidden_distance = z_scored_distance(
            large_signature,
            small_signature,
            metrics=HIDDEN_ALIGNMENT_METRICS,
            scales=hidden_scales,
        )
        logit_distance = z_scored_distance(
            large_signature,
            small_signature,
            metrics=LOGIT_DISRUPTION_METRICS,
            scales=logit_scales,
        )
        anchor_distance = output_anchor_proxy(
            AlignmentWindow("large", large_start, large_end),
            AlignmentWindow("small", small_start, small_end),
            anchors,
        )
        combined_proxy_cost = (
            float(component_weights["stage_signature_distance"]) * _normalize_value(stage_distance, raw_matrices["stage_signature_distance"])
            + float(component_weights["hidden_alignment_proxy"]) * _normalize_value(hidden_distance, raw_matrices["hidden_alignment_proxy"])
            + float(component_weights["logit_disruption_proxy"]) * _normalize_value(logit_distance, raw_matrices["logit_disruption_proxy"])
            + float(component_weights["output_anchor_proxy"]) * _normalize_value(anchor_distance, raw_matrices["output_anchor_proxy"])
        )
        rows.append(
            {
                "name": name,
                "mapping": f"{large_start}..{large_end} -> {small_start}..{small_end}",
                "large_start": large_start,
                "large_end": large_end,
                "small_start": small_start,
                "small_end": small_end,
                "stage_signature_distance": stage_distance,
                "hidden_alignment_proxy": hidden_distance,
                "logit_disruption_proxy": logit_distance,
                "output_anchor_proxy": anchor_distance,
                "combined_proxy_cost": combined_proxy_cost,
            }
        )
    rows.sort(key=lambda row: float(row["combined_proxy_cost"]))
    return rows


def build_cost_payload(config_path: str, output_dir: str | Path) -> dict[str, Any]:
    """Build and persist the Idea 5 local pairwise cost matrices."""

    config = load_config(config_path)
    settings = load_idea5_discovery_settings(config)
    output_root = ensure_dir(output_dir)

    signature_payload = load_json(settings.stage_signature_artifact)
    anchors = load_phase1_shortlist_anchors(settings.phase1_ranking_artifact)

    large_windows = enumerate_local_windows(
        "large",
        layer_min=settings.large_layer_min,
        layer_max=settings.large_layer_max,
        lengths=settings.large_segment_lengths,
    )
    small_windows = enumerate_local_windows(
        "small",
        layer_min=settings.small_layer_min,
        layer_max=settings.small_layer_max,
        lengths=settings.small_segment_lengths,
    )

    large_signatures = _window_signatures(signature_payload["layer_signatures"]["large"], large_windows)
    small_signatures = _window_signatures(signature_payload["layer_signatures"]["small"], small_windows)

    stage_scales = build_metric_scales(large_signatures, small_signatures, STAGE_SIGNATURE_COMPONENT_METRICS)
    hidden_scales = build_metric_scales(large_signatures, small_signatures, HIDDEN_ALIGNMENT_METRICS)
    logit_scales = build_metric_scales(large_signatures, small_signatures, LOGIT_DISRUPTION_METRICS)

    raw_matrices = {component_name: [] for component_name in DISCOVERY_COMPONENTS}
    for large_window, large_signature in zip(large_windows, large_signatures, strict=True):
        stage_row: list[float] = []
        hidden_row: list[float] = []
        logit_row: list[float] = []
        anchor_row: list[float] = []
        for small_window, small_signature in zip(small_windows, small_signatures, strict=True):
            stage_row.append(
                z_scored_distance(
                    large_signature,
                    small_signature,
                    metrics=STAGE_SIGNATURE_COMPONENT_METRICS,
                    scales=stage_scales,
                )
            )
            hidden_row.append(
                z_scored_distance(
                    large_signature,
                    small_signature,
                    metrics=HIDDEN_ALIGNMENT_METRICS,
                    scales=hidden_scales,
                )
            )
            logit_row.append(
                z_scored_distance(
                    large_signature,
                    small_signature,
                    metrics=LOGIT_DISRUPTION_METRICS,
                    scales=logit_scales,
                )
            )
            anchor_row.append(output_anchor_proxy(large_window, small_window, anchors))
        raw_matrices["stage_signature_distance"].append(stage_row)
        raw_matrices["hidden_alignment_proxy"].append(hidden_row)
        raw_matrices["logit_disruption_proxy"].append(logit_row)
        raw_matrices["output_anchor_proxy"].append(anchor_row)

    normalized_matrices = {
        component_name: normalize_matrix(matrix)
        for component_name, matrix in raw_matrices.items()
    }
    combined_matrix = _combine_component_matrices(normalized_matrices, settings.component_weights)

    payload = {
        "config_path": config_path,
        "stage_signature_artifact": settings.stage_signature_artifact,
        "phase1_ranking_artifact": settings.phase1_ranking_artifact,
        "local_region": {
            "large": [settings.large_layer_min, settings.large_layer_max],
            "small": [settings.small_layer_min, settings.small_layer_max],
        },
        "segment_lengths": {
            "large": settings.large_segment_lengths,
            "small": settings.small_segment_lengths,
        },
        "allowed_moves": [[large_length, small_length] for large_length, small_length in settings.allowed_moves],
        "component_weights": settings.component_weights,
        "windows": {
            "large": [window.to_dict() for window in large_windows],
            "small": [window.to_dict() for window in small_windows],
        },
        "anchors": [anchor.to_dict() for anchor in anchors],
        "raw_matrices": raw_matrices,
        "normalized_matrices": normalized_matrices,
        "combined_matrix": combined_matrix,
        "top_pairs": _top_pairs(
            large_windows,
            small_windows,
            combined_matrix,
            raw_matrices,
            top_pairs=settings.top_pairs,
        ),
        "window_diagnostics": _window_diagnostics(
            signature_payload=signature_payload,
            stage_scales=stage_scales,
            hidden_scales=hidden_scales,
            logit_scales=logit_scales,
            raw_matrices=raw_matrices,
            component_weights=settings.component_weights,
            anchors=anchors,
        ),
    }

    save_json(output_root / "cost_payload.json", payload)
    save_json(output_root / "candidate_windows.json", payload["windows"])
    for component_name, matrix in raw_matrices.items():
        save_json(
            output_root / f"{component_name}.json",
            {
                "large_windows": payload["windows"]["large"],
                "small_windows": payload["windows"]["small"],
                "matrix": matrix,
            },
        )
        save_csv(
            output_root / f"{component_name}.csv",
            matrix_to_rows(large_windows, small_windows, matrix, value_key=component_name),
        )
    save_json(
        output_root / "combined_cost_matrix.json",
        {
            "large_windows": payload["windows"]["large"],
            "small_windows": payload["windows"]["small"],
            "matrix": combined_matrix,
        },
    )
    save_csv(
        output_root / "combined_cost_matrix.csv",
        matrix_to_rows(large_windows, small_windows, combined_matrix, value_key="combined_cost"),
    )
    save_csv(output_root / "top_pairs.csv", payload["top_pairs"])
    save_json(output_root / "top_pairs.json", payload["top_pairs"])
    save_json(output_root / "window_diagnostics.json", payload["window_diagnostics"])
    save_csv(output_root / "window_diagnostics.csv", payload["window_diagnostics"])
    return payload


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    build_cost_payload(args.config, args.output_dir)


if __name__ == "__main__":
    main()
