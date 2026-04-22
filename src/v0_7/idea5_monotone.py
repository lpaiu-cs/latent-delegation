"""Monotone path discovery over local large/small layer correspondences."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, load_config, save_json, save_text
from src.v0_7.common import (
    AlignmentWindow,
    layer_overlap_score,
    load_idea5_discovery_settings,
    load_json,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cost-payload", default="artifacts/v0_7/idea5_discovery/costs/cost_payload.json")
    parser.add_argument("--output-dir", default="artifacts/v0_7/idea5_discovery/solver")
    parser.add_argument("--report-path", default="notes/v0_7/idea5_monotone_alignment_report.md")
    return parser.parse_args()


def _window_from_dict(payload: dict[str, Any]) -> AlignmentWindow:
    return AlignmentWindow(
        family=str(payload["family"]),
        start=int(payload["start"]),
        end=int(payload["end"]),
    )


def _build_cost_lookup(cost_payload: dict[str, Any]) -> dict[tuple[int, int, int, int], dict[str, float]]:
    large_windows = [_window_from_dict(row) for row in cost_payload["windows"]["large"]]
    small_windows = [_window_from_dict(row) for row in cost_payload["windows"]["small"]]
    lookup: dict[tuple[int, int, int, int], dict[str, float]] = {}
    for large_index, large_window in enumerate(large_windows):
        for small_index, small_window in enumerate(small_windows):
            key = (large_window.start, large_window.end, small_window.start, small_window.end)
            lookup[key] = {
                "combined_cost": float(cost_payload["combined_matrix"][large_index][small_index]),
                **{
                    component_name: float(cost_payload["raw_matrices"][component_name][large_index][small_index])
                    for component_name in cost_payload["raw_matrices"].keys()
                },
            }
    return lookup


def solve_top_monotone_paths(cost_payload: dict[str, Any], *, top_paths: int) -> list[dict[str, Any]]:
    """Return the top few monotone paths under the configured move set."""

    settings = cost_payload["local_region"]
    allowed_moves = [tuple(move) for move in cost_payload["allowed_moves"]]
    large_min, large_max = int(settings["large"][0]), int(settings["large"][1])
    small_min, small_max = int(settings["small"][0]), int(settings["small"][1])
    large_total = large_max - large_min + 1
    small_total = small_max - small_min + 1
    cost_lookup = _build_cost_lookup(cost_payload)

    @lru_cache(maxsize=None)
    def solve_from(large_offset: int, small_offset: int) -> tuple[dict[str, Any], ...]:
        if large_offset == large_total and small_offset == small_total:
            return (
                {
                    "segments": [],
                    "total_cost": 0.0,
                    "component_totals": {
                        component_name: 0.0
                        for component_name in cost_payload["raw_matrices"].keys()
                    },
                },
            )
        paths: list[dict[str, Any]] = []
        for large_length, small_length in allowed_moves:
            if large_offset + large_length > large_total or small_offset + small_length > small_total:
                continue
            large_start = large_min + large_offset
            large_end = large_start + large_length - 1
            small_start = small_min + small_offset
            small_end = small_start + small_length - 1
            key = (large_start, large_end, small_start, small_end)
            if key not in cost_lookup:
                continue
            segment_costs = cost_lookup[key]
            suffix_paths = solve_from(large_offset + large_length, small_offset + small_length)
            for suffix in suffix_paths:
                component_totals = {
                    component_name: float(segment_costs[component_name]) + float(suffix["component_totals"][component_name])
                    for component_name in cost_payload["raw_matrices"].keys()
                }
                paths.append(
                    {
                        "segments": [
                            {
                                "large_start": large_start,
                                "large_end": large_end,
                                "large_length": large_length,
                                "small_start": small_start,
                                "small_end": small_end,
                                "small_length": small_length,
                                **segment_costs,
                            },
                            *suffix["segments"],
                        ],
                        "total_cost": float(segment_costs["combined_cost"]) + float(suffix["total_cost"]),
                        "component_totals": component_totals,
                    }
                )
        paths.sort(key=lambda row: float(row["total_cost"]))
        return tuple(paths[:top_paths])

    return list(solve_from(0, 0))


def derive_candidate_from_path(
    path: dict[str, Any],
    *,
    target_large_start: int,
    target_large_end: int,
) -> dict[str, Any] | None:
    """Collapse the top path into one minimal Idea 5 candidate around the target splice."""

    small_layers: set[int] = set()
    for segment in path["segments"]:
        large_overlap = layer_overlap_score(
            int(segment["large_start"]),
            int(segment["large_end"]),
            target_large_start,
            target_large_end,
        )
        if large_overlap <= 0.0:
            continue
        small_layers.update(range(int(segment["small_start"]), int(segment["small_end"]) + 1))
    if not small_layers:
        return None
    derived_small_start = min(small_layers)
    derived_small_end = max(small_layers)
    return {
        "large_start": target_large_start,
        "large_end": target_large_end,
        "small_start": derived_small_start,
        "small_end": derived_small_end,
        "mapping": f"{target_large_start}..{target_large_end} -> {derived_small_start}..{derived_small_end}",
    }


def _path_anchor_overlap(path: dict[str, Any], anchors: list[dict[str, Any]], *, target_large_start: int, target_large_end: int) -> dict[str, Any]:
    best_anchor = None
    best_score = -1.0
    small_layers: set[int] = set()
    for segment in path["segments"]:
        if layer_overlap_score(int(segment["large_start"]), int(segment["large_end"]), target_large_start, target_large_end) > 0.0:
            small_layers.update(range(int(segment["small_start"]), int(segment["small_end"]) + 1))
    if not small_layers:
        return {
            "best_anchor_label": None,
            "best_anchor_overlap": 0.0,
            "small_union_for_target_large": [],
        }
    path_small_start = min(small_layers)
    path_small_end = max(small_layers)
    for anchor in anchors:
        score = layer_overlap_score(path_small_start, path_small_end, int(anchor["small_start"]), int(anchor["small_end"]))
        if score > best_score:
            best_anchor = anchor
            best_score = score
    return {
        "best_anchor_label": best_anchor["label"] if best_anchor else None,
        "best_anchor_overlap": best_score,
        "small_union_for_target_large": [path_small_start, path_small_end],
    }


def _write_report(
    report_path: str | Path,
    *,
    cost_payload: dict[str, Any],
    path_payload: dict[str, Any],
) -> None:
    top_path = path_payload["top_paths"][0]
    derived_candidate = path_payload.get("derived_candidate")
    has_asymmetry = any(
        int(segment["large_length"]) != int(segment["small_length"])
        for segment in top_path["segments"]
    )
    recovered_shortlist = float(top_path["target_large_anchor_overlap"]["best_anchor_overlap"]) >= 0.5
    legacy_is_plausible = path_payload["legacy_vs_shortlist"]["legacy_cost"] <= path_payload["legacy_vs_shortlist"]["best_shortlist_cost"]
    window_diagnostics = path_payload["window_diagnostics"]
    lines = [
        "# Idea 5 Monotone Alignment Report",
        "",
        "## Scope",
        "",
        "- Local discovery region on the large side: `22..30`.",
        "- Local discovery region on the small side: `13..20`.",
        "- Cost components: stage-signature distance, hidden-alignment proxy, logit-disruption proxy, and a disabled-by-default output-anchor proxy.",
        "- Solver moves: `1:1`, `2:1`, `1:2`, `2:2`, and `3:2`.",
        "",
        "## Top Path",
        "",
        f"- Total cost: `{top_path['total_cost']:.6f}`",
        f"- Segment count: `{len(top_path['segments'])}`",
        f"- Best overlap with the confirmed Phase 1 shortlist around `24..27`: `{top_path['target_large_anchor_overlap']['best_anchor_label']}` at overlap `{top_path['target_large_anchor_overlap']['best_anchor_overlap']:.3f}`.",
        "",
        "### Segments",
        "",
    ]
    for index, segment in enumerate(top_path["segments"], start=1):
        lines.append(
            f"{index}. `{segment['large_start']}..{segment['large_end']} -> {segment['small_start']}..{segment['small_end']}` "
            f"| combined=`{segment['combined_cost']:.6f}` "
            f"| stage=`{segment['stage_signature_distance']:.6f}` "
            f"| hidden=`{segment['hidden_alignment_proxy']:.6f}` "
            f"| logit=`{segment['logit_disruption_proxy']:.6f}`"
        )
    lines.extend(["", "## Window Diagnostics", ""])
    for row in window_diagnostics:
        lines.append(
            f"- `{row['mapping']}` | proxy=`{row['combined_proxy_cost']:.6f}` "
            f"| stage=`{row['stage_signature_distance']:.6f}` "
            f"| hidden=`{row['hidden_alignment_proxy']:.6f}` "
            f"| logit=`{row['logit_disruption_proxy']:.6f}`"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Does the monotone solver naturally recover the successful two-path shortlist region? {'Yes' if recovered_shortlist else 'Only weakly'}.",
            f"2. Does it suggest that the two shortlisted windows are adjacent samples from a broader low-cost corridor? {'Yes' if path_payload['corridor_support']['has_shortlist_corridor'] else 'No clear corridor yet'}.",
            f"3. Does it suggest asymmetric mapping pressure or multi-segment local correspondence? {'Yes' if has_asymmetry else 'No'}.",
            f"4. Does it make the old `24..29 -> 14..19` split look structurally implausible in a more principled way? {'Yes' if not legacy_is_plausible else 'Not decisively from this discovery signal alone'}.",
            f"5. Does it justify building an Idea 5 model at all? {'Yes, in a bounded follow-up' if path_payload['justify_model_building'] else 'Not yet'}.",
        ]
    )
    if derived_candidate is not None:
        lines.extend(
            [
                "",
                "## Minimal Derived Candidate",
                "",
                f"- Proposed single-path compression of the top monotone path around the successful splice: `{derived_candidate['mapping']}`.",
                "- This is a bounded follow-up candidate only. It was not promoted to a new architecture in this discovery run.",
            ]
        )
    save_text(report_path, "\n".join(lines))


def build_path_payload(config_path: str, cost_payload_path: str | Path, output_dir: str | Path, report_path: str | Path) -> dict[str, Any]:
    """Run the solver, save artifacts, and write the alignment report."""

    config = load_config(config_path)
    settings = load_idea5_discovery_settings(config)
    cost_payload = load_json(cost_payload_path)
    output_root = ensure_dir(output_dir)

    top_paths = solve_top_monotone_paths(cost_payload, top_paths=settings.top_paths)
    anchors = cost_payload["anchors"]
    for path in top_paths:
        path["target_large_anchor_overlap"] = _path_anchor_overlap(
            path,
            anchors,
            target_large_start=24,
            target_large_end=27,
        )

    shortlist_pairs = cost_payload["top_pairs"]
    window_diagnostics = cost_payload.get("window_diagnostics", [])
    shortlist_corridor = [
        row for row in shortlist_pairs
        if 24 <= int(row["large_start"]) <= 27 and 14 <= int(row["small_start"]) <= 19
    ]
    legacy_cost = next(
        (
            float(row["combined_proxy_cost"])
            for row in window_diagnostics
            if str(row["mapping"]) == "24..29 -> 14..19"
        ),
        float("inf"),
    )
    best_shortlist_cost = min(
        [
            float(row["combined_proxy_cost"])
            for row in window_diagnostics
            if str(row["mapping"]) in {"24..27 -> 14..19", "24..27 -> 16..18"}
        ],
        default=float("inf"),
    )

    derived_candidate = derive_candidate_from_path(
        top_paths[0],
        target_large_start=24,
        target_large_end=27,
    ) if top_paths else None

    path_payload = {
        "config_path": config_path,
        "cost_payload_path": str(cost_payload_path),
        "top_paths": top_paths,
        "derived_candidate": derived_candidate,
        "corridor_support": {
            "has_shortlist_corridor": len(shortlist_corridor) >= 2,
            "top_pair_count_inside_shortlist_region": len(shortlist_corridor),
        },
        "window_diagnostics": window_diagnostics,
        "legacy_vs_shortlist": {
            "legacy_cost": legacy_cost,
            "best_shortlist_cost": best_shortlist_cost,
        },
        "justify_model_building": bool(top_paths) and (
            top_paths[0]["target_large_anchor_overlap"]["best_anchor_overlap"] >= 0.5
            or len(shortlist_corridor) >= 3
        ),
    }

    save_json(output_root / "top_paths.json", path_payload)
    _write_report(report_path, cost_payload=cost_payload, path_payload=path_payload)
    return path_payload


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    build_path_payload(args.config, args.cost_payload, args.output_dir, args.report_path)


if __name__ == "__main__":
    main()
