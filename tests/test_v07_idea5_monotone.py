from __future__ import annotations

from src.v0_7.idea5_monotone import derive_candidate_from_path, solve_top_monotone_paths


def test_solve_top_monotone_paths_returns_order_preserving_segments() -> None:
    cost_payload = {
        "local_region": {
            "large": [24, 27],
            "small": [14, 17],
        },
        "allowed_moves": [[1, 1], [2, 1], [1, 2]],
        "windows": {
            "large": [
                {"family": "large", "start": 24, "end": 24},
                {"family": "large", "start": 25, "end": 25},
                {"family": "large", "start": 26, "end": 26},
                {"family": "large", "start": 27, "end": 27},
                {"family": "large", "start": 24, "end": 25},
                {"family": "large", "start": 26, "end": 27},
            ],
            "small": [
                {"family": "small", "start": 14, "end": 14},
                {"family": "small", "start": 15, "end": 15},
                {"family": "small", "start": 16, "end": 16},
                {"family": "small", "start": 17, "end": 17},
                {"family": "small", "start": 14, "end": 15},
                {"family": "small", "start": 16, "end": 17},
            ],
        },
        "raw_matrices": {
            "stage_signature_distance": [
                [1.0, 2.0, 3.0, 4.0, 1.5, 3.5],
                [2.0, 1.0, 2.5, 3.5, 1.2, 3.2],
                [3.0, 2.0, 1.0, 2.0, 2.7, 1.1],
                [4.0, 3.0, 2.0, 1.0, 3.7, 1.2],
                [0.5, 3.0, 5.0, 6.0, 0.2, 4.5],
                [6.0, 5.0, 2.0, 0.8, 4.0, 0.3],
            ],
            "hidden_alignment_proxy": [
                [1.0, 2.0, 3.0, 4.0, 1.5, 3.5],
                [2.0, 1.0, 2.5, 3.5, 1.2, 3.2],
                [3.0, 2.0, 1.0, 2.0, 2.7, 1.1],
                [4.0, 3.0, 2.0, 1.0, 3.7, 1.2],
                [0.5, 3.0, 5.0, 6.0, 0.2, 4.5],
                [6.0, 5.0, 2.0, 0.8, 4.0, 0.3],
            ],
            "logit_disruption_proxy": [
                [1.0, 2.0, 3.0, 4.0, 1.5, 3.5],
                [2.0, 1.0, 2.5, 3.5, 1.2, 3.2],
                [3.0, 2.0, 1.0, 2.0, 2.7, 1.1],
                [4.0, 3.0, 2.0, 1.0, 3.7, 1.2],
                [0.5, 3.0, 5.0, 6.0, 0.2, 4.5],
                [6.0, 5.0, 2.0, 0.8, 4.0, 0.3],
            ],
            "output_anchor_proxy": [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        },
        "combined_matrix": [
            [1.0, 2.0, 3.0, 4.0, 1.5, 3.5],
            [2.0, 1.0, 2.5, 3.5, 1.2, 3.2],
            [3.0, 2.0, 1.0, 2.0, 2.7, 1.1],
            [4.0, 3.0, 2.0, 1.0, 3.7, 1.2],
            [0.5, 3.0, 5.0, 6.0, 0.2, 4.5],
            [6.0, 5.0, 2.0, 0.8, 4.0, 0.3],
        ],
    }

    paths = solve_top_monotone_paths(cost_payload, top_paths=3)

    assert paths
    assert paths[0]["total_cost"] <= paths[1]["total_cost"]
    assert paths[0]["segments"][0]["large_start"] == 24
    assert paths[0]["segments"][-1]["large_end"] == 27


def test_derive_candidate_from_path_unions_small_layers_for_target_large() -> None:
    path = {
        "segments": [
            {"large_start": 22, "large_end": 23, "small_start": 13, "small_end": 14},
            {"large_start": 24, "large_end": 25, "small_start": 15, "small_end": 16},
            {"large_start": 26, "large_end": 27, "small_start": 17, "small_end": 18},
            {"large_start": 28, "large_end": 30, "small_start": 19, "small_end": 20},
        ]
    }

    candidate = derive_candidate_from_path(path, target_large_start=24, target_large_end=27)

    assert candidate is not None
    assert candidate["mapping"] == "24..27 -> 15..18"
