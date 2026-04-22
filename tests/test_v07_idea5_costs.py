from __future__ import annotations

from src.v0_7.common import (
    AlignmentWindow,
    layer_overlap_score,
    normalize_matrix,
    output_anchor_proxy,
)


def test_normalize_matrix_scales_into_unit_interval() -> None:
    matrix = [[2.0, 4.0], [6.0, 8.0]]

    normalized = normalize_matrix(matrix)

    assert normalized[0][0] == 0.0
    assert normalized[1][1] == 1.0
    assert 0.0 < normalized[0][1] < 1.0


def test_layer_overlap_score_uses_closed_interval_iou() -> None:
    score = layer_overlap_score(24, 27, 26, 29)

    assert score == 2.0 / 6.0


def test_output_anchor_proxy_prefers_overlap_with_shortlist_anchor() -> None:
    large_window = AlignmentWindow("large", 24, 25)
    small_window = AlignmentWindow("small", 16, 18)
    anchors = [
        {
            "label": "24..27 -> 14..19",
            "large_start": 24,
            "large_end": 27,
            "small_start": 14,
            "small_end": 19,
        },
        {
            "label": "24..27 -> 16..18",
            "large_start": 24,
            "large_end": 27,
            "small_start": 16,
            "small_end": 18,
        },
    ]

    exact_overlap = output_anchor_proxy(large_window, small_window, anchors)  # type: ignore[arg-type]
    displaced_overlap = output_anchor_proxy(
        AlignmentWindow("large", 28, 29),
        AlignmentWindow("small", 13, 14),
        anchors,  # type: ignore[arg-type]
    )

    assert exact_overlap < displaced_overlap

