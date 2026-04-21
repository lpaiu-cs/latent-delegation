from __future__ import annotations

from src.v0_6.stage_signatures import build_window_signature, build_window_signatures, rank_window_matches


def _layer_row(layer: int, base: float) -> dict[str, float]:
    return {
        "layer": float(layer),
        "hidden_norm_mean": base,
        "delta_norm_mean": base + 1.0,
        "delta_cosine_mean": base + 2.0,
        "logit_entropy_mean": base + 3.0,
        "logit_kl_to_final_mean": base + 4.0,
    }


def test_build_window_signature_averages_metrics() -> None:
    layer_signatures = [_layer_row(0, 1.0), _layer_row(1, 3.0), _layer_row(2, 5.0)]
    window = build_window_signature(layer_signatures, 0, 1)

    assert window["start"] == 0.0
    assert window["end"] == 1.0
    assert window["length"] == 2.0
    assert window["hidden_norm_mean"] == 2.0
    assert window["logit_kl_to_final_mean"] == 6.0


def test_build_window_signatures_emits_all_contiguous_windows() -> None:
    layer_signatures = [_layer_row(layer, float(layer)) for layer in range(5)]
    windows = build_window_signatures(layer_signatures, [2, 3])

    labels = {(int(window["start"]), int(window["end"])) for window in windows}
    assert labels == {(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4)}


def test_rank_window_matches_prefers_closest_signature() -> None:
    reference = _layer_row(0, 5.0) | {"start": 10.0, "end": 11.0, "length": 2.0}
    close = _layer_row(1, 5.1) | {"start": 1.0, "end": 2.0, "length": 2.0}
    far = _layer_row(2, 9.0) | {"start": 3.0, "end": 4.0, "length": 2.0}

    ranked = rank_window_matches(reference, [far, close], top_k=2)

    assert int(ranked[0]["start"]) == 1
    assert ranked[0]["distance"] < ranked[1]["distance"]

