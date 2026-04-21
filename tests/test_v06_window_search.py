from __future__ import annotations

from pathlib import Path

from src.utils.io import load_config
from src.v0_6.common import WindowCandidate, clone_config
from src.v0_6.window_search import distinct_small_windows, enumerate_window_candidates, load_window_search_settings, shortlist_rows


def test_enumerate_window_candidates_includes_default_debug_split() -> None:
    config = load_config(Path("configs/v0_6/debug_tiny_phase1.yaml"))
    settings = load_window_search_settings(config)

    candidates = enumerate_window_candidates(config, settings, large_num_layers=42, small_num_layers=26)

    assert candidates
    assert any(candidate.label == "L24-27__S14-16" for candidate in candidates)
    assert any(candidate.label == "L24-29__S14-19" for candidate in candidates)


def test_clone_config_updates_split_and_training_steps() -> None:
    config = load_config(Path("configs/v0_6/debug_tiny_phase1.yaml"))
    candidate = WindowCandidate(large_start=26, large_end=29, small_start=16, small_end=18)

    cloned = clone_config(config, candidate=candidate, seed=123, stage_a_steps=7, stage_b_steps=9)

    assert cloned.split.large_prefix_end == 25
    assert cloned.split.large_removed_start == 26
    assert cloned.split.large_removed_end == 29
    assert cloned.split.large_suffix_start == 30
    assert cloned.split.small_entry_target_layer == 15
    assert cloned.training.seed == 123
    assert cloned.training.stage_a.max_steps == 7
    assert cloned.training.stage_b.max_steps == 9


def test_shortlist_rows_uses_primary_output_metrics_first() -> None:
    rows = [
        {
            "label": "candidate_a",
            "hybrid_logit_kl_to_teacher_mean": 0.5,
            "hybrid_nll_mean": 1.0,
            "hybrid_perplexity_mean": 2.0,
            "hybrid_hidden_mse_mean": 0.1,
            "hybrid_hidden_cosine_mean": 0.9,
        },
        {
            "label": "candidate_b",
            "hybrid_logit_kl_to_teacher_mean": 0.4,
            "hybrid_nll_mean": 1.2,
            "hybrid_perplexity_mean": 2.1,
            "hybrid_hidden_mse_mean": 0.01,
            "hybrid_hidden_cosine_mean": 0.99,
        },
    ]

    shortlist = shortlist_rows(rows, shortlist_size=1)

    assert shortlist[0]["label"] == "candidate_b"


def test_distinct_small_windows_removes_duplicate_small_ranges() -> None:
    candidates = [
        WindowCandidate(24, 27, 14, 16),
        WindowCandidate(26, 29, 14, 16),
        WindowCandidate(24, 29, 16, 18),
    ]

    selected = distinct_small_windows(candidates, limit=3)

    assert [candidate.label for candidate in selected] == ["L24-27__S14-16", "L24-29__S16-18"]
