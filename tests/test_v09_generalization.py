from __future__ import annotations

import math

import torch

from src.v0_9.aggregate_generalization import paired_bootstrap_delta
from src.v0_9.task_scoring import continuation_logprob_summaries, sample_indices


def test_sample_indices_is_deterministic() -> None:
    first = sample_indices(20, 5, 1234)
    second = sample_indices(20, 5, 1234)

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5


def test_continuation_logprob_summaries_respects_continuation_mask() -> None:
    vocab_size = 5
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    continuation_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32)

    summaries = continuation_logprob_summaries(logits, input_ids, continuation_mask)
    expected = -math.log(vocab_size)

    assert summaries["token_count"].tolist() == [2.0]
    assert torch.allclose(summaries["sum_logprob"], torch.tensor([2 * expected], dtype=torch.float32))
    assert torch.allclose(summaries["avg_logprob"], torch.tensor([expected], dtype=torch.float32))


def test_paired_bootstrap_delta_returns_expected_sign() -> None:
    result = paired_bootstrap_delta(
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        num_samples=200,
        seed=99,
    )

    assert result["delta_mean"] == 1.0
    assert result["ci_low"] > 0.0
    assert result["ci_high"] >= result["ci_low"]
