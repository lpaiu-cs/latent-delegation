from __future__ import annotations

import math

import torch

from src.eval.eval_stage_b_outputs import compute_batch_output_sums, compute_topk_overlap


def test_compute_batch_output_sums_matches_teacher_for_identical_logits() -> None:
    teacher_logits = torch.tensor(
        [
            [
                [5.0, 1.0, 0.0, -1.0],
                [0.0, 4.0, 1.0, -2.0],
                [0.0, 1.0, 4.0, -2.0],
                [0.0, 1.0, -2.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    totals = compute_batch_output_sums(teacher_logits, teacher_logits, labels, top_k=2)

    assert totals["valid_tokens"] == 3.0
    assert math.isclose(totals["logit_kl_to_teacher_sum"], 0.0, abs_tol=1e-6)
    assert totals["top1_agreement_sum"] == 3.0
    assert totals["top5_overlap_sum"] == 3.0


def test_compute_topk_overlap_returns_fractional_overlap_sum() -> None:
    teacher_logits = torch.tensor(
        [[[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]]],
        dtype=torch.float32,
    )
    student_logits = torch.tensor(
        [[[4.0, 2.5, 3.0, 1.0], [4.0, 2.5, 3.0, 1.0], [4.0, 2.5, 3.0, 1.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1, 2]], dtype=torch.long)

    overlap_sum, valid_tokens = compute_topk_overlap(student_logits, teacher_logits, labels, top_k=2)

    assert valid_tokens == 2
    assert math.isclose(overlap_sum, 1.0, rel_tol=1e-6)
