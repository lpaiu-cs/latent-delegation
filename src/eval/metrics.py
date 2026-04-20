"""Metrics and answer parsers."""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F


def masked_hidden_mse(prediction: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error over non-padding positions."""

    mask = attention_mask.unsqueeze(-1).to(prediction.dtype)
    squared_error = (prediction - target).pow(2) * mask
    denom = mask.sum().clamp_min(1.0) * prediction.shape[-1]
    return squared_error.sum() / denom


def masked_hidden_cosine_loss(prediction: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute 1 - cosine similarity over non-padding positions."""

    cosine = F.cosine_similarity(prediction, target, dim=-1)
    mask = attention_mask.to(cosine.dtype)
    return ((1.0 - cosine) * mask).sum() / mask.sum().clamp_min(1.0)


def shifted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token causal LM cross entropy."""

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def shifted_kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """KL divergence on next-token distributions over valid positions."""

    mask = (labels[:, 1:] != -100).reshape(-1)
    student = student_logits[:, :-1, :].reshape(-1, student_logits.size(-1))[mask]
    teacher = teacher_logits[:, :-1, :].reshape(-1, teacher_logits.size(-1))[mask]
    student_log_probs = F.log_softmax(student, dim=-1)
    teacher_probs = F.softmax(teacher, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def perplexity_from_loss(loss_value: float) -> float:
    """Convert average NLL to perplexity."""

    return float(math.exp(loss_value))


def parse_final_number(text: str) -> str | None:
    """Extract the final numeric answer from a generation."""

    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


def parse_yes_no(text: str) -> str | None:
    """Extract a normalized yes/no answer."""

    lowered = text.lower()
    yes_index = lowered.rfind("yes")
    no_index = lowered.rfind("no")
    if yes_index == -1 and no_index == -1:
        return None
    if yes_index > no_index:
        return "yes"
    return "no"


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Run simple greedy decoding by repeated full-sequence forwards."""

    generated_ids = input_ids
    generated_mask = attention_mask
    for _ in range(max_new_tokens):
        outputs = model(generated_ids, attention_mask=generated_mask)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        next_mask = torch.ones_like(next_token, dtype=generated_mask.dtype, device=generated_mask.device)
        generated_mask = torch.cat([generated_mask, next_mask], dim=1)
    return generated_ids
