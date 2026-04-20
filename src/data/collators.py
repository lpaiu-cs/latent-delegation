"""Batch collation utilities."""

from __future__ import annotations

from typing import Any

import torch


class CausalLMCollator:
    """Stack pre-tokenized fixed-length language modeling batches."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([feature["input_ids"] for feature in features], dim=0)
        attention_mask = torch.stack([feature["attention_mask"] for feature in features], dim=0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
