from __future__ import annotations

from pathlib import Path

import torch

from src.models.backbone_loader import load_backbones
from src.models.hybrid_gemma import HybridDelegationModel
from src.utils.io import load_config


def test_hybrid_forward_shapes_debug_tiny() -> None:
    config = load_config(Path("configs/debug_tiny.yaml"))
    backbones = load_backbones(config)
    model = HybridDelegationModel(config, backbones.large_model, backbones.small_model).to(backbones.device)

    batch = backbones.tokenizer(
        ["latent delegation test", "bridge-only baseline check"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.training.seq_len,
    )
    input_ids = batch["input_ids"].to(backbones.device)
    attention_mask = batch["attention_mask"].to(backbones.device)

    outputs = model(input_ids, attention_mask=attention_mask)
    assert outputs.logits.shape[:2] == input_ids.shape
    assert outputs.logits.shape[-1] == config.model.debug_vocab_size
    assert outputs.hidden_after_prefix.shape[-1] == config.model.debug_large_hidden_size
    assert outputs.delegated_small_hidden is not None
    assert outputs.delegated_small_hidden.shape[-1] == config.model.debug_small_hidden_size
    assert outputs.delta_large is not None
    assert outputs.delta_large.shape[-1] == config.model.debug_large_hidden_size
    assert torch.isfinite(outputs.logits).all()
