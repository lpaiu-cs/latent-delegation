from __future__ import annotations

from pathlib import Path

import torch

from src.models.adapters import EntryProjector, LowRankAdapter
from src.models.backbone_loader import load_backbones
from src.models.baselines import BridgeOnlyParamMatchedModel
from src.models.hybrid_gemma import HybridDelegationModel, HybridNoSmallModel
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


def test_adapters_preserve_incoming_activation_dtype() -> None:
    hidden_large = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    hidden_small = torch.randn(2, 8, 12, dtype=torch.bfloat16)

    entry = EntryProjector(input_dim=16, output_dim=12, use_rmsnorm=True, rms_norm_eps=1e-6)
    bridge = LowRankAdapter(input_dim=12, output_dim=16, rank=4)

    projected = entry(hidden_large)
    returned = bridge(hidden_small)

    assert projected.dtype == torch.bfloat16
    assert returned.dtype == torch.bfloat16


def test_hybrid_no_small_uses_identity_small_path_and_preserves_shapes() -> None:
    config = load_config(Path("configs/debug_tiny.yaml"))
    backbones = load_backbones(config)
    model = HybridNoSmallModel(config, backbones.large_model, backbones.small_model).to(backbones.device)

    batch = backbones.tokenizer(
        ["hybrid no small control"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.training.seq_len,
    )
    outputs = model(
        batch["input_ids"].to(backbones.device),
        attention_mask=batch["attention_mask"].to(backbones.device),
    )

    assert outputs.projected_small_hidden is not None
    assert outputs.delegated_small_hidden is not None
    assert outputs.projected_small_hidden.shape == outputs.delegated_small_hidden.shape
    assert torch.allclose(outputs.projected_small_hidden, outputs.delegated_small_hidden)
    assert outputs.hidden_after_removed.shape[-1] == config.model.debug_large_hidden_size


def test_bridge_only_param_matched_forward_shapes_debug_tiny() -> None:
    config = load_config(Path("configs/debug_tiny.yaml"))
    backbones = load_backbones(config, load_large=True, load_small=False)
    model = BridgeOnlyParamMatchedModel(config, backbones.large_model, rank=3).to(backbones.device)

    batch = backbones.tokenizer(
        ["parameter matched bridge control"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.training.seq_len,
    )
    outputs = model(
        batch["input_ids"].to(backbones.device),
        attention_mask=batch["attention_mask"].to(backbones.device),
    )

    assert outputs.delta_large is not None
    assert outputs.delta_large.shape[-1] == config.model.debug_large_hidden_size
    assert outputs.hidden_after_removed.shape[-1] == config.model.debug_large_hidden_size
