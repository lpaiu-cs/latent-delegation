from __future__ import annotations

from pathlib import Path

import torch

from src.models.backbone_loader import load_backbones
from src.utils.io import load_config
from src.v0_6.idea4_common import load_mixture_path_specs
from src.v0_8.idea2_models import (
    SublayerAttributionTokenwiseHybrid,
    ablation_spec_by_name,
    idea2_ablation_specs,
)


def _build_debug_model() -> tuple[object, object, SublayerAttributionTokenwiseHybrid]:
    config = load_config(Path("configs/v0_8/debug_tiny_idea2_attribution.yaml"))
    backbones = load_backbones(config)
    path_specs = load_mixture_path_specs(config)
    model = SublayerAttributionTokenwiseHybrid(config, backbones.large_model, backbones.small_model, path_specs).to(backbones.device)
    return config, backbones, model


def test_idea2_ablation_specs_include_required_global_variants() -> None:
    specs = idea2_ablation_specs(include_path_specific=True)
    labels = [spec.name for spec in specs]

    assert "tokenwise_full" in labels
    assert "tokenwise_attn_suppressed" in labels
    assert "tokenwise_mlp_suppressed" in labels
    assert "tokenwise_both_suppressed" in labels
    assert "tokenwise_attn_suppressed_path_a" in labels
    assert "tokenwise_mlp_suppressed_path_b" in labels


def test_both_suppressed_keeps_each_path_hidden_equal_to_projected_small() -> None:
    config, backbones, model = _build_debug_model()
    batch = backbones.tokenizer(
        ["idea two attribution test"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.training.seq_len,
    )
    prefix_state = model.large_runner.prepare_from_input_ids(
        batch["input_ids"].to(backbones.device),
        attention_mask=batch["attention_mask"].to(backbones.device),
    )
    prefix_state = model.large_runner.run_layers(prefix_state, 0, config.split.large_prefix_end)
    model.set_active_ablation(ablation_spec_by_name(idea2_ablation_specs(include_path_specific=True), "tokenwise_both_suppressed"))
    path_outputs = model.compute_path_outputs(prefix_state.hidden_states.detach(), prefix_state.attention_mask_2d, train_entry_projector=True)

    assert torch.allclose(path_outputs["path_b"].projected_small_hidden, path_outputs["path_b"].delegated_small_hidden)
    assert torch.allclose(path_outputs["path_a"].projected_small_hidden, path_outputs["path_a"].delegated_small_hidden)
    assert path_outputs["path_b"].applied_attention_norm_mean == 0.0
    assert path_outputs["path_b"].applied_mlp_norm_mean == 0.0


def test_attention_suppression_changes_path_statistics_but_preserves_shapes() -> None:
    config, backbones, model = _build_debug_model()
    batch = backbones.tokenizer(
        ["tokenwise attention suppression"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.training.seq_len,
    )
    token_batch = {
        "input_ids": batch["input_ids"].to(backbones.device),
        "attention_mask": batch["attention_mask"].to(backbones.device),
    }
    full_spec = ablation_spec_by_name(idea2_ablation_specs(include_path_specific=True), "tokenwise_full")
    attn_spec = ablation_spec_by_name(idea2_ablation_specs(include_path_specific=True), "tokenwise_attn_suppressed")

    model.set_active_ablation(full_spec)
    full_outputs = model(**token_batch)
    model.set_active_ablation(attn_spec)
    ablated_outputs = model(**token_batch)

    assert full_outputs.logits.shape == ablated_outputs.logits.shape
    assert torch.isfinite(ablated_outputs.logits).all()
    assert full_outputs.path_outputs is not None
    assert ablated_outputs.path_outputs is not None
    assert full_outputs.path_outputs["path_b"].applied_attention_norm_mean > 0.0
    assert ablated_outputs.path_outputs["path_b"].applied_attention_norm_mean == 0.0
    assert not torch.allclose(
        full_outputs.path_outputs["path_b"].delegated_small_hidden,
        ablated_outputs.path_outputs["path_b"].delegated_small_hidden,
    )
