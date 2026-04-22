from __future__ import annotations

from pathlib import Path

import torch

from src.models.backbone_loader import load_backbones
from src.train.trainer_utils import zero_requires_grad
from src.v0_6.idea4_common import MixturePathSpec, gated_return_adapter_state_dict, load_mixture_path_specs
from src.v0_6.idea4_models import (
    TwoPathStaticMixtureHybrid,
    TwoPathStaticMixtureNoSmallModel,
    static_mixture_trainable_prefixes,
)
from src.utils.io import load_config


def _build_debug_model(model_cls: type[torch.nn.Module]) -> tuple[object, object, object]:
    config = load_config(Path("configs/v0_6/debug_tiny_idea4.yaml"))
    backbones = load_backbones(config)
    path_specs = load_mixture_path_specs(config)
    model = model_cls(config, backbones.large_model, backbones.small_model, path_specs).to(backbones.device)
    return config, backbones, model


def test_idea4_debug_config_loads() -> None:
    config = load_config(Path("configs/v0_6/debug_tiny_idea4.yaml"))
    path_specs = load_mixture_path_specs(config)
    assert config.split.large_removed_start == 24
    assert config.split.large_removed_end == 27
    assert [spec.name for spec in path_specs] == ["path_b", "path_a"]
    assert path_specs[0].label == "24..27 -> 14..19"
    assert path_specs[1].label == "24..27 -> 16..18"


def test_static_mixture_forward_shapes_debug_tiny() -> None:
    config, backbones, model = _build_debug_model(TwoPathStaticMixtureHybrid)
    batch = backbones.tokenizer(
        ["idea four static mixture path test"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.training.seq_len,
    )
    outputs = model(
        batch["input_ids"].to(backbones.device),
        attention_mask=batch["attention_mask"].to(backbones.device),
    )
    assert outputs.logits.shape[:2] == batch["input_ids"].shape
    assert outputs.logits.shape[-1] == config.model.debug_vocab_size
    assert outputs.delta_large is not None
    assert outputs.delta_large.shape[-1] == config.model.debug_large_hidden_size
    assert torch.isfinite(outputs.logits).all()
    weights = model.mixture_weights().detach().cpu()
    assert weights.shape == (2,)
    assert torch.allclose(weights, torch.tensor([0.5, 0.5]), atol=1.0e-6)


def test_static_mixture_no_small_uses_identity_path_per_branch() -> None:
    config, backbones, model = _build_debug_model(TwoPathStaticMixtureNoSmallModel)
    batch = backbones.tokenizer(
        ["mixture no small control"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.training.seq_len,
    )
    token_batch = {
        "input_ids": batch["input_ids"].to(backbones.device),
        "attention_mask": batch["attention_mask"].to(backbones.device),
    }
    prefix_state = model.large_runner.prepare_from_input_ids(token_batch["input_ids"], attention_mask=token_batch["attention_mask"])
    prefix_state = model.large_runner.run_layers(prefix_state, 0, config.split.large_prefix_end)
    path_outputs = model.compute_path_outputs(
        prefix_state.hidden_states.detach(),
        prefix_state.attention_mask_2d,
        train_entry_projector=False,
    )
    for path_output in path_outputs.values():
        assert torch.allclose(path_output.projected_small_hidden, path_output.delegated_small_hidden)


def test_static_mixture_freezing_keeps_only_mixture_adapters_trainable() -> None:
    config, backbones, model = _build_debug_model(TwoPathStaticMixtureHybrid)
    zero_requires_grad(model, except_prefixes=static_mixture_trainable_prefixes(config))
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable_names
    assert all(
        name.startswith(("return_adapter_a", "return_adapter_b", "alpha"))
        for name in trainable_names
    )


def test_phase1_gate_absorption_scales_return_adapter() -> None:
    payload = {
        "return_adapter": {
            "down.weight": torch.ones(2, 3),
            "up.weight": torch.ones(4, 2),
        },
        "gate": {
            "raw_gate": torch.tensor(0.5),
        },
    }
    state, gate_value = gated_return_adapter_state_dict(payload)
    assert gate_value == float(torch.tanh(torch.tensor(0.5)))
    assert torch.allclose(state["down.weight"], torch.ones(2, 3))
    assert torch.allclose(state["up.weight"], torch.ones(4, 2) * gate_value)
