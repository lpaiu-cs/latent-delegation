from __future__ import annotations

from pathlib import Path

import torch

from src.models.backbone_loader import load_backbones
from src.train.trainer_utils import zero_requires_grad
from src.utils.io import load_config
from src.v0_6.idea4_common import load_mixture_path_specs
from src.v0_6.idea4_tokenwise_models import (
    TwoPathTokenwiseMixtureHybrid,
    TwoPathTokenwiseMixtureNoSmallModel,
    tokenwise_gate_settings,
    tokenwise_mixture_trainable_prefixes,
)


def _build_debug_model(model_cls: type[torch.nn.Module]) -> tuple[object, object, object]:
    config = load_config(Path("configs/v0_6/debug_tiny_idea4.yaml"))
    backbones = load_backbones(config)
    path_specs = load_mixture_path_specs(config)
    model = model_cls(config, backbones.large_model, backbones.small_model, path_specs).to(backbones.device)
    return config, backbones, model


def test_tokenwise_gate_settings_load_from_debug_config() -> None:
    config = load_config(Path("configs/v0_6/debug_tiny_idea4.yaml"))
    settings = tokenwise_gate_settings(config)
    assert settings["hidden_dim"] == 0
    assert settings["use_rmsnorm"] is True
    assert settings["prior_kl_weight"] == 1.0e-3


def test_tokenwise_mixture_forward_shapes_debug_tiny() -> None:
    config, backbones, model = _build_debug_model(TwoPathTokenwiseMixtureHybrid)
    model.set_static_prior_logits(torch.tensor([0.25, -0.25], device=backbones.device))
    batch = backbones.tokenizer(
        ["idea four tokenwise gate test"],
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
    assert torch.isfinite(outputs.logits).all()
    prior = model.static_prior_weights().detach().cpu()
    assert torch.allclose(prior, torch.softmax(torch.tensor([0.25, -0.25]), dim=0), atol=1.0e-6)


def test_tokenwise_no_small_uses_identity_path_per_branch() -> None:
    config, backbones, model = _build_debug_model(TwoPathTokenwiseMixtureNoSmallModel)
    batch = backbones.tokenizer(
        ["tokenwise no small control"],
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


def test_tokenwise_freezing_keeps_gate_and_return_adapters_trainable() -> None:
    config, backbones, model = _build_debug_model(TwoPathTokenwiseMixtureHybrid)
    zero_requires_grad(model, except_prefixes=tokenwise_mixture_trainable_prefixes(config))
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable_names
    assert all(
        name.startswith(("return_adapter_a", "return_adapter_b", "gate_network"))
        for name in trainable_names
    )
