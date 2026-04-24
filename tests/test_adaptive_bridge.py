from __future__ import annotations

from pathlib import Path

import torch

from src.adaptive_bridge.common import adaptive_bridge_gate_settings, adaptive_bridge_settings, adaptive_bridge_trainable_prefixes, adaptive_eval_spec
from src.adaptive_bridge.models import BridgeAwareResidualMoE, BridgeAwareResidualMoENoSmall
from src.models.backbone_loader import load_backbones
from src.train.trainer_utils import zero_requires_grad
from src.utils.io import load_config
from src.v0_6.idea4_common import load_mixture_path_specs


def _build_debug_model(model_cls: type[torch.nn.Module]) -> tuple[object, object, object]:
    config = load_config(Path("configs/adaptive_bridge/debug_tiny.yaml"))
    backbones = load_backbones(config)
    path_specs = load_mixture_path_specs(config)
    model = model_cls(config, backbones.large_model, backbones.small_model, path_specs).to(backbones.device)
    return config, backbones, model


def test_adaptive_bridge_debug_config_loads() -> None:
    config = load_config(Path("configs/adaptive_bridge/debug_tiny.yaml"))
    settings = adaptive_bridge_settings(config)
    gate_settings = adaptive_bridge_gate_settings(config)
    eval_spec = adaptive_eval_spec(config)

    assert config.split.large_removed_start == 24
    assert config.split.large_removed_end == 27
    assert settings.bridge_rank == 16
    assert settings.warm_start_from_v060 is False
    assert gate_settings.bridge_init_logit == -0.5
    assert [task.name for task in eval_spec.internal_tasks] == ["development_holdout", "confirmation_holdout"]
    assert [task.name for task in eval_spec.multichoice_tasks] == ["piqa", "arc_easy"]


def test_adaptive_bridge_real_config_loads() -> None:
    config = load_config(Path("configs/adaptive_bridge/gemma2_first_milestone.yaml"))
    settings = adaptive_bridge_settings(config)
    eval_spec = adaptive_eval_spec(config)

    assert config.model.large_model_name == "google/gemma-2-9b"
    assert config.model.small_model_name == "google/gemma-2-2b"
    assert settings.bridge_rank == 128
    assert settings.warm_start_from_v060 is True
    assert [task.name for task in eval_spec.lm_tasks] == ["lambada_openai"]


def test_adaptive_bridge_forward_shapes_debug_tiny() -> None:
    config, backbones, model = _build_debug_model(BridgeAwareResidualMoE)
    batch = backbones.tokenizer(
        ["adaptive bridge token wise mixture"],
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
    assert outputs.hidden_after_removed.shape[-1] == config.model.debug_large_hidden_size
    assert outputs.delta_large is not None
    assert outputs.delta_large.shape[-1] == config.model.debug_large_hidden_size
    assert torch.isfinite(outputs.logits).all()


def test_adaptive_bridge_gate_weights_sum_to_one() -> None:
    config, backbones, model = _build_debug_model(BridgeAwareResidualMoE)
    batch = backbones.tokenizer(
        ["bridge expert path a path b"],
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
    _, _, gate_weights, _ = model.compute_mixed_delta(
        prefix_state.hidden_states.detach(),
        prefix_state.attention_mask_2d,
        train_entry_projector=False,
    )

    assert torch.allclose(gate_weights.sum(dim=-1), torch.ones_like(gate_weights[..., 0]), atol=1.0e-6)


def test_adaptive_bridge_no_small_uses_identity_small_paths() -> None:
    config, backbones, model = _build_debug_model(BridgeAwareResidualMoENoSmall)
    batch = backbones.tokenizer(
        ["adaptive bridge no small control"],
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
    expert_outputs = model.compute_expert_outputs(
        prefix_state.hidden_states.detach(),
        prefix_state.attention_mask_2d,
        train_entry_projector=False,
    )

    assert torch.allclose(
        expert_outputs["path_b"].projected_small_hidden,
        expert_outputs["path_b"].delegated_small_hidden,
    )
    assert torch.allclose(
        expert_outputs["path_a"].projected_small_hidden,
        expert_outputs["path_a"].delegated_small_hidden,
    )


def test_adaptive_bridge_freezing_keeps_expected_modules_trainable() -> None:
    config, _, model = _build_debug_model(BridgeAwareResidualMoE)
    zero_requires_grad(model, except_prefixes=adaptive_bridge_trainable_prefixes(config))
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    assert trainable_names
    assert all(
        name.startswith(("return_adapter_a", "return_adapter_b", "bridge_expert", "gate_network"))
        for name in trainable_names
    )
