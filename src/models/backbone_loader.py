"""Backbone loading utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

# Keep the Windows-native path on PyTorch only. The installed Transformers build
# otherwise probes TensorFlow/Flax during model imports and slows or blocks
# bring-up on this machine.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Gemma2Config, Gemma2ForCausalLM

from src.models.hooks import assert_split_fits_model
from src.utils.io import ExperimentConfig


class DebugTokenizer:
    """A simple whitespace tokenizer for debug smoke tests."""

    def __init__(self, vocab_size: int, max_length: int = 512) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.model_max_length = max_length

    def _encode_text(self, text: str, max_length: int | None) -> list[int]:
        length = max_length or self.model_max_length
        token_ids = [self.bos_token_id]
        for token in text.strip().split():
            token_ids.append(4 + (hash(token) % max(1, self.vocab_size - 4)))
        token_ids.append(self.eos_token_id)
        return token_ids[:length]

    def __call__(
        self,
        texts: str | list[str],
        return_tensors: str = "pt",
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode_text(text, max_length if truncation else None) for text in texts]
        target_length = max(len(sequence) for sequence in encoded)
        if isinstance(padding, str) and padding == "max_length" and max_length is not None:
            target_length = max_length

        padded_ids: list[list[int]] = []
        padded_masks: list[list[int]] = []
        for sequence in encoded:
            clipped = sequence[:target_length]
            pad_count = target_length - len(clipped)
            padded_ids.append(clipped + [self.pad_token_id] * pad_count)
            padded_masks.append([1] * len(clipped) + [0] * pad_count)

        if return_tensors != "pt":
            raise ValueError("DebugTokenizer only supports return_tensors='pt'.")
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        }

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True, **_: Any) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        tokens: list[str] = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }:
                continue
            tokens.append(f"tok{token_id}")
        return " ".join(tokens)

    def batch_decode(self, sequences: list[list[int]] | torch.Tensor, skip_special_tokens: bool = True, **kwargs: Any) -> list[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return [self.decode(sequence, skip_special_tokens=skip_special_tokens, **kwargs) for sequence in sequences]


@dataclass
class LoadedBackbones:
    """Loaded models, tokenizer, and execution device."""

    large_model: nn.Module | None
    small_model: nn.Module | None
    tokenizer: Any | None
    device: torch.device
    is_debug: bool


def select_device() -> torch.device:
    """Select the execution device."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _quantization_config(config: ExperimentConfig) -> BitsAndBytesConfig | None:
    if not config.model.load_in_4bit:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit loading requires CUDA in this v1 implementation.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_torch_dtype(config.model.bnb_4bit_compute_dtype),
    )


def _debug_gemma2_config(config: ExperimentConfig, is_large: bool) -> Gemma2Config:
    hidden_size = config.model.debug_large_hidden_size if is_large else config.model.debug_small_hidden_size
    intermediate_size = (
        config.model.debug_large_intermediate_size if is_large else config.model.debug_small_intermediate_size
    )
    num_attention_heads = (
        config.model.debug_large_num_attention_heads if is_large else config.model.debug_small_num_attention_heads
    )
    num_key_value_heads = (
        config.model.debug_large_num_key_value_heads if is_large else config.model.debug_small_num_key_value_heads
    )
    num_layers = 42 if is_large else 26
    assert hidden_size is not None
    assert intermediate_size is not None
    assert num_attention_heads is not None
    assert num_key_value_heads is not None
    assert config.model.debug_vocab_size is not None
    assert config.model.debug_max_position_embeddings is not None
    head_dim = hidden_size // num_attention_heads
    return Gemma2Config(
        vocab_size=config.model.debug_vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        query_pre_attn_scalar=head_dim,
        max_position_embeddings=config.model.debug_max_position_embeddings,
        sliding_window=min(config.training.seq_len, config.model.debug_max_position_embeddings),
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )


def _assert_loaded_split_compatibility(config: ExperimentConfig, large_model: nn.Module | None, small_model: nn.Module | None) -> None:
    if large_model is not None and small_model is not None:
        assert_split_fits_model(
            config,
            large_num_layers=large_model.config.num_hidden_layers,
            small_num_layers=small_model.config.num_hidden_layers,
        )
        return
    if large_model is not None:
        large_layers = int(large_model.config.num_hidden_layers)
        assert config.split.large_prefix_end < large_layers
        assert config.split.large_removed_end < large_layers
        assert config.split.large_suffix_start < large_layers
    if small_model is not None:
        small_layers = int(small_model.config.num_hidden_layers)
        assert config.split.small_entry_target_layer < small_layers
        assert config.split.small_delegate_end < small_layers


def _load_debug_backbones(
    config: ExperimentConfig,
    device: torch.device,
    *,
    load_large: bool,
    load_small: bool,
    load_tokenizer: bool,
) -> LoadedBackbones:
    large_model = Gemma2ForCausalLM(_debug_gemma2_config(config, is_large=True)).to(device) if load_large else None
    small_model = Gemma2ForCausalLM(_debug_gemma2_config(config, is_large=False)).to(device) if load_small else None
    tokenizer = (
        DebugTokenizer(vocab_size=config.model.debug_vocab_size or 320, max_length=config.training.seq_len)
        if load_tokenizer
        else None
    )
    if config.model.freeze_backbones:
        if large_model is not None:
            _freeze_module(large_model)
        if small_model is not None:
            _freeze_module(small_model)
    _assert_loaded_split_compatibility(config, large_model, small_model)
    return LoadedBackbones(
        large_model=large_model,
        small_model=small_model,
        tokenizer=tokenizer,
        device=device,
        is_debug=True,
    )


def _assert_family(model: nn.Module, expected_family: str, model_name: str) -> None:
    model_type = getattr(model.config, "model_type", None)
    if model_type != expected_family:
        raise RuntimeError(f"Loaded model {model_name} resolved to {model_type}, expected {expected_family}.")


def _load_real_backbone(model_name: str, config: ExperimentConfig, device: torch.device) -> nn.Module:
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
        "dtype": _torch_dtype(config.model.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    quant_config = _quantization_config(config)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        kwargs["device_map"] = {"": 0} if device.type == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if quant_config is None:
        model = model.to(device)
    if config.model.freeze_backbones:
        _freeze_module(model)
    _assert_family(model, config.model.family, model_name)
    return model


def load_backbones(
    config: ExperimentConfig,
    *,
    load_large: bool = True,
    load_small: bool = True,
    load_tokenizer: bool = True,
) -> LoadedBackbones:
    """Load the large model, small model, and tokenizer."""

    if not any((load_large, load_small, load_tokenizer)):
        raise ValueError("At least one of load_large, load_small, or load_tokenizer must be True.")

    device = select_device()
    if config.model.debug_random_init:
        return _load_debug_backbones(
            config,
            device,
            load_large=load_large,
            load_small=load_small,
            load_tokenizer=load_tokenizer,
        )

    tokenizer = None
    if load_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name,
            trust_remote_code=config.model.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    large_model = _load_real_backbone(config.model.large_model_name, config, device) if load_large else None
    small_model = _load_real_backbone(config.model.small_model_name, config, device) if load_small else None
    _assert_loaded_split_compatibility(config, large_model, small_model)
    return LoadedBackbones(
        large_model=large_model,
        small_model=small_model,
        tokenizer=tokenizer,
        device=device,
        is_debug=False,
    )
