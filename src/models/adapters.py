"""Adapter modules for latent delegation."""

from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """A minimal RMSNorm used for the entry projector."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized * self.weight.to(normalized.dtype)


class EntryProjector(nn.Module):
    """Affine map from large hidden space into small hidden space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_rmsnorm: bool,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.norm = RMSNorm(output_dim, eps=rms_norm_eps) if use_rmsnorm else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        projected = self.linear(hidden_states.to(self.linear.weight.dtype))
        projected = self.norm(projected)
        return projected.to(input_dtype)


class LowRankAdapter(nn.Module):
    """Two-layer low-rank adapter."""

    def __init__(self, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.down.weight.dtype)
        hidden_states = self.down(hidden_states)
        hidden_states = self.up(hidden_states)
        return hidden_states.to(input_dtype)


class ScalarGate(nn.Module):
    """A bounded scalar gate over the returned delta."""

    def __init__(self, init_value: float = 0.0) -> None:
        super().__init__()
        self.raw_gate = nn.Parameter(torch.tensor(float(init_value)))

    def value(self) -> torch.Tensor:
        return torch.tanh(self.raw_gate)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.value().to(delta.dtype) * delta
