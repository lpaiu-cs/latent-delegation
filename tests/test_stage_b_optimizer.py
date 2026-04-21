from __future__ import annotations

import copy
from pathlib import Path

from torch import nn

from src.train.stage_b_train_utils import stage_b_trainable_prefixes
from src.train.trainer_utils import build_stage_b_optimizer, zero_requires_grad
from src.utils.io import load_config


class _DummyStageBModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.entry_projector = nn.Linear(4, 4)
        self.return_adapter = nn.Linear(4, 4)
        self.gate = nn.Linear(4, 1)
        self.bridge = nn.Linear(4, 4)
        self.frozen = nn.Linear(4, 4)


def test_stage_b_optimizer_uses_module_specific_learning_rates() -> None:
    config = copy.deepcopy(load_config(Path("configs/debug_tiny.yaml")))
    config.training.stage_b.train_entry_projector = True
    config.training.stage_b.entry_lr = 1.0e-4
    config.training.stage_b.return_lr = 2.0e-4
    config.training.stage_b.gate_lr = 3.0e-4

    module = _DummyStageBModule()
    zero_requires_grad(module, except_prefixes=stage_b_trainable_prefixes("hybrid", config))
    optimizer = build_stage_b_optimizer(module, config)

    learning_rates = sorted(group["lr"] for group in optimizer.param_groups)
    assert learning_rates == [1.0e-4, 2.0e-4, 3.0e-4]
