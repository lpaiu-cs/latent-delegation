"""Write the current milestone snapshot and Stage B parameter audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.models.baselines import BridgeOnlyLargeModel, SkipOnlyLargeModel
from src.models.hooks import count_parameters
from src.models.hybrid_gemma import HybridDelegationModel
from src.models.backbone_loader import load_backbones
from src.train.trainer_utils import trainable_parameter_names, zero_requires_grad
from src.utils.io import ensure_dir, load_config, save_json
from src.utils.reporting import write_real_hardware_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gemma2_conservative_pilot_256.yaml")
    parser.add_argument("--audit-path", default="artifacts/milestone_parameter_audit.json")
    parser.add_argument("--report-path", default="notes/real_hardware_report.md")
    return parser.parse_args()


def _module_roots(names: list[str]) -> list[str]:
    roots = sorted({name.split(".", maxsplit=1)[0] for name in names})
    return roots


def _summarize_model(module: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    parameter_names = trainable_parameter_names(module)
    counts = count_parameters(module)
    summary = {
        "total_params": counts.total_params,
        "trainable_params": counts.trainable_params,
        "frozen_params": counts.frozen_params,
        "trainable_parameter_names": parameter_names,
        "trainable_modules": _module_roots(parameter_names),
    }
    if extra:
        summary.update(extra)
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=False)

    skip_only = SkipOnlyLargeModel(config, backbones.large_model)

    bridge_only = BridgeOnlyLargeModel(config, backbones.large_model)
    zero_requires_grad(bridge_only, except_prefixes=["bridge", "gate"])

    hybrid = HybridDelegationModel(config, backbones.large_model, backbones.small_model)
    zero_requires_grad(hybrid, except_prefixes=["return_adapter", "gate"])

    audit_payload = {
        "config_path": args.config,
        "models": {
            "skip_only": _summarize_model(skip_only),
            "bridge_only": _summarize_model(
                bridge_only,
                extra={
                    "bridge_rank": config.adapters.bridge_rank,
                    "gate_init": config.adapters.gate_init,
                },
            ),
            "hybrid": _summarize_model(
                hybrid,
                extra={
                    "return_adapter_rank": config.adapters.return_adapter_rank,
                    "gate_init": config.adapters.gate_init,
                },
            ),
        },
    }

    ensure_dir(Path(args.audit_path).parent)
    save_json(args.audit_path, audit_payload)
    write_real_hardware_report(args.report_path, parameter_audit_path=args.audit_path)


if __name__ == "__main__":
    main()
