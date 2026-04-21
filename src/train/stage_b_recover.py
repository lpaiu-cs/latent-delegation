"""Stage B: hidden recovery."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from src.models.backbone_loader import load_backbones
from src.models.baselines import BridgeOnlyLargeModel
from src.models.hybrid_gemma import HybridDelegationModel
from src.train.stage_b_objective import compute_stage_b_loss_breakdown, prepare_stage_b_teacher_targets
from src.train.trainer_utils import (
    build_dataloader,
    build_optimizer,
    initialize_run_dir,
    load_checkpoint,
    move_batch_to_device,
    save_checkpoint,
    save_history,
    zero_requires_grad,
)
from src.utils.io import ensure_dir, load_config, save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="hybrid", choices=["hybrid", "bridge_only"])
    parser.add_argument("--stage-a-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.training.seed)

    backbones = load_backbones(
        config,
        load_large=True,
        load_small=(args.variant == "hybrid"),
        load_tokenizer=True,
    )
    if args.variant == "hybrid":
        if args.stage_a_checkpoint is None:
            raise ValueError("--stage-a-checkpoint is required for hybrid Stage B runs.")
        model = HybridDelegationModel(config, backbones.large_model, backbones.small_model)
        stage_a_payload = load_checkpoint(args.stage_a_checkpoint, backbones.device)
        model.entry_projector.load_state_dict(stage_a_payload["entry_projector"])
        zero_requires_grad(model, except_prefixes=["return_adapter", "gate"])
    else:
        model = BridgeOnlyLargeModel(config, backbones.large_model)
        zero_requires_grad(model, except_prefixes=["bridge", "gate"])
    optimizer = build_optimizer(model, config)

    dataloader, corpus = build_dataloader(config, backbones.tokenizer, stage_name="stage_b", split_name="train")
    run_dir = ensure_dir(args.output_dir) if args.output_dir else initialize_run_dir(config, "stage_b")
    save_json(Path(run_dir) / "sample_ids.json", corpus.sample_metadata)

    history: list[dict[str, float]] = []
    batch_iterator = itertools.cycle(dataloader)
    model.train()

    for step in range(1, config.training.stage_b.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.training.grad_accum_steps):
            batch = move_batch_to_device(next(batch_iterator), backbones.device)
            teacher_targets = prepare_stage_b_teacher_targets(model.large_runner, batch, config)
            large_hidden = teacher_targets.hidden_after_prefix

            if args.variant == "hybrid":
                with torch.no_grad():
                    projected_hidden = model.entry_projector(large_hidden)
                    delegated_small_hidden = model.run_delegated_small_block(projected_hidden, batch["attention_mask"])
                delta_large = model.return_adapter(delegated_small_hidden.detach())
                predicted_hidden = large_hidden + model.gate(delta_large)
            else:
                delta_large = model.bridge(large_hidden)
                predicted_hidden = large_hidden + model.gate(delta_large)

            loss_terms = compute_stage_b_loss_breakdown(
                model.large_runner,
                config,
                teacher_targets,
                predicted_hidden,
                batch["attention_mask"],
                batch["labels"],
                delta_large,
            )
            (loss_terms.total_loss / config.training.grad_accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()

        row = {
            "step": float(step),
            "loss": float(loss_terms.total_loss.detach().cpu()),
            "mse_loss": float(loss_terms.mse_loss.detach().cpu()),
            "cosine_loss": float(loss_terms.cosine_loss.detach().cpu()),
            "kl_loss": float(loss_terms.kl_loss.detach().cpu()),
            "ce_loss": float(loss_terms.ce_loss.detach().cpu()),
            "delta_reg": float(loss_terms.delta_reg.detach().cpu()),
            "gate_value": float(model.gate.value().detach().cpu()),
        }
        history.append(row)
        if step % config.training.log_every == 0 or step == config.training.stage_b.max_steps:
            LOGGER.info(
                "stage_b step=%s loss=%.6f mse=%.6f cosine=%.6f kl=%.6f ce=%.6f delta=%.6f gate=%.6f",
                step,
                row["loss"],
                row["mse_loss"],
                row["cosine_loss"],
                row["kl_loss"],
                row["ce_loss"],
                row["delta_reg"],
                row["gate_value"],
            )

        if step % config.training.stage_b.save_every == 0 or step == config.training.stage_b.max_steps:
            save_checkpoint(
                Path(run_dir) / "stage_b_checkpoint.pt",
                {
                    "variant": args.variant,
                    **({"entry_projector": model.entry_projector.state_dict()} if args.variant == "hybrid" else {}),
                    **({"return_adapter": model.return_adapter.state_dict()} if args.variant == "hybrid" else {}),
                    **({"bridge": model.bridge.state_dict()} if args.variant == "bridge_only" else {}),
                    "gate": model.gate.state_dict(),
                    "step": step,
                    "config_path": args.config,
                },
            )

    final_metrics = {
        "final_loss": history[-1]["loss"],
        "final_mse_loss": history[-1]["mse_loss"],
        "final_cosine_loss": history[-1]["cosine_loss"],
        "final_kl_loss": history[-1]["kl_loss"],
        "final_ce_loss": history[-1]["ce_loss"],
        "final_delta_reg": history[-1]["delta_reg"],
        "final_gate_value": history[-1]["gate_value"],
        "stage_b_kl_weight": config.training.stage_b.kl_weight or 0.0,
        "stage_b_ce_weight": config.training.stage_b.ce_weight or 0.0,
        "stage_b_delta_reg_weight": config.training.stage_b.delta_reg_weight or 0.0,
    }
    save_history(Path(run_dir), history, final_metrics)


if __name__ == "__main__":
    main()
