"""Stage C: output distillation."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from src.eval.metrics import shifted_cross_entropy, shifted_kl_divergence
from src.models.backbone_loader import load_backbones
from src.models.baselines import BridgeOnlyLargeModel
from src.models.hybrid_gemma import HybridDelegationModel
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
    parser.add_argument("--stage-b-checkpoint", required=True)
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
    stage_b_payload = load_checkpoint(args.stage_b_checkpoint, backbones.device)
    if args.variant == "hybrid":
        if args.stage_a_checkpoint is None:
            raise ValueError("--stage-a-checkpoint is required for hybrid Stage C runs.")
        model = HybridDelegationModel(config, backbones.large_model, backbones.small_model)
        stage_a_payload = load_checkpoint(args.stage_a_checkpoint, backbones.device)
        model.entry_projector.load_state_dict(stage_a_payload["entry_projector"])
        model.return_adapter.load_state_dict(stage_b_payload["return_adapter"])
        model.gate.load_state_dict(stage_b_payload["gate"])
        zero_requires_grad(model, except_prefixes=["entry_projector", "return_adapter", "gate"])
    else:
        model = BridgeOnlyLargeModel(config, backbones.large_model)
        model.bridge.load_state_dict(stage_b_payload["bridge"])
        model.gate.load_state_dict(stage_b_payload["gate"])
        zero_requires_grad(model, except_prefixes=["bridge", "gate"])
    optimizer = build_optimizer(model, config)

    dataloader, corpus = build_dataloader(config, backbones.tokenizer, stage_name="stage_c", split_name="train")
    run_dir = ensure_dir(args.output_dir) if args.output_dir else initialize_run_dir(config, "stage_c")
    save_json(Path(run_dir) / "sample_ids.json", corpus.sample_metadata)

    history: list[dict[str, float]] = []
    batch_iterator = itertools.cycle(dataloader)
    model.train()

    for step in range(1, config.training.stage_c.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.training.grad_accum_steps):
            batch = move_batch_to_device(next(batch_iterator), backbones.device)
            with torch.no_grad():
                teacher_state = model.large_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
                teacher_state = model.large_runner.run_layers(teacher_state, 0, model.large_runner.num_layers - 1)
                teacher_logits = model.large_runner.logits_from_hidden(teacher_state.hidden_states).detach()

            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            kl_loss = shifted_kl_divergence(outputs.logits, teacher_logits, batch["labels"])
            ce_loss = shifted_cross_entropy(outputs.logits, batch["labels"])
            delta_reg = outputs.delta_large.pow(2).mean()
            total_loss = (
                config.training.stage_c.kl_weight * kl_loss
                + config.training.stage_c.ce_weight * ce_loss
                + config.training.stage_c.delta_reg_weight * delta_reg
            ) / config.training.grad_accum_steps
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()

        row = {
            "step": float(step),
            "loss": float(
                (
                    config.training.stage_c.kl_weight * kl_loss
                    + config.training.stage_c.ce_weight * ce_loss
                    + config.training.stage_c.delta_reg_weight * delta_reg
                )
                .detach()
                .cpu()
            ),
            "kl_loss": float(kl_loss.detach().cpu()),
            "ce_loss": float(ce_loss.detach().cpu()),
            "delta_reg": float(delta_reg.detach().cpu()),
            "gate_value": float(model.gate.value().detach().cpu()),
        }
        history.append(row)
        if step % config.training.log_every == 0 or step == config.training.stage_c.max_steps:
            LOGGER.info(
                "stage_c step=%s loss=%.6f kl=%.6f ce=%.6f delta=%.6f gate=%.6f",
                step,
                row["loss"],
                row["kl_loss"],
                row["ce_loss"],
                row["delta_reg"],
                row["gate_value"],
            )

        if step % config.training.stage_c.save_every == 0 or step == config.training.stage_c.max_steps:
            save_checkpoint(
                Path(run_dir) / "stage_c_checkpoint.pt",
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
        "final_kl_loss": history[-1]["kl_loss"],
        "final_ce_loss": history[-1]["ce_loss"],
        "final_delta_reg": history[-1]["delta_reg"],
        "final_gate_value": history[-1]["gate_value"],
    }
    save_history(Path(run_dir), history, final_metrics)


if __name__ == "__main__":
    main()
