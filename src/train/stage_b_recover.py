"""Stage B: hidden recovery."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from src.eval.metrics import masked_hidden_cosine_loss, masked_hidden_mse
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
            with torch.no_grad():
                prefix_state = model.large_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
                prefix_state = model.large_runner.run_layers(prefix_state, 0, config.split.large_prefix_end)
                large_hidden = prefix_state.hidden_states.detach()

                teacher_state = prefix_state.with_hidden(large_hidden)
                teacher_state = model.large_runner.run_layers(
                    teacher_state,
                    config.split.large_removed_start,
                    config.split.large_removed_end,
                )
                teacher_hidden = teacher_state.hidden_states.detach()

            if args.variant == "hybrid":
                projected_hidden = model.entry_projector(large_hidden)
                small_state = model.small_runner.prepare_from_hidden(
                    projected_hidden,
                    attention_mask=batch["attention_mask"],
                    apply_input_scaling=False,
                )
                small_state = model.small_runner.run_layers(
                    small_state,
                    config.split.small_delegate_start,
                    config.split.small_delegate_end,
                )
                delta_large = model.return_adapter(small_state.hidden_states)
                predicted_hidden = large_hidden + model.gate(delta_large)
            else:
                delta_large = model.bridge(large_hidden)
                predicted_hidden = large_hidden + model.gate(delta_large)

            mse_loss = masked_hidden_mse(predicted_hidden, teacher_hidden, batch["attention_mask"])
            cosine_loss = masked_hidden_cosine_loss(predicted_hidden, teacher_hidden, batch["attention_mask"])
            total_loss = (mse_loss + cosine_loss) / config.training.grad_accum_steps
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()

        row = {
            "step": float(step),
            "loss": float((mse_loss + cosine_loss).detach().cpu()),
            "mse_loss": float(mse_loss.detach().cpu()),
            "cosine_loss": float(cosine_loss.detach().cpu()),
            "gate_value": float(model.gate.value().detach().cpu()),
        }
        history.append(row)
        if step % config.training.log_every == 0 or step == config.training.stage_b.max_steps:
            LOGGER.info(
                "stage_b step=%s loss=%.6f mse=%.6f cosine=%.6f gate=%.6f",
                step,
                row["loss"],
                row["mse_loss"],
                row["cosine_loss"],
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
        "final_gate_value": history[-1]["gate_value"],
    }
    save_history(Path(run_dir), history, final_metrics)


if __name__ == "__main__":
    main()
