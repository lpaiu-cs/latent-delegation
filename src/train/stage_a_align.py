"""Stage A: representation alignment."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from src.eval.metrics import masked_hidden_cosine_loss, masked_hidden_mse
from src.models.backbone_loader import load_backbones
from src.models.hybrid_gemma import HybridDelegationModel
from src.train.trainer_utils import (
    build_dataloader,
    build_optimizer,
    initialize_run_dir,
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
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.training.seed)

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    model = HybridDelegationModel(config, backbones.large_model, backbones.small_model)
    zero_requires_grad(model, except_prefixes=["entry_projector"])
    optimizer = build_optimizer(model, config)

    dataloader, corpus = build_dataloader(config, backbones.tokenizer, stage_name="stage_a", split_name="train")
    run_dir = ensure_dir(args.output_dir) if args.output_dir else initialize_run_dir(config, "stage_a")
    save_json(Path(run_dir) / "sample_ids.json", corpus.sample_metadata)

    history: list[dict[str, float]] = []
    batch_iterator = itertools.cycle(dataloader)
    model.train()

    for step in range(1, config.training.stage_a.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.training.grad_accum_steps):
            batch = move_batch_to_device(next(batch_iterator), backbones.device)
            with torch.no_grad():
                large_state = model.large_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
                large_state = model.large_runner.run_layers(large_state, 0, config.split.large_prefix_end)
                large_hidden = large_state.hidden_states.detach()

                small_state = model.small_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
                small_state = model.small_runner.run_layers(small_state, 0, config.split.small_entry_target_layer)
                small_hidden = small_state.hidden_states.detach()

            projected_hidden = model.entry_projector(large_hidden)
            mse_loss = masked_hidden_mse(projected_hidden, small_hidden, batch["attention_mask"])
            cosine_loss = masked_hidden_cosine_loss(projected_hidden, small_hidden, batch["attention_mask"])
            total_loss = (mse_loss + cosine_loss) / config.training.grad_accum_steps
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()

        row = {
            "step": float(step),
            "loss": float((mse_loss + cosine_loss).detach().cpu()),
            "mse_loss": float(mse_loss.detach().cpu()),
            "cosine_loss": float(cosine_loss.detach().cpu()),
        }
        history.append(row)
        if step % config.training.log_every == 0 or step == config.training.stage_a.max_steps:
            LOGGER.info("stage_a step=%s loss=%.6f mse=%.6f cosine=%.6f", step, row["loss"], row["mse_loss"], row["cosine_loss"])

        if step % config.training.stage_a.save_every == 0 or step == config.training.stage_a.max_steps:
            save_checkpoint(
                Path(run_dir) / "stage_a_checkpoint.pt",
                {
                    "entry_projector": model.entry_projector.state_dict(),
                    "step": step,
                    "config_path": args.config,
                },
            )

    final_metrics = {
        "final_loss": history[-1]["loss"],
        "final_mse_loss": history[-1]["mse_loss"],
        "final_cosine_loss": history[-1]["cosine_loss"],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    save_history(Path(run_dir), history, final_metrics)


if __name__ == "__main__":
    main()
