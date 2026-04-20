"""Stage A pilot runner with held-out alignment checks."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import torch

from src.eval.metrics import masked_hidden_cosine_loss, masked_hidden_mse
from src.models.backbone_loader import load_backbones
from src.models.hybrid_gemma import HybridDelegationModel
from src.train.trainer_utils import (
    build_dataloader,
    build_optimizer,
    move_batch_to_device,
    save_checkpoint,
    zero_requires_grad,
)
from src.utils.io import (
    ensure_dir,
    export_run_metadata,
    load_config,
    save_config_snapshot,
    save_csv,
    save_json,
)
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gemma2_conservative_pilot_256.yaml")
    parser.add_argument("--output-dir", default="artifacts/stage_a_pilot_ckpt")
    parser.add_argument("--metrics-path", default="artifacts/stage_a_pilot_metrics.json")
    parser.add_argument("--history-path", default="artifacts/stage_a_pilot_history.csv")
    return parser.parse_args()


@torch.no_grad()
def _evaluate_alignment(
    model: HybridDelegationModel,
    dataloader: Any,
    device: torch.device,
    config: Any,
) -> dict[str, float]:
    model.eval()
    mse_total = 0.0
    cosine_loss_total = 0.0
    batches = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        large_state = model.large_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
        large_state = model.large_runner.run_layers(large_state, 0, config.split.large_prefix_end)
        small_state = model.small_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
        small_state = model.small_runner.run_layers(small_state, 0, config.split.small_entry_target_layer)
        projected_hidden = model.entry_projector(large_state.hidden_states)
        mse_total += float(masked_hidden_mse(projected_hidden, small_state.hidden_states, batch["attention_mask"]).cpu())
        cosine_loss_total += float(
            masked_hidden_cosine_loss(projected_hidden, small_state.hidden_states, batch["attention_mask"]).cpu()
        )
        batches += 1
    model.train()
    avg_mse = mse_total / max(1, batches)
    avg_cosine_loss = cosine_loss_total / max(1, batches)
    return {
        "heldout_mse": avg_mse,
        "heldout_cosine_loss": avg_cosine_loss,
        "heldout_cosine": 1.0 - avg_cosine_loss,
    }


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.training.seed)

    output_dir = ensure_dir(args.output_dir)
    save_config_snapshot(output_dir / "config_snapshot.yaml", config)
    export_run_metadata(output_dir / "metadata.json", config, {"stage": "stage_a_pilot"})

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    model = HybridDelegationModel(config, backbones.large_model, backbones.small_model)
    zero_requires_grad(model, except_prefixes=["entry_projector"])
    optimizer = build_optimizer(model, config)

    train_dataloader, train_corpus = build_dataloader(config, backbones.tokenizer, stage_name="stage_a", split_name="train")
    val_dataloader, val_corpus = build_dataloader(config, backbones.tokenizer, stage_name="stage_a", split_name="validation")
    save_json(output_dir / "train_sample_ids.json", train_corpus.sample_metadata)
    save_json(output_dir / "validation_sample_ids.json", val_corpus.sample_metadata)

    initial_eval = _evaluate_alignment(model, val_dataloader, backbones.device, config)
    history: list[dict[str, float]] = []
    batch_iterator = itertools.cycle(train_dataloader)
    model.train()

    for step in range(1, config.training.stage_a.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.training.grad_accum_steps):
            batch = move_batch_to_device(next(batch_iterator), backbones.device)
            with torch.no_grad():
                large_state = model.large_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
                large_state = model.large_runner.run_layers(large_state, 0, config.split.large_prefix_end)
                small_state = model.small_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
                small_state = model.small_runner.run_layers(small_state, 0, config.split.small_entry_target_layer)

            projected_hidden = model.entry_projector(large_state.hidden_states)
            mse_loss = masked_hidden_mse(projected_hidden, small_state.hidden_states, batch["attention_mask"])
            cosine_loss = masked_hidden_cosine_loss(projected_hidden, small_state.hidden_states, batch["attention_mask"])
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
            LOGGER.info(
                "stage_a_pilot step=%s loss=%.6f mse=%.6f cosine=%.6f",
                step,
                row["loss"],
                row["mse_loss"],
                row["cosine_loss"],
            )

    final_eval = _evaluate_alignment(model, val_dataloader, backbones.device, config)
    checkpoint_payload = {
        "entry_projector": model.entry_projector.state_dict(),
        "step": config.training.stage_a.max_steps,
        "config_path": args.config,
    }
    save_checkpoint(output_dir / "stage_a_checkpoint.pt", checkpoint_payload)

    metrics = {
        "pilot_stage": "stage_a",
        "config_path": args.config,
        "seq_len": config.training.seq_len,
        "max_train_steps": config.training.stage_a.max_steps,
        "train_loss_start": history[0]["loss"],
        "train_loss_end": history[-1]["loss"],
        "train_loss_delta": history[-1]["loss"] - history[0]["loss"],
        "heldout_mse_before": initial_eval["heldout_mse"],
        "heldout_mse_after": final_eval["heldout_mse"],
        "heldout_mse_improvement": initial_eval["heldout_mse"] - final_eval["heldout_mse"],
        "heldout_cosine_before": initial_eval["heldout_cosine"],
        "heldout_cosine_after": final_eval["heldout_cosine"],
        "heldout_cosine_improvement": final_eval["heldout_cosine"] - initial_eval["heldout_cosine"],
        "meaningful_loss_decrease": history[-1]["loss"] < history[0]["loss"],
        "heldout_alignment_improved": (
            final_eval["heldout_mse"] < initial_eval["heldout_mse"]
            and final_eval["heldout_cosine"] > initial_eval["heldout_cosine"]
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }

    save_csv(output_dir / "history.csv", history)
    save_json(output_dir / "metrics.json", metrics)
    ensure_dir(Path(args.history_path).parent)
    ensure_dir(Path(args.metrics_path).parent)
    save_csv(args.history_path, history)
    save_json(args.metrics_path, metrics)


if __name__ == "__main__":
    main()
