"""Stage B pilot runner with held-out skip/bridge/hybrid comparison."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import torch

from src.eval.metrics import masked_hidden_cosine_loss, masked_hidden_mse
from src.models.backbone_loader import LoadedBackbones, load_backbones
from src.models.baselines import BridgeOnlyLargeModel, SkipOnlyLargeModel
from src.models.hybrid_gemma import HybridDelegationModel
from src.train.trainer_utils import (
    build_dataloader,
    build_optimizer,
    load_checkpoint,
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
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/stage_b_pilot_ckpt")
    parser.add_argument("--metrics-path", default="artifacts/stage_b_pilot_metrics.json")
    parser.add_argument("--history-path", default="artifacts/stage_b_pilot_history.csv")
    return parser.parse_args()


def _init_stage_b_model(
    variant: str,
    config: Any,
    backbones: LoadedBackbones,
    stage_a_payload: dict[str, Any],
) -> torch.nn.Module:
    if variant == "hybrid":
        model = HybridDelegationModel(config, backbones.large_model, backbones.small_model)
        model.entry_projector.load_state_dict(stage_a_payload["entry_projector"])
        zero_requires_grad(model, except_prefixes=["return_adapter", "gate"])
        return model
    if variant == "bridge_only":
        model = BridgeOnlyLargeModel(config, backbones.large_model)
        zero_requires_grad(model, except_prefixes=["bridge", "gate"])
        return model
    raise ValueError(f"Unsupported variant: {variant}")


def _train_variant(
    variant: str,
    config: Any,
    backbones: LoadedBackbones,
    train_dataloader: Any,
    stage_a_payload: dict[str, Any],
) -> tuple[torch.nn.Module, list[dict[str, float]], dict[str, Any]]:
    model = _init_stage_b_model(variant, config, backbones, stage_a_payload)
    optimizer = build_optimizer(model, config)
    batch_iterator = itertools.cycle(train_dataloader)
    history: list[dict[str, float]] = []
    model.train()

    for step in range(1, config.training.stage_b.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.training.grad_accum_steps):
            batch = move_batch_to_device(next(batch_iterator), backbones.device)
            with torch.no_grad():
                prefix_state = model.large_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
                prefix_state = model.large_runner.run_layers(prefix_state, 0, config.split.large_prefix_end)
                teacher_state = prefix_state.with_hidden(prefix_state.hidden_states.detach())
                teacher_state = model.large_runner.run_layers(
                    teacher_state,
                    config.split.large_removed_start,
                    config.split.large_removed_end,
                )
                teacher_hidden = teacher_state.hidden_states.detach()

            if variant == "hybrid":
                projected_hidden = model.entry_projector(prefix_state.hidden_states.detach())
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
                predicted_hidden = prefix_state.hidden_states.detach() + model.gate(delta_large)
            else:
                delta_large = model.bridge(prefix_state.hidden_states.detach())
                predicted_hidden = prefix_state.hidden_states.detach() + model.gate(delta_large)

            mse_loss = masked_hidden_mse(predicted_hidden, teacher_hidden, batch["attention_mask"])
            cosine_loss = masked_hidden_cosine_loss(predicted_hidden, teacher_hidden, batch["attention_mask"])
            total_loss = (mse_loss + cosine_loss) / config.training.grad_accum_steps
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()

        row = {
            "variant": variant,
            "step": float(step),
            "loss": float((mse_loss + cosine_loss).detach().cpu()),
            "mse_loss": float(mse_loss.detach().cpu()),
            "cosine_loss": float(cosine_loss.detach().cpu()),
            "gate_value": float(model.gate.value().detach().cpu()),
        }
        history.append(row)
        if step % config.training.log_every == 0 or step == config.training.stage_b.max_steps:
            LOGGER.info(
                "stage_b_pilot variant=%s step=%s loss=%.6f mse=%.6f cosine=%.6f gate=%.6f",
                variant,
                step,
                row["loss"],
                row["mse_loss"],
                row["cosine_loss"],
                row["gate_value"],
            )

    checkpoint_payload = {
        "variant": variant,
        "step": config.training.stage_b.max_steps,
        "config_path": stage_a_payload["config_path"],
        "gate": model.gate.state_dict(),
    }
    if variant == "hybrid":
        checkpoint_payload["entry_projector"] = model.entry_projector.state_dict()
        checkpoint_payload["return_adapter"] = model.return_adapter.state_dict()
    else:
        checkpoint_payload["bridge"] = model.bridge.state_dict()
    return model, history, checkpoint_payload


@torch.no_grad()
def _evaluate_variants(
    config: Any,
    backbones: LoadedBackbones,
    hybrid_model: HybridDelegationModel,
    bridge_model: BridgeOnlyLargeModel,
    val_dataloader: Any,
) -> dict[str, float]:
    skip_model = SkipOnlyLargeModel(config, backbones.large_model).eval()
    hybrid_model.eval()
    bridge_model.eval()

    totals = {
        "skip_only_hidden_mse": 0.0,
        "skip_only_cosine_loss": 0.0,
        "bridge_only_hidden_mse": 0.0,
        "bridge_only_cosine_loss": 0.0,
        "hybrid_hidden_mse": 0.0,
        "hybrid_cosine_loss": 0.0,
    }
    batches = 0

    for batch in val_dataloader:
        batch = move_batch_to_device(batch, backbones.device)
        prefix_state = hybrid_model.large_runner.prepare_from_input_ids(batch["input_ids"], batch["attention_mask"])
        prefix_state = hybrid_model.large_runner.run_layers(prefix_state, 0, config.split.large_prefix_end)
        teacher_state = prefix_state.with_hidden(prefix_state.hidden_states.detach())
        teacher_state = hybrid_model.large_runner.run_layers(
            teacher_state,
            config.split.large_removed_start,
            config.split.large_removed_end,
        )
        teacher_hidden = teacher_state.hidden_states.detach()

        skip_outputs = skip_model(batch["input_ids"], attention_mask=batch["attention_mask"])
        bridge_outputs = bridge_model(batch["input_ids"], attention_mask=batch["attention_mask"])
        hybrid_outputs = hybrid_model(batch["input_ids"], attention_mask=batch["attention_mask"])

        totals["skip_only_hidden_mse"] += float(
            masked_hidden_mse(skip_outputs.hidden_after_removed, teacher_hidden, batch["attention_mask"]).cpu()
        )
        totals["skip_only_cosine_loss"] += float(
            masked_hidden_cosine_loss(skip_outputs.hidden_after_removed, teacher_hidden, batch["attention_mask"]).cpu()
        )
        totals["bridge_only_hidden_mse"] += float(
            masked_hidden_mse(bridge_outputs.hidden_after_removed, teacher_hidden, batch["attention_mask"]).cpu()
        )
        totals["bridge_only_cosine_loss"] += float(
            masked_hidden_cosine_loss(bridge_outputs.hidden_after_removed, teacher_hidden, batch["attention_mask"]).cpu()
        )
        totals["hybrid_hidden_mse"] += float(
            masked_hidden_mse(hybrid_outputs.hidden_after_removed, teacher_hidden, batch["attention_mask"]).cpu()
        )
        totals["hybrid_cosine_loss"] += float(
            masked_hidden_cosine_loss(hybrid_outputs.hidden_after_removed, teacher_hidden, batch["attention_mask"]).cpu()
        )
        batches += 1

    metrics = {key: value / max(1, batches) for key, value in totals.items()}
    metrics["skip_only_cosine"] = 1.0 - metrics.pop("skip_only_cosine_loss")
    metrics["bridge_only_cosine"] = 1.0 - metrics.pop("bridge_only_cosine_loss")
    metrics["hybrid_cosine"] = 1.0 - metrics.pop("hybrid_cosine_loss")
    metrics["hybrid_beats_skip_only"] = (
        metrics["hybrid_hidden_mse"] < metrics["skip_only_hidden_mse"]
        and metrics["hybrid_cosine"] > metrics["skip_only_cosine"]
    )
    metrics["hybrid_beats_bridge_only"] = (
        metrics["hybrid_hidden_mse"] < metrics["bridge_only_hidden_mse"]
        and metrics["hybrid_cosine"] > metrics["bridge_only_cosine"]
    )
    metrics["positive_pilot"] = metrics["hybrid_beats_skip_only"]
    return metrics


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.training.seed)

    output_dir = ensure_dir(args.output_dir)
    save_config_snapshot(output_dir / "config_snapshot.yaml", config)
    export_run_metadata(output_dir / "metadata.json", config, {"stage": "stage_b_pilot"})

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    stage_a_payload = load_checkpoint(args.stage_a_checkpoint, backbones.device)

    train_dataloader, train_corpus = build_dataloader(config, backbones.tokenizer, stage_name="stage_b", split_name="train")
    val_dataloader, val_corpus = build_dataloader(config, backbones.tokenizer, stage_name="stage_b", split_name="validation")
    save_json(output_dir / "train_sample_ids.json", train_corpus.sample_metadata)
    save_json(output_dir / "validation_sample_ids.json", val_corpus.sample_metadata)

    hybrid_model, hybrid_history, hybrid_checkpoint = _train_variant(
        "hybrid",
        config,
        backbones,
        train_dataloader,
        stage_a_payload,
    )
    save_checkpoint(output_dir / "hybrid_stage_b_checkpoint.pt", hybrid_checkpoint)

    bridge_model, bridge_history, bridge_checkpoint = _train_variant(
        "bridge_only",
        config,
        backbones,
        train_dataloader,
        stage_a_payload,
    )
    save_checkpoint(output_dir / "bridge_only_stage_b_checkpoint.pt", bridge_checkpoint)

    history = hybrid_history + bridge_history
    eval_metrics = _evaluate_variants(config, backbones, hybrid_model, bridge_model, val_dataloader)
    metrics = {
        "pilot_stage": "stage_b",
        "config_path": args.config,
        "seq_len": config.training.seq_len,
        "max_train_steps": config.training.stage_b.max_steps,
        "hybrid_train_loss_start": hybrid_history[0]["loss"],
        "hybrid_train_loss_end": hybrid_history[-1]["loss"],
        "bridge_only_train_loss_start": bridge_history[0]["loss"],
        "bridge_only_train_loss_end": bridge_history[-1]["loss"],
        **eval_metrics,
    }

    save_csv(output_dir / "history.csv", history)
    save_json(output_dir / "metrics.json", metrics)
    ensure_dir(Path(args.history_path).parent)
    ensure_dir(Path(args.metrics_path).parent)
    save_csv(args.history_path, history)
    save_json(args.metrics_path, metrics)


if __name__ == "__main__":
    main()
