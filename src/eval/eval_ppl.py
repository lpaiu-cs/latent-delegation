"""Lightweight perplexity evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.eval.metrics import perplexity_from_loss, shifted_cross_entropy
from src.models.backbone_loader import load_backbones
from src.train.trainer_utils import (
    build_dataloader,
    build_model_variant,
    load_checkpoint,
    move_batch_to_device,
    required_backbones_for_variant,
)
from src.utils.io import ensure_dir, load_config, save_json
from src.utils.logging_utils import configure_logging
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="hybrid", choices=["full_large", "skip_only", "bridge_only", "hybrid"])
    parser.add_argument("--stage-a-checkpoint", default=None)
    parser.add_argument("--stage-b-checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/eval_ppl")
    return parser.parse_args()


def maybe_load_variant_weights(variant: str, variants: dict[str, torch.nn.Module], args: argparse.Namespace, device: torch.device) -> None:
    if variant == "hybrid":
        if args.stage_a_checkpoint:
            payload = load_checkpoint(args.stage_a_checkpoint, device)
            variants["hybrid"].entry_projector.load_state_dict(payload["entry_projector"])
        if args.stage_b_checkpoint:
            payload = load_checkpoint(args.stage_b_checkpoint, device)
            if "return_adapter" in payload:
                variants["hybrid"].return_adapter.load_state_dict(payload["return_adapter"])
            variants["hybrid"].gate.load_state_dict(payload["gate"])
    if variant == "bridge_only" and args.stage_b_checkpoint:
        payload = load_checkpoint(args.stage_b_checkpoint, device)
        if "bridge" in payload:
            variants["bridge_only"].bridge.load_state_dict(payload["bridge"])
            variants["bridge_only"].gate.load_state_dict(payload["gate"])


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.training.seed)

    load_large, load_small = required_backbones_for_variant(args.variant)
    backbones = load_backbones(config, load_large=load_large, load_small=load_small, load_tokenizer=True)
    model = build_model_variant(args.variant, config, backbones)
    variants = {args.variant: model}
    maybe_load_variant_weights(args.variant, variants, args, backbones.device)
    model = model.eval()

    dataloader, corpus = build_dataloader(config, backbones.tokenizer, stage_name="eval_ppl", split_name="validation")

    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, backbones.device)
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = shifted_cross_entropy(outputs.logits, batch["labels"])
            losses.append(float(loss.detach().cpu()))

    mean_loss = sum(losses) / max(1, len(losses))
    payload = {
        "variant": args.variant,
        "mean_loss": mean_loss,
        "perplexity": perplexity_from_loss(mean_loss),
        "num_batches": len(losses),
        "sample_count": len(corpus.sample_metadata),
    }

    output_dir = ensure_dir(args.output_dir)
    save_json(Path(output_dir) / f"{args.variant}_metrics.json", payload)


if __name__ == "__main__":
    main()
