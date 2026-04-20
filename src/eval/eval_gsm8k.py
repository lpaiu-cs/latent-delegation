"""Lightweight GSM8K subset evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.build_corpus import build_eval_examples
from src.eval.metrics import greedy_generate, parse_final_number
from src.models.backbone_loader import load_backbones
from src.train.trainer_utils import build_model_variant, load_checkpoint, required_backbones_for_variant
from src.utils.io import ensure_dir, load_config, save_json
from src.utils.logging_utils import configure_logging
from src.utils.seed import seed_everything


PROMPT_TEMPLATE = "Solve the math problem.\nQuestion: {question}\nAnswer:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="hybrid", choices=["full_large", "skip_only", "bridge_only", "hybrid"])
    parser.add_argument("--stage-a-checkpoint", default=None)
    parser.add_argument("--stage-b-checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/eval_gsm8k")
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

    examples = build_eval_examples("gsm8k", config)
    predictions: list[dict[str, object]] = []

    for example in examples:
        prompt = PROMPT_TEMPLATE.format(question=example["question"])
        encoded = backbones.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.training.seq_len,
        )
        input_ids = encoded["input_ids"].to(backbones.device)
        attention_mask = encoded["attention_mask"].to(backbones.device)
        generated_ids = greedy_generate(model, input_ids, attention_mask, config.eval.max_new_tokens)
        decoded = backbones.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        parsed_prediction = parse_final_number(decoded)
        gold = parse_final_number(str(example["answer"]))
        predictions.append(
            {
                "id": example["id"],
                "question": example["question"],
                "gold_answer": gold,
                "prediction_text": decoded,
                "parsed_prediction": parsed_prediction,
                "correct": parsed_prediction == gold,
            }
        )

    accuracy = sum(1 for row in predictions if row["correct"]) / max(1, len(predictions))
    output_dir = ensure_dir(args.output_dir)
    save_json(Path(output_dir) / f"{args.variant}_predictions.json", predictions)
    save_json(
        Path(output_dir) / f"{args.variant}_metrics.json",
        {"variant": args.variant, "accuracy": accuracy, "num_examples": len(predictions)},
    )


if __name__ == "__main__":
    main()
