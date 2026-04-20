"""Latency and memory evaluation for the four variants."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.data.build_corpus import build_corpus_bundle
from src.eval.metrics import greedy_generate
from src.models.backbone_loader import load_backbones
from src.train.trainer_utils import build_model_variant, load_checkpoint
from src.utils.io import ensure_dir, load_config, save_json
from src.utils.logging_utils import configure_logging
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-a-checkpoint", default=None)
    parser.add_argument("--stage-b-checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/eval_speed")
    return parser.parse_args()


def maybe_load_hybrid_weights(variants: dict[str, torch.nn.Module], args: argparse.Namespace, device: torch.device) -> None:
    if args.stage_a_checkpoint:
        payload = load_checkpoint(args.stage_a_checkpoint, device)
        variants["hybrid"].entry_projector.load_state_dict(payload["entry_projector"])
    if args.stage_b_checkpoint:
        payload = load_checkpoint(args.stage_b_checkpoint, device)
        if "return_adapter" in payload:
            variants["hybrid"].return_adapter.load_state_dict(payload["return_adapter"])
            variants["hybrid"].gate.load_state_dict(payload["gate"])
        if "bridge" in payload:
            variants["bridge_only"].bridge.load_state_dict(payload["bridge"])
            variants["bridge_only"].gate.load_state_dict(payload["gate"])


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.training.seed)

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    variants = {
        name: build_model_variant(name, config, backbones)
        for name in ("full_large", "skip_only", "bridge_only", "hybrid")
    }
    maybe_load_hybrid_weights(variants, args, backbones.device)

    corpus = build_corpus_bundle(config, backbones.tokenizer, stage_name="eval_speed", split_name="validation")
    example = corpus.dataset[0]
    input_ids = example["input_ids"].unsqueeze(0).to(backbones.device)
    attention_mask = example["attention_mask"].unsqueeze(0).to(backbones.device)

    metrics: dict[str, dict[str, float | None]] = {}
    for variant_name, model in variants.items():
        model.eval()
        if backbones.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(backbones.device)

        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
        prefill_latency = time.perf_counter() - start

        start = time.perf_counter()
        with torch.no_grad():
            generated = greedy_generate(model, input_ids, attention_mask, config.eval.speed_decode_tokens)
        decode_latency = time.perf_counter() - start
        generated_tokens = max(1, generated.shape[1] - input_ids.shape[1])

        peak_vram_mb = None
        if backbones.device.type == "cuda":
            peak_vram_mb = float(torch.cuda.max_memory_allocated(backbones.device) / (1024**2))

        metrics[variant_name] = {
            "prefill_latency_sec": prefill_latency,
            "decode_latency_sec": decode_latency,
            "decode_tokens": float(generated_tokens),
            "decode_tokens_per_sec": generated_tokens / decode_latency,
            "peak_vram_mb": peak_vram_mb,
        }

    output_dir = ensure_dir(args.output_dir)
    save_json(Path(output_dir) / "speed_metrics.json", metrics)


if __name__ == "__main__":
    main()
