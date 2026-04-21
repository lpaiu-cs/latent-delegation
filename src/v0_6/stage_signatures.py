"""Phase 1B functional-stage signature extraction and matching."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.eval.metrics import shifted_kl_divergence
from src.models.backbone_loader import load_backbones
from src.models.hybrid_gemma import GemmaCausalLMRunner
from src.train.trainer_utils import build_dataloader, move_batch_to_device
from src.utils.io import ensure_dir, load_config, save_json, save_text
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import seed_everything
from src.v0_6.common import WindowCandidate, clone_config


LOGGER = get_logger(__name__)
SIGNATURE_METRICS = [
    "hidden_norm_mean",
    "delta_norm_mean",
    "delta_cosine_mean",
    "logit_entropy_mean",
    "logit_kl_to_final_mean",
]


@dataclass(frozen=True)
class StageSignatureSettings:
    """Signature extraction settings."""

    window_lengths: list[int]
    top_k: int
    max_batches: int | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="artifacts/v0_6/phase1_stage_signatures")
    parser.add_argument("--report-path", default="notes/v0_6/phase1_stage_signatures_report.md")
    return parser.parse_args()


def load_stage_signature_settings(config: Any) -> StageSignatureSettings:
    """Load the optional `stage_signatures` section from config."""

    values = config.raw.get("stage_signatures", {})
    return StageSignatureSettings(
        window_lengths=list(values.get("window_lengths", [2, 3, 4, 5, 6, 8])),
        top_k=int(values.get("top_k", 5)),
        max_batches=values.get("max_batches"),
    )


def _valid_next_token_count(labels: torch.Tensor) -> int:
    return int((labels[:, 1:] != -100).sum().item())


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def _layer_signature_row(layer_idx: int) -> dict[str, float]:
    return {
        "layer": float(layer_idx),
        "hidden_norm_sum": 0.0,
        "delta_norm_sum": 0.0,
        "delta_cosine_sum": 0.0,
        "logit_entropy_sum": 0.0,
        "logit_kl_to_final_sum": 0.0,
        "token_count": 0.0,
        "valid_next_tokens": 0.0,
    }


def _finalize_layer_signature(row: dict[str, float]) -> dict[str, float]:
    token_count = max(1.0, row["token_count"])
    valid_next_tokens = max(1.0, row["valid_next_tokens"])
    return {
        "layer": row["layer"],
        "hidden_norm_mean": row["hidden_norm_sum"] / token_count,
        "delta_norm_mean": row["delta_norm_sum"] / token_count,
        "delta_cosine_mean": row["delta_cosine_sum"] / token_count,
        "logit_entropy_mean": row["logit_entropy_sum"] / token_count,
        "logit_kl_to_final_mean": row["logit_kl_to_final_sum"] / valid_next_tokens,
    }


def collect_layer_signatures(
    runner: GemmaCausalLMRunner,
    dataloader: Any,
    *,
    device: torch.device,
    max_batches: int | None,
) -> list[dict[str, float]]:
    """Collect per-layer functional signatures for one backbone."""

    rows = {_layer: _layer_signature_row(_layer) for _layer in range(runner.num_layers)}

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            state = runner.prepare_from_input_ids(batch["input_ids"], attention_mask=batch["attention_mask"])
            previous_hidden = state.hidden_states.detach()
            hidden_by_layer: list[torch.Tensor] = []
            for layer_idx in range(runner.num_layers):
                state = runner.run_layers(state, layer_idx, layer_idx)
                hidden = state.hidden_states.detach()
                hidden_by_layer.append(hidden)
            final_logits = runner.logits_from_hidden(hidden_by_layer[-1])
            attention_mask = batch["attention_mask"].float()
            labels = batch["labels"]
            token_count = float(attention_mask.sum().item())
            valid_next_tokens = float(_valid_next_token_count(labels))

            previous = previous_hidden
            for layer_idx, hidden in enumerate(hidden_by_layer):
                layer_logits = runner.logits_from_hidden(hidden)
                hidden_norm = hidden.float().norm(dim=-1)
                delta = hidden - previous
                delta_norm = delta.float().norm(dim=-1)
                delta_cosine = F.cosine_similarity(hidden.float(), previous.float(), dim=-1)
                logit_entropy = _entropy_from_logits(layer_logits)
                logit_kl = shifted_kl_divergence(layer_logits, final_logits, labels)

                row = rows[layer_idx]
                row["hidden_norm_sum"] += float((hidden_norm * attention_mask).sum().cpu())
                row["delta_norm_sum"] += float((delta_norm * attention_mask).sum().cpu())
                row["delta_cosine_sum"] += float((delta_cosine * attention_mask).sum().cpu())
                row["logit_entropy_sum"] += float((logit_entropy * attention_mask).sum().cpu())
                row["logit_kl_to_final_sum"] += float(logit_kl.cpu()) * valid_next_tokens
                row["token_count"] += token_count
                row["valid_next_tokens"] += valid_next_tokens
                previous = hidden

    return [_finalize_layer_signature(rows[layer_idx]) for layer_idx in range(runner.num_layers)]


def build_window_signature(layer_signatures: list[dict[str, float]], start: int, end: int) -> dict[str, float]:
    """Aggregate one contiguous window into a stage signature."""

    window_rows = layer_signatures[start : end + 1]
    signature: dict[str, float] = {
        "start": float(start),
        "end": float(end),
        "length": float(end - start + 1),
    }
    for metric in SIGNATURE_METRICS:
        signature[metric] = sum(float(row[metric]) for row in window_rows) / float(len(window_rows))
    return signature


def build_window_signatures(layer_signatures: list[dict[str, float]], window_lengths: list[int]) -> list[dict[str, float]]:
    """Aggregate all contiguous windows for the requested lengths."""

    windows: list[dict[str, float]] = []
    total_layers = len(layer_signatures)
    for length in window_lengths:
        if length <= 0 or length > total_layers:
            continue
        for start in range(0, total_layers - length + 1):
            windows.append(build_window_signature(layer_signatures, start, start + length - 1))
    return windows


def rank_window_matches(
    reference_signature: dict[str, float],
    candidate_signatures: list[dict[str, float]],
    *,
    top_k: int,
) -> list[dict[str, float]]:
    """Rank candidate windows by z-scored Euclidean distance to the reference."""

    all_rows = [reference_signature, *candidate_signatures]
    scales: dict[str, float] = {}
    for metric in SIGNATURE_METRICS:
        values = [float(row[metric]) for row in all_rows]
        mean = sum(values) / float(len(values))
        variance = sum((value - mean) ** 2 for value in values) / float(len(values))
        scales[metric] = math.sqrt(variance) or 1.0

    ranked: list[dict[str, float]] = []
    for row in candidate_signatures:
        distance = 0.0
        for metric in SIGNATURE_METRICS:
            normalized = (float(row[metric]) - float(reference_signature[metric])) / scales[metric]
            distance += normalized * normalized
        ranked.append({**row, "distance": math.sqrt(distance)})
    ranked.sort(key=lambda row: (float(row["distance"]), int(row["length"]), int(row["start"])))
    return ranked[:top_k]


def _default_small_window_rank(
    ranked_small_matches: list[dict[str, float]],
    config: Any,
) -> int | None:
    for index, row in enumerate(ranked_small_matches, start=1):
        if int(row["start"]) == config.split.small_delegate_start and int(row["end"]) == config.split.small_delegate_end:
            return index
    return None


def _write_report(
    report_path: str | Path,
    *,
    config_path: str,
    is_debug: bool,
    large_reference: dict[str, float],
    large_matches: list[dict[str, float]],
    small_matches: list[dict[str, float]],
    default_small_rank: int | None,
) -> None:
    asymmetric = any(int(row["length"]) != int(large_reference["length"]) for row in small_matches[:3])
    default_wrong = default_small_rank is None or default_small_rank > 1
    large_match_labels = ", ".join(f"{int(row['start'])}..{int(row['end'])}" for row in large_matches)
    small_match_labels = ", ".join(f"{int(row['start'])}..{int(row['end'])}" for row in small_matches)
    lines = [
        "# Phase 1B Stage Signature Report",
        "",
        f"- Config: `{config_path}`",
        f"- Backbones: `{'debug_tiny' if is_debug else 'real_model'}`",
        "- Signature metrics: hidden norm, hidden drift norm, hidden drift cosine, logit-lens entropy, and KL to final logits.",
        "",
        "## Closest Large Windows",
        "",
    ]
    for row in large_matches:
        lines.append(
            f"- `{int(row['start'])}..{int(row['end'])}` (len={int(row['length'])}) distance={row['distance']:.6f}"
        )
    lines.extend(["", "## Closest Small Windows", ""])
    for row in small_matches:
        lines.append(
            f"- `{int(row['start'])}..{int(row['end'])}` (len={int(row['length'])}) distance={row['distance']:.6f}"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Large windows functionally closest to the current removed block: {large_match_labels if large_match_labels else 'none beyond the reference window'}",
            f"2. Small windows functionally closest to the removed large block: {small_match_labels if small_match_labels else 'none'}",
            f"3. Does stage-aware matching suggest the current 6 -> 6 split is probably wrong? {'Yes' if default_wrong else 'No'}.",
            f"4. Does stage-aware matching suggest asymmetric mapping? {'Yes' if asymmetric else 'No'}.",
        ]
    )
    if is_debug:
        lines.extend(
            [
                "",
                "## Caveat",
                "",
                "- This report was generated on the debug-tiny path in the current environment. It validates the phase workflow only and should not be read as a Gemma research conclusion.",
                "- Real Gemma continuation runs remain blocked by the documented gated-model and no-CUDA environment issues in `notes/blockers.md`.",
            ]
        )
    save_text(report_path, "\n".join(lines))


def main() -> None:
    """Run stage-signature extraction and window matching."""

    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    settings = load_stage_signature_settings(config)
    seed_everything(config.training.seed)
    output_dir = ensure_dir(args.output_dir)

    backbones = load_backbones(config, load_large=True, load_small=True, load_tokenizer=True)
    signature_config = clone_config(config, experiment_name=f"{config.experiment.name}_phase1_signatures")
    dataloader, _ = build_dataloader(
        signature_config,
        backbones.tokenizer,
        stage_name="phase1_stage_signatures",
        split_name="validation",
    )

    LOGGER.info("collecting large-model stage signatures")
    large_runner = GemmaCausalLMRunner(backbones.large_model)
    large_layer_signatures = collect_layer_signatures(
        large_runner,
        dataloader,
        device=backbones.device,
        max_batches=settings.max_batches,
    )

    LOGGER.info("collecting small-model stage signatures")
    small_runner = GemmaCausalLMRunner(backbones.small_model)
    small_layer_signatures = collect_layer_signatures(
        small_runner,
        dataloader,
        device=backbones.device,
        max_batches=settings.max_batches,
    )

    large_reference = build_window_signature(
        large_layer_signatures,
        config.split.large_removed_start,
        config.split.large_removed_end,
    )
    large_windows = build_window_signatures(large_layer_signatures, settings.window_lengths)
    small_windows = build_window_signatures(small_layer_signatures, settings.window_lengths)

    large_alternatives = [
        row for row in large_windows if not (int(row["start"]) == config.split.large_removed_start and int(row["end"]) == config.split.large_removed_end)
    ]
    ranked_large_matches = rank_window_matches(large_reference, large_alternatives, top_k=settings.top_k)
    ranked_small_matches = rank_window_matches(large_reference, small_windows, top_k=settings.top_k)
    default_small_rank = _default_small_window_rank(ranked_small_matches, config)

    payload = {
        "config_path": args.config,
        "is_debug": backbones.is_debug,
        "settings": {
            "window_lengths": settings.window_lengths,
            "top_k": settings.top_k,
            "max_batches": settings.max_batches,
        },
        "reference_large_window": large_reference,
        "layer_signatures": {
            "large": large_layer_signatures,
            "small": small_layer_signatures,
        },
        "window_signatures": {
            "large": large_windows,
            "small": small_windows,
        },
        "matches": {
            "large_alternatives": ranked_large_matches,
            "small_to_large_reference": ranked_small_matches,
            "default_small_window": WindowCandidate(
                large_start=config.split.large_removed_start,
                large_end=config.split.large_removed_end,
                small_start=config.split.small_delegate_start,
                small_end=config.split.small_delegate_end,
            ).to_dict(),
            "default_small_rank": default_small_rank,
        },
    }
    save_json(output_dir / "signatures.json", payload)
    _write_report(
        args.report_path,
        config_path=args.config,
        is_debug=backbones.is_debug,
        large_reference=large_reference,
        large_matches=ranked_large_matches,
        small_matches=ranked_small_matches,
        default_small_rank=default_small_rank,
    )


if __name__ == "__main__":
    main()
