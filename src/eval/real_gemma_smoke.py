"""Real-Gemma single-GPU bring-up smoke matrix."""

from __future__ import annotations

import argparse
import gc
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from src.models.backbone_loader import LoadedBackbones, load_backbones
from src.train.trainer_utils import build_model_variant, required_backbones_for_variant
from src.utils.io import ensure_dir, load_config, save_json
from src.utils.reporting import write_real_hardware_report
from src.utils.seed import seed_everything


LOAD_ONLY_CASES = ("load_small_only", "load_large_only")
FORWARD_CASE_TO_VARIANT = {
    "full_large_forward": "full_large",
    "skip_only_forward": "skip_only",
    "bridge_only_forward": "bridge_only",
    "hybrid_forward": "hybrid",
}
FORWARD_SEQ_LENS = (64, 128, 256)
EXPECTED_CASES = len(LOAD_ONLY_CASES) + len(FORWARD_CASE_TO_VARIANT) * len(FORWARD_SEQ_LENS)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gemma2_conservative.yaml")
    parser.add_argument("--output-path", default="artifacts/real_gemma_smoke.json")
    parser.add_argument("--report-path", default="notes/real_hardware_report.md")
    return parser.parse_args()


def _clear_runtime_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_vram_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def _synthetic_batch(backbones: LoadedBackbones, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    loaded_models = [model for model in (backbones.large_model, backbones.small_model) if model is not None]
    vocab_size = min(int(model.config.vocab_size) for model in loaded_models)
    input_ids = torch.randint(
        low=0,
        high=max(8, vocab_size),
        size=(batch_size, seq_len),
        device=backbones.device,
        dtype=torch.long,
    )
    attention_mask = torch.ones((batch_size, seq_len), device=backbones.device, dtype=torch.long)
    return input_ids, attention_mask


def _requested_runtime(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    return {
        "torch_dtype": config.model.torch_dtype,
        "load_in_4bit": config.model.load_in_4bit,
        "bnb_4bit_quant_type": config.model.bnb_4bit_quant_type,
        "bnb_4bit_compute_dtype": config.model.bnb_4bit_compute_dtype,
    }


def _run_single_case(config_path: str, case: str, seq_len: int | None, batch_size: int) -> dict[str, Any]:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {
        "case": case,
        "seq_len": seq_len,
        "batch_size": batch_size if seq_len is not None else None,
        "runtime": _requested_runtime(config_path),
        "success": False,
        "error": None,
        "error_type": None,
        "traceback": None,
        "peak_vram_mb": None,
        "wall_time_sec": None,
    }

    backbones: LoadedBackbones | None = None
    model: torch.nn.Module | None = None
    start = time.perf_counter()
    _reset_peak_memory(device)
    try:
        if case == "load_small_only":
            backbones = load_backbones(config, load_large=False, load_small=True, load_tokenizer=False)
        elif case == "load_large_only":
            backbones = load_backbones(config, load_large=True, load_small=False, load_tokenizer=False)
        else:
            variant = FORWARD_CASE_TO_VARIANT[case]
            load_large, load_small = required_backbones_for_variant(variant)
            backbones = load_backbones(config, load_large=load_large, load_small=load_small, load_tokenizer=False)
            model = build_model_variant(variant, config, backbones).eval()
            input_ids, attention_mask = _synthetic_batch(backbones, batch_size=batch_size, seq_len=seq_len or 0)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            result["logits_shape"] = list(outputs.logits.shape)

        result["success"] = True
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=12)
    finally:
        if backbones is not None:
            result["device"] = backbones.device.type
        result["wall_time_sec"] = time.perf_counter() - start
        result["peak_vram_mb"] = _peak_vram_mb(backbones.device if backbones is not None else device)
        del model
        del backbones
        _clear_runtime_state()
    return result


def main() -> None:
    """Run the full real-Gemma smoke matrix and update the report."""

    args = parse_args()
    seed_everything(42)

    results: list[dict[str, Any]] = []
    ensure_dir(Path(args.output_path).parent)
    save_json(
        args.output_path,
        {
            "config_path": args.config,
            "expected_cases": EXPECTED_CASES,
            "completed_cases": 0,
            "overall_success": False,
            "results": results,
        },
    )
    for case in LOAD_ONLY_CASES:
        results.append(_run_single_case(args.config, case, seq_len=None, batch_size=1))
        save_json(
            args.output_path,
            {
                "config_path": args.config,
                "expected_cases": EXPECTED_CASES,
                "completed_cases": len(results),
                "overall_success": False,
                "results": results,
            },
        )
    for case in FORWARD_CASE_TO_VARIANT:
        for seq_len in FORWARD_SEQ_LENS:
            results.append(_run_single_case(args.config, case, seq_len=seq_len, batch_size=1))
            save_json(
                args.output_path,
                {
                    "config_path": args.config,
                    "expected_cases": EXPECTED_CASES,
                    "completed_cases": len(results),
                    "overall_success": False,
                    "results": results,
                },
            )

    overall_success = all(result["success"] for result in results)
    payload = {
        "config_path": args.config,
        "expected_cases": EXPECTED_CASES,
        "completed_cases": len(results),
        "overall_success": overall_success,
        "results": results,
    }
    save_json(args.output_path, payload)

    blockers = [f"{result['case']} seq_len={result['seq_len']} failed: {result['error']}" for result in results if not result["success"]]
    write_real_hardware_report(
        args.report_path,
        smoke_path=args.output_path,
        blockers=blockers,
    )
    raise SystemExit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
