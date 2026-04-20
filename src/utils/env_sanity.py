"""Environment and Hugging Face auth sanity checks for real Gemma bring-up."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep non-PyTorch backends disabled for native Windows bring-up.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import datasets
import torch
import transformers
from huggingface_hub import HfFolder, hf_hub_download

from src.utils.io import ensure_dir, load_config, save_json
from src.utils.reporting import write_real_hardware_report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gemma2_conservative.yaml")
    parser.add_argument("--output-path", default="artifacts/env_sanity.json")
    parser.add_argument("--report-path", default="notes/real_hardware_report.md")
    return parser.parse_args()


def _bitsandbytes_status() -> dict[str, Any]:
    try:
        module = importlib.import_module("bitsandbytes")
        return {
            "available": True,
            "version": getattr(module, "__version__", None),
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _hf_token_status() -> tuple[str | None, str | None]:
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token, "environment"
    cached_token = HfFolder.get_token()
    if cached_token:
        return cached_token, "huggingface_hub_cache"
    return None, None


def _gemma_access_status(model_names: list[str], token: str | None) -> dict[str, Any]:
    per_model: dict[str, dict[str, Any]] = {}
    overall_success = True
    for model_name in model_names:
        try:
            downloaded = hf_hub_download(repo_id=model_name, filename="config.json", token=token)
            per_model[model_name] = {
                "success": True,
                "downloaded_path": downloaded,
                "error": None,
            }
        except Exception as exc:
            overall_success = False
            per_model[model_name] = {
                "success": False,
                "downloaded_path": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    summary = "ok" if overall_success else "; ".join(
        f"{name} failed: {payload['error']}" for name, payload in per_model.items() if not payload["success"]
    )
    return {
        "success": overall_success,
        "summary": summary,
        "per_model": per_model,
    }


def collect_env_sanity(config_path: str) -> dict[str, Any]:
    """Collect environment, CUDA, bitsandbytes, and Gemma auth details."""

    config = load_config(config_path)
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory if cuda_available else None
    bf16_supported = bool(cuda_available and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    token, token_source = _hf_token_status()
    gemma_access = _gemma_access_status([config.model.large_model_name, config.model.small_model_name], token)

    payload = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "total_vram_bytes": total_vram_bytes,
        "total_vram_gb": round(total_vram_bytes / (1024**3), 3) if total_vram_bytes is not None else None,
        "bf16_supported": bf16_supported,
        "bitsandbytes": _bitsandbytes_status(),
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "hf_auth": {
            "token_present": token is not None,
            "token_source": token_source,
        },
        "gemma_access": gemma_access,
    }
    quantization_ready = (not config.model.load_in_4bit) or payload["bitsandbytes"]["available"]
    payload["overall_pass"] = bool(
        payload["cuda_available"]
        and quantization_ready
        and payload["hf_auth"]["token_present"]
        and payload["gemma_access"]["success"]
    )
    return payload


def main() -> None:
    """Run the sanity check, write JSON, update the report, and exit nonzero on failure."""

    args = parse_args()
    payload = collect_env_sanity(args.config)
    ensure_dir(Path(args.output_path).parent)
    save_json(args.output_path, payload)

    blockers: list[str] = []
    if not payload["cuda_available"]:
        blockers.append("CUDA is not available on the current machine.")
    if load_config(args.config).model.load_in_4bit and not payload["bitsandbytes"]["available"]:
        blockers.append(f"bitsandbytes is unavailable for the configured 4-bit load path: {payload['bitsandbytes']['error']}")
    if not payload["hf_auth"]["token_present"]:
        blockers.append("No Hugging Face authentication token was detected.")
    if not payload["gemma_access"]["success"]:
        blockers.append(f"Gemma access check failed: {payload['gemma_access']['summary']}")

    write_real_hardware_report(
        args.report_path,
        env_path=args.output_path,
        blockers=blockers,
    )

    raise SystemExit(0 if payload["overall_pass"] else 1)


if __name__ == "__main__":
    main()
