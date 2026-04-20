"""Helpers for writing the real-hardware bring-up report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir


MEMORY_AUDIT_LINES = [
    "Stage A instantiates one shared frozen Gemma 2 9B and one shared frozen Gemma 2 2B. No extra teacher model is loaded; the large prefix and small reference pass reuse those same backbone objects inside `torch.no_grad()`.",
    "Stage B hybrid instantiates one shared 9B and one shared 2B. The teacher hidden after large layer 29 is produced by a second serial forward through the already-loaded large backbone, not by a duplicate 9B instance.",
    "Stage B bridge-only instantiates only the 9B backbone. The 2B model is not loaded on that code path anymore.",
    "Stage C hybrid instantiates one shared 9B and one shared 2B. Teacher logits come from a serial `torch.no_grad()` full-large pass on the same 9B object that the student hybrid later reuses.",
    "Single-variant eval paths now load only the required backbones: `full_large`, `skip_only`, and `bridge_only` load only the 9B; `hybrid` loads the 9B and 2B. They no longer instantiate all variants up front.",
    "The speed eval still keeps one 9B and one 2B resident for the whole comparison because the hybrid case requires both. Wrapper modules are lightweight and share the same backbone objects rather than cloning weights.",
    "Expected VRAM pressure points are: large-model load in 4-bit; hybrid forward at seq_len 256 because both frozen backbones are resident; and Stage C because the student path must retain activations through the frozen small block and large suffix for adapter gradients.",
]


def _load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    return json.loads(path_obj.read_text(encoding="utf-8"))


def _format_env_section(env_payload: dict[str, Any] | None) -> tuple[str, str]:
    if env_payload is None:
        return "Not run.", "Not run."

    target_environment = "\n".join(
        [
            f"- Python: {env_payload.get('python_version')}",
            f"- PyTorch: {env_payload.get('torch_version')}",
            f"- CUDA available: {env_payload.get('cuda_available')}",
            f"- Device: {env_payload.get('device_name')}",
            f"- Total VRAM (GB): {env_payload.get('total_vram_gb')}",
            f"- BF16 available: {env_payload.get('bf16_supported')}",
            f"- bitsandbytes import: {env_payload.get('bitsandbytes', {}).get('available')}",
            f"- transformers: {env_payload.get('transformers_version')}",
            f"- datasets: {env_payload.get('datasets_version')}",
        ]
    )

    auth_status = "\n".join(
        [
            f"- HF token present: {env_payload.get('hf_auth', {}).get('token_present')}",
            f"- HF token source: {env_payload.get('hf_auth', {}).get('token_source')}",
            f"- Gemma access success: {env_payload.get('gemma_access', {}).get('success')}",
            f"- Gemma access detail: {env_payload.get('gemma_access', {}).get('summary')}",
        ]
    )
    return target_environment, auth_status


def _format_smoke_section(smoke_payload: dict[str, Any] | None) -> str:
    if smoke_payload is None:
        return "Not run."
    lines = [
        f"- Overall success: {smoke_payload.get('overall_success')}",
        f"- Cases attempted: {len(smoke_payload.get('results', []))}",
    ]
    for result in smoke_payload.get("results", []):
        seq = result.get("seq_len")
        seq_label = f"seq_len={seq}" if seq is not None else "load-only"
        lines.append(
            f"- {result.get('case')} ({seq_label}): success={result.get('success')}, "
            f"peak_vram_mb={result.get('peak_vram_mb')}, wall_time_sec={result.get('wall_time_sec')}, "
            f"error={result.get('error')}"
        )
    return "\n".join(lines)


def _format_pilot_section(pilot_payload: dict[str, Any] | None, blocked_message: str) -> str:
    if pilot_payload is None:
        return blocked_message
    return "\n".join([f"- {key}: {value}" for key, value in pilot_payload.items()])


def write_real_hardware_report(
    report_path: str | Path,
    *,
    env_path: str | Path = "artifacts/env_sanity.json",
    smoke_path: str | Path = "artifacts/real_gemma_smoke.json",
    stage_a_path: str | Path = "artifacts/stage_a_pilot_metrics.json",
    stage_b_path: str | Path = "artifacts/stage_b_pilot_metrics.json",
    blockers: list[str] | None = None,
    next_action: str | None = None,
) -> None:
    """Write the real-hardware bring-up report from the current artifacts."""

    env_payload = _load_json_if_exists(env_path)
    smoke_payload = _load_json_if_exists(smoke_path)
    stage_a_payload = _load_json_if_exists(stage_a_path)
    stage_b_payload = _load_json_if_exists(stage_b_path)

    target_environment, auth_status = _format_env_section(env_payload)
    smoke_results = _format_smoke_section(smoke_payload)
    stage_a_results = _format_pilot_section(stage_a_payload, "Not run.")
    stage_b_results = _format_pilot_section(stage_b_payload, "Not run.")

    blocker_lines = blockers[:] if blockers else []
    if env_payload is not None and not env_payload.get("overall_pass", False):
        blocker_lines.append("Environment or Gemma auth sanity checks did not pass.")
    if smoke_payload is not None and not smoke_payload.get("overall_success", False):
        blocker_lines.append("Real Gemma smoke matrix has one or more failures.")
    if not blocker_lines:
        blocker_lines.append("No blocker recorded.")

    if next_action is None:
        if env_payload is None or not env_payload.get("overall_pass", False):
            next_action = "Provide a CUDA-capable single-GPU machine with Hugging Face-authenticated Gemma access, then rerun `./scripts/env_sanity.sh`."
        elif smoke_payload is None:
            next_action = "Run `./scripts/real_gemma_smoke.sh` on the target GPU machine."
        elif not smoke_payload.get("overall_success", False):
            next_action = "Fix the failing smoke-matrix case with the smallest patch possible, then rerun the smoke matrix before any pilot training."
        else:
            next_action = "If hybrid forward succeeds at seq_len 256, run the minimal Stage A pilot next."

    markdown = "\n".join(
        [
            "# Real Hardware Report",
            "",
            "## 1. target environment",
            "",
            target_environment,
            "",
            "## 2. auth status",
            "",
            auth_status,
            "",
            "## 3. memory topology audit",
            "",
            "\n".join(f"- {line}" for line in MEMORY_AUDIT_LINES),
            "",
            "## 4. smoke matrix results",
            "",
            smoke_results,
            "",
            "## 5. Stage A pilot results",
            "",
            stage_a_results,
            "",
            "## 6. Stage B pilot results",
            "",
            stage_b_results,
            "",
            "## 7. blockers",
            "",
            "\n".join(f"- {line}" for line in blocker_lines),
            "",
            "## 8. exact next recommended action",
            "",
            next_action,
            "",
        ]
    )

    ensure_dir(Path(report_path).parent)
    Path(report_path).write_text(markdown, encoding="utf-8")
