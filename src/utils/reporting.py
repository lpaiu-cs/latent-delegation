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
EXPECTED_SMOKE_CASES = 14


def _load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    return json.loads(path_obj.read_text(encoding="utf-8"))


def _format_env_section(env_payload: dict[str, Any] | None) -> tuple[str, str]:
    if env_payload is None:
        return "Not run.", "Not run."

    gemma_models = env_payload.get("gemma_access", {}).get("per_model", {})
    target_environment = "\n".join(
        [
            f"- Python: {env_payload.get('python_version')}",
            f"- PyTorch: {env_payload.get('torch_version')}",
            f"- CUDA available: {env_payload.get('cuda_available')}",
            f"- Device: {env_payload.get('device_name')}",
            f"- Total VRAM (GB): {env_payload.get('total_vram_gb')}",
            f"- BF16 available: {env_payload.get('bf16_supported')}",
            f"- bitsandbytes available: {env_payload.get('bitsandbytes', {}).get('available')}",
            f"- transformers: {env_payload.get('transformers_version')}",
            f"- HF token present: {env_payload.get('hf_auth', {}).get('token_present')}",
            f"- Large Gemma access: {gemma_models.get('google/gemma-2-9b', {}).get('success')}",
            f"- Small Gemma access: {gemma_models.get('google/gemma-2-2b', {}).get('success')}",
        ]
    )

    auth_status = "\n".join(
        [
            f"- HF token source: {env_payload.get('hf_auth', {}).get('token_source')}",
            f"- Gemma access success: {env_payload.get('gemma_access', {}).get('success')}",
            f"- Gemma access detail: {env_payload.get('gemma_access', {}).get('summary')}",
            f"- bitsandbytes version: {env_payload.get('bitsandbytes', {}).get('version')}",
            f"- bitsandbytes error: {env_payload.get('bitsandbytes', {}).get('error')}",
        ]
        + (
            [
                "- Operational note: Windows Hugging Face cache symlink warnings were observed during downloads and were treated as non-blocking.",
                "- Operational note: native wrappers set `USE_TF=0` and `USE_FLAX=0` to keep the bring-up on the PyTorch path.",
            ]
            if "Windows" in str(env_payload.get("platform"))
            else []
        )
    )
    return target_environment, auth_status


def _largest_successful_seq_len(smoke_payload: dict[str, Any] | None, case_name: str) -> str:
    if smoke_payload is None:
        return "unknown"
    successes = [
        int(result["seq_len"])
        for result in smoke_payload.get("results", [])
        if result.get("case") == case_name and result.get("success") and result.get("seq_len") is not None
    ]
    return str(max(successes)) if successes else "none"


def _smoke_is_incomplete(smoke_payload: dict[str, Any] | None) -> bool:
    if smoke_payload is None:
        return True
    results = smoke_payload.get("results")
    if not isinstance(results, list):
        return True
    expected = int(smoke_payload.get("expected_cases", EXPECTED_SMOKE_CASES))
    completed = int(smoke_payload.get("completed_cases", len(results)))
    return completed < expected or len(results) < expected


def _diagnosis(env_payload: dict[str, Any] | None, smoke_payload: dict[str, Any] | None) -> tuple[str, str]:
    if smoke_payload is None or _smoke_is_incomplete(smoke_payload):
        return "D", "artifact files are missing, partial, or invalid, so the smoke result is still unknown"
    if env_payload is not None and not env_payload.get("overall_pass", False):
        return "C", "env_sanity failed on a real blocker"
    if env_payload is not None and env_payload.get("overall_pass", False) and smoke_payload.get("overall_success", False):
        return "A", "env_sanity passed and real_gemma_smoke completed successfully"
    if env_payload is not None and env_payload.get("overall_pass", False):
        return "B", "env_sanity passed, but real_gemma_smoke failed or did not finish cleanly"
    return "D", "artifact files are not sufficient to classify the run with confidence"


def _format_smoke_section(smoke_payload: dict[str, Any] | None) -> str:
    if smoke_payload is None:
        return "Not run."
    lines = [
        f"- Overall success: {smoke_payload.get('overall_success')}",
        f"- Cases completed: {smoke_payload.get('completed_cases', len(smoke_payload.get('results', [])))} / {smoke_payload.get('expected_cases', EXPECTED_SMOKE_CASES)}",
    ]
    for result in smoke_payload.get("results", []):
        seq = result.get("seq_len")
        seq_label = f"seq_len={seq}" if seq is not None else "load-only"
        lines.append(
            f"- {result.get('case')} ({seq_label}): success={result.get('success')}, "
            f"peak_vram_mb={result.get('peak_vram_mb')}, wall_time_sec={result.get('wall_time_sec')}, "
            f"runtime={result.get('runtime')}, error={result.get('error')}"
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
    diagnosis_code, diagnosis_text = _diagnosis(env_payload, smoke_payload)
    smoke_results = _format_smoke_section(smoke_payload)
    stage_a_results = _format_pilot_section(stage_a_payload, "Not run.")
    stage_b_results = _format_pilot_section(stage_b_payload, "Not run.")

    blocker_lines = blockers[:] if blockers else []
    if env_payload is not None and not env_payload.get("overall_pass", False):
        blocker_lines.append("Environment or Gemma auth sanity checks did not pass.")
    if _smoke_is_incomplete(smoke_payload):
        blocker_lines.append("real_gemma_smoke artifact is incomplete or stale and needs a clean rerun.")
    elif smoke_payload is not None and not smoke_payload.get("overall_success", False):
        blocker_lines.append("Real Gemma smoke matrix has one or more failures.")
    if stage_a_payload is not None and not stage_a_payload.get("heldout_alignment_improved", False):
        blocker_lines.append("Stage A pilot did not improve held-out large-to-small alignment.")
    if stage_b_payload is not None and not stage_b_payload.get("positive_pilot", False):
        blocker_lines.append("Stage B pilot did not beat skip-only on the held-out hidden recovery metrics.")
    if not blocker_lines:
        blocker_lines.append("No blocker recorded.")

    if next_action is None:
        if env_payload is None or not env_payload.get("overall_pass", False):
            next_action = "Fix the environment blocker, then rerun `scripts\\env_sanity.ps1`."
        elif _smoke_is_incomplete(smoke_payload):
            next_action = "Rerun `scripts\\real_gemma_smoke.ps1` in native Windows mode and overwrite the stale smoke artifact."
        elif not smoke_payload.get("overall_success", False):
            next_action = "Fix the failing smoke-matrix case with the smallest patch possible, then rerun the smoke matrix before any pilot training."
        elif stage_a_payload is None:
            next_action = "Run `scripts\\run_stage_a_pilot.ps1` with the 256-token pilot config."
        elif not stage_a_payload.get("heldout_alignment_improved", False):
            next_action = "Do not start Stage B yet; fix the Stage A alignment path and rerun the pilot."
        elif stage_b_payload is None:
            next_action = "Run `scripts\\run_stage_b_pilot.ps1` with the Stage A pilot checkpoint."
        elif not stage_b_payload.get("positive_pilot", False):
            next_action = "Do not start Stage C. Investigate why the Stage B pilot failed to beat skip-only."
        elif not stage_b_payload.get("hybrid_beats_bridge_only", False):
            next_action = "Do not start Stage C yet. Record that the hybrid beats skip-only at seq_len 256 but does not clearly beat bridge-only, then decide whether to tune Stage B or keep bridge-only as the stronger baseline."
        else:
            next_action = "Do not start Stage C in this task. Record the positive 256-token pilot result and prepare the next run plan."

    markdown = "\n".join(
        [
            "# Real Hardware Report",
            "",
            "## 0. diagnosis",
            "",
            f"- Status: {diagnosis_code}",
            f"- Summary: {diagnosis_text}.",
            "",
            "## 1. environment summary",
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
            "## 5. largest successful seq_len",
            "",
            "\n".join(
                [
                    f"- full_large: {_largest_successful_seq_len(smoke_payload, 'full_large_forward')}",
                    f"- skip_only: {_largest_successful_seq_len(smoke_payload, 'skip_only_forward')}",
                    f"- bridge_only: {_largest_successful_seq_len(smoke_payload, 'bridge_only_forward')}",
                    f"- hybrid: {_largest_successful_seq_len(smoke_payload, 'hybrid_forward')}",
                ]
            ),
            "",
            "## 6. Stage A pilot results",
            "",
            stage_a_results,
            "",
            "## 7. Stage B pilot results",
            "",
            stage_b_results,
            "",
            "## 8. blockers",
            "",
            "\n".join(f"- {line}" for line in blocker_lines),
            "",
            "## 9. exact next recommended action",
            "",
            next_action,
            "",
        ]
    )

    ensure_dir(Path(report_path).parent)
    Path(report_path).write_text(markdown, encoding="utf-8")
