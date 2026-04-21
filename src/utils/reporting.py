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


def _format_milestone_snapshot(
    smoke_payload: dict[str, Any] | None,
    stage_a_payload: dict[str, Any] | None,
    stage_b_payload: dict[str, Any] | None,
    ablation_payload: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    if smoke_payload is None:
        lines.append("- Smoke: Not run.")
    else:
        lines.append(
            "- Smoke: "
            f"overall_success={smoke_payload.get('overall_success')}, "
            f"completed_cases={smoke_payload.get('completed_cases', len(smoke_payload.get('results', [])))}/"
            f"{smoke_payload.get('expected_cases', EXPECTED_SMOKE_CASES)}, "
            f"largest_full_large_seq_len={_largest_successful_seq_len(smoke_payload, 'full_large_forward')}, "
            f"largest_skip_only_seq_len={_largest_successful_seq_len(smoke_payload, 'skip_only_forward')}, "
            f"largest_bridge_only_seq_len={_largest_successful_seq_len(smoke_payload, 'bridge_only_forward')}, "
            f"largest_hybrid_seq_len={_largest_successful_seq_len(smoke_payload, 'hybrid_forward')}."
        )
    if stage_a_payload is None:
        lines.append("- Stage A: Not run.")
    else:
        lines.append(
            "- Stage A: "
            f"train_loss_start={stage_a_payload.get('train_loss_start')}, "
            f"train_loss_end={stage_a_payload.get('train_loss_end')}, "
            f"heldout_mse_before={stage_a_payload.get('heldout_mse_before')}, "
            f"heldout_mse_after={stage_a_payload.get('heldout_mse_after')}, "
            f"heldout_cosine_before={stage_a_payload.get('heldout_cosine_before')}, "
            f"heldout_cosine_after={stage_a_payload.get('heldout_cosine_after')}, "
            f"heldout_alignment_improved={stage_a_payload.get('heldout_alignment_improved')}."
        )
    if stage_b_payload is None:
        lines.append("- Stage B: Not run.")
    else:
        lines.append(
            "- Stage B: "
            f"skip_only_hidden_mse={stage_b_payload.get('skip_only_hidden_mse')}, "
            f"skip_only_cosine={stage_b_payload.get('skip_only_cosine')}, "
            f"bridge_only_hidden_mse={stage_b_payload.get('bridge_only_hidden_mse')}, "
            f"bridge_only_cosine={stage_b_payload.get('bridge_only_cosine')}, "
            f"hybrid_hidden_mse={stage_b_payload.get('hybrid_hidden_mse')}, "
            f"hybrid_cosine={stage_b_payload.get('hybrid_cosine')}, "
            f"hybrid_beats_skip_only={stage_b_payload.get('hybrid_beats_skip_only')}, "
            f"hybrid_beats_bridge_only={stage_b_payload.get('hybrid_beats_bridge_only')}."
        )
    if ablation_payload is not None:
        pairwise_wins = ablation_payload.get("summary", {}).get("pairwise_wins", {})
        lines.append(
            "- Stage B ablation: "
            f"hybrid_vs_skip_only_wins={pairwise_wins.get('skip_only', {}).get('hybrid_wins_on_both_metrics')}/"
            f"{pairwise_wins.get('skip_only', {}).get('seeds')}, "
            f"hybrid_vs_hybrid_no_small_wins={pairwise_wins.get('hybrid_no_small', {}).get('hybrid_wins_on_both_metrics')}/"
            f"{pairwise_wins.get('hybrid_no_small', {}).get('seeds')}, "
            f"hybrid_vs_bridge_only_wins={pairwise_wins.get('bridge_only', {}).get('hybrid_wins_on_both_metrics')}/"
            f"{pairwise_wins.get('bridge_only', {}).get('seeds')}, "
            f"hybrid_vs_param_matched_bridge_wins={pairwise_wins.get('bridge_only_param_matched', {}).get('hybrid_wins_on_both_metrics')}/"
            f"{pairwise_wins.get('bridge_only_param_matched', {}).get('seeds')}."
        )
    return "\n".join(lines)


def _format_parameter_audit(parameter_audit_payload: dict[str, Any] | None) -> str:
    if parameter_audit_payload is None:
        return "Parameter audit not run."
    lines = [f"- Config: {parameter_audit_payload.get('config_path')}"]
    for label in ("skip_only", "bridge_only", "hybrid"):
        summary = parameter_audit_payload.get("models", {}).get(label)
        if summary is None:
            continue
        trainable_modules = ", ".join(summary.get("trainable_modules", [])) or "none"
        extra_bits = []
        if "bridge_rank" in summary:
            extra_bits.append(f"bridge_rank={summary['bridge_rank']}")
        if "return_adapter_rank" in summary:
            extra_bits.append(f"return_adapter_rank={summary['return_adapter_rank']}")
        if "gate_init" in summary:
            extra_bits.append(f"gate_init={summary['gate_init']}")
        extras = f" ({', '.join(extra_bits)})" if extra_bits else ""
        lines.append(
            f"- {label}{extras}: total_params={summary.get('total_params')}, "
            f"trainable_params={summary.get('trainable_params')}, "
            f"frozen_params={summary.get('frozen_params')}, "
            f"trainable_modules={trainable_modules}"
        )
    return "\n".join(lines)


def _format_v040_milestone() -> str:
    return "\n".join(
        [
            "- Milestone: `v0.4.0`",
            "- Strongest defensible claim: delegated small-model computation is useful relative to passthrough / no-small controls, but it has not yet shown a clean win over strong large-space bridge controls.",
            "- Frozen reference artifacts:",
            "- `artifacts/real_gemma_smoke.json`",
            "- `artifacts/stage_a_pilot_metrics.json`",
            "- `artifacts/stage_b_pilot_metrics.json`",
            "- `artifacts/stage_b_ablation_results.json`",
            "- `artifacts/stage_b_diagnostics.json`",
            "- `notes/stage_b_ablation_report.md`",
        ]
    )


def write_real_hardware_report(
    report_path: str | Path,
    *,
    env_path: str | Path = "artifacts/env_sanity.json",
    smoke_path: str | Path = "artifacts/real_gemma_smoke.json",
    stage_a_path: str | Path = "artifacts/stage_a_pilot_metrics.json",
    stage_b_path: str | Path = "artifacts/stage_b_pilot_metrics.json",
    parameter_audit_path: str | Path = "artifacts/milestone_parameter_audit.json",
    stage_b_ablation_path: str | Path = "artifacts/stage_b_ablation_results.json",
    blockers: list[str] | None = None,
    next_action: str | None = None,
) -> None:
    """Write the real-hardware bring-up report from the current artifacts."""

    env_payload = _load_json_if_exists(env_path)
    smoke_payload = _load_json_if_exists(smoke_path)
    stage_a_payload = _load_json_if_exists(stage_a_path)
    stage_b_payload = _load_json_if_exists(stage_b_path)
    parameter_audit_payload = _load_json_if_exists(parameter_audit_path)
    stage_b_ablation_payload = _load_json_if_exists(stage_b_ablation_path)

    target_environment, auth_status = _format_env_section(env_payload)
    diagnosis_code, diagnosis_text = _diagnosis(env_payload, smoke_payload)
    smoke_results = _format_smoke_section(smoke_payload)
    milestone_snapshot = _format_milestone_snapshot(smoke_payload, stage_a_payload, stage_b_payload, stage_b_ablation_payload)
    parameter_audit = _format_parameter_audit(parameter_audit_payload)

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
    if stage_b_ablation_payload is not None:
        pairwise_wins = stage_b_ablation_payload.get("summary", {}).get("pairwise_wins", {})
        if pairwise_wins.get("bridge_only", {}).get("hybrid_wins_on_both_metrics", 0) < 2:
            blocker_lines.append("Stage B ablation did not show a reproducible win over the original bridge-only control.")
        if pairwise_wins.get("bridge_only_param_matched", {}).get("hybrid_wins_on_both_metrics", 0) < 2:
            blocker_lines.append("Stage B ablation did not show a reproducible win over the parameter-matched bridge-only control.")
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
        elif stage_b_ablation_payload is not None:
            pairwise_wins = stage_b_ablation_payload.get("summary", {}).get("pairwise_wins", {})
            bridge_wins = pairwise_wins.get("bridge_only", {}).get("hybrid_wins_on_both_metrics", 0)
            param_wins = pairwise_wins.get("bridge_only_param_matched", {}).get("hybrid_wins_on_both_metrics", 0)
            if bridge_wins >= 2 and param_wins >= 2:
                next_action = "Stage B ablation cleared the stronger bridge controls. Stage C is now justified on the current evidence."
            else:
                next_action = "Do not start Stage C yet. The delegated path is active and beats skip-only and hybrid_no_small, but the three-seed ablation still leaves bridge-only as the stronger control."
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
            "## 6. milestone snapshot",
            "",
            milestone_snapshot,
            "",
            "## 7. parameter audit",
            "",
            parameter_audit,
            "",
            "## 8. v0.4.0 milestone",
            "",
            _format_v040_milestone(),
            "",
            "## 9. blockers",
            "",
            "\n".join(f"- {line}" for line in blocker_lines),
            "",
            "## 10. exact next recommended action",
            "",
            next_action,
            "",
        ]
    )

    ensure_dir(Path(report_path).parent)
    Path(report_path).write_text(markdown, encoding="utf-8")
