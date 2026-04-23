"""Build a paper-release reproducibility manifest from frozen artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, git_commit_hash, save_json, save_text


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="artifacts/paper_release")
    parser.add_argument("--note-path", default="notes/paper/reproducibility.md")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sample_bundle(root: Path, *, task_name: str, sample_ids_path: str, slice_definition_path: str) -> dict[str, Any]:
    sample_path = root / sample_ids_path
    slice_path = root / slice_definition_path
    return {
        "task_name": task_name,
        "sample_ids_path": _rel(sample_path, root),
        "slice_definition_path": _rel(slice_path, root),
        "slice_definition": _load_json(slice_path),
        "sample_ids": _load_json(sample_path),
    }


def _git_worktree_dirty(root: Path) -> bool | None:
    """Return whether the current git worktree is dirty, or None if unavailable."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def build_manifest(root: Path) -> dict[str, Any]:
    """Collect the paper-release reproducibility payload."""

    env = _load_json(root / "artifacts" / "env_sanity.json")
    main_holdout = _sample_bundle(
        root,
        task_name="v0_6_main_holdout",
        sample_ids_path="artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/seed_42/sample_ids.json",
        slice_definition_path="artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/slice_definition.json",
    )
    fresh_holdout = _sample_bundle(
        root,
        task_name="v0_6_fresh_holdout",
        sample_ids_path="artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/sample_ids.json",
        slice_definition_path="artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/slice_definition.json",
    )

    generalization_tasks = [
        ("hellaswag", "artifacts/v0_9/generalization/raw/multichoice/hellaswag/sample_ids.json", "artifacts/v0_9/generalization/raw/multichoice/hellaswag/slice_definition.json"),
        ("piqa", "artifacts/v0_9/generalization/raw/multichoice/piqa/sample_ids.json", "artifacts/v0_9/generalization/raw/multichoice/piqa/slice_definition.json"),
        ("winogrande", "artifacts/v0_9/generalization/raw/multichoice/winogrande/sample_ids.json", "artifacts/v0_9/generalization/raw/multichoice/winogrande/slice_definition.json"),
        ("arc_easy", "artifacts/v0_9/generalization/raw/multichoice/arc_easy/sample_ids.json", "artifacts/v0_9/generalization/raw/multichoice/arc_easy/slice_definition.json"),
        ("arc_challenge", "artifacts/v0_9/generalization/raw/multichoice/arc_challenge/sample_ids.json", "artifacts/v0_9/generalization/raw/multichoice/arc_challenge/slice_definition.json"),
        ("lambada_openai", "artifacts/v0_9/generalization/raw/lm/lambada_openai/sample_ids.json", "artifacts/v0_9/generalization/raw/lm/lambada_openai/slice_definition.json"),
    ]
    generalization_bundles = [
        _sample_bundle(root, task_name=task_name, sample_ids_path=sample_ids_path, slice_definition_path=slice_definition_path)
        for task_name, sample_ids_path, slice_definition_path in generalization_tasks
    ]

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit_hash": git_commit_hash(),
        "git_worktree_dirty": _git_worktree_dirty(root),
        "best_result_freeze": {
            "canonical_result": "v0.6.0",
            "analysis_only_branches": ["v0_7", "v0_8"],
            "bounded_generalization_branch": "v0_9",
            "stage_c_started": False,
        },
        "artifact_roots": {
            "best_result": "artifacts/v0_6/idea4_tokenwise",
            "phase1": "artifacts/v0_6/phase1_real",
            "idea5_analysis": "artifacts/v0_7/idea5_discovery",
            "idea2_analysis": "artifacts/v0_8/idea2_attribution",
            "generalization": "artifacts/v0_9/generalization",
            "paper_tables": "artifacts/paper_tables",
            "paper_figures": "artifacts/paper_figures",
        },
        "seed_policy": {
            "core_multiseed_runs": [42, 43, 44],
            "generalization_bootstrap_seed": 9090,
        },
        "environment_summary": {
            "overall_pass": env["overall_pass"],
            "platform": env["platform"],
            "device_name": env["device_name"],
            "total_vram_gb": env["total_vram_gb"],
            "python_version": env["python_version"],
            "torch_version": env["torch_version"],
            "transformers_version": env["transformers_version"],
            "datasets_version": env["datasets_version"],
            "bitsandbytes_version": env["bitsandbytes"]["version"],
            "cuda_available": env["cuda_available"],
            "bf16_supported": env["bf16_supported"],
            "hf_auth_token_present": env["hf_auth"]["token_present"],
            "gemma_access_success": env["gemma_access"]["success"],
        },
        "exact_slice_ids": {
            "main_holdout": main_holdout,
            "fresh_holdout": fresh_holdout,
            "generalization_tasks": generalization_bundles,
        },
        "windows_native_reproduction": {
            "install": [
                "py -3.12 -m pip install --upgrade pip",
                "py -3.12 -m pip install -r requirements.txt",
            ],
            "sanity": [
                "powershell -ExecutionPolicy Bypass -File .\\scripts\\env_sanity.ps1",
                "powershell -ExecutionPolicy Bypass -File .\\scripts\\real_gemma_smoke.ps1",
            ],
            "paper_assets": [
                "powershell -ExecutionPolicy Bypass -File .\\scripts\\run_paper_assets.ps1",
            ],
            "generalization": [
                "powershell -ExecutionPolicy Bypass -File .\\scripts\\v0_9\\run_generalization_eval.ps1",
            ],
            "tests": [
                "py -3.12 -m pytest -q",
            ],
        },
    }


def build_note(manifest: dict[str, Any]) -> str:
    """Render the paper reproducibility note."""

    lines = [
        "# Reproducibility Package",
        "",
        "## Freeze",
        "",
        f"- Canonical result: `{manifest['best_result_freeze']['canonical_result']}`",
        f"- Analysis-only branches: `{', '.join(manifest['best_result_freeze']['analysis_only_branches'])}`",
        f"- Bounded generalization branch: `{manifest['best_result_freeze']['bounded_generalization_branch']}`",
        f"- Stage C started: `{manifest['best_result_freeze']['stage_c_started']}`",
        "",
        "## Commit And Artifact Roots",
        "",
        f"- Git commit hash: `{manifest['git_commit_hash']}`",
        f"- Git worktree dirty: `{manifest['git_worktree_dirty']}`",
    ]
    for label, value in manifest["artifact_roots"].items():
        lines.append(f"- {label.replace('_', ' ')}: `{value}`")
    lines.extend(
        [
            "",
            "## Seeds",
            "",
            f"- Core multi-seed runs: `{manifest['seed_policy']['core_multiseed_runs']}`",
            f"- Generalization bootstrap seed: `{manifest['seed_policy']['generalization_bootstrap_seed']}`",
            "",
            "## Environment",
            "",
        ]
    )
    for key, value in manifest["environment_summary"].items():
        lines.append(f"- {key.replace('_', ' ')}: `{value}`")
    lines.extend(
        [
            "",
            "## Exact Slice ID Files",
            "",
            f"- Main holdout sample IDs: `{manifest['exact_slice_ids']['main_holdout']['sample_ids_path']}`",
            f"- Main holdout slice definition: `{manifest['exact_slice_ids']['main_holdout']['slice_definition_path']}`",
            f"- Fresh holdout sample IDs: `{manifest['exact_slice_ids']['fresh_holdout']['sample_ids_path']}`",
            f"- Fresh holdout slice definition: `{manifest['exact_slice_ids']['fresh_holdout']['slice_definition_path']}`",
        ]
    )
    for task in manifest["exact_slice_ids"]["generalization_tasks"]:
        lines.append(f"- {task['task_name']} sample IDs: `{task['sample_ids_path']}`")
        lines.append(f"- {task['task_name']} slice definition: `{task['slice_definition_path']}`")
    lines.extend(["", "## Windows-Native Commands", ""])
    for section_name, commands in manifest["windows_native_reproduction"].items():
        lines.append(f"### {section_name.replace('_', ' ').title()}")
        lines.append("")
        for command in commands:
            lines.append(f"- `{command}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    root = Path(args.repo_root).resolve()
    output_dir = ensure_dir(root / args.output_dir)
    note_path = root / args.note_path
    ensure_dir(note_path.parent)

    manifest = build_manifest(root)
    save_json(output_dir / "repro_manifest.json", manifest)
    save_text(note_path, build_note(manifest))


if __name__ == "__main__":
    main()
