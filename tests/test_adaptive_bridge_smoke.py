from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_adaptive_bridge_debug_smoke_runs_end_to_end() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    train_dir = repo_root / "outputs" / "adaptive_bridge" / "debug_train"
    eval_dir = repo_root / "outputs" / "adaptive_bridge" / "debug_eval"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if eval_dir.exists():
        shutil.rmtree(eval_dir)

    env = os.environ.copy()
    env["USE_TF"] = "0"
    env["USE_FLAX"] = "0"
    python = sys.executable
    subprocess.run(
        [
            python,
            "-m",
            "src.adaptive_bridge.train",
            "--config",
            "configs/adaptive_bridge/debug_tiny.yaml",
            "--output-dir",
            "outputs/adaptive_bridge/debug_train",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            python,
            "-m",
            "src.adaptive_bridge.evaluate",
            "--config",
            "configs/adaptive_bridge/debug_tiny.yaml",
            "--train-dir",
            "outputs/adaptive_bridge/debug_train",
            "--output-dir",
            "outputs/adaptive_bridge/debug_eval",
            "--report-path",
            "outputs/adaptive_bridge/debug_eval/summary_note.md",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )

    assert (train_dir / "seed_42" / "adaptive_bridge_moe_checkpoint.pt").exists()
    assert (eval_dir / "summary.csv").exists()
    assert (eval_dir / "summary_note.md").exists()
