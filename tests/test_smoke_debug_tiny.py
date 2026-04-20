from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_smoke_script_runs_end_to_end() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / "outputs" / "debug_smoke"
    if output_root.exists():
        shutil.rmtree(output_root)

    env = os.environ.copy()
    env["USE_TF"] = "0"
    env["USE_FLAX"] = "0"
    python = sys.executable
    subprocess.run([python, "-m", "src.train.stage_a_align", "--config", "configs/debug_tiny.yaml", "--output-dir", "outputs/debug_smoke/stage_a"], cwd=repo_root, env=env, check=True)
    subprocess.run([python, "-m", "src.train.stage_b_recover", "--config", "configs/debug_tiny.yaml", "--stage-a-checkpoint", "outputs/debug_smoke/stage_a/stage_a_checkpoint.pt", "--output-dir", "outputs/debug_smoke/stage_b"], cwd=repo_root, env=env, check=True)
    subprocess.run([python, "-m", "src.train.stage_c_distill", "--config", "configs/debug_tiny.yaml", "--stage-a-checkpoint", "outputs/debug_smoke/stage_a/stage_a_checkpoint.pt", "--stage-b-checkpoint", "outputs/debug_smoke/stage_b/stage_b_checkpoint.pt", "--output-dir", "outputs/debug_smoke/stage_c"], cwd=repo_root, env=env, check=True)
    subprocess.run([python, "-m", "src.eval.eval_ppl", "--config", "configs/debug_tiny.yaml", "--variant", "hybrid", "--stage-a-checkpoint", "outputs/debug_smoke/stage_a/stage_a_checkpoint.pt", "--stage-b-checkpoint", "outputs/debug_smoke/stage_b/stage_b_checkpoint.pt", "--output-dir", "outputs/debug_smoke/eval_ppl"], cwd=repo_root, env=env, check=True)

    assert (output_root / "stage_a" / "stage_a_checkpoint.pt").exists()
    assert (output_root / "stage_b" / "stage_b_checkpoint.pt").exists()
    assert (output_root / "stage_c" / "stage_c_checkpoint.pt").exists()
    assert (output_root / "eval_ppl" / "hybrid_metrics.json").exists()
