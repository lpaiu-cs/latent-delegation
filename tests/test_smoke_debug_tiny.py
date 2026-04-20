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
    env["PYTHON_BIN"] = sys.executable
    subprocess.run(
        ["bash", str(repo_root / "scripts" / "smoke_test.sh")],
        cwd=repo_root,
        env=env,
        check=True,
    )

    assert (output_root / "stage_a" / "stage_a_checkpoint.pt").exists()
    assert (output_root / "stage_b" / "stage_b_checkpoint.pt").exists()
    assert (output_root / "stage_c" / "stage_c_checkpoint.pt").exists()
    assert (output_root / "eval_ppl" / "hybrid_metrics.json").exists()
