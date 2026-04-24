#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/gemma2_conservative.yaml}"
STAGE_A_CHECKPOINT="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if [[ -z "${STAGE_A_CHECKPOINT}" ]]; then
  echo "usage: ./scripts/run_stage_b.sh <config> <stage_a_checkpoint>" >&2
  exit 1
fi

"${PYTHON_BIN}" -m src.train.stage_b_recover --config "${CONFIG_PATH}" --stage-a-checkpoint "${STAGE_A_CHECKPOINT}"
