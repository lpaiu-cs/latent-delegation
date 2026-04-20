#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/gemma2_conservative.yaml}"
STAGE_A_CHECKPOINT="${2:-}"
STAGE_B_CHECKPOINT="${3:-}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if [[ -z "${STAGE_A_CHECKPOINT}" || -z "${STAGE_B_CHECKPOINT}" ]]; then
  echo "usage: ./scripts/run_stage_c.sh <config> <stage_a_checkpoint> <stage_b_checkpoint>" >&2
  exit 1
fi

"${PYTHON_BIN}" -m src.train.stage_c_distill --config "${CONFIG_PATH}" --stage-a-checkpoint "${STAGE_A_CHECKPOINT}" --stage-b-checkpoint "${STAGE_B_CHECKPOINT}"
