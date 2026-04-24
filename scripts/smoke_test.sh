#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/debug_tiny.yaml"
RUN_ROOT="${ROOT_DIR}/outputs/debug_smoke"

rm -rf "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/stage_a" "${RUN_ROOT}/stage_b" "${RUN_ROOT}/stage_c"

"${PYTHON_BIN}" -m src.train.stage_a_align \
  --config "${CONFIG_PATH}" \
  --output-dir "${RUN_ROOT}/stage_a"

"${PYTHON_BIN}" -m src.train.stage_b_recover \
  --config "${CONFIG_PATH}" \
  --stage-a-checkpoint "${RUN_ROOT}/stage_a/stage_a_checkpoint.pt" \
  --output-dir "${RUN_ROOT}/stage_b"

"${PYTHON_BIN}" -m src.train.stage_c_distill \
  --config "${CONFIG_PATH}" \
  --stage-a-checkpoint "${RUN_ROOT}/stage_a/stage_a_checkpoint.pt" \
  --stage-b-checkpoint "${RUN_ROOT}/stage_b/stage_b_checkpoint.pt" \
  --output-dir "${RUN_ROOT}/stage_c"

"${PYTHON_BIN}" -m src.eval.eval_ppl \
  --config "${CONFIG_PATH}" \
  --variant hybrid \
  --stage-a-checkpoint "${RUN_ROOT}/stage_a/stage_a_checkpoint.pt" \
  --stage-b-checkpoint "${RUN_ROOT}/stage_b/stage_b_checkpoint.pt" \
  --output-dir "${RUN_ROOT}/eval_ppl"

"${PYTHON_BIN}" -m src.eval.eval_gsm8k \
  --config "${CONFIG_PATH}" \
  --variant hybrid \
  --stage-a-checkpoint "${RUN_ROOT}/stage_a/stage_a_checkpoint.pt" \
  --stage-b-checkpoint "${RUN_ROOT}/stage_b/stage_b_checkpoint.pt" \
  --output-dir "${RUN_ROOT}/eval_gsm8k"

"${PYTHON_BIN}" -m src.eval.eval_strategyqa \
  --config "${CONFIG_PATH}" \
  --variant hybrid \
  --stage-a-checkpoint "${RUN_ROOT}/stage_a/stage_a_checkpoint.pt" \
  --stage-b-checkpoint "${RUN_ROOT}/stage_b/stage_b_checkpoint.pt" \
  --output-dir "${RUN_ROOT}/eval_strategyqa"

"${PYTHON_BIN}" -m src.eval.eval_speed \
  --config "${CONFIG_PATH}" \
  --stage-a-checkpoint "${RUN_ROOT}/stage_a/stage_a_checkpoint.pt" \
  --stage-b-checkpoint "${RUN_ROOT}/stage_b/stage_b_checkpoint.pt" \
  --output-dir "${RUN_ROOT}/eval_speed"
