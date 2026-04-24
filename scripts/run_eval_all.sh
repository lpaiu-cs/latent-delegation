#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/gemma2_conservative.yaml}"
STAGE_A_CHECKPOINT="${2:-}"
STAGE_B_CHECKPOINT="${3:-}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

for variant in full_large skip_only bridge_only hybrid; do
  ARGS=(--config "${CONFIG_PATH}" --variant "${variant}")
  if [[ "${variant}" == "hybrid" && -n "${STAGE_A_CHECKPOINT}" ]]; then
    ARGS+=(--stage-a-checkpoint "${STAGE_A_CHECKPOINT}")
  fi
  if [[ "${variant}" == "hybrid" && -n "${STAGE_B_CHECKPOINT}" ]]; then
    ARGS+=(--stage-b-checkpoint "${STAGE_B_CHECKPOINT}")
  fi
  "${PYTHON_BIN}" -m src.eval.eval_ppl "${ARGS[@]}"
  "${PYTHON_BIN}" -m src.eval.eval_gsm8k "${ARGS[@]}"
  "${PYTHON_BIN}" -m src.eval.eval_strategyqa "${ARGS[@]}"
done

SPEED_ARGS=(--config "${CONFIG_PATH}")
if [[ -n "${STAGE_A_CHECKPOINT}" ]]; then
  SPEED_ARGS+=(--stage-a-checkpoint "${STAGE_A_CHECKPOINT}")
fi
if [[ -n "${STAGE_B_CHECKPOINT}" ]]; then
  SPEED_ARGS+=(--stage-b-checkpoint "${STAGE_B_CHECKPOINT}")
fi
"${PYTHON_BIN}" -m src.eval.eval_speed "${SPEED_ARGS[@]}"
