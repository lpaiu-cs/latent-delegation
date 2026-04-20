#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/gemma2_conservative.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"${PYTHON_BIN}" -m src.train.stage_a_align --config "${CONFIG_PATH}"
