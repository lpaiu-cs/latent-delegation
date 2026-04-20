#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/gemma2_conservative.yaml}"

mkdir -p "${ROOT_DIR}/artifacts"
"${PYTHON_BIN}" -m src.utils.env_sanity \
  --config "${CONFIG_PATH}" \
  --output-path "${ROOT_DIR}/artifacts/env_sanity.json" \
  --report-path "${ROOT_DIR}/notes/real_hardware_report.md"
