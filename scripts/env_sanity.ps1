param(
    [string]$Config = "configs/gemma2_conservative.yaml"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

python -m src.utils.env_sanity `
    --config $Config `
    --output-path "artifacts/env_sanity.json" `
    --report-path "notes/real_hardware_report.md"
