param(
    [string]$Config = "configs/gemma2_conservative.yaml"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

& "$PSScriptRoot\env_sanity.ps1" -Config $Config

python -m src.eval.real_gemma_smoke `
    --config $Config `
    --output-path "artifacts/real_gemma_smoke.json" `
    --report-path "notes/real_hardware_report.md"
