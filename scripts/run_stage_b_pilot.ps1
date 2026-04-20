param(
    [string]$Config = "configs/gemma2_conservative_pilot_256.yaml",
    [Parameter(Mandatory = $true)][string]$StageACheckpoint,
    [string]$OutputDir = "artifacts/stage_b_pilot_ckpt"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

python -m src.pilots.stage_b_pilot `
    --config $Config `
    --stage-a-checkpoint $StageACheckpoint `
    --output-dir $OutputDir `
    --metrics-path "artifacts/stage_b_pilot_metrics.json" `
    --history-path "artifacts/stage_b_pilot_history.csv"

python -c "from src.utils.reporting import write_real_hardware_report; write_real_hardware_report('notes/real_hardware_report.md')"
