param(
    [string]$Config = "configs/gemma2_conservative_pilot_256.yaml",
    [Parameter(Mandatory = $true)][string]$StageACheckpoint,
    [string]$OutputDir = "artifacts/stage_b_ablation"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

python -m src.pilots.stage_b_ablation `
    --config $Config `
    --stage-a-checkpoint $StageACheckpoint `
    --output-dir $OutputDir `
    --results-path "artifacts/stage_b_ablation_results.json" `
    --summary-path "artifacts/stage_b_ablation_summary.csv" `
    --diagnostics-path "artifacts/stage_b_diagnostics.json" `
    --report-path "notes/stage_b_ablation_report.md" `
    --seeds 42 43 44
