param(
    [string]$Config = "configs/gemma2_conservative_pilot_256.yaml",
    [string]$AblationDir = "artifacts/stage_b_ablation",
    [string]$AblationResults = "artifacts/stage_b_ablation_results.json",
    [string]$OutputDir = "artifacts/stage_b_output_probe"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

python -m src.eval.eval_stage_b_outputs `
    --config $Config `
    --ablation-dir $AblationDir `
    --ablation-results $AblationResults `
    --output-dir $OutputDir `
    --results-path "artifacts/stage_b_output_probe_results.json" `
    --summary-path "artifacts/stage_b_output_probe_summary.csv" `
    --report-path "notes/stage_b_output_probe_report.md"
