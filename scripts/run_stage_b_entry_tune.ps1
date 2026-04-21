param(
    [string]$FrozenConfig = "configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml",
    [string]$TunedConfig = "configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml",
    [Parameter(Mandatory = $true)][string]$StageACheckpoint,
    [string]$FrozenAblationDir = "artifacts/stage_b_ablation_output_aware",
    [string]$FrozenAblationResults = "artifacts/stage_b_ablation_output_aware_results.json",
    [string]$FrozenOutputProbeResults = "artifacts/stage_b_output_probe_output_aware_results.json",
    [string]$TunedRawDir = "artifacts/stage_b_ablation_output_aware_train_entry_raw",
    [string]$TunedAblationResults = "artifacts/stage_b_ablation_output_aware_train_entry_results.json",
    [string]$TunedAblationSummary = "artifacts/stage_b_ablation_output_aware_train_entry_summary.csv",
    [string]$TunedTrainDiagnostics = "artifacts/stage_b_ablation_output_aware_train_entry_diagnostics.json",
    [string]$TunedRawReport = "notes/stage_b_ablation_output_aware_train_entry_raw_report.md",
    [string]$ProbeMergedDir = "artifacts/stage_b_ablation_output_aware_train_entry_probe",
    [string]$TunedProbeDir = "artifacts/stage_b_output_probe_output_aware_train_entry",
    [string]$TunedProbeResults = "artifacts/stage_b_output_probe_output_aware_train_entry_results.json",
    [string]$TunedProbeSummary = "artifacts/stage_b_output_probe_output_aware_train_entry_summary.csv",
    [string]$TunedProbeRawReport = "notes/stage_b_output_probe_output_aware_train_entry_raw_report.md"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

python -m src.pilots.stage_b_ablation `
    --config $TunedConfig `
    --stage-a-checkpoint $StageACheckpoint `
    --output-dir $TunedRawDir `
    --results-path $TunedAblationResults `
    --summary-path $TunedAblationSummary `
    --diagnostics-path $TunedTrainDiagnostics `
    --report-path $TunedRawReport `
    --variants hybrid hybrid_no_small `
    --seeds 42 43 44

if (Test-Path $ProbeMergedDir) {
    Remove-Item -Recurse -Force $ProbeMergedDir
}
New-Item -ItemType Directory -Path $ProbeMergedDir | Out-Null

foreach ($seed in 42, 43, 44) {
    $seedDir = Join-Path $ProbeMergedDir "seed_$seed"
    New-Item -ItemType Directory -Path $seedDir | Out-Null
    Copy-Item (Join-Path $FrozenAblationDir "seed_$seed\\bridge_only_checkpoint.pt") $seedDir
    Copy-Item (Join-Path $FrozenAblationDir "seed_$seed\\bridge_only_param_matched_checkpoint.pt") $seedDir
    Copy-Item (Join-Path $TunedRawDir "seed_$seed\\hybrid_checkpoint.pt") $seedDir
    Copy-Item (Join-Path $TunedRawDir "seed_$seed\\hybrid_no_small_checkpoint.pt") $seedDir
}

python -m src.eval.eval_stage_b_outputs `
    --config $TunedConfig `
    --ablation-dir $ProbeMergedDir `
    --ablation-results $TunedAblationResults `
    --output-dir $TunedProbeDir `
    --results-path $TunedProbeResults `
    --summary-path $TunedProbeSummary `
    --report-path $TunedProbeRawReport

python -m src.analysis.stage_b_entry_tune_report `
    --frozen-results $FrozenAblationResults `
    --tuned-results $TunedAblationResults `
    --tuned-diagnostics $TunedTrainDiagnostics `
    --results-path "artifacts/stage_b_entry_tune_results.json" `
    --summary-path "artifacts/stage_b_entry_tune_summary.csv" `
    --report-path "notes/stage_b_entry_tune_report.md"

python -m src.analysis.stage_b_entry_tune_output_probe_report `
    --frozen-ablation-results $FrozenAblationResults `
    --tuned-ablation-results $TunedAblationResults `
    --tuned-train-diagnostics $TunedTrainDiagnostics `
    --frozen-probe-results $FrozenOutputProbeResults `
    --tuned-probe-results $TunedProbeResults `
    --diagnostics-path "artifacts/stage_b_entry_tune_diagnostics.json" `
    --results-path "artifacts/stage_b_entry_tune_output_probe_results.json" `
    --summary-path "artifacts/stage_b_entry_tune_output_probe_summary.csv" `
    --report-path "notes/stage_b_entry_tune_output_probe_report.md"
