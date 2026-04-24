param(
    [string]$Config = "configs/adaptive_bridge/gemma2_eval_scaleup.yaml",
    [string]$TrainDir = "outputs/adaptive_bridge/real_seed42_43_44_warm_start/train",
    [string]$OutputDir = "outputs/adaptive_bridge/eval_scaleup",
    [string]$ResultsPath = "outputs/adaptive_bridge/eval_scaleup/results.json",
    [string]$SummaryPath = "outputs/adaptive_bridge/eval_scaleup/summary.csv",
    [string]$ReportPath = "outputs/adaptive_bridge/eval_scaleup/summary_note.md",
    [string]$UncertaintyPath = "outputs/adaptive_bridge/eval_scaleup/paired_uncertainty.json",
    [int]$BootstrapSamples = 4000
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"

py -3.12 -m src.adaptive_bridge.evaluate `
    --config $Config `
    --train-dir $TrainDir `
    --output-dir $OutputDir `
    --results-path $ResultsPath `
    --summary-path $SummaryPath `
    --report-path $ReportPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

py -3.12 -m src.adaptive_bridge.eval_hardening `
    --config $Config `
    --train-dir $TrainDir `
    --eval-dir $OutputDir `
    --output-path $UncertaintyPath `
    --bootstrap-samples $BootstrapSamples `
    --skip-expert-usage

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
