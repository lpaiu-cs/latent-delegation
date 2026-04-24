param(
    [string]$Config = "configs/adaptive_bridge/gemma2_first_milestone.yaml",
    [string]$TrainDir = "outputs/adaptive_bridge/train",
    [string]$OutputDir = "outputs/adaptive_bridge/eval"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"

py -3.12 -m src.adaptive_bridge.evaluate `
    --config $Config `
    --train-dir $TrainDir `
    --output-dir $OutputDir `
    --results-path "$OutputDir/results.json" `
    --summary-path "$OutputDir/summary.csv" `
    --report-path "$OutputDir/summary_note.md"
