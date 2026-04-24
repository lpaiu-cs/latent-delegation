param(
    [string]$Config = "configs/adaptive_bridge/gemma2_first_milestone.yaml",
    [string]$OutputDir = "outputs/adaptive_bridge/train"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"

py -3.12 -m src.adaptive_bridge.train `
    --config $Config `
    --output-dir $OutputDir `
    --results-path "$OutputDir/results.json" `
    --summary-path "$OutputDir/summary.csv" `
    --diagnostics-path "$OutputDir/diagnostics.json"
