param(
    [string]$Config = "configs/adaptive_bridge/debug_tiny.yaml",
    [string]$TrainDir = "outputs/adaptive_bridge/debug_train",
    [string]$EvalDir = "outputs/adaptive_bridge/debug_eval"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"

py -3.12 -m src.adaptive_bridge.train `
    --config $Config `
    --output-dir $TrainDir `
    --results-path "$TrainDir/results.json" `
    --summary-path "$TrainDir/summary.csv" `
    --diagnostics-path "$TrainDir/diagnostics.json"

py -3.12 -m src.adaptive_bridge.evaluate `
    --config $Config `
    --train-dir $TrainDir `
    --output-dir $EvalDir `
    --results-path "$EvalDir/results.json" `
    --summary-path "$EvalDir/summary.csv" `
    --report-path "$EvalDir/summary_note.md"
