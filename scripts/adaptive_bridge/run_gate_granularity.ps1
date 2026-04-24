param(
    [string]$Config = "configs/adaptive_bridge/gemma2_three_seed_replication.yaml",
    [string]$TrainDir = "outputs/adaptive_bridge/real_seed42_43_44_warm_start/train",
    [string]$OutputDir = "outputs/adaptive_bridge/gate_granularity",
    [string]$ResultsPath = "outputs/adaptive_bridge/gate_granularity/results.json"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"

py -3.12 -m src.adaptive_bridge.gate_granularity `
    --config $Config `
    --train-dir $TrainDir `
    --output-dir $OutputDir `
    --results-path $ResultsPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
