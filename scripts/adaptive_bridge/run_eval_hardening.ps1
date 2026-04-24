param(
    [string]$Config = "configs/adaptive_bridge/gemma2_three_seed_replication.yaml",
    [string]$TrainDir = "outputs/adaptive_bridge/real_seed42_43_44_warm_start/train",
    [string]$EvalDir = "outputs/adaptive_bridge/real_seed42_43_44_warm_start/eval",
    [string]$OutputPath = "outputs/adaptive_bridge/real_seed42_43_44_warm_start/eval/paired_uncertainty.json",
    [int]$BootstrapSamples = 4000
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"

py -3.12 -m src.adaptive_bridge.eval_hardening `
    --config $Config `
    --train-dir $TrainDir `
    --eval-dir $EvalDir `
    --output-path $OutputPath `
    --bootstrap-samples $BootstrapSamples

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
