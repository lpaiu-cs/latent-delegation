param(
    [string]$Config = "configs/adaptive_bridge/gemma2_three_seed_replication.yaml",
    [string]$OutputDir = "outputs/adaptive_bridge/real_seed42_43_44_warm_start/train",
    [string]$Seed42SourceDir = "outputs/adaptive_bridge/real_seed42_warm_start/train/seed_42",
    [int[]]$Seeds = @(43, 44)
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"

$seed42Destination = Join-Path $OutputDir "seed_42"
if (-not (Test-Path -LiteralPath $seed42Destination)) {
    if (-not (Test-Path -LiteralPath $Seed42SourceDir)) {
        throw "Missing seed_42 trained checkpoints at $Seed42SourceDir"
    }
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Copy-Item -LiteralPath $Seed42SourceDir -Destination $OutputDir -Recurse -Force
}

$requiredWarmStarts = @(42) + $Seeds
foreach ($Seed in $requiredWarmStarts) {
    $warmStartPath = "artifacts/v0_6/idea4_tokenwise/confirm/stage_b/seed_$Seed/tokenwise_mixture_checkpoint.pt"
    if (-not (Test-Path -LiteralPath $warmStartPath)) {
        throw "Missing frozen warm-start checkpoint at $warmStartPath"
    }
}

$seedArgs = @()
foreach ($Seed in $Seeds) {
    $seedArgs += $Seed.ToString()
}

py -3.12 -m src.adaptive_bridge.train `
    --config $Config `
    --seeds $seedArgs `
    --output-dir $OutputDir `
    --results-path "$OutputDir/results.json" `
    --summary-path "$OutputDir/summary.csv" `
    --diagnostics-path "$OutputDir/diagnostics.json"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
