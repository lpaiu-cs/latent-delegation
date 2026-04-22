param(
    [string]$Config,
    [string]$StageDir,
    [string]$StageResultsPath,
    [string]$StageSummaryPath,
    [string]$DiagnosticsPath,
    [string]$StageReportPath,
    [string]$OutputProbeDir,
    [string]$OutputProbeResultsPath,
    [string]$OutputProbeSummaryPath,
    [string]$OutputProbeReportPath,
    [int[]]$Seeds = @(42)
)

$seedArgs = @()
foreach ($seed in $Seeds) {
    $seedArgs += [string]$seed
}

py -3.12 -m src.v0_6.idea4_static_mixture `
    --config $Config `
    --output-dir $StageDir `
    --results-path $StageResultsPath `
    --summary-path $StageSummaryPath `
    --diagnostics-path $DiagnosticsPath `
    --report-path $StageReportPath `
    --seeds $seedArgs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

py -3.12 -m src.v0_6.idea4_output_probe `
    --config $Config `
    --stage-dir $StageDir `
    --stage-results $StageResultsPath `
    --output-dir $OutputProbeDir `
    --results-path $OutputProbeResultsPath `
    --summary-path $OutputProbeSummaryPath `
    --report-path $OutputProbeReportPath `
    --seeds $seedArgs

exit $LASTEXITCODE
