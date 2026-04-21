param(
  [string]$Config = "configs/v0_6/gemma2_phase1.yaml",
  [string]$OutputDir = "artifacts/v0_6/phase1_stage_signatures",
  [string]$ReportPath = "notes/v0_6/phase1_stage_signatures_report.md"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"

py -3.12 -m src.v0_6.stage_signatures `
  --config $Config `
  --output-dir $OutputDir `
  --report-path $ReportPath

