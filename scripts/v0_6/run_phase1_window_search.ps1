param(
  [string]$Config = "configs/v0_6/gemma2_phase1.yaml",
  [ValidateSet("pilot", "confirm")]
  [string]$Mode = "pilot",
  [string]$OutputDir = "artifacts/v0_6/phase1_window_search",
  [string]$ReportPath = "notes/v0_6/phase1_window_search_report.md",
  [string]$ShortlistPath = ""
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"

$argsList = @(
  "-m", "src.v0_6.phase1_window_search",
  "--config", $Config,
  "--mode", $Mode,
  "--output-dir", $OutputDir,
  "--report-path", $ReportPath
)

if ($ShortlistPath -ne "") {
  $argsList += @("--shortlist-path", $ShortlistPath)
}

py -3.12 @argsList

