$ErrorActionPreference = "Stop"

$config = "configs/v0_7/idea5_candidate_24_27_to_15_18.yaml"
$outputDir = "artifacts/v0_7/idea5_discovery/empirical_check"
$reportPath = "notes/v0_7/idea5_empirical_check.md"

py -3.12 -m src.v0_6.phase1_window_search `
  --config $config `
  --mode pilot `
  --output-dir $outputDir `
  --report-path $reportPath
