$ErrorActionPreference = "Stop"

$config = "configs/v0_8/idea2_attribution.yaml"
$outputDir = "artifacts/v0_8/idea2_attribution"
$resultsPath = "$outputDir/results.json"
$summaryPath = "$outputDir/summary.csv"
$reportPath = "notes/v0_8/idea2_attribution_report.md"
$decisionPath = "notes/v0_8/idea2_combined_decision.md"

py -3.12 -m src.v0_8.idea2_attribution `
  --config $config `
  --output-dir $outputDir `
  --results-path $resultsPath `
  --summary-path $summaryPath `
  --report-path $reportPath `
  --combined-decision-path $decisionPath

