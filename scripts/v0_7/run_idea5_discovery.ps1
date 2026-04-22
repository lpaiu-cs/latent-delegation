$ErrorActionPreference = "Stop"

$config = "configs/v0_7/idea5_discovery.yaml"
$costDir = "artifacts/v0_7/idea5_discovery/costs"
$solverDir = "artifacts/v0_7/idea5_discovery/solver"
$reportPath = "notes/v0_7/idea5_monotone_alignment_report.md"

py -3.12 -m src.v0_7.idea5_costs `
  --config $config `
  --output-dir $costDir

py -3.12 -m src.v0_7.idea5_monotone `
  --config $config `
  --cost-payload "$costDir/cost_payload.json" `
  --output-dir $solverDir `
  --report-path $reportPath

