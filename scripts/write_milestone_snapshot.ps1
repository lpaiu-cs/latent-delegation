param(
    [string]$Config = "configs/gemma2_conservative_pilot_256.yaml",
    [string]$AuditPath = "artifacts/milestone_parameter_audit.json",
    [string]$ReportPath = "notes/real_hardware_report.md"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

python -m src.tools.write_milestone_snapshot `
    --config $Config `
    --audit-path $AuditPath `
    --report-path $ReportPath
