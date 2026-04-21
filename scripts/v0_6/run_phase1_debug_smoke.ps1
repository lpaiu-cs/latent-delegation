$ErrorActionPreference = "Stop"

powershell -ExecutionPolicy Bypass -File .\scripts\v0_6\run_phase1_window_search.ps1 `
  -Config "configs/v0_6/debug_tiny_phase1.yaml" `
  -Mode "pilot" `
  -OutputDir "artifacts/v0_6/phase1_window_search" `
  -ReportPath "notes/v0_6/phase1_window_search_report.md"

powershell -ExecutionPolicy Bypass -File .\scripts\v0_6\run_phase1_stage_signatures.ps1 `
  -Config "configs/v0_6/debug_tiny_phase1.yaml" `
  -OutputDir "artifacts/v0_6/phase1_stage_signatures" `
  -ReportPath "notes/v0_6/phase1_stage_signatures_report.md"
