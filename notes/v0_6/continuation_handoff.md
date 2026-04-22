# v0.6 Continuation Handoff

This repository remains frozen at `v0.5.1` as a qualified feasibility result.

Status note:

- debug shortlist exists from the `debug_tiny` Phase 1 smoke path
- real Gemma Phase 1 is now the next required step
- `v0_6` is a development track, not a release result

The `v0_6` namespace is the continuation track. It exists to test whether the remaining gap to strong bridge baselines is mainly caused by coarse, structurally misaligned block substitution rather than by a failure of same-family delegation itself.

Immediate execution order:

1. Phase 1B: functional-stage signature matching on the real Gemma path
2. Phase 1A: candidate-specific real shortlist screening
3. Phase 1C: combine the two into a short real-only decision note

Scope guardrails for this continuation:

- keep the repo and the Windows-native workflow
- keep the same-family Gemma-2 9B -> 2B default pair
- keep frozen-backbone policy
- do not revise `v0.5.1` claims
- do not start Stage C in this continuation unless a later explicit note says so
- do not move to soft mixtures, monotone alignment, or sublayer substitution until Phase 1 is complete

Primary new entrypoints:

- `configs/v0_6/gemma2_phase1.yaml`
- `configs/v0_6/debug_tiny_phase1.yaml`
- `scripts/v0_6/run_phase1_window_search.ps1`
- `scripts/v0_6/run_phase1_stage_signatures.ps1`
- `scripts/v0_6/run_phase1_debug_smoke.ps1`

Artifact targets:

- `artifacts/v0_6/phase1_window_search/`
- `artifacts/v0_6/phase1_stage_signatures/`

Reports:

- `notes/v0_6/research_plan.md`
- `notes/v0_6/phase1_window_search_report.md`
- `notes/v0_6/phase1_stage_signatures_report.md`
- `notes/v0_6/phase1_combined_decision.md`

Real Phase 1 execution is tracked separately under:

- `notes/v0_6/phase1_real_status.md`
- `notes/v0_6/phase1b_real_stage_signature.md`
- `notes/v0_6/phase1a_real_screening.md`
- `notes/v0_6/phase1_real_combined_decision.md`
