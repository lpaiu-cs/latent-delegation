# Phase 1 Combined Decision

## Status

Phase 1A and Phase 1B are implemented and runnable in this repository.

In the current workspace, only the debug-tiny smoke path was executed:

- `artifacts/v0_6/phase1_window_search/pilot_summary.csv`
- `artifacts/v0_6/phase1_window_search/pilot_distinct_small_windows.json`
- `artifacts/v0_6/phase1_stage_signatures/signatures.json`

Real Gemma Phase 1 remains blocked here by the existing `notes/blockers.md` conditions:

- no CUDA target device in this environment
- gated Gemma access unavailable

## Best Candidate Pairs

Debug pilot ranking produced these top output-aware candidates:

1. `24..27 -> 16..18`
2. `24..27 -> 14..19`
3. `24..27 -> 16..21`

Stage-signature matching produced these nearest priors:

- large alternatives near the frozen removed block: `25..30`, `24..27`, `26..31`
- small alternatives near the large reference signature: `12..14`, `10..12`, `10..13`, `11..14`, `10..15`

Operational shortlist for the first real Gemma continuation run:

1. `24..27 -> 16..18`
2. `24..27 -> 14..19`
3. `25..30 -> 10..15`

The first two come from the output-aware pilot harness. The third is a stage-aware exploratory pair to test the mismatch diagnosis directly.

## Decisions

The current `v0.5.1` split should remain as a reference only, not as the sole continuation hypothesis.

Phase 2 should **not** start yet in this blocked environment. The correct next move is a real Gemma Phase 1 run, carrying 2-3 candidate small windows rather than a single frozen `14..19` choice.

Recommended small-window shortlist for the first real confirmation pass:

1. `16..18`
2. `14..19`
3. `10..15`

The evidence already leans toward hard discrete replacement being part of the bottleneck, but only weakly in this workspace. That judgment still rests mainly on the frozen `v0.5.1` bridge gap plus the fact that the debug output-aware ranking and the debug stage-signature prior disagree on the best small-window location. Real Gemma Phase 1 is still required before turning that into a research conclusion.
