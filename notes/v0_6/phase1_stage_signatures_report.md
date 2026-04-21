# Phase 1B Stage Signature Report

- Config: `configs/v0_6/debug_tiny_phase1.yaml`
- Backbones: `debug_tiny`
- Signature metrics: hidden norm, hidden drift norm, hidden drift cosine, logit-lens entropy, and KL to final logits.

## Closest Large Windows

- `25..30` (len=6) distance=0.137518
- `24..27` (len=4) distance=0.223419
- `26..31` (len=6) distance=0.243333
- `26..28` (len=3) distance=0.389708
- `28..31` (len=4) distance=0.473216

## Closest Small Windows

- `12..14` (len=3) distance=10.947673
- `10..12` (len=3) distance=11.027409
- `10..13` (len=4) distance=11.036743
- `11..14` (len=4) distance=11.084898
- `10..15` (len=6) distance=11.098575

## Answers

1. Large windows functionally closest to the current removed block: 25..30, 24..27, 26..31, 26..28, 28..31
2. Small windows functionally closest to the removed large block: 12..14, 10..12, 10..13, 11..14, 10..15
3. Does stage-aware matching suggest the current 6 -> 6 split is probably wrong? Yes.
4. Does stage-aware matching suggest asymmetric mapping? Yes.

## Caveat

- This report was generated on the debug-tiny path in the current environment. It validates the phase workflow only and should not be read as a Gemma research conclusion.
- Real Gemma continuation runs remain blocked by the documented gated-model and no-CUDA environment issues in `notes/blockers.md`.