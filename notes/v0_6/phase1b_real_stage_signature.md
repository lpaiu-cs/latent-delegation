# Phase 1B Real Stage Signature

## Run

- runner: Windows-native PowerShell path via `scripts/v0_6/run_phase1_stage_signatures.ps1`
- config: `configs/v0_6/gemma2_phase1.yaml`
- artifact root: `artifacts/v0_6/phase1_real/stage_signature/`
- scope: real Gemma 2B/9B only; no broad search expansion
- signature metrics: hidden norm, hidden drift norm, hidden drift cosine, logit-lens entropy, and KL to final logits

## Closest Large Windows

- `25..28` (len=4) distance=`0.077011`
- `24..28` (len=5) distance=`0.078716`
- `25..29` (len=5) distance=`0.096068`
- `23..30` (len=8) distance=`0.100344`
- `25..27` (len=3) distance=`0.141753`

## Closest Small Windows

- `16..17` (len=2) distance=`1.238411`
- `15..17` (len=3) distance=`1.353851`
- `15..19` (len=5) distance=`1.412366`
- `15..18` (len=4) distance=`1.417906`
- `16..18` (len=3) distance=`1.418965`

## Interpretation

- Real Gemma again rejects the frozen contiguous `24..29 -> 14..19` split as the best structural default.
- `24..27 -> 16..18` remains directly supported by the small-window signature ranking.
- `24..27 -> 14..19` stayed in the screening set as a conservative local perturbation of the legacy split, not because the small-window signatures preferred `14..19`.
- The debug exploratory candidate `25..30 -> 10..15` was not locally supported on the real path. The large side moved toward `25..29`, while the small side concentrated around `15..19` and `16..18`.
- I used the one allowed swap and replaced `25..30 -> 10..15` with the nearby conservative candidate `25..29 -> 15..19`.
- The screened set remained fixed at 4 total candidates.

## Screened Set

1. legacy baseline: `24..29 -> 14..19`
2. candidate A: `24..27 -> 16..18`
3. candidate B: `24..27 -> 14..19`
4. candidate C: `25..29 -> 15..19`
