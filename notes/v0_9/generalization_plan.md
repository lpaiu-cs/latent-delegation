# Generalization Plan

## Goal

- Evaluate the external validity of the frozen `v0.6.0` token-wise Idea 4 model.
- Keep this task evaluation-only unless a benchmark format forces a tiny compatibility step.

## Frozen Comparison Set

1. `v0.6.0` token-wise Idea 4
2. `v0.6.0` static mixture
3. token-wise no-small control
4. `bridge_only`
5. parameter-matched bridge
6. `skip_only` as a sanity floor only

## Benchmark Set

- HellaSwag
- PIQA
- WinoGrande
- ARC-Easy
- ARC-Challenge
- LAMBADA OpenAI test slice for LM-style evaluation

## Scoring Policy

- Multiple-choice tasks use conditional answer-option log-likelihood.
- Choice ranking uses length-normalized continuation log-prob so longer answers are not penalized purely by token count.
- LM evaluation uses deterministic held-out slices with NLL, PPL, and KL to the full large teacher where meaningful.

## Reporting Requirements

- Save per-example predictions and raw scores.
- Save exact sampled IDs and seeds.
- Compute paired deltas between token-wise and the key baselines.
- Add simple paired bootstrap uncertainty for:
  - token-wise vs static mixture
  - token-wise vs `bridge_only`
  - token-wise vs parameter-matched bridge

## Stop / Go Rule

- Proceed to bounded cross-pair replication only if token-wise remains better than both bridge controls on a clear majority of the new tasks or evaluation families.
- Otherwise stop and write the paper around the current result plus this broader benchmark check.
