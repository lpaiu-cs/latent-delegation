# Generalization Handoff

## Fixed Facts From v0.6.0

- `v0.6.0` is the frozen current best result.
- The frozen best model is the real Gemma token-wise Idea 4 mixture over the validated two-path shortlist:
  - `24..27 -> 14..19`
  - `24..27 -> 16..18`
- On the original held-out output probe and the fresh untouched holdout, the token-wise model beat:
  - static mixture
  - token-wise no-small overall
  - `bridge_only`
  - the parameter-matched bridge

## Branch Boundaries

- `v0_7` remains a discovery/analysis branch only.
- `v0_7` strengthened the monotone asymmetric corridor explanation, but its bounded empirical candidate did not beat `v0.6.0`.
- `v0_8` remains a discovery/analysis branch only.
- `v0_8` showed that both delegated attention and delegated MLP matter for the `v0.6.0` gain, with MLP contributing more, but not strongly enough to justify sublayer-only model-building.

## v0_9 Question

- The next question is generalization, not architecture.
- We need to test whether the frozen `v0.6.0` token-wise model remains better than its strongest controls outside the original Wikitext-style probe regime.

## Benchmark Set In This Task

- Multiple-choice log-likelihood tasks:
  - HellaSwag
  - PIQA
  - WinoGrande
  - ARC-Easy
  - ARC-Challenge
- Additional held-out LM-style evaluation:
  - LAMBADA OpenAI test slice (`EleutherAI/lambada_openai`, `test`)

## Decision Rule

- Recommend bounded cross-pair replication only if the token-wise model remains better than both bridge controls on a clear majority of the new tasks or evaluation families.
- Otherwise stop and write the paper around `v0.6.0` plus benchmark generalization, without claiming broad generality.
