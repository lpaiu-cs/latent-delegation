# Idea 2 Research Plan

## Fixed Facts From v0.6.0

- The legacy contiguous split is no longer the best structural default.
- The successful local path family is the validated two-path shortlist:
  - `24..27 -> 14..19`
  - `24..27 -> 16..18`
- Static mixture was the first branch to beat both bridge controls.
- Token-wise Idea 4 improved further and is the frozen current best result.

## Working Question

- Does the `v0.6.0` token-wise gain come mainly from delegated attention, delegated MLP, or their interaction?
- If whole-block delegated substitution is still too coarse, Idea 2 should show that through selective sublayer suppression before any new model is trained.

## Discovery Scope

1. Load the existing `v0.6.0` token-wise checkpoints.
2. Suppress delegated attention or delegated MLP inside the frozen small delegated block while keeping the same gate and splice context.
3. Measure degradation on both the original holdout and the fresh untouched holdout.
4. Compare against the token-wise no-small control and the bridge controls.
5. Only if the attribution is clearly directional, consider one bounded sublayer-specific variant.

## Stop Rules

- Stop before new model-building if both subcomponents look necessary or the attribution is unstable.
- Stop after one bounded sublayer variant if it is not at least competitive with full `v0.6.0` on KL and NLL.

