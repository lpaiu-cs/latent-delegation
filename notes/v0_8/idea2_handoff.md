# Idea 2 Handoff

- `v0.6.0` remains the frozen current best result.
- `v0.6.0` token-wise Idea 4 beats static mixture, the token-wise no-small control overall, and both bridge controls on both the main holdout and the fresh untouched holdout.
- Idea 5 discovery strengthened the monotone asymmetric corridor story, but its bounded empirical candidate did not beat `v0.6.0`.
- Idea 5 stops here as a discovery-only branch.

## Rationale

- The next uncertainty is no longer which local window family to use.
- The next uncertainty is whether the `v0.6.0` gain comes primarily from delegated attention, delegated MLP, or the interaction of both.
- Idea 2 therefore starts as a sublayer-attribution track before any new architecture is trained.

## Guardrails

- Keep `v0.6.0` frozen as the current best release.
- Keep Idea 5 frozen as discovery only.
- Use the existing `v0.6.0` token-wise checkpoint family as the subject.
- Do not start Stage C.
- Do not resume Idea 5 model-building.
- Do not widen benchmarks.
- Only consider one bounded sublayer variant if attribution is clearly directional.

