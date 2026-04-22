# Idea 5 Research Plan

## Fixed Facts From v0.6.0

- The legacy contiguous `24..29 -> 14..19` split is rejected as the best default.
- Real Phase 1 narrowed the continuation set to `{24..27 -> 14..19, 24..27 -> 16..18}`.
- Static mixture is the first result that beat both bridge controls on KL and NLL, and the bridge win survived a fresh untouched holdout.
- Token-wise Idea 4 improved further and is now the strongest branch so far.

## Working Hypothesis

- The successful two-path token-wise gate is not just averaging two arbitrary windows.
- Instead, it may be exploiting a broader monotone but asymmetric alignment corridor between local large-model and small-model layer regions.
- If that story is correct, a monotone discovery tool should recover the shortlisted region without being explicitly forced to do so.

## Discovery Scope

1. Build local pairwise window costs from the real Phase 1B stage-signature artifact.
2. Combine stage-signature, hidden-alignment, and logit-disruption proxies into one monotone-search signal.
3. Solve for the top low-cost order-preserving paths with asymmetric moves.
4. Compare the discovered paths against the validated Idea 4 shortlist.
5. Only if the structure is clear enough, derive one minimal Idea 5 candidate for a later empirical check.

## Stop Rules

- Stop before model-building if the monotone corridor does not naturally recover the shortlisted region.
- Stop before model-building if the solver only reproduces the old hard-window story without additional structure.
- Even if the discovery signal is positive, keep any model-building follow-up bounded to one minimal candidate.
