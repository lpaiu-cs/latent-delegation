# One-Page Summary

## Frozen State

- `v0.6.0` is the frozen current best branch and result.
- `v0_7` and `v0_8` are analysis-only branches.
- `v0_9` is bounded external generalization.
- Stage C was not started.
- The repo is now in paperization mode, not model-building mode.

## Core Question

Can a large same-family model keep the master residual stream while delegating a middle slice of computation to a smaller same-family model through latent-space transfer?

Default pair:

- large: Gemma-2 9B
- small: Gemma-2 2B

## What Changed Since v0.5.1

`v0.5.1` was a qualified feasibility result: the hybrid beat `skip_only` and no-small controls, but not the strong bridge baselines.

The continuation track changed the structural prior:

- Phase 1 rejected the old fixed `24..29 -> 14..19` split as the best default.
- The real shortlist became `{24..27 -> 14..19, 24..27 -> 16..18}`.
- A static two-path mixture was the first result that beat both bridge controls on KL/NLL.
- The token-wise two-path gate improved further and became the frozen best model.

## Current Best Result

The current best model is the `v0.6.0` token-wise Idea 4 hybrid:

- removed large window: `24..27`
- delegated path B: `24..27 -> 14..19`
- delegated path A: `24..27 -> 16..18`
- low-capacity token-wise gate over the two delegated deltas

Original holdout:

- token-wise: KL `0.255739`, NLL `2.980182`
- static mixture: KL `0.267095`, NLL `3.000438`
- `bridge_only`: KL `0.288448`, NLL `3.072051`

Fresh untouched holdout:

- token-wise: KL `0.248886`, NLL `3.185004`
- static mixture: KL `0.267244`, NLL `3.213048`
- `bridge_only`: KL `0.289564`, NLL `3.295081`

So the strongest repo claim is now:

> in the bounded same-family Gemma-2 9B -> 2B setting, a two-path token-wise delegated hybrid beats both bridge controls on the main held-out LM-style probes, and that win survives a fresh untouched holdout

## What Later Branches Mean

`v0_7`:

- strengthened the monotone corridor story
- explained why the shortlisted windows worked
- did not produce a bounded empirical candidate that beat `v0.6.0`

`v0_8`:

- showed that both delegated attention and delegated MLP matter
- MLP suppression hurts more than attention suppression
- did not justify a clean attention-only or MLP-only replacement model

Those branches are scientifically useful, but they do not replace the `v0.6.0` best-model claim.

## Broader Evaluation

`v0_9` asked whether `v0.6.0` generalizes outside the original Wikitext-style probe regime.

Outcome:

- yes on bounded LM-style evaluation
- mixed on multiple-choice tasks
- not strong enough for a broad “better than bridge baselines everywhere” claim

Examples:

- LAMBADA OpenAI: token-wise KL/NLL `0.251354 / 3.423984`, better than both bridge controls
- ARC-Challenge: token-wise accuracy `0.442708`, slightly above both bridge controls
- PIQA and ARC-Easy: bridge baselines recover clearly

## Final Claim Boundary

Supported:

- one-way same-family latent delegation is real under the repo’s single-GPU constraints
- delegated small-model computation can beat skip/no-small controls
- the frozen `v0.6.0` token-wise model beats both bridge controls on the main held-out LM-style probes

Not supported:

- broad downstream superiority
- broad multiple-choice generalization
- Stage C justification
- thought-transfer framing

## Final Recommendation

Stop and write the paper around `v0.6.0` plus bounded generalization.
