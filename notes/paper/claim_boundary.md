# Claim Boundary

## Supported Claims

- A same-family one-way latent delegation system between Gemma-2 9B and Gemma-2 2B is feasible on a single RTX 5090-class GPU with frozen backbones.
- Delegated small-model computation improves over `skip_only` and no-small controls.
- The frozen `v0.6.0` token-wise two-path hybrid beats both strong bridge controls on the repo’s main held-out LM-style probes.
- That bridge win survives a fresh untouched holdout slice.
- Bounded external generalization exists, with the clearest carryover on held-out LM-style evaluation.
- Idea 5 discovery strengthens the interpretation that the successful windows sit inside a broader local monotone asymmetric corridor.
- Idea 2 attribution shows that both delegated attention and delegated MLP matter, with larger degradation when MLP is suppressed.

## Unsupported Claims

- Delegated small-model computation is broadly better than strong bridge baselines across downstream tasks.
- The token-wise model generalizes cleanly across multiple-choice reasoning benchmarks.
- The result is already robust across model families, scales, or tokenizer families.
- Hidden-space improvements alone are sufficient evidence of output-level improvement.
- The repo demonstrates full thought transfer, model equivalence, or a general-purpose delegation architecture.
- Stage C is justified by the current evidence.

## Exact Wording To Prefer In Abstract, Introduction, And Conclusion

Preferred abstract wording:

> We report a bounded positive result in the same-family Gemma-2 9B -> 2B setting: a frozen two-path token-wise delegated hybrid beats strong bridge controls on held-out LM-style probes, and that win survives a fresh untouched holdout slice.

Preferred introduction wording:

> The goal is not to claim thought transfer or broad downstream superiority, but to test whether delegated same-family latent computation can recover useful work under frozen-backbone, single-GPU constraints.

Preferred results wording:

> The continuation work shows that the original fixed contiguous substitution window was the wrong structural prior; once the delegated path is reformulated as a bounded two-path token-wise mixture, the hybrid becomes better than the bridge controls on the repo’s main held-out LM-style probes.

Preferred limitation wording:

> Broader external validity is mixed rather than broad, so the evidence supports a strong in-setting claim, not a general downstream-superiority claim.

Preferred conclusion wording:

> The strongest defensible conclusion is a bounded one: same-family one-way latent delegation can beat strong bridge baselines inside this frozen-backbone Gemma-2 setting, but the current evidence does not justify broad claims beyond that regime.
