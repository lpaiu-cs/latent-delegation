# Claim Boundary

## Supported Claims

- A same-family one-way latent delegation system between Gemma-2 9B and Gemma-2 2B is feasible on a single RTX 5090-class GPU with frozen backbones.
- Delegated small-model computation improves over `skip_only` and no-small controls.
- The final two-path token-wise hybrid beats both strong bridge controls on the development holdout and on the untouched confirmation holdout.
- The strongest external carryover appears on held-out LM-style evaluation.
- Monotone-corridor analysis strengthens the interpretation that the successful shortlist lies in a broader local asymmetric alignment corridor.
- Sublayer attribution shows that both delegated attention and delegated MLP matter, with larger degradation when MLP is suppressed.

## Unsupported Claims

- Delegated small-model computation is broadly better than strong bridge baselines across downstream tasks.
- The token-wise model generalizes cleanly across multiple-choice reasoning benchmarks.
- The result is already robust across model families, scales, or tokenizer families.
- Hidden-space improvements alone are sufficient evidence of output-level improvement.
- The paper demonstrates full thought transfer, model equivalence, or a general-purpose delegation architecture.
- Stage C is justified by the current evidence.

## Exact Wording To Prefer

Preferred abstract wording:

> We report a bounded positive result in the same-family Gemma-2 9B -> 2B setting: a two-path token-wise delegated hybrid beats strong bridge controls on held-out LM-style probes, and that win survives an untouched confirmation holdout.

Preferred introduction wording:

> The goal is not to claim thought transfer or broad downstream superiority, but to test whether delegated same-family latent computation can recover useful work under frozen-backbone, single-GPU constraints.

Preferred evaluation wording:

> The development holdout is the reused model-selection slice; the untouched confirmation holdout is the stricter external check and should be treated as the primary confirmation surface.

Preferred results wording:

> The fixed contiguous substitution is the wrong structural prior; once the delegated computation is reformulated as a bounded two-path routing problem, the hybrid becomes better than the bridge controls on the main LM-style probes.

Preferred limitation wording:

> Broader external validity is mixed rather than broad, so the evidence supports a strong in-setting claim, not a general downstream-superiority claim.

Preferred conclusion wording:

> The strongest defensible conclusion is a bounded one: same-family one-way latent delegation can beat strong bridge baselines inside this frozen-backbone Gemma-2 setting, but the current evidence does not justify broad claims beyond that regime.
