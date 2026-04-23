# Results

## Fixed-Window Hybrid as a Qualified Feasibility Baseline

The fixed-window baseline establishes that same-family latent delegation is feasible, but not yet sufficient. It beats `skip_only` and `hybrid_no_small`, especially after output-aware Stage B is introduced, but it does not beat the strong bridge baselines. That is the right interpretation of the early result: positive feasibility evidence with a clear structural limit.

## Asymmetric Window Search Rejects the Fixed Contiguous Prior

A real-model local window search shows that the original fixed contiguous substitution is structurally wrong. On coarse screening, the legacy `24..29 -> 14..19` candidate reaches KL/NLL `0.725030 / 3.425046`, far worse than the two candidates that later become the shortlist. On 3-seed confirmation, the shortlist remains near-tied:

- `24..27 -> 14..19`: KL/NLL `0.281641 / 3.078029`
- `24..27 -> 16..18`: KL/NLL `0.282215 / 3.074461`

This result matters because it changes the structural prior before any mixture model is introduced.

## Static Two-Path Mixture Surpasses Bridge Baselines

The static two-path mixture is the first model to beat both bridge controls on the main output metrics.

On the development holdout:

- static mixture: KL/NLL `0.267095 / 3.000438`
- `bridge_only`: `0.288448 / 3.072051`
- parameter-matched bridge: `0.283258 / 3.045527`

On the untouched confirmation holdout:

- static mixture: KL/NLL `0.267244 / 3.213048`
- `bridge_only`: `0.289564 / 3.295081`
- parameter-matched bridge: `0.284433 / 3.262601`

This is the first clean bridge win in the project, which is why it matters more than a simple single-window re-ranking.

## Token-Wise Two-Path Routing Gives the Best Final Model

The token-wise model improves further over the static mixture and remains ahead of both bridge controls on both holdout policies.

Untouched confirmation holdout:

- token-wise routing: KL/NLL `0.248886 / 3.185004`
- static mixture: `0.267244 / 3.213048`
- `bridge_only`: `0.289564 / 3.295081`
- parameter-matched bridge: `0.301746 / 3.327024`

Development holdout:

- token-wise routing: KL/NLL `0.255739 / 2.980182`
- static mixture: `0.267095 / 3.000438`
- `bridge_only`: `0.288448 / 3.072051`
- parameter-matched bridge: `0.302323 / 3.102081`

The bridge comparison is clean: token-wise routing wins on the primary metrics in all `3/3` seeds on both holdouts against both bridge controls. The no-small comparison is also favorable in aggregate KL/NLL on both holdouts, but it is slightly less clean on the untouched confirmation holdout: the token-wise model wins the joint primary-metric seed comparison `2/3` rather than `3/3`. That caveat is weaker than the bridge result but should still be stated explicitly.

## Explanatory Follow-Ups Strengthen Interpretation Rather Than Replacing the Model

The monotone-corridor analysis recovers a broader low-cost asymmetric region around the successful two-path shortlist. This strengthens the interpretation that the final model is exploiting a local alignment corridor rather than a lucky hard window.

The sublayer-attribution analysis shows that both delegated attention and delegated MLP matter. Suppressing MLP hurts more, but suppressing attention also hurts materially. On the development holdout, attention suppression adds `+0.103797` KL / `+0.218670` NLL, while MLP suppression adds `+0.182743` KL / `+0.350897` NLL. On the untouched confirmation holdout the pattern remains stable at `+0.109496 / +0.224608` for attention suppression and `+0.182872 / +0.379610` for MLP suppression.

## Bounded Generalization Is Real but Mixed

The bounded external suite supports a narrower claim than the internal holdouts do.

The cleanest external strength remains on LM-style scoring. On LAMBADA OpenAI:

- token-wise routing: KL/NLL `0.251354 / 3.423984`
- `bridge_only`: `0.254975 / 3.433371`
- parameter-matched bridge: `0.266066 / 3.446407`

Under paired bootstrap, the KL improvement over both bridge controls is favorable, and the NLL improvement is clearly separated against the parameter-matched bridge. The NLL comparison against `bridge_only` is favorable by point estimate but not cleanly separated under the paired interval.

The multiple-choice picture is mixed and should be stated that way:

- HellaSwag and ARC-Challenge are positive by point estimate, but paired uncertainty intervals cross zero.
- WinoGrande is mixed.
- PIQA and ARC-Easy are negative relative to the bridge baselines.

So the strongest defensible external-validity claim is that the final model is not just a single-slice artifact and retains the clearest advantage on held-out LM-style scoring, not that it broadly dominates strong bridge baselines across downstream tasks.
