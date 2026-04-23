# Results

## Fixed-Window Hybrid as a Qualified Feasibility Baseline

The fixed-window hybrid establishes feasibility but not the final claim. In its best output-aware form, it improves over `skip_only` removal and over the no-small interface control, showing that delegated small-model computation is real and functionally used. However, it does not beat the strong bridge baselines. The correct interpretation of this stage is therefore positive feasibility evidence with a clear structural limit.

## Asymmetric Window Search Rejects the Fixed Contiguous Prior

A real-model local window search shows that the original fixed contiguous substitution is the wrong structural prior. The legacy `24..29 -> 14..19` candidate is substantially worse than the two candidates that later become the shortlist. After confirmation, the shortlist remains near-tied:

- `24..27 -> 14..19`: KL/NLL `0.281641 / 3.078029`
- `24..27 -> 16..18`: KL/NLL `0.282215 / 3.074461`

Here `24..27 -> 14..19` is slightly better on KL and top-5 overlap, whereas `24..27 -> 16..18` is slightly better on NLL, perplexity, and top-1 agreement. This result matters because it changes the structural prior before any mixture model is introduced.

## Static Two-Path Mixture Surpasses Bridge Baselines

The static two-path mixture is the first model to beat both bridge controls on the primary output metrics. On the development holdout, it improves over both bridge baselines in KL and NLL, and the same pattern survives on the untouched confirmation holdout. This is the first clean bridge win in the project, which is why it matters more than a simple re-ranking among single-window candidates. The matched no-small mixture also improves over the earlier no-small baselines, but the full static mixture remains stronger, indicating that the gain cannot be reduced to route mixing alone.

## Token-Wise Two-Path Routing Gives the Best Final Model

The token-wise two-path router improves further over the static mixture and remains ahead of both bridge controls on both holdout policies.

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

The bridge comparison is clean: on both the development holdout and the untouched confirmation holdout, the token-wise model wins on the primary metrics in all three seeds against both bridge baselines. The no-small comparison is also favorable in aggregate KL and NLL on both holdouts, but it is slightly less clean on the untouched confirmation holdout, where the joint seed-level KL/NLL win count is `2/3` rather than `3/3`. That caveat is weaker than the bridge result, but it should still be stated explicitly.

The gate diagnostics are consistent with real routing rather than trivial collapse. The gate favors path A on average, retains substantial mass on path B, and remains far from full collapse. The matched no-small control uses the same gate family but is sharper, more collapsed, and materially weaker, which argues against a pure route-capacity explanation.

## Explanatory Follow-Ups

The explanatory follow-ups strengthen interpretation rather than replacing the final model. Monotone-corridor analysis recovers a broader low-cost asymmetric region around the successful two-path shortlist, suggesting that the final model exploits a local alignment corridor rather than a lucky hard window. Sublayer attribution shows that both delegated attention and delegated MLP matter. Suppressing MLP hurts more, but suppressing attention is also materially harmful, so the gain is not well explained by a single-component simplification.

## Bounded Generalization Is Real but Mixed

Bounded external generalization is real but mixed. The cleanest external strength remains on LM-style scoring: on the held-out LAMBADA slice, the token-wise model is favorable against both bridge baselines in KL and favorable against the parameter-matched bridge in NLL, while the NLL comparison against `bridge_only` is positive by point estimate but not cleanly separated under paired uncertainty. The multiple-choice picture is weaker and should be stated that way. HellaSwag and ARC-Challenge are positive by point estimate, but their paired intervals cross zero; WinoGrande is mixed; and PIQA and ARC-Easy are negative relative to the bridge baselines. The strongest defensible external-validity claim is therefore that the final model is not just a single-slice artifact and retains a real LM-style held-out advantage, not that it broadly dominates strong bridge baselines across downstream tasks.
