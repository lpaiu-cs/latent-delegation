# Results

## 1. v0.5.x Established Feasibility But Not the Final Best Result

The original project line ended at a qualified feasibility result. Bring-up succeeded, Stage A alignment stabilized, and output-aware Stage B established an output-level gain over `skip_only` and `hybrid_no_small`. But the hybrid still did not beat the strongest bridge baselines. Entry-projector finetuning improved hidden recovery while worsening KL/NLL relative to the frozen-entry output-aware baseline. That is the correct interpretation of `v0.5.1`: positive feasibility evidence with a hard limit, not the final repo claim.

## 2. Phase 1 Rejected the Legacy Fixed Split

The continuation work tested whether the old fixed contiguous `24..29 -> 14..19` substitution was structurally wrong. Real Gemma Phase 1 showed that it was. On coarse screening, the legacy candidate posted KL/NLL `0.725030 / 3.425046`, far worse than the two local asymmetric candidates that later became the shortlist. On 3-seed confirmation, the shortlist remained near-tied:

- `24..27 -> 14..19`: KL/NLL `0.281641 / 3.078029`
- `24..27 -> 16..18`: KL/NLL `0.282215 / 3.074461`

That result is scientifically important even before the mixture models: the old fixed `6 -> 6` prior was not the right structural default.

## 3. Static Mixture Was the First Clean Bridge Win

The static two-path mixture was the first pilot result to beat both bridge controls on KL and NLL. On the original holdout it reached KL/NLL `0.267095 / 3.000438`, compared with `0.288448 / 3.072051` for `bridge_only` and `0.283258 / 3.045527` for the parameter-matched bridge. That win survived the fresh untouched holdout as well, with KL/NLL `0.267244 / 3.213048`, versus `0.289564 / 3.295081` and `0.284433 / 3.262601` for the bridge controls.

This matters because it separates two different conclusions:

- the successful result is not just “pick a better single hard window”
- a local structural mixture over the real shortlist is already better than either keeping the old hard window or staying purely in large-space bridge mode

## 4. Token-Wise Gating Produced the Frozen v0.6.0 Best Result

The token-wise two-path gate improved further over the static mixture.

Original holdout:

- token-wise: KL/NLL `0.255739 / 2.980182`
- static mixture: `0.267095 / 3.000438`
- `bridge_only`: `0.288448 / 3.072051`
- parameter-matched bridge: `0.302323 / 3.102081`

Fresh untouched holdout:

- token-wise: KL/NLL `0.248886 / 3.185004`
- static mixture: `0.267244 / 3.213048`
- `bridge_only`: `0.289564 / 3.295081`
- parameter-matched bridge: `0.301746 / 3.327024`

The corresponding deltas are stable and directionally clean:

- versus static mixture: `-0.011356 / -0.020256` on the original holdout and `-0.018358 / -0.028044` on the fresh holdout
- versus `bridge_only`: `-0.032709 / -0.091869` on the original holdout and `-0.040677 / -0.110077` on the fresh holdout
- versus parameter-matched bridge: `-0.046584 / -0.121899` on the original holdout and `-0.052859 / -0.142020` on the fresh holdout

This is why `v0.6.0` is the frozen current best result.

## 5. Idea 5 and Idea 2 Added Explanation, Not a Better Model

Idea 5 discovery recovered a local monotone asymmetric corridor around the successful splice region and made the shortlist easier to explain mechanistically. But its one bounded derived candidate did not beat `v0.6.0`. So the branch is analytically valuable without changing the best-model claim.

Idea 2 attribution showed that both delegated attention and delegated MLP matter. The degradation is larger when MLP is suppressed, but attention suppression is also clearly harmful. That makes the result scientifically stronger by ruling out a trivial explanation, but it does not justify replacing the full token-wise model with a narrower attention-only or MLP-only variant.

## 6. Broader Evaluation Is Real but Mixed

The bounded `v0_9` generalization suite shows that `v0.6.0` is not just a Wikitext artifact, but it also shows that the strongest claim must remain narrow.

The clearest external strength is on the held-out LM family:

- LAMBADA OpenAI, token-wise: KL/NLL `0.251354 / 3.423984`
- `bridge_only`: `0.254975 / 3.433371`
- parameter-matched bridge: `0.266066 / 3.446407`

The multiple-choice picture is mixed:

- HellaSwag and ARC-Challenge are favorable by point estimate
- WinoGrande is split
- PIQA and ARC-Easy are negative relative to the bridge baselines

So the strongest defensible external-validity statement is that the frozen token-wise model retains a real held-out LM advantage and stays competitive on part of a bounded commonsense benchmark set, not that it broadly dominates strong bridge baselines across tasks.
