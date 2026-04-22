# Idea 4 Token-Wise Combined Decision

1. Does the static mixture win survive on the fresh untouched holdout?
Yes. On the untouched `wikitext-103-v1` test slice, the static mixture still beat both bridge controls on KL and NLL in all 3 seeds. The aggregate deltas were `-0.022320 / -0.082033` versus `bridge_only` and `-0.017189 / -0.049553` versus the mixture-budget parameter-matched bridge.

2. Does token-wise gating beat static mixture on KL/NLL?
Yes. It beat static mixture on KL and NLL in all 3 seeds on the original holdout and again in all 3 seeds on the untouched holdout. The aggregate deltas were `-0.011356 / -0.020256` on the original holdout and `-0.018358 / -0.028044` on the untouched holdout.

3. Does token-wise gating beat its no-small control?
Yes, with one caveat. On the original holdout it beat the token-wise no-small control on KL and NLL in all 3 seeds. On the untouched holdout the aggregate KL and NLL still favored the hybrid, but the joint primary-metric win count was `2/3` rather than `3/3`.

4. Does token-wise gating beat the bridge controls on the original holdout?
Yes. It beat both `bridge_only` and the updated parameter-matched bridge on KL and NLL in all 3 seeds. The aggregate deltas were `-0.032709 / -0.091869` and `-0.046584 / -0.121899`.

5. Does token-wise gating beat the bridge controls on the fresh untouched holdout?
Yes. It again beat both bridge controls on KL and NLL in all 3 seeds. The aggregate deltas were `-0.040677 / -0.110077` and `-0.052859 / -0.142020`.

6. Do the gate diagnostics suggest real path specialization, or just noisy extra capacity?
They suggest real path specialization rather than noisy extra capacity. The token-wise gate stays far from full collapse (`collapse_score = 0.054150`), keeps moderate entropy (`0.602399`), shifts mean usage toward path A without eliminating path B (`0.588602 / 0.411437`), and shows stable per-token variance. The matched no-small control uses the same gate family but is sharper, more collapsed, and materially weaker on both hidden recovery and output metrics.

7. Is Idea 4 now exhausted?
Yes. Static mixture already established the two-path structural claim, and the minimal token-wise gate now adds a reproducible output-level gain over static mixture while preserving wins over the bridge controls on both holdout policies. More Idea 4 gate elaboration would be lower-signal than moving to the next research branch.

Proceed to Idea 5
