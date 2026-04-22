# Idea 4 Combined Decision

## Milestone

- static mixture is the first pilot result that beats both bridge controls on KL/NLL
- this is strong enough to justify continuation
- but a fresh untouched holdout is now required for stricter confidence

1. Does the static two-path mixture beat the best single-path candidate?
Yes. On the 3-seed confirmation run, the static mixture beat the best single-path reference (`24..27 -> 14..19` by mean KL) on both KL and NLL in all 3 seeds. The aggregate deltas were dKL `-0.014546` and dNLL `-0.077591`.

2. Does the static two-path mixture beat its no-small control?
Yes. The static mixture beat the two-path no-small control on both KL and NLL in all 3 seeds. The aggregate deltas were dKL `-0.002231` and dNLL `-0.068552`.

3. Does the static two-path mixture materially reduce the gap to the strong bridge controls?
Yes. Within this pilot it does more than reduce the gap: it beats both `bridge_only` and the mixture-budget parameter-matched bridge on KL and NLL in all 3 seeds. The aggregate deltas were `-0.021352 / -0.071613` versus `bridge_only` and `-0.016163 / -0.045089` versus the mixture-budget parameter-matched bridge.

4. Is the gain large enough to justify adding a token-wise mixture gate?
Yes. The rule for continuation was that the static mixture must beat the no-small control and be at least competitive with the best single-path candidate on the main output metrics. It clears that bar comfortably, and it also beats the bridge controls at this budget.

5. What is the strongest defensible claim after Idea 4 static mixture?
The strongest defensible claim is that the near-tied Phase 1 shortlist was a real structural signal: combining the two shortlisted delegated windows with one global static mixture recovers more output-level performance than either hard single-window choice, and the gain is not explained by route mixing alone or by simply adding a larger large-space bridge.

Proceed to token-wise Idea 4 mixture gate
