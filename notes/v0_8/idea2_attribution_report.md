# Idea 2 Attribution Report

## Scope

- Subject model: frozen `v0.6.0` token-wise Idea 4 checkpoint family.
- Attribution method: explicit suppression of delegated attention residuals and delegated MLP residuals inside the frozen small delegated block.
- Holdout policies: original `main_validation` and the `fresh_untouched` slice from `v0.6.0`.

## Main Holdout

- tokenwise_full: KL=0.255739, NLL=2.980182, top1=0.763351, top5=0.744268
- attn_suppressed delta from full: dKL=0.103797, dNLL=0.218670
- mlp_suppressed delta from full: dKL=0.182743, dNLL=0.350897
- both_suppressed delta from full: dKL=0.147248, dNLL=0.332872

## Fresh Holdout

- tokenwise_full: KL=0.248886, NLL=3.185004, top1=0.758319, top5=0.742734
- attn_suppressed delta from full: dKL=0.109496, dNLL=0.224608
- mlp_suppressed delta from full: dKL=0.182872, dNLL=0.379610
- both_suppressed delta from full: dKL=0.154198, dNLL=0.379402

## Path-Specific Deltas

- main holdout attention suppression: path B dKL=0.036265, path A dKL=0.060497
- main holdout MLP suppression: path B dKL=0.049808, path A dKL=0.100613
- fresh holdout attention suppression: path B dKL=0.052240, path A dKL=0.058770
- fresh holdout MLP suppression: path B dKL=0.068471, path A dKL=0.089358

## Interpretation

1. Is delegated attention necessary for the current `v0.6.0` gain?
Yes. Suppressing delegated attention raises KL by `+0.103797` and NLL by `+0.218670` on the main holdout, and by `+0.109496` / `+0.224608` on the fresh untouched holdout. That degradation is materially larger than the gap between the full token-wise model and the route-only no-small control.

2. Is delegated MLP necessary for the current `v0.6.0` gain?
Yes. Suppressing delegated MLP is even more damaging: `+0.182743` KL and `+0.350897` NLL on the main holdout, and `+0.182872` / `+0.379610` on the fresh untouched holdout.

3. Does one subcomponent dominate, or are both needed?
Both are needed, but the delegated MLP is the more important contributor under the primary output metrics. The MLP ablation is consistently worse than the attention ablation on both holdouts, while the attention ablation is still far from negligible.

4. Is the answer stable across both holdout policies?
Yes. The ranking `full < attention-suppressed < MLP-suppressed` by KL/NLL is stable across the original holdout and the fresh untouched holdout. The magnitude is also similar across the two policies.

5. Does the answer differ by path A vs path B?
Yes. Path A (`24..27 -> 16..18`) is the more sensitive route. On both holdouts, suppressing either attention or MLP on path A hurts more than suppressing the same subcomponent on path B, with the largest penalty coming from path A MLP suppression.

## Decision For This Task

- The attribution is informative, but it does not justify a bounded attention-only or MLP-only model in this task.
- Reason: the signal is directional toward MLP, but attention is also clearly necessary, so a single-component delegated variant would start from a structurally weakened prior.
- Result: stop Idea 2 at attribution for now and preserve `v0.6.0` as the current best branch.
