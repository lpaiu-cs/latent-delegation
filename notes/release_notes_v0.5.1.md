# Release Notes: v0.5.1

`v0.5.1` is a clarification and freeze milestone. It does not introduce a larger research claim than `v0.5.0`; it closes the main remaining ambiguity from `v0.5.0` and freezes the repo for write-up.

## What Changed Since v0.5.0

- Added optional Stage B entry-projector finetuning with config control:
  - `training.stage_b.train_entry_projector`
  - `training.stage_b.entry_lr`
  - `training.stage_b.return_lr`
  - `training.stage_b.gate_lr`
- Added entry-projector diagnostics:
  - update norm from Stage A initialization
  - gradient norm statistics
  - gate and delta usage summaries
- Ran a focused 3-seed comparison of frozen-entry vs train-entry for:
  - `hybrid`
  - `hybrid_no_small`
- Ran the output probe on the resulting entry-tuned checkpoints.
- Added final freeze outputs:
  - final report
  - release notes
  - abstract
  - one-page summary
  - reproducibility manifest
  - final figures and summary table

## Why This Is Not A Larger Research Leap

The main question after `v0.5.0` was whether the fixed Stage A entry projector was the main reason the hybrid still trailed the strong bridge controls. `v0.5.1` answers that question directly:

- hidden recovery improved when the entry projector was allowed to train during Stage B
- output quality did not improve on the main metrics
- the gap to the bridge controls widened rather than shrinking

So `v0.5.1` sharpens the claim boundary instead of expanding the claim.

## Key Numbers Worth Quoting

From the `v0.5.0` output-aware Stage B output probe:

- `hybrid`: KL `0.6553`, NLL `3.4235`
- `hybrid_no_small`: KL `0.6730`, NLL `3.5018`
- `bridge_only`: KL `0.6463`, NLL `3.3939`
- `bridge_only_param_matched`: KL `0.6471`, NLL `3.3954`

Interpretation:

- `hybrid` beat `skip_only` and `hybrid_no_small`
- `hybrid` did not beat either bridge control

From the `v0.5.1` entry-tune follow-up:

- `hybrid_frozen_entry`: KL `0.6553`, NLL `3.4235`
- `hybrid_train_entry`: KL `0.6686`, NLL `3.4518`
- `hybrid_no_small_frozen_entry`: KL `0.6730`, NLL `3.5018`
- `hybrid_no_small_train_entry`: KL `0.6810`, NLL `3.4862`

Interpretation:

- entry tuning improved hidden metrics
- entry tuning did not improve hybrid output KL/NLL
- tuned `hybrid` still beat tuned `hybrid_no_small`
- entry tuning did not reduce the gap to the bridge controls

## Final Recommendation

Stop experiments and write up the qualified result.

The supported claim is:

> same-family one-way latent delegation is feasible and improves over skip/no-small controls

The unsupported stronger claim is:

> delegated small-model computation is better than strong large-space bridge alternatives
