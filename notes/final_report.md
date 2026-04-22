# Final Report

## Frozen Repo State

- `v0.6.0` is the frozen current best model/result in this repo.
- `v0_7` and `v0_8` remain analysis-only branches and do not replace the `v0.6.0` best-model claim.
- `v0_9` is a bounded external generalization branch and does not introduce a new best model.
- Stage C was not started.
- The correct next step is paperization, not more model-building.

## Objective

This project asked a narrow feasibility question:

Can a large same-family model keep ownership of the master residual stream while delegating a middle slice of computation to a smaller same-family model through latent-space transfer?

The default pair was:

- `google/gemma-2-9b`
- `google/gemma-2-2b`

The intended claim was never full thought transfer or broad downstream superiority. The intended claim was that a frozen large model might recover useful computation by routing a middle block through a frozen smaller same-family model plus learned interface modules.

## Constraints

- one RTX 5090-class GPU
- same-family open models only
- frozen backbones
- train only small interface modules
- 4-bit frozen backbones when practical
- no multi-GPU, no FSDP, no DeepSpeed, no LoRA by default
- no Stage C unless earlier evidence justified it

## v0.5.x Baseline Story

The `v0.5.x` line established the qualified feasibility result but not the final repo best model.

What worked:

- real-hardware Gemma bring-up succeeded on the target Windows machine
- the smoke matrix passed `14/14` cases
- Stage A alignment stabilized
- hidden-only Stage B beat `skip_only` and `hybrid_no_small`
- output-aware Stage B produced an output-level gain over `skip_only` and `hybrid_no_small`

What failed:

- the hybrid still did not beat the strong bridge baselines
- entry-projector finetuning improved hidden metrics but worsened KL/NLL relative to the frozen-entry output-aware baseline

So `v0.5.1` remained a qualified feasibility result, not the final best branch.

## Phase 1 Continuation

The `v0_6` continuation asked whether the old fixed contiguous `24..29 -> 14..19` substitution was simply the wrong structural prior.

Real Gemma Phase 1 rejected that legacy split as the best default.

The confirmed shortlist was exactly:

- `24..27 -> 14..19`
- `24..27 -> 16..18`

On the confirmation pilot:

- `24..27 -> 14..19`: KL `0.281641`, NLL `3.078029`
- `24..27 -> 16..18`: KL `0.282215`, NLL `3.074461`

The old legacy candidate was much worse even on coarse screening:

- `24..29 -> 14..19`: KL `0.725030`, NLL `3.425046`

That was the point where the repo stopped treating the original fixed `6 -> 6` split as the right default structural prior.

## v0.6.0 Best Result

The strongest result in the repo is the frozen token-wise Idea 4 model.

Architecture:

- removed large window fixed to `24..27`
- two delegated paths:
  - path B: `24..27 -> 14..19`
  - path A: `24..27 -> 16..18`
- frozen large and small backbones
- learnable low-capacity token-wise gate over the two delegated deltas

Intermediate step:

- the static two-path mixture was the first pilot result that beat both bridge controls on KL/NLL
- that bridge win survived a fresh untouched holdout slice

Final token-wise `v0.6.0` results:

Original holdout:

- `tokenwise_mixture`: KL `0.255739`, NLL `2.980182`, PPL `19.763760`
- `static_mixture`: KL `0.267095`, NLL `3.000438`, PPL `20.156769`
- `bridge_only`: KL `0.288448`, NLL `3.072051`, PPL `21.673345`
- `bridge_only_param_matched`: KL `0.302323`, NLL `3.102081`, PPL `22.330668`

Fresh untouched holdout:

- `tokenwise_mixture`: KL `0.248886`, NLL `3.185004`, PPL `24.167632`
- `static_mixture`: KL `0.267244`, NLL `3.213048`, PPL `24.854807`
- `bridge_only`: KL `0.289564`, NLL `3.295081`, PPL `26.979976`
- `bridge_only_param_matched`: KL `0.301746`, NLL `3.327024`, PPL `27.855381`

Pairwise deltas for `tokenwise_mixture`:

- versus `static_mixture`, original holdout: KL `-0.011356`, NLL `-0.020256`
- versus `static_mixture`, fresh holdout: KL `-0.018358`, NLL `-0.028044`
- versus `bridge_only`, original holdout: KL `-0.032709`, NLL `-0.091869`
- versus `bridge_only`, fresh holdout: KL `-0.040677`, NLL `-0.110077`
- versus `bridge_only_param_matched`, original holdout: KL `-0.046584`, NLL `-0.121899`
- versus `bridge_only_param_matched`, fresh holdout: KL `-0.052859`, NLL `-0.142020`

Interpretation:

- the strongest result is not just “delegation beats skip”
- it is specifically that a bounded two-path token-wise delegated hybrid beats the strongest bridge controls on the repo’s main held-out LM-style probes, and that this win survives an untouched holdout recheck

## Analysis Branches Only

### v0_7: Idea 5

Idea 5 asked whether the successful two-path shortlist was a local sample from a broader monotone asymmetric cross-scale corridor.

That discovery branch succeeded analytically:

- it recovered a monotone local corridor around the successful splice region
- it strengthened the explanation for why the two-path shortlist worked
- it made the old legacy fixed split look structurally implausible in a more principled way

But its bounded empirical candidate `24..27 -> 15..18` was not competitive enough with `v0.6.0`.

So `v0_7` remains analysis-only.

### v0_8: Idea 2

Idea 2 asked whether the `v0.6.0` gain came mainly from delegated attention, delegated MLP, or both.

Result:

- suppressing delegated attention hurts materially
- suppressing delegated MLP hurts more
- both are clearly needed
- the pattern is stable across the original and fresh holdouts

Main-holdout degradation relative to the full token-wise model:

- attention suppressed: `+0.103797` KL, `+0.218670` NLL
- MLP suppressed: `+0.182743` KL, `+0.350897` NLL

Fresh-holdout degradation:

- attention suppressed: `+0.109496` KL, `+0.224608` NLL
- MLP suppressed: `+0.182872` KL, `+0.379610` NLL

That was directional but not clean enough to justify a single attention-only or MLP-only architecture branch, so `v0_8` also remains analysis-only.

## Bounded Generalization

The `v0_9` branch evaluated the frozen `v0.6.0` family of baselines outside the original Wikitext-style probes.

Benchmarks:

- HellaSwag
- PIQA
- WinoGrande
- ARC-Easy
- ARC-Challenge
- LAMBADA OpenAI held-out LM slice

Main result:

- external validity is real but mixed
- the strongest carryover remains on LM-style evaluation
- the multiple-choice pattern is not broad enough to support a strong “better than bridge controls across tasks” claim

Selected generalization results:

- LAMBADA OpenAI, token-wise: KL `0.251354`, NLL `3.423984`
- LAMBADA OpenAI, `bridge_only`: KL `0.254975`, NLL `3.433371`
- LAMBADA OpenAI, `bridge_only_param_matched`: KL `0.266066`, NLL `3.446407`
- HellaSwag accuracy: token-wise `0.671875`, `bridge_only` `0.656250`
- ARC-Challenge accuracy: token-wise `0.442708`, `bridge_only` `0.432292`
- PIQA accuracy: token-wise `0.723958`, `bridge_only` `0.734375`
- ARC-Easy accuracy: token-wise `0.791667`, `bridge_only` `0.828125`

Recommendation from `v0_9`:

Stop and write the paper around `v0.6.0` plus benchmark generalization.

## Supported Claims

- Same-family one-way latent delegation is feasible on a single RTX 5090-class GPU.
- Delegated small-model computation improves over skip/no-small controls.
- In the bounded Gemma-2 9B -> 2B setting, the frozen `v0.6.0` token-wise hybrid beats strong bridge controls on the main held-out LM-style probes.
- That bridge win survives an untouched fresh holdout slice.
- Bounded external generalization exists, with the clearest carryover on LM-style evaluation.

## Unsupported Claims

- This work does not support a broad claim of downstream superiority over bridge baselines.
- This work does not support a claim that token-wise delegation is uniformly better across external task families.
- This work does not support cross-family robustness, broad replication, or Stage C scaling.
- This work does not support a full thought-transfer framing.

## Final Interpretation

The repo no longer ends at the `v0.5.1` qualified feasibility result. The continuation work produced a stronger bounded result: a two-path token-wise delegated hybrid that beats both bridge controls on the main held-out LM-style probes in the same-family Gemma-2 setting.

At the same time, the broader evaluation remains mixed. So the strongest defensible paper framing is:

- strong positive evidence inside the bounded Gemma-2 setting
- mixed but nonzero broader generalization
- careful refusal to claim broad downstream superiority

## Paperization Pointers

Paper prose sources:

- [abstract.md](</E:/lab/latent-delegation/notes/paper/abstract.md>)
- [introduction.md](</E:/lab/latent-delegation/notes/paper/introduction.md>)
- [method.md](</E:/lab/latent-delegation/notes/paper/method.md>)
- [experiments.md](</E:/lab/latent-delegation/notes/paper/experiments.md>)
- [results.md](</E:/lab/latent-delegation/notes/paper/results.md>)
- [limitations.md](</E:/lab/latent-delegation/notes/paper/limitations.md>)
- [conclusion.md](</E:/lab/latent-delegation/notes/paper/conclusion.md>)
- [claim_boundary.md](</E:/lab/latent-delegation/notes/paper/claim_boundary.md>)

Generated paper assets:

- [tables.md](</E:/lab/latent-delegation/notes/paper/tables.md>)
- [figures.md](</E:/lab/latent-delegation/notes/paper/figures.md>)
- [artifacts/paper_tables](</E:/lab/latent-delegation/artifacts/paper_tables>)
- [artifacts/paper_figures](</E:/lab/latent-delegation/artifacts/paper_figures>)
