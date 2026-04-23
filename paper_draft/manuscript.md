# Bounded Same-Family Latent Delegation in Gemma-2: Two-Path Token-Wise Routing Beats Bridge Baselines on Held-Out LM Probes

## Abstract

We study whether a large language model can preserve its master residual stream while delegating part of its intermediate computation to a smaller model from the same architecture family, without converting intermediate states back into text tokens. We instantiate this question on Gemma-2 9B and Gemma-2 2B under a strict single-GPU budget and freeze both backbones throughout. An initial fixed contiguous substitution provides only a qualified feasibility result: delegated computation improves over skip-only and no-small controls, but not over strong learned bridge baselines in large hidden space. A continuation branch shows that the legacy fixed split is the wrong structural prior. Real-model Phase 1 rejects the old window and identifies a near-tied shortlist of local asymmetric delegated windows. A static mixture over those shortlisted paths becomes the first pilot result to beat the strong bridge controls on KL and NLL, and a low-capacity token-wise gate improves further. The resulting frozen `v0.6.0` model beats skip-only, no-small, and both bridge baselines on the original held-out LM-style probe and again on a fresh untouched holdout slice. Later Idea 5 and Idea 2 branches strengthen the explanation of why the model works but do not replace the best-model claim. A bounded generalization suite shows that the signal carries beyond the original Wikitext-style probes, especially on held-out LM-style evaluation, but multiple-choice transfer is mixed rather than broad. The strongest defensible claim is therefore a bounded same-family positive result with mixed external generalization, not a broad downstream-superiority claim.

## 1. Introduction

Most multi-model language-model collaboration happens through text. One model emits tokens, another consumes them, and the boundary between the two systems is linguistically legible but computationally expensive. A stricter alternative is latent delegation: a large model keeps ownership of the master hidden state and final logits while delegating part of its internal computation directly in hidden space to a smaller model.

This idea is only plausible if the two models are structurally compatible enough for hidden-space transfer to mean something. That is why this project deliberately stays in a same-family regime. The default pair is Gemma-2 9B as the large model and Gemma-2 2B as the small model. The system is also intentionally constrained: one RTX 5090-class GPU, frozen backbones, lightweight learned interface modules only, and no Stage C unless earlier evidence already justifies going that far.

The project began with a conservative fixed-window substitution and produced a qualified feasibility result. Delegated small-model computation improved over `skip_only` and no-small controls, but not over strong bridge baselines that stayed entirely in large hidden space. That left a structural question unresolved: did delegation fail in principle, or was the original hard window match simply the wrong prior?

The continuation answer is the main result of the repo. Real-model Phase 1 rejected the legacy fixed split as the best default and identified a near-tied two-window shortlist. A static mixture over those windows was the first result to beat both bridge controls on KL and NLL, and a low-capacity token-wise gate improved further. That token-wise model is now frozen as `v0.6.0`, the current best result in the repo.

This manuscript therefore makes a bounded claim. It does not claim thought transfer, broad downstream superiority, or cross-family generality. It claims that in a constrained same-family Gemma-2 setting, structural mismatch can be handled well enough for a token-wise two-path delegated hybrid to beat strong bridge baselines on held-out LM-style probes, while broader external generalization remains mixed rather than broad.

## 2. Setup And Problem Statement

We study one-way latent delegation.

- The large model owns the input path, master residual stream, suffix, and final logits.
- The small model owns delegated latent computation only.
- Communication is through learned latent interfaces, not token re-injection.
- All backbone weights remain frozen.

The default models are:

- large model: `google/gemma-2-9b`
- small model: `google/gemma-2-2b`

The original conservative split was:

- large prefix: layers `0..23`
- removed large middle block: layers `24..29`
- large suffix: layers `30..41`
- small reference hidden: after layer `13`
- delegated small block: layers `14..19`

The scientific question is not merely whether such a hybrid can run. The stronger question is whether delegated small-model computation adds value beyond strong controls:

- `skip_only`
- no-small interface controls
- learned bridge baselines in large hidden space
- parameter-matched bridge baselines

## 3. Method

### 3.1 Base Interface

The base hybrid path is:

1. run the frozen large prefix
2. map the large hidden state into small latent space with an entry projector
3. run a frozen delegated small-model window
4. map the delegated result back into large space with a return adapter
5. apply a learned gate to the returned delta
6. continue through the frozen large suffix and large LM head

### 3.2 Training Stages

Stage A trains the entry projector so that the large hidden state aligns with the small-family latent space. Stage B first exists in a hidden-only form and later in an output-aware form that directly includes teacher-logit KL and next-token CE/NLL. Stage C is intentionally not started in this paper because the decisive structural questions are answerable inside the Stage B regime.

### 3.3 Controls

The control family is central rather than peripheral:

- `skip_only`: remove the large middle block and continue directly
- `hybrid_no_small`: preserve the interface scaffold but remove actual delegated computation
- `bridge_only`: learn a replacement entirely in large space
- `bridge_only_param_matched`: match the trainable budget of the delegated system as closely as practical

These controls separate three claims:

1. delegation beats skipping
2. delegation beats an interface-only route
3. delegation beats a strong large-space alternative with similar trainable capacity

## 4. From v0.5.1 to v0.6.0

### 4.1 v0.5.1 Was a Qualified Feasibility Result

The original repo line succeeded at a bounded feasibility question. Real Gemma bring-up worked on the target Windows machine, the smoke matrix passed `14/14`, Stage A stabilized, and output-aware Stage B produced a clear output-level win over `skip_only` and `hybrid_no_small`.

But it did not beat the bridge controls. Entry-projector finetuning improved hidden recovery while worsening KL/NLL relative to the frozen-entry output-aware baseline. So `v0.5.1` remained a qualified feasibility result, not the final best model.

### 4.2 Phase 1 Rejected the Legacy Structural Prior

The continuation branch asked whether the old hard split `24..29 -> 14..19` was simply structurally wrong. Real Gemma Phase 1 showed that it was.

The confirmed shortlist became:

- path B: `24..27 -> 14..19`
- path A: `24..27 -> 16..18`

On confirmation:

- `24..27 -> 14..19`: KL/NLL `0.281641 / 3.078029`
- `24..27 -> 16..18`: KL/NLL `0.282215 / 3.074461`

The legacy split was much worse even on coarse screening:

- `24..29 -> 14..19`: KL/NLL `0.725030 / 3.425046`

So the old fixed contiguous substitution is no longer the default structural story for the project.

### 4.3 Static Mixture Was the First Clean Bridge Win

The first successful continuation model was a static two-path mixture over the shortlisted windows. Both delegated paths were kept active and their large-space deltas were combined with a learnable global softmax mixture.

That model was already scientifically important. On the original holdout it achieved:

- static mixture: KL/NLL `0.267095 / 3.000438`
- `bridge_only`: `0.288448 / 3.072051`
- parameter-matched bridge: `0.283258 / 3.045527`

On the fresh untouched holdout:

- static mixture: KL/NLL `0.267244 / 3.213048`
- `bridge_only`: `0.289564 / 3.295081`
- parameter-matched bridge: `0.284433 / 3.262601`

This was the first point where delegation beat the strongest bridge controls rather than merely skip/no-small controls.

### 4.4 Token-Wise Mixture Became the Frozen Best Result

The final `v0.6.0` model replaced the static global mixture with a low-capacity token-wise gate over the same two delegated paths.

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

The token-wise model improves over static mixture on both holdouts and stays ahead of both bridge controls on KL and NLL. That is why `v0.6.0` is the canonical result for this repo.

## 5. Evaluation Protocol

### 5.1 Primary Metrics

The primary ranking is output-first:

1. KL to the full large teacher
2. NLL
3. perplexity
4. top-1 agreement
5. top-5 overlap

Hidden-space MSE and cosine are used diagnostically, not as final ranking criteria.

### 5.2 Holdout Policies

The repo uses two main held-out LM-style probe policies:

- original main holdout
- fresh untouched holdout slice

The fresh slice exists to avoid treating the reused validation policy as the only external check.

### 5.3 Bounded External Generalization

The bounded generalization branch evaluates frozen `v0.6.0` baselines on:

- HellaSwag
- PIQA
- WinoGrande
- ARC-Easy
- ARC-Challenge
- LAMBADA OpenAI held-out LM slice

This branch is evaluation-only. It does not train a new model and it does not change the best-model claim.

## 6. Main Results

### 6.1 Strong Internal Result

The strongest internal result is a bounded positive one:

> in the same-family Gemma-2 9B -> 2B setting, a two-path token-wise delegated hybrid beats both strong bridge baselines on the main held-out LM-style probes, and that win survives a fresh untouched holdout slice

The clearest numerical deltas are:

- versus static mixture, original holdout: KL `-0.011356`, NLL `-0.020256`
- versus static mixture, fresh holdout: KL `-0.018358`, NLL `-0.028044`
- versus `bridge_only`, original holdout: KL `-0.032709`, NLL `-0.091869`
- versus `bridge_only`, fresh holdout: KL `-0.040677`, NLL `-0.110077`
- versus parameter-matched bridge, original holdout: KL `-0.046584`, NLL `-0.121899`
- versus parameter-matched bridge, fresh holdout: KL `-0.052859`, NLL `-0.142020`

### 6.2 Why the Result Is Not Just Extra Routing Capacity

The matched no-small controls matter. Static mixture beats static no-small, and token-wise mixture beats token-wise no-small overall. The successful token-wise gate also does not collapse:

- path A mean weight: `0.588602`
- path B mean weight: `0.411437`
- gate entropy: `0.602399`
- collapse score: `0.054150`

That is consistent with real path specialization rather than a trivial extra-capacity story.

### 6.3 Idea 5 and Idea 2 Are Explanatory Follow-Ups

Idea 5 discovery recovered a broader local monotone asymmetric corridor around the successful region and made the shortlist easier to explain, but its bounded empirical candidate did not beat `v0.6.0`.

Idea 2 attribution showed that both delegated attention and delegated MLP matter, with larger degradation when MLP is suppressed:

- main holdout, attention suppressed: `+0.103797` KL, `+0.218670` NLL
- main holdout, MLP suppressed: `+0.182743` KL, `+0.350897` NLL
- fresh holdout, attention suppressed: `+0.109496` KL, `+0.224608` NLL
- fresh holdout, MLP suppressed: `+0.182872` KL, `+0.379610` NLL

These branches strengthen the explanation of the main result, but they do not produce a better model than `v0.6.0`.

### 6.4 Bounded Generalization Is Mixed

The `v0_9` generalization suite gives a mixed answer.

The clearest external strength remains on LM-style evaluation:

- LAMBADA OpenAI, token-wise: KL/NLL `0.251354 / 3.423984`
- `bridge_only`: `0.254975 / 3.433371`
- parameter-matched bridge: `0.266066 / 3.446407`

The multiple-choice picture is selective rather than broad:

- HellaSwag: token-wise accuracy `0.671875`, above both bridge point estimates
- ARC-Challenge: token-wise accuracy `0.442708`, slightly above both bridge point estimates
- WinoGrande: mixed
- PIQA and ARC-Easy: bridge baselines recover clearly

So the strongest external-validity claim is not broad superiority. It is that the token-wise result is not just a single-slice artifact, while broader external gains remain mixed.

## 7. Tables, Figures, And Reproducibility

Canonical generated tables:

- [Table bundle](</E:/lab/latent-delegation/notes/paper/tables.md>)
- [Machine-readable tables](</E:/lab/latent-delegation/artifacts/paper_tables>)

Canonical figure specs:

- [Figure bundle](</E:/lab/latent-delegation/notes/paper/figures.md>)
- [Machine-readable figure specs](</E:/lab/latent-delegation/artifacts/paper_figures>)

Reproducibility package:

- [Paper reproducibility note](</E:/lab/latent-delegation/notes/paper/reproducibility.md>)
- [Reproducibility manifest](</E:/lab/latent-delegation/artifacts/paper_release/repro_manifest.json>)

The reproducibility package fixes:

- final git commit hash
- artifact roots
- multi-seed policy
- exact held-out slice definition files
- exact benchmark sample ID files
- Windows-native reproduction commands

## 8. Limitations

This is a bounded systems-and-mechanism result, not a broad benchmark result.

- only one family/pair is carried through the full continuation path
- the strongest evidence remains on held-out LM-style probes
- external generalization is mixed rather than broad
- Stage C was intentionally not started
- the paper does not establish cross-family robustness or universal delegation principles

These limitations are central to the claim boundary, not footnotes.

## 9. Conclusion

The repo’s strongest result is a bounded positive one. Same-family latent delegation did not succeed because a fixed hard layer match happened to work. It succeeded only after the project rejected the old fixed substitution window, moved to a real shortlist of asymmetric local windows, and then replaced a hard choice with a low-capacity two-path token-wise mixture.

That frozen `v0.6.0` model beats both strong bridge baselines on the main held-out LM-style probes and again on a fresh untouched holdout. This is stronger than the original `v0.5.1` feasibility story and is the correct canonical result for the paper.

At the same time, bounded external generalization is mixed. The right paper framing is therefore:

- strong internal result
- mixed external generalization
- no broad superiority claim
- Idea 5 and Idea 2 treated as explanatory follow-up branches, not as new best models

## Appendix A. Paper Asset Index

- [Abstract source](</E:/lab/latent-delegation/notes/paper/abstract.md>)
- [Introduction source](</E:/lab/latent-delegation/notes/paper/introduction.md>)
- [Method source](</E:/lab/latent-delegation/notes/paper/method.md>)
- [Experiments source](</E:/lab/latent-delegation/notes/paper/experiments.md>)
- [Results source](</E:/lab/latent-delegation/notes/paper/results.md>)
- [Limitations source](</E:/lab/latent-delegation/notes/paper/limitations.md>)
- [Conclusion source](</E:/lab/latent-delegation/notes/paper/conclusion.md>)
- [Claim boundary source](</E:/lab/latent-delegation/notes/paper/claim_boundary.md>)
