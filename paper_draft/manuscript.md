# Bounded Same-Family Latent Delegation in Gemma-2: Two-Path Token-Wise Routing Beats Bridge Baselines on Held-Out LM Probes

## Abstract

We study a bounded same-family latent delegation setting in which a frozen large language model keeps ownership of the master residual stream while replacing part of its middle computation with delegated computation from a frozen smaller model plus learned interface modules. The target pair is Gemma-2 9B and Gemma-2 2B under a single RTX 5090-class GPU budget, with frozen backbones throughout.

The starting point is a fixed contiguous substitution baseline. That baseline establishes qualified feasibility: delegated computation improves over skip-only and no-small controls, but not over strong learned bridge baselines that remain entirely in large hidden space. A local asymmetric window search then rejects the fixed contiguous prior and produces a stable two-window shortlist, `24..27 -> 14..19` and `24..27 -> 16..18`. A static two-path mixture over those windows is the first model to beat both bridge controls on KL and NLL, and a low-capacity token-wise gate over the same two paths improves further.

The final token-wise two-path model is the strongest model evaluated here. On the development holdout it reaches KL/NLL `0.255739 / 2.980182`, versus `0.288448 / 3.072051` for `bridge_only`; on an untouched confirmation holdout it reaches `0.248886 / 3.185004`, versus `0.289564 / 3.295081` for `bridge_only`. Follow-up monotone-corridor and sublayer-attribution analyses strengthen the explanation of why the model works but do not produce a better model. A bounded external generalization suite shows real but mixed external validity: the cleanest carryover remains on held-out LM-style evaluation, while multiple-choice gains are selective rather than broad. The strongest defensible claim is therefore a bounded same-family positive result with mixed external generalization, not a broad downstream-superiority claim.

## 1. Introduction

This paper asks a narrow question: can a large same-family model keep the master residual stream and final logits while delegating part of its middle computation to a smaller same-family model through latent-space transfer? The framing is intentionally bounded. We do not claim thought transfer, model equivalence, or broad downstream superiority. We ask whether delegated latent computation can recover useful work under strict practical constraints.

Those constraints are deliberate. The target setting is a single RTX 5090-class GPU, same-family open models only, frozen backbones, and lightweight learned interface modules. The default pair is Gemma-2 9B and Gemma-2 2B. This keeps the experiment interpretable: if the hybrid fails, it fails in a realistic frozen-backbone regime; if it succeeds, the success can be tied to a concrete latent substitution mechanism rather than to large-scale finetuning or distributed training.

The fixed-window baseline gives only a qualified feasibility result. Delegated small-model computation improves over `skip_only` and no-small controls, but not over strong bridge baselines that remain entirely in large hidden space. That leaves a structural question unresolved: does delegation fail in principle, or is the original fixed contiguous layer match simply the wrong prior?

The answer is the main result of the paper. A real-model asymmetric window search rejects the original fixed contiguous substitution and identifies a near-tied two-window shortlist. A static two-path mixture over that shortlist is the first model to beat both bridge controls on KL and NLL, and a low-capacity token-wise gate over the same two paths improves further. The resulting final model beats both bridge controls on the development holdout and again on an untouched confirmation holdout.

This paper is closest in spirit to recent representation-transfer results such as the Linear Representation Transferability Hypothesis and language-model model stitching, but asks a harder operational question: not whether features or steering directions can be mapped between models, but whether a small model can replace a missing block of a larger model's forward computation under frozen same-family constraints. At the same time, recent neural incompatibility results motivate strong bridge controls and a conservative claim boundary rather than an assumption of easy cross-scale interchangeability.

The contributions are:

1. We show that the original fixed contiguous substitution is not the right structural prior; a bounded asymmetric window search identifies a near-tied local shortlist instead.
2. We show that a static two-path mixture and then a low-capacity token-wise two-path routing model can beat strong bridge baselines on held-out LM-style probes in the frozen same-family Gemma-2 9B -> 2B setting.
3. We strengthen the explanation of the result with monotone-corridor analysis and sublayer attribution, and we show that broader external generalization is mixed rather than broad.

## 2. Related Work And Positioning

This project sits between positive results on representation alignment and negative results on cross-scale transfer.

Recent representation-transfer work argues that related models can admit simple affine maps between hidden spaces. The Linear Representation Transferability Hypothesis studies affine maps between models of different scales and shows that transferred directions can preserve useful behavior. Recent language-model stitching work similarly shows that residual-stream features can be transferred across models with lightweight mappings. Our interface design follows that line, but the operational question here is harder: we are not only transferring features or steering vectors, but replacing a missing block of a larger model's forward computation with delegated computation from a smaller one.

Negative results matter just as much. Neural Incompatibility argues that cross-scale parametric knowledge transfer faces a real structural barrier even with alignment. We treat that as a design warning. The system therefore stays within one model family, keeps all backbones frozen, and is evaluated against strong large-space bridge controls rather than against weak skip-only baselines alone.

The same-family choice is grounded in the Gemma 2 technical report, which makes the 2B and 9B models a plausible alignment pair while still leaving room for genuine cross-scale mismatch. Gemma-2 2B and 9B share the same family structure, including RMSNorm, RoPE, grouped-query attention, GeGLU MLP blocks, and the alternating local/global attention pattern.

For bounded external evaluation, we use standard log-likelihood-compatible benchmarks: HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi et al., 2020), ARC (Clark et al., 2018), and LAMBADA (Paperno et al., 2016). Exact source URLs used during repo preparation are collected in [notes/references.md](</E:/lab/latent-delegation/notes/references.md>).

## 3. Setup And Problem Statement

We study one-way latent delegation.

- The large model owns the input path, master residual stream, suffix, and final logits.
- The small model owns delegated latent computation only.
- Communication is through learned latent interfaces, not token re-injection.
- All backbone weights remain frozen.

The default models are:

- large model: `google/gemma-2-9b`
- small model: `google/gemma-2-2b`

The original fixed-window baseline removes large layers `24..29` and replaces them with delegated small layers `14..19`, entered from the hidden state after small layer `13`:

- large prefix: `0..23`
- removed large block: `24..29`
- large suffix: `30..41`
- small reference hidden: after layer `13`
- delegated small block: `14..19`

The scientific question is not whether such a hybrid can merely run. The stronger question is whether delegated small-model computation adds value beyond strong controls:

- `skip_only`
- no-small interface controls
- learned bridge baselines in large hidden space
- parameter-matched bridge baselines

## 4. Method

### 4.1 Single-Path Interface

Let `h_t^L` denote the large-model hidden state at token position `t` after the frozen large prefix. Let `N(.)` denote RMSNorm, `E_p` the entry projector for path `p`, `S_p` the frozen delegated small-model window, and `R_p` the return adapter back into large hidden space.

For each delegated path `p in {A, B}`, the path delta is:

`Delta_{p,t} = R_p(S_p(E_p(N(h_t^L))))`

The hybrid hidden state after the removed large window is:

`h_t^H = h_t^L + Delta_t`

and the frozen large suffix then produces final logits.

### 4.2 Static Two-Path Mixture

The first successful continuation model keeps two delegated paths active in parallel:

- path B: `24..27 -> 14..19`
- path A: `24..27 -> 16..18`

It combines their returned deltas with a learnable global two-logit softmax:

`w = softmax(alpha)`

`Delta_t = w_B * Delta_{B,t} + w_A * Delta_{A,t}`

The matched no-small control keeps the same entry projectors, return adapters, and global mixture weights, but removes the actual delegated small-model computation.

### 4.3 Token-Wise Two-Path Routing

The final model replaces the global mixture with a low-capacity per-token gate. The gate reads only the large-prefix hidden state at the splice boundary. In the confirmed final configuration, the gate is a direct linear head over RMS-normalized prefix states because the configured gate hidden size is `0`:

`g_t = softmax(W_g N(h_t^L) + b_g)`

`Delta_t = g_{t,B} * Delta_{B,t} + g_{t,A} * Delta_{A,t}`

The gate bias is initialized from the learned static-mixture prior. The token-wise no-small control uses the same gate family and interface routes but removes delegated small-model computation.

### 4.4 Training Objectives

Stage A trains only the entry projector. Its exact objective in code is:

`L_A = MSE(E(h^L), h^S_ref) + CosineLoss(E(h^L), h^S_ref)`

where `h^S_ref` is the frozen small-model reference hidden state before the delegated small block.

The decisive training regime is output-aware Stage B. Its implemented objective is:

`L_B = MSE(h^H, h^T) + CosineLoss(h^H, h^T) + 5.0 * L_KL + 1.0 * L_CE + 1e-4 * ||Delta||_2^2`

Here `h^T` is the frozen large-model hidden state after the removed large block, `L_KL` is the teacher-logit KL term, and `L_CE` is shifted next-token cross-entropy. The token-wise gate adds a small prior/stability package:

- entropy penalty weight: `1e-4`
- KL-to-static-prior weight: `1e-3`
- smoothness penalty weight: `0.0`

Stage C is intentionally not used in this paper.

### 4.5 Fairness And Parameter Budgets

Parameter matching is explicit because routing adds capacity beyond a plain large-space bridge:

| model | trainable parameters |
| --- | ---: |
| static two-path mixture | `753666` |
| static two-path mixture no-small | `753666` |
| token-wise two-path routing | `764418` |
| token-wise no-small | `764418` |
| `bridge_only` | `458753` |
| updated parameter-matched bridge | `766977` |

The updated parameter-matched bridge uses rank `107`, which is the closest saved match to the token-wise routing budget.

## 5. Experimental Protocol

### 5.1 Hardware, Seeds, And Workflow

All experiments were run in a native Windows workflow on a single RTX 5090-class GPU using same-family Gemma-2 9B and 2B backbones. The core confirmed result families use seeds `42, 43, 44`. Deterministic evaluation subsets and saved sample IDs are used throughout.

### 5.2 Development Holdout And Untouched Confirmation Holdout

The paper distinguishes two LM-style holdout policies:

- development holdout: the original held-out slice reused during model development and screening
- untouched confirmation holdout: a fresh Wikitext test-split slice sampled only after the winning continuation structure was fixed

The untouched confirmation holdout uses the saved policy:

- dataset: `wikitext`, config `wikitext-103-v1`, split `test`
- sample count: `32`
- sampling seed: `7606`

The main claim should be read through the untouched confirmation holdout first, because it is the stricter check against repeated reuse of the development slice.

### 5.3 Primary Metrics And Ranking

The primary ranking is output-first:

1. KL to the frozen full-large teacher
2. NLL
3. perplexity
4. top-1 agreement
5. top-5 overlap

KL is ranked first because the core research question is whether delegated computation reproduces the functional role of the removed large-model block relative to the frozen large teacher. Hidden-space MSE and cosine remain diagnostic only.

### 5.4 Bounded Generalization

The bounded external generalization suite evaluates frozen baselines on:

- HellaSwag
- PIQA
- WinoGrande
- ARC-Easy
- ARC-Challenge
- LAMBADA OpenAI held-out LM slice

Multiple-choice tasks are scored by conditional answer log-likelihood with a fixed normalization policy. The LM-style external slice uses next-token scoring with KL, NLL, and PPL. Uncertainty is reported with paired bootstrap estimates for token-wise routing versus static mixture, `bridge_only`, and the parameter-matched bridge.

## 6. Results

### 6.1 Fixed-Window Hybrid as a Qualified Feasibility Baseline

The fixed-window baseline establishes that same-family latent delegation is feasible, but not yet sufficient. It beats `skip_only` and `hybrid_no_small`, especially after output-aware Stage B is introduced, but it does not beat the strong bridge baselines. That is the right interpretation of the early result: positive feasibility evidence with a clear structural limit.

### 6.2 Asymmetric Window Search Rejects the Fixed Contiguous Prior

A real-model local window search shows that the original fixed contiguous substitution is structurally wrong. On coarse screening, the legacy `24..29 -> 14..19` candidate reaches KL/NLL `0.725030 / 3.425046`, far worse than the two candidates that later become the shortlist. On 3-seed confirmation, the shortlist remains near-tied:

- `24..27 -> 14..19`: KL/NLL `0.281641 / 3.078029`
- `24..27 -> 16..18`: KL/NLL `0.282215 / 3.074461`

This result matters because it changes the structural prior before any mixture model is introduced.

### 6.3 Static Two-Path Mixture Surpasses Bridge Baselines

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

### 6.4 Token-Wise Two-Path Routing Gives the Best Final Model

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

The gate diagnostics are consistent with real routing rather than trivial collapse:

- path A mean weight: `0.588602`
- path B mean weight: `0.411437`
- gate entropy: `0.602399`
- collapse score: `0.054150`

### 6.5 Explanatory Follow-Ups Strengthen Interpretation Rather Than Replacing the Model

The monotone-corridor analysis recovers a broader low-cost asymmetric region around the successful two-path shortlist. This strengthens the interpretation that the final model is exploiting a local alignment corridor rather than a lucky hard window.

The sublayer-attribution analysis shows that both delegated attention and delegated MLP matter. Suppressing MLP hurts more, but suppressing attention also hurts materially. On the development holdout, attention suppression adds `+0.103797` KL / `+0.218670` NLL, while MLP suppression adds `+0.182743` KL / `+0.350897` NLL. On the untouched confirmation holdout the pattern remains stable at `+0.109496 / +0.224608` for attention suppression and `+0.182872 / +0.379610` for MLP suppression.

### 6.6 Bounded Generalization Is Real but Mixed

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

So the strongest defensible external-validity statement is that the final model is not just a single-slice artifact and retains the clearest advantage on held-out LM-style scoring, not that it broadly dominates strong bridge baselines across downstream tasks.

## 7. Reproducibility And Paper Assets

Canonical generated tables:

- [notes/paper/tables.md](</E:/lab/latent-delegation/notes/paper/tables.md>)
- [artifacts/paper_tables](</E:/lab/latent-delegation/artifacts/paper_tables>)

Canonical figure specs:

- [notes/paper/figures.md](</E:/lab/latent-delegation/notes/paper/figures.md>)
- [artifacts/paper_figures](</E:/lab/latent-delegation/artifacts/paper_figures>)

Reproducibility package:

- [notes/paper/reproducibility.md](</E:/lab/latent-delegation/notes/paper/reproducibility.md>)
- [artifacts/paper_release/repro_manifest.json](</E:/lab/latent-delegation/artifacts/paper_release/repro_manifest.json>)

Versioned artifact names such as `v0.6.0`, `v0_7`, `v0_8`, and `v0_9` are kept in the release package and appendix for reproducibility. The main body uses experimental phase names and model names instead of repo branch history.

## 8. Limitations

This is a bounded systems-and-mechanism result, not a broad benchmark result.

- only one family/pair is carried through the full continuation path
- the strongest evidence remains on LM-style scoring
- broader external generalization is mixed rather than broad
- the no-small comparison is weaker on the untouched confirmation holdout than the bridge comparison
- Stage C is intentionally not started
- the paper does not establish cross-family robustness or universal delegation principles

These limitations are central to the claim boundary, not footnotes.

## 9. Conclusion

The strongest result in this project is a bounded positive one. Same-family latent delegation does not work because a fixed hard layer match happens to be adequate. It works only after the fixed contiguous substitution is rejected, a local asymmetric shortlist is identified, and the delegated computation is reformulated as a low-capacity two-path routing problem.

The resulting final token-wise model beats both strong bridge controls on the development holdout and again on an untouched confirmation holdout in the frozen Gemma-2 9B -> 2B setting. That is stronger than the original fixed-window feasibility result and is the correct main claim for the paper.

At the same time, the paper should remain disciplined. The monotone-corridor and sublayer-attribution branches explain the result better, but they do not replace the final model. The bounded generalization suite shows real external carryover, especially on LM-style evaluation, but the broader multiple-choice picture is mixed rather than broad.

The correct conclusion is therefore narrow and strong at the same time: one-way same-family latent delegation can beat strong bridge baselines inside this frozen-backbone Gemma-2 setting, but the current evidence does not justify a broad downstream-superiority claim outside that regime.

## Appendix A. Paper Asset Index

- [Abstract source](</E:/lab/latent-delegation/notes/paper/abstract.md>)
- [Introduction source](</E:/lab/latent-delegation/notes/paper/introduction.md>)
- [Related-work source](</E:/lab/latent-delegation/notes/paper/related_work.md>)
- [Method source](</E:/lab/latent-delegation/notes/paper/method.md>)
- [Experiments source](</E:/lab/latent-delegation/notes/paper/experiments.md>)
- [Results source](</E:/lab/latent-delegation/notes/paper/results.md>)
- [Limitations source](</E:/lab/latent-delegation/notes/paper/limitations.md>)
- [Conclusion source](</E:/lab/latent-delegation/notes/paper/conclusion.md>)
- [Claim boundary source](</E:/lab/latent-delegation/notes/paper/claim_boundary.md>)
- [Tables bundle](</E:/lab/latent-delegation/notes/paper/tables.md>)
- [Figures bundle](</E:/lab/latent-delegation/notes/paper/figures.md>)
- [Reproducibility package](</E:/lab/latent-delegation/notes/paper/reproducibility.md>)
