# Bounded Same-Family Latent Delegation in Gemma-2: Asymmetric Two-Path Routing Surpasses Bridge Baselines on Held-Out Language-Model Probes

## Abstract

We study a bounded same-family latent delegation setting in which a frozen large language model keeps ownership of the master residual stream and final logits while replacing part of its middle computation with delegated computation from a frozen smaller model plus learned interface modules. We instantiate this setting with Gemma-2 9B and Gemma-2 2B under a single RTX 5090-class GPU budget and frozen backbones throughout.

A fixed contiguous delegation baseline provides only a qualified feasibility result: it improves over skip-only and no-small controls, but not over strong learned bridge baselines that remain entirely in large hidden space. A local asymmetric window search then rejects the fixed contiguous prior and yields a stable two-window shortlist. A static two-path mixture over that shortlist is the first model to beat both bridge controls on teacher KL and held-out NLL, and a low-capacity token-wise router over the same two paths improves further.

The final token-wise model beats both bridge controls on a development holdout and again on an untouched confirmation holdout. Follow-up monotone-corridor and sublayer-attribution analyses strengthen the explanation of why the model works but do not produce a better model. A bounded external generalization suite shows real but mixed external validity: the cleanest carryover remains on held-out LM-style scoring, while multiple-choice gains are selective rather than broad. The strongest defensible claim is therefore a bounded same-family positive result with mixed external generalization, not a broad downstream-superiority claim.

## 1. Introduction

This paper asks a narrow question: can a large same-family model preserve the master residual stream and final logits while delegating part of its middle computation to a smaller same-family model through latent-space transfer? The framing is intentionally bounded. We do not claim thought transfer, model equivalence, or broad downstream superiority. We ask whether delegated latent computation can recover useful work under strict practical constraints.

Those constraints are deliberate. We study a single RTX 5090-class GPU setting with frozen Gemma-2 9B and Gemma-2 2B backbones, learned interface modules only, and no large-scale finetuning or distributed training. This makes the result interpretable. If the hybrid fails, it fails in a realistic frozen-backbone regime; if it succeeds, the success can be tied to a concrete delegated-computation mechanism rather than to broad adaptation capacity.

Our main finding is that the original fixed contiguous substitution is the wrong structural prior. A real-model asymmetric window search produces a near-tied two-window shortlist, and a static two-path mixture over that shortlist is the first model to beat strong large-space bridge baselines on held-out LM-style probes. A low-capacity token-wise router over the same two paths improves further and gives the best final model. The bridge win survives not only on a development holdout but also on an untouched confirmation holdout.

At the same time, the paper remains deliberately conservative. Monotone-corridor analysis and sublayer attribution strengthen the explanation of why the final model works, but neither produces a better model than the final token-wise two-path router. A bounded external generalization suite also shows that the result does not translate into broad external superiority: the clearest carryover remains on held-out LM-style scoring, while multiple-choice accuracy is mixed.

Our contributions are threefold. First, we show that fixed contiguous delegation is a poor structural prior in this same-family Gemma-2 setting, and that a bounded asymmetric window search identifies a much stronger local shortlist. Second, we show that a static two-path mixture and then a low-capacity token-wise router can beat strong bridge baselines on both a development holdout and an untouched confirmation holdout. Third, we provide explanatory follow-up analyses, monotone-corridor discovery and sublayer attribution, that strengthen the interpretation of the best model while also clarifying the limits of its external generalization.

## 2. Related Work

Our work sits between positive results on representation transfer and negative results on cross-scale incompatibility. On the positive side, recent work on same-family or closely related models suggests that hidden spaces can often be connected by simple affine or lightweight learned maps, and that transferred representations can preserve useful behavior. In language models, model stitching and related feature-transfer results similarly suggest that residual-stream features can remain compatible across models under suitable interfaces. Our design inherits that perspective, but the operational question here is stricter: we ask whether delegated small-model computation can replace a missing block of a larger model's forward pass under frozen-backbone constraints.

At the same time, recent negative results on cross-scale transfer motivate a conservative design and a strong control set. We therefore remain within one model family, keep all backbone weights frozen, and evaluate against learned bridge baselines that stay entirely in large hidden space. These controls matter because a positive result against weak baselines alone would not distinguish useful delegated computation from a better use of trainable capacity in the large model's own latent space.

The same-family choice is grounded in Gemma-2 itself. Gemma-2 2B and 9B share the same family structure while still differing enough in scale to make cross-scale mismatch meaningful. For bounded external evaluation, we use standard log-likelihood-compatible benchmarks: HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi et al., 2020), ARC-Easy and ARC-Challenge (Clark et al., 2018), and LAMBADA (Paperno et al., 2016). A formatted bibliography is included below and mirrored in [notes/paper/references.md](</E:/lab/latent-delegation/notes/paper/references.md>).

## 3. Setup And Problem Statement

We study one-way latent delegation. The large model owns the input path, the master residual stream, the suffix, and the final logits. The small model owns delegated latent computation only. Communication between the two models is through learned latent interfaces rather than token re-injection, and all backbone weights remain frozen.

The default models are Gemma-2 9B as the large model and Gemma-2 2B as the small model. The original fixed-window baseline removes large layers `24..29` and replaces them with delegated small layers `14..19`, entered from the small hidden state immediately before that delegated block. Concretely, the large prefix is layers `0..23`, the removed large block is `24..29`, the large suffix is `30..41`, the small reference hidden is taken after layer `13`, and the delegated small block is `14..19`.

The scientific question is not whether such a hybrid can merely run. The stronger question is whether delegated small-model computation adds value beyond strong controls, including `skip_only` removal, no-small interface controls, learned bridge baselines that remain in large hidden space, and parameter-matched bridge baselines. The paper should be read as an answer to that stronger question.

## 4. Method

### 4.1 Base Interface

Let `h_t^L` denote the large-model hidden state at token position `t` after the frozen large prefix, and let `N(.)` denote RMSNorm. For each delegated path `p`, let `E_p` be the entry projector into small latent space, `S_p` the frozen delegated small-model window, and `R_p` the return adapter back into large hidden space. The path-specific returned delta is

`Delta_{p,t} = R_p(S_p(E_p(N(h_t^L))))`.

In the single-path case, the hybrid hidden state after the removed large block is

`h_t^H = h_t^L + Delta_t`,

after which the frozen large suffix produces the final logits.

### 4.2 Static Two-Path Mixture

The first successful continuation model activates two delegated paths in parallel: path B, which maps `24..27 -> 14..19`, and path A, which maps `24..27 -> 16..18`. In the static two-path mixture, the returned deltas are combined by a learned global two-logit softmax,

`w = softmax(alpha)`,

`Delta_t = w_B Delta_{B,t} + w_A Delta_{A,t}`.

The matched no-small control keeps the same entry projectors, return adapters, and global mixture weights but removes actual delegated small-model computation.

### 4.3 Token-Wise Two-Path Routing

The final model replaces the global mixture with a low-capacity token-wise router. The gate reads only the large-prefix hidden state at the splice boundary. In the confirmed final configuration, the gate is a direct linear head over RMS-normalized prefix states:

`g_t = softmax(W_g N(h_t^L) + b_g)`,

`Delta_t = g_{t,B} Delta_{B,t} + g_{t,A} Delta_{A,t}`.

The token-wise no-small control uses the same gate family and the same interface routes but removes delegated small-model computation.

### 4.4 Training Objectives

Stage A trains only the entry projector and aligns the large-model splice state to the frozen small-model reference state:

`L_A = MSE(E(h^L), h_ref^S) + CosineLoss(E(h^L), h_ref^S)`.

The decisive training regime is output-aware Stage B. Its implemented objective is

`L_B = MSE(h^H, h^T) + CosineLoss(h^H, h^T) + lambda_KL L_KL + lambda_CE L_CE + lambda_D ||Delta||_2^2`,

where `h^T` is the frozen large-model hidden state after the removed large block, `L_KL` is teacher-logit KL, and `L_CE` is shifted next-token cross-entropy. In the confirmed runs, `lambda_KL = 5.0`, `lambda_CE = 1.0`, and `lambda_D = 1e-4`. The token-wise gate adds only a small stability package: a weak entropy term, a weak KL-to-static-prior term, and no temporal smoothness term in the confirmed final configuration. Stage C is intentionally not used in this paper.

### 4.5 Fairness And Parameter Budgets

Parameter matching is explicit because routing adds trainable capacity beyond a plain large-space bridge. The static two-path mixture uses `753,666` trainable parameters, the token-wise two-path router uses `764,418`, and the matched token-wise no-small control uses the same budget. The plain bridge baseline uses `458,753` trainable parameters, so we also include an updated parameter-matched bridge with `766,977` trainable parameters. This fairness audit matters because the core scientific comparison is not whether the hybrid works at all, but whether delegated small-model computation adds value beyond a strong use of comparable trainable capacity in the large model's own latent space.

## 5. Experimental Protocol

### 5.1 Hardware, Seeds, And Workflow

All experiments were run in a native Windows workflow on a single RTX 5090-class GPU using same-family Gemma-2 9B and 2B backbones. The core confirmed result families use seeds `42, 43, 44`. Deterministic evaluation subsets and saved sample IDs are used throughout.

### 5.2 Development Holdout And Untouched Confirmation Holdout

We distinguish two LM-style holdout policies. The first is a development holdout: the original held-out slice reused during model development and model-selection decisions. The second is an untouched confirmation holdout: a fresh `wikitext-103-v1` test-split slice sampled only after the winning continuation structure had been fixed. The untouched confirmation holdout contains `32` sequences at `seq_len = 256` sampled with seed `7606`. The paper's strongest claim should be read through this confirmation holdout first, because it is the stricter safeguard against repeated reuse of the development slice.

### 5.3 Primary Metrics

Primary ranking is output-first: teacher KL, then NLL, then perplexity, then top-1 agreement, then top-5 overlap. KL is ranked first because the central question is whether delegated computation reproduces the functional role of the removed large-model block relative to the frozen full-large teacher. Hidden-space MSE and cosine are reported only as diagnostics.

### 5.4 Bounded Generalization

For bounded external generalization, we evaluate the frozen final model and key controls on HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, and a held-out LAMBADA slice. Multiple-choice tasks are scored by normalized conditional answer log-likelihood; the LM-style slice is scored by KL, NLL, and perplexity. Uncertainty is reported with paired bootstrap estimates against the main internal baselines.

## 6. Results

### 6.1 Fixed-Window Hybrid as a Qualified Feasibility Baseline

The fixed-window hybrid establishes feasibility but not the final claim. In its best output-aware form, it improves over `skip_only` removal and over the no-small interface control, showing that delegated small-model computation is real and functionally used. However, it does not beat the strong bridge baselines. The correct interpretation of this stage is therefore positive feasibility evidence with a clear structural limit.

### 6.2 Asymmetric Window Search Rejects the Fixed Contiguous Prior

A real-model local window search shows that the original fixed contiguous substitution is the wrong structural prior. The legacy `24..29 -> 14..19` candidate is substantially worse than the two candidates that later become the shortlist. After confirmation, the shortlist remains near-tied:

- `24..27 -> 14..19`: KL/NLL `0.281641 / 3.078029`
- `24..27 -> 16..18`: KL/NLL `0.282215 / 3.074461`

Here `24..27 -> 14..19` is slightly better on KL and top-5 overlap, whereas `24..27 -> 16..18` is slightly better on NLL, perplexity, and top-1 agreement. This result matters because it changes the structural prior before any mixture model is introduced.

### 6.3 Static Two-Path Mixture Surpasses Bridge Baselines

The static two-path mixture is the first model to beat both bridge controls on the primary output metrics. On the development holdout, it improves over both bridge baselines in KL and NLL, and the same pattern survives on the untouched confirmation holdout. This is the first clean bridge win in the project, which is why it matters more than a simple re-ranking among single-window candidates. The matched no-small mixture also improves over the earlier no-small baselines, but the full static mixture remains stronger, indicating that the gain cannot be reduced to route mixing alone.

### 6.4 Token-Wise Two-Path Routing Gives the Best Final Model

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

### 6.5 Explanatory Follow-Ups

The explanatory follow-ups strengthen interpretation rather than replacing the final model. Monotone-corridor analysis recovers a broader low-cost asymmetric region around the successful two-path shortlist, suggesting that the final model exploits a local alignment corridor rather than a lucky hard window. Sublayer attribution shows that both delegated attention and delegated MLP matter. Suppressing MLP hurts more, but suppressing attention is also materially harmful, so the gain is not well explained by a single-component simplification.

### 6.6 Bounded Generalization Is Real but Mixed

Bounded external generalization is real but mixed. The cleanest external strength remains on LM-style scoring: on the held-out LAMBADA slice, the token-wise model is favorable against both bridge baselines in KL and favorable against the parameter-matched bridge in NLL, while the NLL comparison against `bridge_only` is positive by point estimate but not cleanly separated under paired uncertainty. The multiple-choice picture is weaker and should be stated that way. HellaSwag and ARC-Challenge are positive by point estimate, but their paired intervals cross zero; WinoGrande is mixed; and PIQA and ARC-Easy are negative relative to the bridge baselines. The strongest defensible external-validity claim is therefore that the final model is not just a single-slice artifact and retains a real LM-style held-out advantage, not that it broadly dominates strong bridge baselines across downstream tasks.

## 7. Reproducibility

All canonical tables, figure specifications, sample IDs, and reproducibility manifests are generated directly from frozen artifacts. The public release includes machine-readable result tables, figure-ready summaries, exact benchmark slice definitions and seeds, and a reproducibility manifest with artifact roots and commit-level provenance. Version labels such as `v0.6.0`, `v0_7`, `v0_8`, and `v0_9` are retained in the release package only to identify frozen artifact families; the main text refers to experimental phases and model families rather than repository branch history.

## 8. Limitations

This paper reports a bounded systems-and-mechanism result, not a broad benchmark result. The strongest evidence is confined to one same-family pair, Gemma-2 9B and 2B, under a frozen-backbone single-GPU regime. The strongest external carryover remains on LM-style scoring, while broader multiple-choice generalization is mixed rather than broad. The no-small comparison is also slightly weaker on the untouched confirmation holdout than the bridge comparison. Stage C is intentionally not used, and the paper does not establish cross-family robustness, broad downstream superiority, or a universal delegation principle. These limitations are part of the claim boundary, not footnotes to it.

## 9. Conclusion

The strongest result in this paper is a bounded positive one. Same-family latent delegation does not work here because a fixed hard layer match happens to be adequate. It works only after the fixed contiguous substitution is rejected, a local asymmetric shortlist is identified, and delegated computation is reformulated as a low-capacity two-path routing problem.

The resulting token-wise two-path model beats both strong bridge controls on a development holdout and again on an untouched confirmation holdout in the frozen Gemma-2 9B -> 2B setting. This is stronger than the earlier fixed-window feasibility result and is the correct main claim of the paper. At the same time, the paper remains deliberately narrow. The explanatory follow-up branches clarify why the model works but do not replace it, and the bounded external suite shows mixed rather than broad generalization. The right conclusion is therefore narrow and strong at once: one-way same-family latent delegation can surpass strong bridge baselines inside this frozen-backbone Gemma-2 setting, but the current evidence does not justify a broad downstream-superiority claim outside that regime.

## References

- Bisk, Y., Zellers, R., Gao, J., Choi, Y. 2020. *PIQA: Reasoning about Physical Commonsense in Natural Language*. AAAI.
- Clark, P., Cowhey, I., Etzioni, O., et al. 2018. *Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*. arXiv:1803.05457.
- DeepMind. 2024. *Gemma 2 Technical Report*.
- *Linear Representation Transferability Hypothesis*. 2025. arXiv:2506.00653.
- *Model Stitching for Language Models*. 2025. arXiv:2506.06609.
- *Neural Incompatibility*. 2025. arXiv:2505.14436.
- Paperno, D., Kruszewski, G., Lazaridou, A., et al. 2016. *The LAMBADA Dataset: Word Prediction Requiring a Broad Discourse Context*. ACL.
- Sakaguchi, K., Bras, R. L., Bhagavatula, C., Choi, Y. 2020. *WinoGrande: An Adversarial Winograd Schema Challenge at Scale*. AAAI.
- Zellers, R., Bisk, Y., Schwartz, R., Choi, Y. 2019. *HellaSwag: Can a Machine Really Finish Your Sentence?* ACL.

## Appendix A

Appendix A lists the paper-facing source files, canonical tables, figure specifications, bibliography, and reproducibility manifest included in the public release. In the final camera-ready version, each item should be linked either to an appendix section, a supplementary PDF bundle, or the public repository release page.
