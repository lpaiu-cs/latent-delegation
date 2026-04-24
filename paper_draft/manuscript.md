# Bounded Same-Family Latent Delegation in Gemma-2: Asymmetric Two-Path Routing Surpasses Bridge Baselines on Held-Out Language-Model Probes

## Abstract

We study a bounded same-family latent delegation setting in which a frozen large language model keeps ownership of the master residual stream and final logits while replacing part of its middle computation with delegated computation from a frozen smaller model plus learned interface modules. We instantiate this setting with Gemma-2 9B and Gemma-2 2B under a single RTX 5090-class GPU budget and frozen backbones throughout.

A fixed contiguous delegation baseline provides only a qualified feasibility result: it improves over skip-only and no-small controls, but not over strong learned bridge baselines that remain entirely in large hidden space. A local asymmetric window search then rejects the fixed contiguous prior and yields a stable two-window shortlist. A static two-path mixture over that shortlist is the first model to beat both bridge controls on the primary internal LM-style output metrics, specifically teacher KL and held-out NLL, and a low-capacity token-wise router over the same two paths improves further.

The final token-wise model beats both bridge controls on the primary internal LM-style output metrics on a development holdout and again on an untouched confirmation holdout. Follow-up monotone-corridor and sublayer-attribution analyses strengthen the explanation of why the model works but do not produce a better model. A bounded external generalization suite shows real but mixed external validity: the cleanest carryover remains on held-out LM-style scoring, while multiple-choice gains are selective rather than broad. The strongest defensible claim is therefore a bounded same-family positive result with mixed external generalization, not a broad downstream-superiority claim.

## 1. Introduction

This paper asks a narrow question: can a large same-family model preserve the master residual stream and final logits while delegating part of its middle computation to a smaller same-family model through latent-space transfer? The framing is intentionally bounded. We do not claim thought transfer, model equivalence, or broad downstream superiority. We ask whether delegated latent computation can recover useful work under strict practical constraints.

Those constraints are deliberate. We study a single RTX 5090-class GPU setting with frozen Gemma-2 9B and Gemma-2 2B backbones, learned interface modules only, and no large-scale finetuning or distributed training. This makes the result interpretable. If the hybrid fails, it fails in a realistic frozen-backbone regime; if it succeeds, the success can be tied to a concrete delegated-computation mechanism rather than to broad adaptation capacity.

Our main finding is that the original fixed contiguous substitution is the wrong structural prior. A real-model asymmetric window search produces a near-tied two-window shortlist, and a static two-path mixture over that shortlist is the first model to beat strong large-space bridge baselines on held-out LM-style probes. A low-capacity token-wise router over the same two paths improves further and gives the best final model. The bridge win survives not only on a development holdout but also on an untouched confirmation holdout.

At the same time, the paper remains deliberately conservative. Monotone-corridor analysis and sublayer attribution strengthen the explanation of why the final model works, but neither produces a better model than the final token-wise two-path router. A bounded external generalization suite also shows that the result does not translate into broad external superiority: the clearest carryover remains on held-out LM-style scoring, while multiple-choice accuracy is mixed.

Our contributions are threefold. First, we show that fixed contiguous delegation is a poor structural prior in this same-family Gemma-2 setting, and that a bounded asymmetric window search identifies a much stronger local shortlist. Second, we show that a static two-path mixture and then a low-capacity token-wise router can beat strong bridge baselines on both a development holdout and an untouched confirmation holdout. Third, we provide explanatory follow-up analyses, monotone-corridor discovery and sublayer attribution, that strengthen the interpretation of the best model while also clarifying the limits of its external generalization.

## 2. Related Work

Our work sits between positive results on representation transfer and negative results on cross-scale incompatibility. On the positive side, recent work on same-family or closely related models suggests that hidden spaces can often be connected by simple affine or lightweight learned maps, and that transferred representations can preserve useful behavior. In language models, model stitching and related feature-transfer results similarly suggest that residual-stream features can remain compatible across models under suitable interfaces. Our design inherits that perspective, but the operational question here is stricter: we ask whether delegated small-model computation can replace a missing block of a larger model's forward pass under frozen-backbone constraints.

At the same time, recent negative results on cross-scale transfer motivate a conservative design and a strong control set. We therefore remain within one model family, keep all backbone weights frozen, and evaluate against learned bridge baselines that stay entirely in large hidden space. These controls matter because a positive result against weak baselines alone would not distinguish useful delegated computation from a better use of trainable capacity in the large model's own latent space.

Our contribution is not to show that same-family latent alignment is possible in principle. The contribution is operational: under frozen-backbone single-GPU constraints, fixed contiguous delegation fails, whereas asymmetric shortlist selection followed by static and token-wise two-path routing can surpass strong large-space bridges on held-out LM-style probes. The same-family choice is grounded in Gemma-2 itself. Gemma-2 2B and 9B share the same family structure while still differing enough in scale to make cross-scale mismatch meaningful. For bounded external evaluation, we use standard log-likelihood-compatible benchmarks: HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi et al., 2020), ARC-Easy and ARC-Challenge (Clark et al., 2018), and LAMBADA (Paperno et al., 2016).

## 3. Setup And Problem Statement

We study one-way latent delegation. The large model owns the input path, the master residual stream, the suffix, and the final logits. The small model owns delegated latent computation only. Communication between the two models is through learned latent interfaces rather than token re-injection, and all backbone weights remain frozen.

The default models are Gemma-2 9B as the large model and Gemma-2 2B as the small model. The original fixed-window baseline removes large layers `24..29` and replaces them with delegated small layers `14..19`, entered from the small hidden state immediately before that delegated block. Concretely, the large prefix is layers `0..23`, the removed large block is `24..29`, the large suffix is `30..41`, the small reference hidden is taken after layer `13`, and the delegated small block is `14..19`.

The scientific question is not whether such a hybrid can merely run. The stronger question is whether delegated small-model computation adds value beyond strong controls, including `skip_only` removal, no-small interface controls, learned bridge baselines that remain in large hidden space, and parameter-matched bridge baselines. The empirical protocol is designed to answer that stronger question directly.

## 4. Method

### 4.1 Base Interface

Let `h_t^L` denote the large-model hidden state at token position `t` after the frozen large prefix, and let `N(.)` denote RMSNorm. For each delegated path `p`, let `E_p` be the entry projector into small latent space, `S_p` the frozen delegated small-model window, and `R_p` the return adapter back into large hidden space. The path-specific returned delta is

`Delta_{p,t} = R_p(S_p(E_p(N(h_t^L))))`.

For the single-path fixed-window hybrid, we write `Delta_t := Delta_{p,t}` for the active delegated path. The hybrid hidden state after the removed large block is then

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

### 4.4 Bridge Controls

The bridge baselines stay entirely in large hidden space and use no small-model computation. For large hidden width `d_L` and bridge rank `r`, the bridge is a bias-free two-layer low-rank adapter:

`B_r(h_t^L) = U_r D_r h_t^L`, with `D_r in R^{r x d_L}` and `U_r in R^{d_L x r}`.

The bridged state is

`h_t^B = h_t^L + gamma B_r(h_t^L)`, with `gamma = tanh(a)`.

The scalar gate `a` is learned and initialized from the same small positive raw value as the hybrid gates. The down projection is Kaiming-initialized and the up projection is zero-initialized, so the bridge starts close to skip-only. The plain `bridge_only` baseline uses rank `64`; with Gemma-2 9B hidden width `3584`, this gives `2 * 3584 * 64 + 1 = 458,753` trainable parameters. The parameter-matched bridge uses the same function class with rank `107`, giving `2 * 3584 * 107 + 1 = 766,977` trainable parameters.

### 4.5 Training Objectives

Stage A trains only the entry projector and aligns the large-model splice state to the frozen small-model reference state:

`L_A = MSE(E(h^L), h_ref^S) + CosineLoss(E(h^L), h_ref^S)`.

The decisive training regime is output-aware Stage B. Its implemented objective is

`L_B = MSE(h^H, h^T) + CosineLoss(h^H, h^T) + lambda_KL L_KL + lambda_CE L_CE + lambda_D ||Delta||_2^2`,

where `h^T` is the frozen large-model hidden state after the removed large block, `L_KL` is teacher-logit KL, and `L_CE` is shifted next-token cross-entropy. In the confirmed runs, `lambda_KL = 5.0`, `lambda_CE = 1.0`, and `lambda_D = 1e-4`. The token-wise gate adds only a small stability package: a weak entropy term, a weak KL-to-static-prior term, and no temporal smoothness term in the confirmed final configuration. Stage C is intentionally not used in this paper.

### 4.6 Fairness And Parameter Budgets

Parameter matching is explicit because routing adds trainable capacity beyond a plain large-space bridge. The static two-path mixture uses `753,666` trainable parameters, the token-wise two-path router uses `764,418`, and the matched token-wise no-small control uses the same budget. The plain bridge baseline uses `458,753` trainable parameters, so we also include an updated parameter-matched bridge with `766,977` trainable parameters. This fairness audit matters because the core scientific comparison is not whether the hybrid works at all, but whether delegated small-model computation adds value beyond a strong use of comparable trainable capacity in the large model's own latent space.

## 5. Experimental Protocol

### 5.1 Hardware, Seeds, And Workflow

All experiments were run in a native Windows workflow on a single RTX 5090-class GPU using same-family Gemma-2 9B and 2B backbones. The core confirmed result families use seeds `42, 43, 44`. Deterministic evaluation subsets and saved sample IDs are used throughout.

### 5.2 Adapter Training Data And Optimization

Adapter training uses fixed lightweight corpus slices rather than a broad training mixture. For each seed, the training pool contains `128` non-empty Wikitext-103-v1 train snippets sampled with the experiment seed and `64` GSM8K train question-answer records sampled with seed offset `+17`; all sequences are tokenized to `seq_len = 256`. Stage A uses the first `144` examples of that pool, corresponding to the configured 75% Stage A cutoff, and Stage B uses the full `192`-example pool.

The confirmed single-path shortlist runs train Stage A for `200` optimizer steps and Stage B for `200` optimizer steps. The static mixture warm-starts both paths from those confirmed single-path checkpoints and trains only Stage B for `200` steps. The token-wise model warm-starts from the static mixture checkpoint, freezes the entry projectors, and trains only the return adapters plus gate network for `200` Stage B steps.

All confirmed runs use AdamW, weight decay `0`, gradient clipping at `1.0`, micro-batch size `1`, gradient accumulation `8`, and final fixed-budget checkpoint selection with no validation-based early stopping. Phase 1 and static mixture Stage B use learning rate `3e-4`; the final token-wise Stage B uses return-adapter LR `1.5e-4` and gate LR `3e-4`. Appendix A tabulates the protocol.

### 5.3 Development Holdout And Untouched Confirmation Holdout

We distinguish two LM-style holdout policies. The first is a development holdout: the original held-out slice reused during model development and model-selection decisions. The second is an untouched confirmation holdout: a fresh `wikitext-103-v1` test-split slice sampled only after the winning continuation structure had been fixed. The untouched confirmation holdout contains `32` sequences at `seq_len = 256` sampled with seed `7606`. We treat the untouched confirmation holdout as the primary basis for the strongest internal claim because it is the stricter safeguard against repeated reuse of the development slice.

### 5.4 Primary Metrics

Primary ranking is output-first: teacher KL, then NLL, then perplexity, then top-1 agreement, then top-5 overlap. KL is ranked first because the central question is whether delegated computation reproduces the functional role of the removed large-model block relative to the frozen full-large teacher. Hidden-space MSE and cosine are reported only as diagnostics.

### 5.5 Bounded Generalization

For bounded external generalization, we evaluate the frozen final model and key controls on HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, and a held-out LAMBADA slice. Multiple-choice tasks are scored by normalized conditional answer log-likelihood; the LM-style slice is scored by KL, NLL, and perplexity. Uncertainty is reported with paired bootstrap estimates against the main internal baselines. All six external tasks use deterministic bounded subsets rather than full benchmark sweeps. HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, and LAMBADA each use `64` examples, with fixed sampling seeds `9001`, `9002`, `9003`, `9004`, `9005`, and `9010` respectively; exact sample IDs are saved in the supplementary materials.

## 6. Results

### 6.1 Fixed-Window Hybrid as a Qualified Feasibility Baseline

The fixed-window hybrid establishes feasibility but not the final claim. In its best output-aware form, it improves over `skip_only` removal and over the no-small interface control, showing that delegated small-model computation is real and functionally used. However, it does not beat the strong bridge baselines. The correct interpretation of this stage is therefore positive feasibility evidence with a clear structural limit.

### 6.2 Asymmetric Window Search Rejects the Fixed Contiguous Prior

A real-model local window search shows that the original fixed contiguous substitution is the wrong structural prior. The legacy `24..29 -> 14..19` candidate is substantially worse than the two candidates that later become the shortlist. After confirmation, the shortlist remains near-tied: `24..27 -> 14..19` reaches KL/NLL `0.281641 / 3.078029`, whereas `24..27 -> 16..18` reaches `0.282215 / 3.074461`. The first is slightly better on KL and top-5 overlap; the second is slightly better on NLL, perplexity, and top-1 agreement. Table 1 summarizes how that shortlist then leads to the final routing result.

**Table 1. Structural progression from the fixed-window baseline to the final routing model.**

These rows come from different evaluation stages and holdout policies. The table summarizes the structural progression of the project; it is not a single directly comparable leaderboard.

| phase | compared model | holdout policy | seeds | KL | NLL | PPL |
| --- | --- | --- | --- | --- | --- | --- |
| fixed-window feasibility | output-aware fixed-window hybrid | development holdout | 42, 43, 44 | 0.655263 | 3.423486 | 30.723442 |
| asymmetric shortlist | best single-path `24..27 -> 14..19` | Phase 1 confirmation probe | 42, 43, 44 | 0.281641 | 3.078029 | 21.780681 |
| static mixture | two-path static mixture | development holdout | 42, 43, 44 | 0.267095 | 3.000438 | 20.156769 |
| final model | token-wise two-path routing | development holdout | 42, 43, 44 | 0.255739 | 2.980182 | 19.763760 |

### 6.3 Static Two-Path Mixture Surpasses Bridge Baselines

The static two-path mixture is the first model to beat both bridge controls on the primary output metrics. On the development holdout, it improves over both bridge baselines in KL and NLL, and the same pattern survives on the untouched confirmation holdout. This is the first clean bridge win in the project, which is why it matters more than a simple re-ranking among single-window candidates. The matched no-small mixture also improves over the earlier no-small baselines, but the full static mixture remains stronger, indicating that the gain cannot be reduced to route mixing alone.

### 6.4 Token-Wise Two-Path Routing Gives the Best Final Model

The token-wise two-path router improves further over the static mixture and remains ahead of both bridge controls on both holdout policies. Table 2 gives the main comparison. On the untouched confirmation holdout, token-wise routing reaches KL/NLL `0.248886 / 3.185004`, compared with `0.267244 / 3.213048` for the static mixture, `0.289564 / 3.295081` for `bridge_only`, and `0.301746 / 3.327024` for the parameter-matched bridge. On the development holdout, the same ordering remains: `0.255739 / 2.980182` for token-wise routing, `0.267095 / 3.000438` for the static mixture, `0.288448 / 3.072051` for `bridge_only`, and `0.302323 / 3.102081` for the parameter-matched bridge.

The bridge comparison is clean: on both the development holdout and the untouched confirmation holdout, the token-wise model wins on the primary metrics in all three seeds against both bridge baselines. The no-small comparison is also favorable in aggregate KL and NLL on both holdouts, but it is slightly less clean on the untouched confirmation holdout, where the joint seed-level KL/NLL win count is `2/3` rather than `3/3`. That caveat is weaker than the bridge result, but it should still be stated explicitly.

The gate diagnostics are consistent with real routing rather than trivial collapse. The gate favors path A on average, retains substantial mass on path B, and remains far from full collapse. The matched no-small control uses the same gate family but is sharper, more collapsed, and materially weaker, which argues against a pure route-capacity explanation.

**Table 2. Final internal comparison on the development and untouched confirmation holdouts.**

| model | development KL | development NLL | confirmation KL | confirmation NLL |
| --- | --- | --- | --- | --- |
| token-wise two-path routing | 0.255739 | 2.980182 | 0.248886 | 3.185004 |
| token-wise no-small control | 0.257501 | 3.038605 | 0.251294 | 3.261786 |
| static two-path mixture | 0.267095 | 3.000438 | 0.267244 | 3.213048 |
| `bridge_only` | 0.288448 | 3.072051 | 0.289564 | 3.295081 |
| parameter-matched bridge | 0.302323 | 3.102081 | 0.301746 | 3.327024 |

### 6.5 Explanatory Follow-Ups

The explanatory follow-ups strengthen interpretation rather than replacing the final model. Monotone-corridor analysis recovers a broader low-cost asymmetric region around the successful two-path shortlist, suggesting that the final model exploits a local alignment corridor rather than a lucky hard window. Sublayer attribution shows that both delegated attention and delegated MLP matter. Suppressing MLP hurts more, but suppressing attention is also materially harmful, so the gain is not well explained by a single-component simplification.

### 6.6 Bounded Generalization Is Real but Mixed

Bounded external generalization is real but mixed. The cleanest external strength remains on LM-style scoring: on the held-out LAMBADA slice, the token-wise model is favorable against both bridge baselines in KL and favorable against the parameter-matched bridge in NLL, while the NLL comparison against `bridge_only` is positive by point estimate but not cleanly separated under paired uncertainty. The multiple-choice picture is weaker and should be stated that way. HellaSwag and ARC-Challenge are positive by point estimate, but their paired intervals cross zero; WinoGrande is mixed; and PIQA and ARC-Easy are negative relative to the bridge baselines. The strongest defensible external-validity claim is therefore that the final model is not just a single-slice artifact and retains a real LM-style held-out advantage, not that it broadly dominates strong bridge baselines across downstream tasks.

**Table 3. Bounded external generalization summary.**

All rows use deterministic bounded `64`-example subsets. Positive multiple-choice rows should be read as point-estimate wins only unless paired bootstrap intervals are clearly separated; where noted in the text, some paired intervals still cross zero.

| task | metric | token-wise | static mixture | `bridge_only` | parameter-matched bridge | reading |
| --- | --- | --- | --- | --- | --- | --- |
| HellaSwag | accuracy | 0.671875 | 0.671875 | 0.656250 | 0.656250 | positive by point estimate; paired CI crosses zero |
| PIQA | accuracy | 0.723958 | 0.723958 | 0.734375 | 0.744792 | negative versus bridge baselines |
| WinoGrande | accuracy | 0.645833 | 0.656250 | 0.635417 | 0.656250 | mixed |
| ARC-Easy | accuracy | 0.791667 | 0.796875 | 0.828125 | 0.828125 | negative versus bridge baselines |
| ARC-Challenge | accuracy | 0.442708 | 0.427083 | 0.432292 | 0.437500 | positive by point estimate; paired CI crosses zero |
| LAMBADA | KL / NLL | 0.251354 / 3.423984 | 0.258699 / 3.419273 | 0.254975 / 3.433371 | 0.266066 / 3.446407 | cleanest external LM-style signal |

## 7. Reproducibility Statement

We release exact benchmark slice definitions, saved sample IDs, fixed seeds, canonical result tables, figure specifications, and a reproducibility manifest generated directly from frozen artifacts. Supplementary materials record commit provenance, artifact roots, and the Windows-native commands needed to rerun the reported evaluations.

## 8. Limitations

This paper reports a bounded systems-and-mechanism result, not a broad benchmark result. The strongest evidence is confined to one same-family pair, Gemma-2 9B and 2B, under a frozen-backbone single-GPU regime. The strongest external carryover remains on LM-style scoring, while broader multiple-choice generalization is mixed rather than broad. The no-small comparison is also slightly weaker on the untouched confirmation holdout than the bridge comparison. Stage C is intentionally not used because the main bounded claim was already established before it, while broader generalization remained mixed and an additional distillation stage would have added capacity and confounds without answering a stronger question. The paper does not establish cross-family robustness, broad downstream superiority, or a universal delegation principle. These limitations are part of the claim boundary, not footnotes to it.

## 9. Conclusion

The strongest result in this paper is a bounded positive one. Same-family latent delegation does not work here because a fixed hard layer match happens to be adequate. It works only after the fixed contiguous substitution is rejected, a local asymmetric shortlist is identified, and delegated computation is reformulated as a low-capacity two-path routing problem.

The resulting token-wise two-path model beats both strong bridge controls on the primary internal LM-style output metrics on a development holdout and again on an untouched confirmation holdout in the frozen Gemma-2 9B -> 2B setting. This is stronger than the earlier fixed-window feasibility result and is the correct main claim of the paper. At the same time, the paper remains deliberately narrow. The explanatory follow-up branches clarify why the model works but do not replace it, and the bounded external suite shows mixed rather than broad generalization. The right conclusion is therefore narrow and strong at once: one-way same-family latent delegation can surpass strong bridge baselines inside this frozen-backbone Gemma-2 setting, but the current evidence does not justify a broad downstream-superiority claim outside that regime.

## References

- Bello, F., Das, A., Zeng, F., Yin, F., and Liu, L. 2025. *Linear Representation Transferability Hypothesis: Leveraging Small Models to Steer Large Models*. arXiv:2506.00653.
- Bisk, Y., Zellers, R., Gao, J., and Choi, Y. 2020. *PIQA: Reasoning about Physical Commonsense in Natural Language*. Proceedings of AAAI.
- Chen, A., Merullo, J., Stolfo, A., and Pavlick, E. 2025. *Transferring Linear Features Across Language Models With Model Stitching*. arXiv:2506.06609.
- Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. 2018. *Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*. arXiv:1803.05457.
- Gemma Team. 2024. *Gemma 2 Technical Report*. Technical report.
- Paperno, D., Kruszewski, G., Lazaridou, A., et al. 2016. *The LAMBADA Dataset: Word Prediction Requiring a Broad Discourse Context*. Proceedings of ACL.
- Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. 2020. *WinoGrande: An Adversarial Winograd Schema Challenge at Scale*. Proceedings of AAAI.
- Tan, Y., He, S., Liu, K., and Zhao, J. 2025. *Neural Incompatibility: The Unbridgeable Gap of Cross-Scale Parametric Knowledge Transfer in Large Language Models*. arXiv:2505.14436.
- Zellers, R., Bisk, Y., Schwartz, R., and Choi, Y. 2019. *HellaSwag: Can a Machine Really Finish Your Sentence?* Proceedings of ACL.

## Appendix A

### Adapter Training Protocol

All backbone weights remain frozen, all corpora are deterministic per seed, and final checkpoints are selected by fixed step budget rather than validation-based early stopping.

| run family | training pool | steps | optimizer / LR | batch policy | checkpoint selection |
| --- | --- | --- | --- | --- | --- |
| Phase 1 single-path shortlist | Stage A: 144 examples; Stage B: 192 examples from 128 Wikitext train snippets + 64 GSM8K train QA records | A: 200; B: 200 | AdamW, LR `3e-4` | micro-batch 1, grad accum 8, seq_len 256 | final fixed-budget checkpoint; no early stopping |
| Static two-path mixture | same Stage B pool; paths warm-started from confirmed Phase 1 checkpoints | B: 200 | AdamW, LR `3e-4` | micro-batch 1, grad accum 8, seq_len 256 | final fixed-budget checkpoint; no early stopping |
| Token-wise two-path routing | same Stage B pool; warm-started from static mixture; entry projectors frozen | B: 200 | AdamW, return LR `1.5e-4`, gate LR `3e-4` | micro-batch 1, grad accum 8, seq_len 256 | final fixed-budget checkpoint; no early stopping |

The Wikitext training examples are sampled from non-empty Wikitext-103-v1 train records with the run seed. GSM8K training examples are sampled from the GSM8K train split with seed offset `+17`. Validation diagnostics use `32` Wikitext-103-v1 validation examples sampled with seed offset `+101`; the untouched confirmation holdout is separate and sampled from the Wikitext-103-v1 test split with seed `7606`.

### Bounded Generalization Subset Policy

None of these tasks uses a full benchmark sweep in the current paper. Every task uses a deterministic fixed-size subset with saved sample identifiers.

| task | split | evaluation type | sample count | seed |
| --- | --- | --- | --- | --- |
| HellaSwag | validation | multiple choice | 64 | 9001 |
| PIQA | validation | multiple choice | 64 | 9002 |
| WinoGrande | validation | multiple choice | 64 | 9003 |
| ARC-Easy | validation | multiple choice | 64 | 9004 |
| ARC-Challenge | validation | multiple choice | 64 | 9005 |
| LAMBADA | test | LM-style scoring | 64 | 9010 |

The supplementary release also includes canonical paper tables, figure specifications, exact sample-ID files, slice definitions, and a reproducibility manifest with commit provenance, artifact roots, environment metadata, and Windows-native rerun commands.
