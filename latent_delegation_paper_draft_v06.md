# Draft Paper: Same-Family Large-to-Small Latent Delegation Under Structural Mismatch

## Title candidates

1. **Latent Delegation Without Token Re-injection: Same-Family Large-to-Small Routing Beats Strong Bridge Baselines**
2. **When Fixed Layer Matching Fails: Window Mixtures and Token-Wise Routing for Same-Family Latent Delegation**
3. **Beyond Hard Layer Matching: Structural Shortlists and Token-Wise Delegation Between Same-Family Language Models**

---

## Abstract

We study whether a large language model can preserve its master state while delegating part of its intermediate computation to a smaller model from the same architecture family, without converting intermediate states back into text tokens. Prior work suggests that same-family models can admit affine mappings between hidden-state spaces, but also warns that cross-scale transfer is structurally fragile. We instantiate this question on Gemma 2 9B and Gemma 2 2B under a strict single-GPU budget and freeze both backbones throughout. A direct fixed contiguous substitution of large-model layers with a small-model window is not the best structural prior: real-model screening rejects the legacy split and identifies a near-tied two-window shortlist. We then show that combining the shortlisted paths with a static mixture improves over both hard single-window choices, and that a low-capacity token-wise gate improves further. The resulting token-wise delegated model beats skip-only, no-small, and strong large-space bridge baselines on KL and NLL, including on a fresh untouched holdout slice. Monotone-alignment discovery supports the view that the successful windows lie on a broader asymmetric corridor, but a compressed single-path candidate does not surpass the token-wise mixture. Sublayer attribution further shows that both delegated attention and delegated MLP are necessary, with MLP contributing more under KL/NLL. The current evidence supports a qualified but strong claim: same-family large-to-small latent delegation can work at the output level when structural mismatch is handled with shortlisted windows and lightweight routing, but the resulting gain appears to depend on mixed-function delegated computation rather than a single simple correspondence.

---

## 1. Introduction

Most cross-model collaboration between language models is mediated by text: one model writes tokens, another reads tokens. This is robust but inefficient. A more ambitious alternative is **latent delegation**: a large model hands off a portion of its intermediate hidden-state computation to a smaller model directly in latent space, while preserving ownership of the master state and final logits.

This idea is plausible for same-family models. Recent work argues that hidden states across model scales may be approximately connected by affine maps, and that feature transfer across residual streams is often surprisingly effective. At the same time, other work shows that cross-scale transfer is structurally brittle and may fail when the underlying models are not sufficiently aligned. These two observations suggest a narrow but concrete question:

> Can a large model delegate part of its intermediate computation to a smaller same-family model in latent space, under a strict single-GPU budget, while improving output behavior over strong large-space alternatives?

We answer this question in a deliberately constrained setting. We use Gemma 2 9B as the large model and Gemma 2 2B as the delegated model. We keep both backbones frozen, train only interface and routing modules, and evaluate output fidelity primarily with KL divergence and next-token NLL relative to the full large teacher.

Our findings proceed in stages.

1. A naive fixed contiguous substitution is not the best structural prior.
2. Real-model screening identifies a near-tied two-window shortlist.
3. A static mixture of the shortlisted windows beats the best single-window choice and strong bridge baselines.
4. A low-capacity token-wise gate improves further and remains better than bridge baselines on both the original holdout and a fresh untouched holdout.
5. Monotone alignment provides a useful explanatory analysis but not a better single-path replacement.
6. Delegated attention and delegated MLP are both necessary; MLP matters more, but attention is still materially required.

The result is not a universal theory of cross-scale collaboration. It is a bounded but positive result: **same-family latent delegation can work at the output level, but only after rejecting hard single-window matching and replacing it with structurally informed multi-path routing.**

---

## 2. Related Work

This paper sits at the intersection of three lines of work.

### 2.1 Representation transfer across model scales

The Linear Representation Transferability (LRT) hypothesis argues that models from the same family and training distribution may admit affine transformations between representation spaces. This motivates our use of learned entry and return interfaces between Gemma 2 9B and Gemma 2 2B.

### 2.2 Model stitching and feature transfer

Recent work on model stitching shows that affine mappings between residual streams can transfer useful features, probes, and steering directions across language models. This supports our use of a learned interface rather than token-level communication.

### 2.3 Structural limits of cross-scale transfer

Other work emphasizes that cross-scale transfer is fundamentally constrained by architectural and scale-dependent mismatches. Our results strongly align with this view: fixed contiguous layer substitution is a weak prior, and good performance depends on explicitly handling structural mismatch.

### 2.4 Our position relative to prior work

Our contribution is not to show that affine transfer exists in principle. Rather, we provide an end-to-end systems result under strict constraints:

- same-family only,
- single GPU,
- frozen backbones,
- no token re-injection,
- strong bridge baselines,
- output-level evaluation,
- and a staged analysis that explains why the successful branch works.

---

## 3. Problem Setup

We study **one-way latent delegation**.

- The **large model** owns the master hidden state and final logits.
- The **small model** performs delegated latent computation only.
- Intermediate communication occurs through learned interface modules, not text tokens.
- The system is trained under frozen backbones and a single-GPU budget.

### 3.1 Base models

- Large model: Gemma 2 9B
- Small model: Gemma 2 2B
- Hardware budget: 1× RTX 5090-class GPU
- Precision/load policy: frozen quantized backbones with trainable adapter/routing modules

### 3.2 Baseline family

We compare against several controls.

- **skip_only**: remove the large middle block and continue directly.
- **hybrid_no_small**: keep the same interface/routing scaffold, but remove actual delegated small-model computation.
- **bridge_only**: replace the removed large block with a direct large-space bridge.
- **parameter-matched bridge**: a stronger large-space bridge with trainable capacity matched as closely as possible to the delegated system.

These controls are essential. The claim we care about is not merely that delegation beats a broken model, but that it adds value beyond strong large-space alternatives.

---

## 4. Method

## 4.1 Stages A and B

### Stage A: interface alignment

Stage A learns an entry projector from the large hidden space to the small hidden space. The initial purpose is to make the large prefix hidden state readable by the delegated small-model path.

### Stage B: output-aware delegated recovery

Stage B replaces a removed large-model middle region with delegated computation and trains only the lightweight interface/routing modules. The final successful Stage B objective is output-aware:

- hidden recovery terms,
- teacher-logit KL,
- next-token CE/NLL,
- and regularization on delegated deltas.

We do **not** use Stage C in the current paper. Stage C remains intentionally unexecuted because the key questions were already answerable within the Stage B regime.

## 4.2 Legacy fixed split and why it failed

The original structural prior was a fixed contiguous substitution:

- large removed block: `24..29`
- delegated small block: `14..19`

Real-model screening rejected this as the best default. The failure mode was not that delegation itself was useless. Rather, a coarse single-window prior introduced too much structural mismatch.

## 4.3 Real shortlist discovery

A local real-model search narrowed the continuation set to two near-tied windows:

- Path B: `24..27 -> 14..19`
- Path A: `24..27 -> 16..18`

This shortlist already improved substantially over the legacy split and preserved `hybrid > hybrid_no_small`, but neither path alone cleanly beat the strong bridge controls.

## 4.4 Static mixture

We then built a static two-path mixture:

- shared large prefix and suffix,
- two separate delegated paths,
- one global 2-logit softmax mixture over their large-space deltas.

The static mixture is still simple. It does not do token-wise routing. But it already beats both single-path candidates, beats the no-small control, and beats strong bridge baselines.

## 4.5 Token-wise mixture gate

Our strongest branch adds a minimal per-token gate:

- input: large-prefix hidden state at the splice boundary,
- model: RMSNorm + 2-logit linear head,
- output: token-wise softmax over the two delegated paths,
- controls: identical no-small gate family, fairness audit against parameter-matched bridge.

This token-wise gate remains low-capacity, avoids teacher leakage, and improves further over the static mixture.

---

## 5. Experimental Setup

## 5.1 Evaluation priorities

We rank models primarily by output-level metrics relative to the full large teacher:

1. KL divergence to teacher logits,
2. next-token NLL,
3. perplexity,
4. top-1 agreement,
5. top-5 overlap/agreement.

Hidden-space metrics (MSE, cosine) are used diagnostically, not as final ranking criteria.

## 5.2 Holdouts

We use two evaluation policies.

- **Original holdout**: the main validation path used in the earlier pilot workflow.
- **Fresh untouched holdout**: a distinct `wikitext-103-v1` test slice, sampled with seed `7606`, 32 sequences at `seq_len = 256`.

The fresh holdout is important because static mixture selection and token-wise confirmation otherwise risk overusing the same probe slice.

## 5.3 Training budget

All reported branches use the existing single-GPU pilot regime:

- frozen Gemma backbones,
- candidate-specific Stage A where required,
- output-aware Stage B,
- no Stage C,
- 3-seed confirmation for key claims.

---

## 6. Main Results

## 6.1 Phase 1 rejects the legacy fixed split

Real screening shows that the legacy contiguous split `24..29 -> 14..19` is not the best default. Both shortlisted candidates `24..27 -> 14..19` and `24..27 -> 16..18` outperform the legacy split by a large margin on KL, NLL, PPL, and top-1 agreement.

This is an important negative result: **fixed single-window correspondence is a weak structural prior.**

## 6.2 Static mixture beats single-path, no-small, and bridge controls

On 3-seed confirmation, the static mixture achieves:

- original holdout: KL `0.267095 ± 0.016769`, NLL `3.000438 ± 0.096956`
- fresh holdout: KL `0.267244 ± 0.001267`, NLL `3.213048 ± 0.003226`

Against the best single-path candidate, static mixture wins on KL and NLL in all 3 seeds. Against `bridge_only`, it wins by:

- original holdout: dKL `-0.021352`, dNLL `-0.071613`
- fresh holdout: dKL `-0.022320`, dNLL `-0.082033`

Against the parameter-matched bridge, it also wins in all 3 seeds on both primary metrics.

## 6.3 Token-wise gate improves further

The token-wise model is the strongest branch.

### Original holdout

- tokenwise_mixture: KL `0.255739 ± 0.016955`, NLL `2.980182 ± 0.105470`
- vs static mixture: dKL `-0.011356`, dNLL `-0.020256`
- vs bridge_only: dKL `-0.032709`, dNLL `-0.091869`
- vs updated parameter-matched bridge: dKL `-0.046584`, dNLL `-0.121899`

### Fresh untouched holdout

- tokenwise_mixture: KL `0.248886 ± 0.003762`, NLL `3.185004 ± 0.005495`
- vs static mixture: dKL `-0.018358`, dNLL `-0.028044`
- vs bridge_only: dKL `-0.040677`, dNLL `-0.110077`
- vs updated parameter-matched bridge: dKL `-0.052859`, dNLL `-0.142020`

This is the first branch that robustly beats the bridge controls on both the original and untouched holdouts.

---

## 7. Diagnostics and Analysis

## 7.1 Why the token-wise result is not just extra capacity

The no-small controls are critical.

- static mixture beats static no-small,
- token-wise hybrid beats token-wise no-small overall,
- the no-small gate is sharper and more collapsed,
- while the full token-wise model shows moderate entropy and nontrivial per-token variance.

This argues against the explanation that the improvement is simply caused by adding a larger routing scaffold.

## 7.2 Gate specialization

The successful token-wise gate does not collapse onto one path.

- mean path weights: path B `0.411437`, path A `0.588602`
- entropy: `0.602399`
- collapse score: `0.054150`

This indicates real route specialization rather than a trivial hard switch or uniform averaging.

## 7.3 Monotone alignment explains the shortlist but does not replace it

A local monotone alignment solver recovers a broader asymmetric corridor around the successful region. The top path is multi-segment:

- `22..22 -> 13..14`
- `23..24 -> 15..16`
- `25..27 -> 17..18`
- `28..30 -> 19..20`

This supports the idea that the winning shortlisted windows are adjacent coarse samples from a broader low-cost corridor. However, compressing that corridor into a single derived candidate `24..27 -> 15..18` does not beat the v0.6.0 token-wise model under KL-first ranking.

Interpretation: **monotone alignment is a good explanatory tool here, but not yet a better replacement model.**

## 7.4 Sublayer attribution: both attention and MLP matter

We then ask whether the successful gain is driven primarily by delegated attention, delegated MLP, or both.

Relative to the full token-wise model:

### Main holdout

- attention suppressed: dKL `+0.103797`, dNLL `+0.218670`
- MLP suppressed: dKL `+0.182743`, dNLL `+0.350897`

### Fresh holdout

- attention suppressed: dKL `+0.109496`, dNLL `+0.224608`
- MLP suppressed: dKL `+0.182872`, dNLL `+0.379610`

MLP matters more, but attention still matters materially. Path A is the more sensitive route for both subcomponents. The result is not clean enough to justify an MLP-only or attention-only delegated variant.

Interpretation: **the current best result depends on mixed-function delegated computation.**

---

## 8. Contributions

We claim the following contributions.

1. **A strict same-family latent delegation setup** using Gemma 2 9B and 2B with frozen backbones on a single GPU.
2. **A negative structural result**: fixed contiguous `6 -> 6` substitution is not the best prior.
3. **A positive structural result**: a real-model shortlist of asymmetric local windows improves over the legacy split.
4. **A stronger systems result**: static two-path mixture beats strong bridge baselines on output-level metrics.
5. **Our strongest result**: low-capacity token-wise routing improves further and remains better than strong bridge baselines on both reused and fresh untouched holdouts.
6. **Two analysis results**:
   - monotone alignment explains the successful shortlist as part of a broader asymmetric corridor,
   - sublayer attribution shows that both delegated attention and delegated MLP are necessary.

---

## 9. Limitations

This work has several important limitations.

1. **Single family / single pair**: the main positive result is on Gemma 2 9B -> 2B only.
2. **Probe-centric evaluation**: the current strongest evidence is on LM output probes and holdout slices, not yet on a broad downstream task suite.
3. **No Stage C**: we intentionally stop before a full final-output fine alignment stage.
4. **No latency claim yet**: we do not currently claim end-to-end speedup superiority over bridge baselines.
5. **No cross-family generalization claim**: we do not yet know whether the same recipe transfers to another architecture family.

These are not footnotes; they define the current scope of the paper.

---

## 10. Immediate Generalization Agenda

The next paper-strengthening step should **not** be more local surgery inside the Gemma 2 pair. The current best branch is already identified. The next priority is external validity.

### 10.1 What to generalize next

1. **Task generalization on the same Gemma pair**
   - Add evaluation-only generalization beyond Wikitext-style holdouts.
   - Prefer multiple-choice or scoring-based tasks that base models can support cleanly:
     - HellaSwag
     - PIQA
     - WinoGrande
     - ARC-Easy / ARC-Challenge
     - LAMBADA-style completion or another held-out LM corpus

2. **One additional same-family pair or family replication**
   - Preferred if feasible under the same single-GPU constraint.
   - This can be a second same-family open-weight pair, but it should remain tightly bounded.

3. **Efficiency reporting**
   - Add latency and memory comparisons for the final token-wise model versus bridge controls.

### 10.2 What not to do yet

- Do not start Stage C before establishing external validity.
- Do not revive Idea 5 model-building.
- Do not pretend the current result is already family-agnostic.

---

## 11. Conclusion

Same-family latent delegation is not solved by hard layer matching. In our setting, a fixed contiguous `6 -> 6` substitution is the wrong structural prior. But a nearby pair of asymmetric windows, combined through static mixture and then token-wise routing, is enough to produce a robust output-level gain over strong large-space bridge baselines under a strict single-GPU budget.

This result is both positive and qualified. It does not yet establish a universal cross-scale delegation principle. But it does show that **once structural mismatch is handled explicitly, large-to-small latent delegation can outperform stronger large-space alternatives without token re-injection.**

---

## Appendix A. Suggested figures

1. **System overview**
   - full large
   - legacy fixed split
   - shortlisted two-path static mixture
   - token-wise mixture

2. **Phase progression figure**
   - legacy split -> shortlist -> static mixture -> token-wise -> analysis branches

3. **Output metric table**
   - main holdout and fresh holdout
   - static mixture vs token-wise vs bridges vs no-small

4. **Gate behavior figure**
   - token-wise path usage histogram / entropy / collapse score

5. **Monotone corridor diagram**
   - top path segments and relation to shortlisted windows

6. **Sublayer attribution bar chart**
   - attention suppressed / MLP suppressed / both suppressed

---

## Appendix B. Suggested results table skeleton

| Model | Holdout | KL ↓ | NLL ↓ | PPL ↓ | Top-1 ↑ | Top-5 ↑ | Beats Bridge? |
|---|---:|---:|---:|---:|---:|---:|---:|
| bridge_only | original | ... | ... | ... | ... | ... | - |
| param-matched bridge | original | ... | ... | ... | ... | ... | - |
| static mixture | original | 0.267095 | 3.000438 | 20.156769 | 0.762009 | 0.741646 | yes |
| token-wise mixture | original | 0.255739 | 2.980182 | 19.763760 | 0.763351 | 0.744268 | yes |
| bridge_only | fresh | ... | ... | ... | ... | ... | - |
| param-matched bridge | fresh | ... | ... | ... | ... | ... | - |
| static mixture | fresh | 0.267244 | 3.213048 | 24.854807 | 0.753466 | 0.740155 | yes |
| token-wise mixture | fresh | 0.248886 | 3.185004 | 24.167632 | 0.758319 | 0.742734 | yes |

---

## Appendix C. References to include in the bibliography

- Gemma Team. 2024. *Gemma 2: Improving Open Language Models at a Practical Size.*
- Bello et al. 2025. *Linear Representation Transferability Hypothesis: Leveraging Small Models to Steer Large Models.*
- Chen et al. 2025. *Transferring Features Across Language Models With Model Stitching.*
- Tan et al. 2025. *Neural Incompatibility: The Unbridgeable Gap of Cross-Scale Parametric Knowledge Transfer in Large Language Models.*

(Expand this bibliography in the actual manuscript build step.)
