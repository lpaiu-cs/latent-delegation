# 1. Title

One-Way Latent Delegation Between Same-Family Gemma-2 Models on a Single GPU: A Qualified Feasibility Result

## Result-State Note

- `v0.6.0` is now the frozen current best model/result state for this repo.
- `v0_7` and `v0_8` remain analysis branches only and do not replace the `v0.6.0` best-model claim.
- The next active branch is evaluation generalization, not new model-building and not Stage C.

## 2. Objective

The objective of this project was to test a narrow feasibility question: can a large model keep ownership of the master residual stream while delegating part of the middle computation to a smaller same-family model through latent-space transfer? The default pair was `google/gemma-2-9b` as the large model and `google/gemma-2-2b` as the small model.

The intended claim was never full "thought transfer" or full-model equivalence. The intended claim was that a frozen large model might recover useful computation by routing a middle block through a frozen smaller same-family model plus learned interface modules.

## 3. Constraints

- Hardware: one RTX 5090-class GPU
- Scope: one-month feasibility project, not a benchmark or SOTA effort
- Model family: same-family open models only, defaulting to Gemma-2 9B -> 2B
- Training policy: frozen backbones throughout v1
- Quantization: 4-bit frozen backbones where practical
- Trainable modules only: interface adapters and scalar gate
- No LoRA, no backbone unfreezing, no multi-GPU, no Stage C unless earlier stages justified it

## 4. Model Choice And Rationale

Gemma-2 9B and Gemma-2 2B were chosen because the project depended on same-family latent compatibility rather than cross-family distillation. Using the same family keeps tokenizer conventions, architectural conventions, and hidden-state semantics aligned enough to make the delegated-latent hypothesis meaningful.

The repo remained model-family-specific by design. Gemma access and CUDA both worked on the target Windows machine, so the project did not need to fall back to another family.

## 5. Architecture

The architecture used the conservative split defined in `configs/gemma2_conservative.yaml` and carried forward into the pilot configs.

- Large prefix: large layers `0..23`
- Removed large middle block: large layers `24..29`
- Large suffix: large layers `30..41`
- Small reference state: small hidden after layer `13`
- Delegated small block: small layers `14..19`

Hybrid forward path:

1. Run the frozen large prefix.
2. Map the large hidden state into the small latent space with an affine entry projector.
3. Run the frozen delegated small block.
4. Map the delegated result back into large space with a low-rank return adapter.
5. Add the returned delta through a learned scalar gate.
6. Continue through the frozen large suffix and large LM head.

Default adapter settings:

- Entry projector: affine + optional RMSNorm
- Return adapter rank: `64`
- Bridge rank: `64`
- Gate initialization: `0.01`

## 6. Training Stages

### Stage A

Stage A trained the entry projector to align the large hidden after layer 23 with the small-family latent state before small layer 14. The losses were hidden-state MSE and cosine similarity loss.

### Stage B Hidden-Only

The first Stage B version trained the return path against the large teacher hidden after layer 29 using hidden-state MSE and cosine only. This established whether delegated computation helped recover the removed large block in hidden space.

### Stage B Output-Aware

The second Stage B version kept the same architecture but added teacher-logit KL and held-out next-token CE directly into Stage B, with a small delta regularizer. This was the minimal change needed after the hidden-only Stage B result failed to translate into better output behavior.

### Entry-Tune Follow-Up

The final follow-up kept the output-aware Stage B objective but allowed the Stage A entry projector to continue adapting during Stage B. This tested whether the fixed Stage A projector was the main remaining bottleneck. No backbone layers were unfrozen.

## 7. Controls And Baselines

- `skip_only`: remove the large middle block and continue directly to the suffix
- `hybrid_no_small`: keep the entry projector, return adapter, and gate, but do not run the delegated small-model block
- `bridge_only`: learned large-space bridge baseline with no small model
- `bridge_only_param_matched`: stronger large-space bridge baseline chosen to approximately match the hybrid trainable budget

Parameter counts at the Stage B ablation milestone:

- `skip_only`: `0` trainable params
- `hybrid`: `376,833` trainable params
- `hybrid_no_small`: `376,833` trainable params
- `bridge_only`: `458,753` trainable params
- `bridge_only_param_matched`: `379,905` trainable params

The parameter-matched bridge control matters because it separates "delegated small-model computation" from "simply having more trainable capacity."

## 8. Experimental Timeline / Milestone Progression

### v0.4.0

- Real-hardware bring-up completed on the RTX 5090-class Windows machine.
- Gemma 2B and 9B loaded successfully in the same-family path.
- Stage A stabilized.
- Stage B hidden-only ablation showed that `hybrid` beat `skip_only` and `hybrid_no_small`, but not the bridge controls.

### v0.5.0

- Stage B was upgraded to an output-aware objective.
- Output probe showed that `hybrid` now beat `skip_only` and `hybrid_no_small` at the output level.
- The bridge controls still remained better.

### v0.5.1

- Stage B entry-projector finetuning was added as a focused follow-up.
- Hidden recovery improved, but output KL/NLL did not.
- The gap to the bridge controls widened rather than closing.
- The project was frozen as a qualified feasibility result rather than extended to Stage C.

## 9. Main Quantitative Results

### Smoke Success

- `env_sanity` passed.
- `real_gemma_smoke` completed successfully: `14/14` cases passed.
- Largest successful `seq_len` was `256` for `full_large`, `skip_only`, `bridge_only`, and `hybrid`.
- Hybrid forward at `seq_len=256` completed at about `9.62 GB` peak VRAM in the smoke matrix.

### Stage A

From `artifacts/stage_a_pilot_metrics.json`:

- Train loss: `354.0 -> 91.0`
- Held-out MSE: `111.20 -> 99.95`
- Held-out cosine: `0.0078 -> 0.8447`
- Trainable params: `8,262,144`

Interpretation: the entry projector learned a stable same-family alignment on the lightweight pilot.

### Stage B Hidden-Only

Single pilot:

- `skip_only`: hidden MSE `25.57`, cosine `0.7420`
- `bridge_only`: hidden MSE `20.26`, cosine `0.7957`
- `hybrid`: hidden MSE `20.88`, cosine `0.8064`

Three-seed ablation:

- `hybrid` beat `skip_only` on both hidden metrics in `3/3` seeds
- `hybrid` beat `hybrid_no_small` in `3/3` seeds
- `hybrid` beat `bridge_only` in `0/3` seeds
- `hybrid` beat `bridge_only_param_matched` in `0/3` seeds

Aggregate hidden-only output probe:

- `hybrid`: KL `1.2571`, NLL `4.1061`, PPL `60.96`
- `hybrid_no_small`: KL `1.0012`, NLL `3.8624`, PPL `47.71`
- `bridge_only`: KL `1.0589`, NLL `3.9102`, PPL `50.06`
- `bridge_only_param_matched`: KL `1.0406`, NLL `3.8932`, PPL `49.18`

Interpretation: hidden-only Stage B improved hidden recovery but did not produce the desired output-level advantage.

### Stage B Output-Aware

Three-seed hidden recovery summary:

- `skip_only`: hidden MSE `25.51`, cosine `0.7461`
- `hybrid_no_small`: hidden MSE `24.59`, cosine `0.7637`
- `hybrid`: hidden MSE `22.85`, cosine `0.7868`
- `bridge_only`: hidden MSE `22.16`, cosine `0.7825`
- `bridge_only_param_matched`: hidden MSE `22.44`, cosine `0.7811`

Three-seed output probe summary:

- `skip_only`: KL `1.5273`, NLL `4.4687`, PPL `87.59`
- `hybrid_no_small`: KL `0.6730`, NLL `3.5018`, PPL `33.23`
- `hybrid`: KL `0.6553`, NLL `3.4235`, PPL `30.72`
- `bridge_only`: KL `0.6463`, NLL `3.3939`, PPL `29.83`
- `bridge_only_param_matched`: KL `0.6471`, NLL `3.3954`, PPL `29.87`

Seed-level results:

- `hybrid > skip_only`: `3/3`
- `hybrid > hybrid_no_small`: `3/3`
- `hybrid > bridge_only`: `0/3`
- `hybrid > bridge_only_param_matched`: `0/3`

Interpretation: output-aware Stage B materially strengthened the feasibility claim relative to no-small controls, but did not resolve the stronger bridge comparison.

### Entry-Tune Follow-Up

Three-seed hidden summary after enabling `training.stage_b.train_entry_projector: true`:

- `hybrid_frozen_entry`: hidden MSE `22.85`, cosine `0.7868`
- `hybrid_train_entry`: hidden MSE `21.16`, cosine `0.7960`
- `hybrid_no_small_frozen_entry`: hidden MSE `24.59`, cosine `0.7637`
- `hybrid_no_small_train_entry`: hidden MSE `23.94`, cosine `0.7769`

Three-seed output summary:

- `hybrid_frozen_entry`: KL `0.6553`, NLL `3.4235`, PPL `30.72`
- `hybrid_train_entry`: KL `0.6686`, NLL `3.4518`, PPL `31.61`
- `hybrid_no_small_frozen_entry`: KL `0.6730`, NLL `3.5018`, PPL `33.23`
- `hybrid_no_small_train_entry`: KL `0.6810`, NLL `3.4862`, PPL `32.74`

Key diagnostics:

- Hybrid entry grad norm mean: `0.2292`
- Hybrid final entry update norm mean: `7.62`
- Hybrid no-small final entry update norm mean: `11.70`
- Tuned `hybrid` still beat tuned `hybrid_no_small` on KL/NLL in `2/3` seeds
- Gap to `bridge_only` worsened from KL/NLL `0.0089 / 0.0296` to `0.0223 / 0.0579`

Interpretation: the entry projector was active and trainable, but tuning it did not improve the main output metrics and did not close the bridge gap.

## 10. Interpretation

The project produced a qualified positive result, not a clean positive result.

The positive part is that one-way latent delegation was real in the intended same-family Gemma-2 setting. The delegated path was used, it improved over `skip_only`, and after output-aware Stage B it also improved over the `hybrid_no_small` control at the output level. That means the delegated small-model computation was doing something beyond a pure passthrough interface.

The negative part is equally important. Across hidden-only Stage B, output-aware Stage B, and the entry-tune follow-up, the hybrid did not beat the strong large-space bridge controls on the main held-out output metrics. Entry-projector finetuning made the hidden metrics better but made the hybrid output metrics worse relative to the frozen-entry output-aware baseline. That is strong evidence that better hidden recovery, as measured here, is not sufficient for better output behavior.

## 11. What Is Supported

- Same-family one-way latent delegation is feasible on a single RTX 5090-class GPU.
- The Gemma-2 9B -> 2B hybrid path runs successfully at `seq_len=256` in the conservative split.
- The delegated path is functionally used, not merely bypassed by the gate.
- The hybrid improves over `skip_only`.
- After output-aware Stage B, the hybrid also improves over the no-small control at the output level.
- The project provides evidence that delegated small-model computation can be useful relative to skip/no-small controls under these constraints.

## 12. What Is NOT Supported

- This work does not support the claim that delegated small-model computation is better than strong bridge-based alternatives.
- This work does not support the claim that hidden MSE/cosine improvement reliably predicts output-level quality improvement.
- This work does not support a transition to Stage C under the repo's own gating rules.
- This work does not show end-task reasoning gains, benchmark superiority, or general-purpose thought transfer.

## 13. Limitations

- Only one model family was tested in earnest.
- Only one conservative split was evaluated deeply.
- Training remained lightweight by design and did not include broader data or benchmark sweeps.
- Frozen-backbone constraints were intentional, but they also limit how much representational mismatch can be corrected.
- Output evaluation used lightweight held-out text probes rather than large benchmark suites.
- The project was run under a one-GPU, one-month feasibility budget rather than a scaling-optimized research budget.

## 14. Failure Analysis

The final negative comparison against the bridge controls is not explained by a trivial machine or implementation failure.

- CUDA, bitsandbytes, Hugging Face auth, and the Gemma path all worked.
- The hybrid forward path ran reliably at `seq_len=256`.
- The delegated path was measurably active through nonzero gates and delta norms.
- The bridge control was not winning only because of a tiny parameter budget; the parameter-matched bridge remained competitive.
- Entry-projector finetuning was explicitly tested and did not resolve the gap.

The most plausible current reading is that, in this conservative architecture and data regime, a strong large-space bridge is a better use of limited trainable capacity than routing through the frozen small-model block.

## 15. Conclusion

In a single-GPU same-family Gemma-2 9B -> 2B setting, one-way latent delegation is real and improves over skip/no-small controls, including at the output level after output-aware Stage B, but it does not outperform strong large-space bridge baselines.

Therefore, this work is evidence for feasibility of delegated latent computation, not evidence that delegated small-model computation is superior to strong bridge-based alternatives.

The correct end state for this milestone is to stop experiments, freeze the artifacts, and write up the qualified result rather than widening scope to Stage C.

## 16. Future Work

Future work, if ever resumed, should stay minimal and hypothesis-driven. The next useful question would be a single architectural variable rather than a broader sweep, but that work is intentionally outside the `v0.5.1` freeze. The present milestone is complete as a qualified feasibility result.
