# Method

## Base Framing

The large model owns the input path, the master residual stream, the suffix, and the final logits. The small model owns delegated latent computation only. All backbone weights remain frozen. The only trainable modules are the interface pieces that map into and out of the delegated path plus a scalar or low-capacity routing mechanism.

The original conservative split was:

- large prefix: layers `0..23`
- removed large middle block: layers `24..29`
- large suffix: layers `30..41`
- small reference hidden: after layer `13`
- delegated small block: layers `14..19`

The base hybrid forward path was:

1. run the frozen large prefix
2. project the large hidden state into small latent space
3. run a frozen delegated small window
4. map the delegated result back to large space
5. add the returned delta through a learned gate
6. continue through the frozen large suffix and large LM head

## Training Stages

Stage A trained the entry projector to align the large hidden after layer 23 with the small-family latent state before small layer 14, using hidden-state MSE and cosine losses.

Stage B first existed in a hidden-only form, where the delegated return path was trained to recover the large teacher hidden after the removed middle block. That version improved hidden recovery but did not translate into output-level wins over the strongest controls.

Stage B was then upgraded to an output-aware objective. It retained hidden-state alignment terms but also added teacher-logit KL and next-token cross-entropy. This was the last architecture inside the `v0.5.x` line.

Stage C was intentionally not started. The repo’s own gating rule required the delegated hybrid to first beat or at least match the stronger bridge controls on lightweight output metrics.

## Continuation to v0.6.0

The continuation branch asked whether the fixed contiguous substitution window was the wrong structural prior. Phase 1 held the architecture family fixed and searched conservatively over local asymmetric windows on the real Gemma path. The result was a stable shortlist:

- path B: `24..27 -> 14..19`
- path A: `24..27 -> 16..18`

The resulting Idea 4 progression was intentionally minimal.

### Static Mixture

The first continuation model kept both shortlisted paths active in parallel and combined their returned large-space deltas with a learnable global 2-logit softmax:

`delta_mix = w[0] * delta_B + w[1] * delta_A`

This model used the same frozen backbones and the same output-aware Stage B family. A matched no-small control retained the same interface structure but removed actual delegated small-model computation.

### Token-Wise Mixture

The final `v0.6.0` model replaced the static global mixture with a low-capacity token-wise gate. The gate reads only the large-prefix hidden state at the splice boundary, applies normalization plus a small head to 2 logits, and outputs per-token softmax weights over the same two delegated paths. The gate is initialized from the learned static mixture prior. A matched token-wise no-small control uses the same gate family but removes delegated small-model computation.

## Controls

The controls are central to the method, not optional bookkeeping:

- `skip_only`: remove the target large block and continue directly to the suffix
- `hybrid_no_small`: keep interface structure but remove delegated small-model computation
- `bridge_only`: learn a bridge entirely in large space
- `bridge_only_param_matched`: adjust bridge capacity to match the trainable budget of the delegated alternative as closely as practical

This control stack is what lets the paper separate three questions:

1. does delegation beat skipping?
2. does delegation beat a no-small interface-only route?
3. does delegation beat a strong large-space bridge under a matched or near-matched trainable budget?
