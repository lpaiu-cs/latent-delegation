# One-Page Summary

## Current Frozen State

- `v0.6.0` is the frozen current best branch.
- `v0_7` and `v0_8` are analysis-only follow-ups and do not supersede the `v0.6.0` claim.
- The next repo question is evaluation generalization, not new architecture work and not Stage C.

## What This Project Asked

This project tested a narrow feasibility question, not a broad model-improvement claim:

Can a large model keep control of the main hidden state while delegating a middle slice of computation to a smaller same-family model through latent-space transfer?

The default pair was Gemma-2 9B as the large model and Gemma-2 2B as the small model. The system was intentionally constrained:

- one RTX 5090-class GPU
- same-family open models only
- frozen backbones
- lightweight interface training only
- no multi-GPU, no LoRA, no Stage C unless earlier evidence justified it

## What Was Built

The hybrid model keeps the large model's input path, master residual stream, suffix, and final logits. It removes a middle block of large layers, projects the large hidden state into the small-model latent space, runs a frozen window of small-model layers, maps the result back into large space, gates that delta, and continues through the large suffix.

The key split was:

- large prefix: layers `0..23`
- removed large block: layers `24..29`
- large suffix: layers `30..41`
- small delegated block: layers `14..19`

The main controls were:

- `skip_only`
- `hybrid_no_small`
- `bridge_only`
- `bridge_only_param_matched`

Those controls were essential because the central scientific question was not just "does the hybrid work at all?" but "does the delegated small-model computation add something beyond a simple learned bridge?"

## What Worked

The real-hardware bring-up succeeded. On the target Windows machine, the Gemma-2 9B/2B path worked, CUDA and bitsandbytes worked, and the conservative smoke matrix passed at `seq_len=256` for full large, skip-only, bridge-only, and hybrid.

Stage A was stable:

- loss fell from `354` to `91`
- held-out MSE improved from `111.20` to `99.95`
- held-out cosine improved from `0.0078` to `0.8447`

Stage B hidden-only produced an initial positive signal:

- `hybrid` beat `skip_only`
- `hybrid` beat `hybrid_no_small`
- the delegated path was real and used

That established that delegated small-model computation was not a dead mechanism.

## Where The Stronger Claim Failed

The first problem was that hidden recovery did not automatically translate into better outputs. In the hidden-only Stage B ablation, the hybrid looked good in hidden space, but the output probe showed that it did not beat the stronger controls on teacher-logit KL or held-out NLL.

That led to the minimal next change: make Stage B output-aware by adding KL and CE directly to the Stage B objective. This materially improved the claim. After output-aware Stage B:

- `hybrid` beat `skip_only`
- `hybrid` beat `hybrid_no_small`
- the win held at the output level, not just in hidden space

But the hybrid still did **not** beat the strong bridge controls:

- `bridge_only`
- `bridge_only_param_matched`

This was the central remaining ambiguity.

## Final Follow-Up And Final Decision

The last focused hypothesis was that the fixed Stage A entry projector might be the main bottleneck. So the repo added optional Stage B entry-projector finetuning and reran a focused 3-seed comparison.

That follow-up gave a clear answer:

- hidden recovery improved for both `hybrid` and `hybrid_no_small`
- the entry projector definitely received gradient and changed substantially
- but the hybrid's output KL/NLL got worse relative to the frozen-entry output-aware baseline
- the bridge gap widened rather than narrowing

So the project ended with a qualified result, not a launch point for Stage C.

## Final Claim Boundary

Supported:

> same-family one-way latent delegation is feasible and improves over skip/no-small controls, including at the output level after output-aware Stage B

Not supported:

> delegated small-model computation is better than strong large-space bridge alternatives

## Bottom Line

The repo should be frozen at `v0.5.1`, not extended into Stage C. The scientific contribution here is a cleanly diagnosed feasibility result with clear limits, not a benchmark win. That is still a useful research outcome because it shows both where same-family latent delegation works and where, in this constrained design, a strong large-space bridge remains the better baseline.
