# Adaptive Bridge Phase 3 Decision

Date: `2026-04-25`

Inputs:

- `outputs/adaptive_bridge/eval_scaleup/results.json`
- `outputs/adaptive_bridge/eval_scaleup/paired_uncertainty.json`
- `outputs/adaptive_bridge/gate_granularity/results.json`
- `notes/adaptive_bridge_eval_scaleup.md`
- `notes/adaptive_bridge_gate_granularity.md`

## 1. Does PIQA remain positive versus both bridge baselines after the larger-eval pass, or was the earlier result mainly bounded-sample noise?

Answer:

- `adaptive_bridge_moe` stays slightly positive on point estimate versus both bridge baselines on the full `PIQA` validation pass
- but the gap shrinks to:
  - `+0.002358` vs `bridge_only_param_matched`
  - `+0.001088` vs `bridge_only_strong`
- both paired intervals cross zero

So the earlier `200`-example result was not fully fake, but the bridge-gap was clearly inflated by the smaller sample. After scale-up, it is still not a robust bridge-baseline win.

## 2. Does ARC-Easy remain unresolved?

Answer: `yes`

Evidence:

- scale-up `ARC-Easy`:
  - `adaptive_bridge_moe`: `0.794737`
  - `bridge_only_strong`: `0.800585`
  - `bridge_only_param_matched`: `0.802924`
- paired deltas versus the bridge baselines stay non-positive on point estimate

`ARC-Easy` remains the main unresolved task.

## 3. Does sequence-mean gating preserve most of the full token-wise gain on confirmation/LAMBADA/PIQA?

Answer: `no`

Why:

- confirmation holdout:
  - sequence-mean loses most of the token-wise gain and falls behind the no-small control on NLL
- `LAMBADA`:
  - sequence-mean keeps some KL benefit but loses the token-wise NLL advantage
- `PIQA`:
  - full token-wise gain over no-small: `+0.018333`
  - sequence-mean gain over no-small: `+0.008333`
  - preserved fraction: about `45%`

That is not “most of the gain” across the key tasks.

## 4. Is token-wise gating clearly necessary, or is a simpler granularity already competitive?

Answer:

- token-wise gating is still necessary for the current active reference
- the simpler granularities are only partially competitive

More specifically:

- sequence/global mean do not preserve the current confirmation and `LAMBADA` NLL behavior
- on bounded `PIQA`, both simpler granularities fall behind the full token-wise gate
- on bounded `ARC-Easy`, global-mean gating is competitive, but that task is still unresolved in the main eval

So the simpler granularity is informative, not yet sufficient.

## 5. Is the current evidence strong enough to justify a bridge-conditioned simplification?

Answer: `no`

Reason:

- the bridge-baseline `PIQA` gap is still not uncertainty-robust after scale-up
- the no-small separation on full `PIQA` disappears in the larger pass
- sequence-level averaging does not preserve most of the current token-wise gain where the model is strongest

The current evidence is not strong enough to replace the active reference with a simplification pilot yet.

## 6. If yes, what is the single simplest justified variant?

Answer:

- not justified in this task
- no new bridge-conditioned variant should be implemented yet

## Recommendation

`Keep current adaptive_bridge_moe as active reference and gather more evidence`

Rationale:

- milestone 1 is still the first genuinely positive post-paper continuation branch
- internal and `LAMBADA` preservation remain strong
- but the larger PIQA pass weakens the case that the bridge-baseline gap is already robust
- and the gate-granularity audit says token-wise routing is still doing real work

So the disciplined next move is to keep `adaptive_bridge_moe` frozen as the active reference, not to jump to a bridge-conditioned simplification in this task.
