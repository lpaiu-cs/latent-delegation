# Adaptive Bridge Eval Scaleup

Date: `2026-04-25`

Primary artifacts:

- `outputs/adaptive_bridge/eval_scaleup/results.json`
- `outputs/adaptive_bridge/eval_scaleup/paired_uncertainty.json`

## Scope

This pass reran evaluation on the existing `42/43/44` adaptive-bridge checkpoints without retraining.

Larger or full slices used here:

- `PIQA` validation: full `1838`
- `ARC-Easy` validation: full `570`
- `ARC-Challenge` validation: full `299`
- `LAMBADA`: deterministic larger anchor slice `1024`

Internal guardrail slices stayed fixed:

- development holdout: `64`
- confirmation holdout: `64`

## Same-Run vs Frozen-Reference Separation

Keep these separate.

Frozen-reference comparisons:

- `adaptive_bridge_moe` vs `frozen_v060_tokenwise`
- `adaptive_bridge_no_small` vs `frozen_v060_tokenwise`

Same-run comparisons:

- `adaptive_bridge_moe` vs `bridge_only_strong`
- `adaptive_bridge_moe` vs `bridge_only_param_matched`
- `adaptive_bridge_moe` vs `adaptive_bridge_no_small`

This note keeps that separation explicit.

## Check 1: Frozen Reference Integrity

Status: `passed`

Verified for seeds `42/43/44`:

- frozen `v0.6.0` checkpoints exist locally
- warm-start was recorded as `loaded`
- the scale-up eval loaded `frozen_v060_tokenwise` as a live comparison model
- every scale-up task artifact contains frozen-reference outputs

The larger-slice comparison is valid.

## Frozen-Reference Results

### Internal And LAMBADA Guardrail

Frozen-reference comparison only.

- development holdout:
  - `adaptive_bridge_moe` KL `0.239405` vs frozen `0.254750`
  - `adaptive_bridge_moe` NLL `2.880880` vs frozen `2.908419`
  - preserved under the configured tolerances
- confirmation holdout:
  - `adaptive_bridge_moe` KL `0.244074` vs frozen `0.261244`
  - `adaptive_bridge_moe` NLL `2.860653` vs frozen `2.888155`
  - preserved under the configured tolerances
- LAMBADA:
  - KL delta `-0.006625`, CI `[-0.006867, -0.006371]`
  - NLL delta `-0.003642`, CI `[-0.004601, -0.002675]`
  - preservation remains robust

### PIQA

Frozen-reference comparison only.

- `adaptive_bridge_moe`: `0.786181`
- frozen `v0.6.0`: `0.784186`
- accuracy delta: `+0.001995`
- paired CI: `[-0.002358, +0.006166]`

Interpretation:

- the larger-slice PIQA edge over frozen `v0.6.0` is small
- it is not uncertainty-robust in this pass

### ARC-Easy

Frozen-reference comparison only.

- `adaptive_bridge_moe`: `0.794737`
- frozen `v0.6.0`: `0.793567`
- accuracy delta: `+0.001170`
- paired CI: `[-0.005848, +0.008772]`

Interpretation:

- no robust external recovery appeared here

### ARC-Challenge

Frozen-reference comparison only.

- `adaptive_bridge_moe`: `0.521739`
- frozen `v0.6.0`: `0.529543`
- accuracy delta: `-0.007804`
- paired CI: `[-0.020067, +0.003344]`

Interpretation:

- the optional scale-up task does not add a new positive result

## Same-Run Bridge-Baseline Results

Same-run comparison only.

### PIQA

- vs `bridge_only_param_matched`:
  - accuracy delta `+0.002358`
  - CI `[-0.001818, +0.006533]`
  - probability of improvement `0.864`
- vs `bridge_only_strong`:
  - accuracy delta `+0.001088`
  - CI `[-0.003446, +0.005622]`
  - probability of improvement `0.665`
- vs `adaptive_bridge_no_small`:
  - accuracy delta `-0.000544`
  - CI `[-0.004534, +0.003627]`

Interpretation:

- the earlier `200`-example PIQA bridge-gap shrinks materially on the full validation pass
- the PIQA edge over both bridge baselines remains only a tiny point estimate
- the PIQA edge over the no-small control does not survive this larger pass

### ARC-Easy

- vs `bridge_only_param_matched`:
  - accuracy delta `-0.008187`
  - CI `[-0.016959, +0.000585]`
- vs `bridge_only_strong`:
  - accuracy delta `-0.005848`
  - CI `[-0.015789, +0.003509]`
- vs `adaptive_bridge_no_small`:
  - accuracy delta `+0.000585`
  - CI `[-0.008187, +0.009357]`

Interpretation:

- the scale-up pass does not move `ARC-Easy` toward a recovery claim
- the bridge baselines still look stronger on this task

### ARC-Challenge

- vs `bridge_only_param_matched`:
  - accuracy delta `+0.004459`
  - CI `[-0.008919, +0.017837]`
- vs `bridge_only_strong`:
  - accuracy delta `-0.005574`
  - CI `[-0.020067, +0.008919]`
- vs `adaptive_bridge_no_small`:
  - accuracy delta `-0.008919`
  - CI `[-0.021182, +0.003344]`

Interpretation:

- the optional task stays mixed and does not support a simplification move

## Main Answer From The Scale-Up Pass

1. `PIQA` does not become a robust win over the bridge baselines after the larger/full recheck.
2. The earlier bounded-slice result was directionally real, but the bridge-gap was inflated by the smaller sample.
3. `LAMBADA` preservation remains strong and robust.
4. `ARC-Easy` remains unresolved.

## Explicit ARC-Easy Status

`ARC-Easy` remains unresolved after the larger-eval pass.

- `adaptive_bridge_moe`: `0.794737`
- `bridge_only_strong`: `0.800585`
- `bridge_only_param_matched`: `0.802924`

Do not treat `ARC-Easy` as recovered.
