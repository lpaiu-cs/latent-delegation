# Adaptive Bridge Summary

Date: `2026-04-24`

## Milestone 1 Freeze

Milestone 1 is a bounded positive result.

- internal LM-style strengths and `LAMBADA` preservation are robust relative to frozen `v0.6.0`
- `PIQA` recovery is robust versus frozen `v0.6.0` and the no-small control, but not yet robust versus the bridge baselines
- `ARC-Easy` remains unresolved
- the active reference in this fork remains `adaptive_bridge_moe`
- this is not yet a global replacement for frozen `v0.6.0`

## Question

Can a bridge-aware three-expert residual MoE:

- preserve the internal LM-style strengths of frozen `v0.6.0`
- preserve LAMBADA strength
- recover at least one mixed-generalization weakness on PIQA or ARC-Easy relative to bridge baselines

without changing the frozen Gemma-2 backbone pair or widening the benchmark scope?

## Runs Used

- 1-seed milestone:
  - config: `configs/adaptive_bridge/gemma2_first_milestone.yaml`
  - train outputs: `outputs/adaptive_bridge/real_seed42_warm_start/train/`
  - eval outputs: `outputs/adaptive_bridge/real_seed42_warm_start/eval/`
- 3-seed replication:
  - config: `configs/adaptive_bridge/gemma2_three_seed_replication.yaml`
  - train outputs: `outputs/adaptive_bridge/real_seed42_43_44_warm_start/train/`
  - eval outputs: `outputs/adaptive_bridge/real_seed42_43_44_warm_start/eval/`
  - seeds: `42`, `43`, `44`
- warm-start: loaded from frozen `v0.6.0` token-wise checkpoints for all three seeds

## Internal LM-Style Preservation

The 3-seed adaptive bridge MoE preserved and slightly improved over frozen `v0.6.0` on both internal holdouts.

- development holdout:
  - `adaptive_bridge_moe` KL `0.239405`, NLL `2.880880`
  - `frozen_v060_tokenwise` KL `0.254750`, NLL `2.908419`
- confirmation holdout:
  - `adaptive_bridge_moe` KL `0.244074`, NLL `2.860653`
  - `frozen_v060_tokenwise` KL `0.261244`, NLL `2.888155`

Relative to the no-small adaptive control and both bridge baselines, the MoE is clearly better on both holdouts.

## External Results

LAMBADA was preserved relative to frozen `v0.6.0` across the 3-seed aggregate, and that preservation remained robust in the later larger-slice recheck.

- `adaptive_bridge_moe` KL `0.241733`, NLL `3.381540`
- `frozen_v060_tokenwise` KL `0.248918`, NLL `3.385684`

Multiple-choice transfer remained mixed. On the bounded milestone-1 slice, `PIQA` was positive against both bridge baselines in the 3-seed aggregate.

- PIQA:
  - `adaptive_bridge_moe` accuracy `0.785`
  - `bridge_only_param_matched` accuracy `0.778333`
  - `bridge_only_strong` accuracy `0.776667`
  - `adaptive_bridge_no_small` accuracy `0.766667`
  - `frozen_v060_tokenwise` accuracy `0.763333`
- ARC-Easy:
  - `adaptive_bridge_moe` accuracy `0.805`
  - `bridge_only_strong` accuracy `0.813333`
  - `bridge_only_param_matched` accuracy `0.818333`
  - `frozen_v060_tokenwise` accuracy `0.803333`

Interpretation:

- on the bounded milestone-1 slice, the adaptive bridge recovered `PIQA` relative to both bridge baselines on point estimate
- it did not recover `ARC-Easy`
- the no-small adaptive control does not explain the PIQA gain
- the PIQA gain is no longer a 1-seed-only observation

## Current Evidence Boundary

After the larger-slice evidence-consolidation pass:

- internal preservation remains robust relative to frozen `v0.6.0`
- `LAMBADA` preservation remains robust relative to frozen `v0.6.0`
- `PIQA` stays positive versus frozen `v0.6.0` and the bridge baselines on point estimate, but it is not yet uncertainty-robust versus the bridge baselines
- `PIQA` is no longer clearly better than `adaptive_bridge_no_small` on the full validation scale-up
- `ARC-Easy` remains unresolved
- the active reference remains `adaptive_bridge_moe`

## Recommendation

`keep adaptive_bridge_moe as active reference and gather more evidence`

Reason:

- all internal preservation checks passed
- all LAMBADA preservation checks passed
- the adaptive MoE produced the first bounded positive continuation result in this fork
- the bridge-baseline `PIQA` gap still needs stronger evidence before any architecture simplification

## Caveats

- `ARC-Easy` remains unresolved
- the 3-seed result is stronger than the 1-seed pilot, but still bounded to the fixed evaluation suite and current machine setup
- the larger-slice pass reduced the `PIQA` bridge-gap from a visible bounded-slice effect to a small point-estimate edge
- the PIQA run used the mirror dataset `nthngdy/piqa` because the legacy `piqa` dataset script entry is no longer supported by the installed `datasets` stack on this machine
