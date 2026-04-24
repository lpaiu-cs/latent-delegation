# Adaptive Bridge Summary

Date: `2026-04-24`

## Milestone 1 Freeze

Milestone 1 is a bounded positive result.

- internal LM-style strengths are preserved relative to frozen `v0.6.0`
- `PIQA` is recovered over both bridge baselines
- `ARC-Easy` remains unresolved
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

LAMBADA was preserved relative to frozen `v0.6.0` across the 3-seed aggregate.

- `adaptive_bridge_moe` KL `0.241733`, NLL `3.381540`
- `frozen_v060_tokenwise` KL `0.248918`, NLL `3.385684`

Multiple-choice transfer remained mixed, but the PIQA gain survived replication.

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

- the adaptive bridge recovered `PIQA` relative to both bridge baselines
- it did not recover `ARC-Easy`
- the no-small adaptive control does not explain the PIQA gain
- the PIQA gain is no longer a 1-seed-only observation

## Recommendation

`continue adaptive-bridge`

Reason:

- all internal preservation checks passed
- all LAMBADA preservation checks passed
- the adaptive MoE recovered one target mixed-generalization task, `PIQA`, over the bridge baselines in the 3-seed aggregate

## Caveats

- `ARC-Easy` remains unresolved
- the 3-seed result is stronger than the 1-seed pilot, but still bounded to the fixed evaluation suite and current machine setup
- the PIQA run used the mirror dataset `nthngdy/piqa` because the legacy `piqa` dataset script entry is no longer supported by the installed `datasets` stack on this machine
