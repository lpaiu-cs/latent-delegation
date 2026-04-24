# Adaptive Bridge Fairness Audit

Date: `2026-04-24`

## Fixed Comparison Rules Used

- same-family backbones only:
  - `google/gemma-2-9b`
  - `google/gemma-2-2b`
- frozen backbones only
- same large removed window for every compared model:
  - `24..27`
- same delegated path set for every adaptive model:
  - path B: `24..27 -> 14..19`
  - path A: `24..27 -> 16..18`
- no Stage C
- same bounded task suite only:
  - development holdout
  - confirmation holdout
  - LAMBADA
  - PIQA
  - ARC-Easy
- same replication seed set:
  - `42`
  - `43`
  - `44`

## Compared Models

- `frozen_v060_tokenwise`
- `bridge_only_strong`
- `bridge_only_param_matched`
- `adaptive_bridge_no_small`
- `adaptive_bridge_moe`

## Parameter Budget Readout

Source:
- `outputs/adaptive_bridge/real_seed42_43_44_warm_start/train/results.json`

Recorded values:

- `frozen_v060_tokenwise_trainable_params`: `764418`
- `adaptive_bridge_moe_trainable_params`: `1685507`
- `adaptive_bridge_no_small_trainable_params`: `1685507`
- `bridge_only_strong_rank`: `128`
- `bridge_only_strong_trainable_params`: `917505`
- `bridge_only_param_matched_rank`: `235`
- `bridge_only_param_matched_trainable_params`: `1684481`

Interpretation:

- `bridge_only_param_matched` is the fair budget-matched large-only comparison for the adaptive MoE
- `bridge_only_strong` is a stronger but smaller-parameter bridge control

## Warm-Start Audit

Warm-start source:
- `artifacts/v0_6/idea4_tokenwise/confirm/stage_b/seed_42/tokenwise_mixture_checkpoint.pt`
- `artifacts/v0_6/idea4_tokenwise/confirm/stage_b/seed_43/tokenwise_mixture_checkpoint.pt`
- `artifacts/v0_6/idea4_tokenwise/confirm/stage_b/seed_44/tokenwise_mixture_checkpoint.pt`

Recorded status:

- `adaptive_bridge_moe`: `loaded` for seeds `42/43/44`
- `adaptive_bridge_no_small`: `loaded` for seeds `42/43/44`
- `frozen_v060_tokenwise` comparison checkpoints: `loaded` for seeds `42/43/44`

Interpretation:

- the 3-seed replication used true warm starts for every seed
- the evaluation used the same frozen `v0.6.0` token-wise checkpoint family rather than a scratch proxy

## Dataset Audit

Internal and LM-style tasks used the configured public datasets directly.

For PIQA:

- configured dataset id used in the final run: `nthngdy/piqa`
- reason: the legacy `piqa` dataset-script entry now fails in the installed `datasets` version with `RuntimeError: Dataset scripts are no longer supported, but found piqa.py`

ARC-Easy used:

- `allenai/ai2_arc`
- config: `ARC-Easy`

LAMBADA used:

- `EleutherAI/lambada_openai`

## Decision-Relevant Findings

- internal dev KL/NLL preserved relative to frozen `v0.6.0` in the 3-seed aggregate
- internal confirmation KL/NLL preserved relative to frozen `v0.6.0` in the 3-seed aggregate
- LAMBADA KL/NLL preserved relative to frozen `v0.6.0` in the 3-seed aggregate
- `PIQA` recovered over both bridge baselines in the 3-seed aggregate:
  - `adaptive_bridge_moe`: `0.785000`
  - `bridge_only_param_matched`: `0.778333`
  - `bridge_only_strong`: `0.776667`
- `ARC-Easy` did not recover over bridge baselines in the 3-seed aggregate:
  - `adaptive_bridge_moe`: `0.805000`
  - `bridge_only_param_matched`: `0.818333`
  - `bridge_only_strong`: `0.813333`
- the no-small adaptive control underperformed the adaptive MoE on `PIQA`

## Audit Conclusion

The first bounded milestone and its 3-seed replication are fair enough to support a bounded continuation decision.

Current decision:

- `continue adaptive-bridge`
