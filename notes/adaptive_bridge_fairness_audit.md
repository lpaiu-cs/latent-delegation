# Adaptive Bridge Fairness Audit

This note is the audit frame for the first adaptive-bridge milestone.

## Fixed Comparison Rules

- Same-family backbones only:
  - `google/gemma-2-9b`
  - `google/gemma-2-2b`
- Frozen backbones only
- Same large removed window for every compared model:
  - `24..27`
- Same delegated path set for every adaptive model:
  - path B: `24..27 -> 14..19`
  - path A: `24..27 -> 16..18`
- No Stage C
- Same bounded task suite only:
  - development holdout
  - confirmation holdout
  - LAMBADA
  - PIQA
  - ARC-Easy
- Same tokenizer family and same max sequence length within one run
- No prompt retuning across models inside the bounded evaluation

## Compared Models

- `frozen_v060_tokenwise`
- `bridge_only_strong`
- `bridge_only_param_matched`
- `adaptive_bridge_no_small`
- `adaptive_bridge_moe`

## Parameter Budget Checklist

Fill from:

- `outputs/adaptive_bridge/train/results.json`
- `outputs/adaptive_bridge/train/diagnostics.json`

Record:

- `frozen_v060_tokenwise_trainable_params`
- `adaptive_bridge_moe_trainable_params`
- `adaptive_bridge_no_small_trainable_params`
- `bridge_only_strong_rank`
- `bridge_only_strong_trainable_params`
- `bridge_only_param_matched_rank`
- `bridge_only_param_matched_trainable_params`

## Warm-Start Audit

Warm-start is allowed only from the frozen `v0.6.0` token-wise delegated path modules.

Record:

- whether warm-start was enabled
- exact checkpoint path template
- whether each seed actually loaded the checkpoint
- whether the frozen comparison checkpoint existed at eval time

If the frozen reference checkpoint is missing, the main continue/stop decision is blocked and must be stated explicitly.

## Interpretation Guardrails

- A gain on PIQA or ARC-Easy does not count if internal KL/NLL or LAMBADA materially regress versus frozen `v0.6.0`.
- A no-small gain does not count as delegation evidence.
- If the adaptive model beats bridge baselines but cannot be compared to frozen `v0.6.0`, the result is promising but incomplete.
