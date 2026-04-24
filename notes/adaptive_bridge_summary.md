# Adaptive Bridge Summary

This note is the first-milestone summary template for the post-paper adaptive-bridge fork.

## Question

Can a bridge-aware three-expert residual MoE:

- preserve the internal LM-style strengths of frozen `v0.6.0`
- preserve LAMBADA strength
- recover at least one mixed-generalization weakness on PIQA or ARC-Easy relative to bridge baselines

without changing the frozen Gemma-2 backbone pair or widening the benchmark scope?

## Fill From

- `outputs/adaptive_bridge/train/results.json`
- `outputs/adaptive_bridge/train/diagnostics.json`
- `outputs/adaptive_bridge/eval/results.json`
- `outputs/adaptive_bridge/eval/summary.csv`
- `outputs/adaptive_bridge/eval/summary_note.md`

## Required Summary Points

- development holdout KL / NLL versus frozen `v0.6.0`
- confirmation holdout KL / NLL versus frozen `v0.6.0`
- LAMBADA KL / NLL versus frozen `v0.6.0`
- PIQA accuracy versus `bridge_only_strong` and `bridge_only_param_matched`
- ARC-Easy accuracy versus `bridge_only_strong` and `bridge_only_param_matched`
- whether the adaptive no-small control explains the gain

## Binary Recommendation

- `continue adaptive-bridge`
- `stop this fork`

Do not fill this section unless the frozen `v0.6.0` comparison actually ran. If the frozen reference is missing, write that the decision is blocked rather than forcing a conclusion.
