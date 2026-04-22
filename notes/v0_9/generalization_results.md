# Generalization Results

## Frozen Context

- `v0.6.0` remains the frozen current best result.
- `v0_7` and `v0_8` remain analysis branches only and do not replace the `v0.6.0` claim.
- This task was evaluation-only. No new architecture was trained and Stage C was not started.

## Benchmark Set

- HellaSwag
- PIQA
- WinoGrande
- ARC-Easy
- ARC-Challenge
- LAMBADA OpenAI test slice for LM-style evaluation

## Readout

- Multiple-choice tasks were scored by length-normalized conditional answer log-likelihood.
- The LM slice used held-out next-token scoring with NLL, PPL, and KL to the frozen full-large teacher.
- All tasks used deterministic sampling with saved IDs and 3 frozen `v0.6.0` seeds.
- Paired bootstrap CIs were computed for token-wise vs `static_mixture`, `bridge_only`, and `bridge_only_param_matched`.

## Task Summary

- HellaSwag: token-wise accuracy `0.6719`, static `0.6719`, bridge_only `0.6562`, bridge_param `0.6562`.
  Token-wise beats both bridge point estimates on accuracy, but the paired accuracy CIs still cross zero (`+0.0156`, CI `[-0.0104, +0.0469]` vs `bridge_only`).
- PIQA: token-wise accuracy `0.7240`, bridge_only `0.7344`, bridge_param `0.7448`.
  This is a negative task for token-wise relative to the bridge baselines.
- WinoGrande: token-wise accuracy `0.6458`, bridge_only `0.6354`, bridge_param `0.6562`.
  Token-wise is slightly above `bridge_only` on accuracy but below the parameter-matched bridge.
- ARC-Easy: token-wise accuracy `0.7917`, bridge_only `0.8281`, bridge_param `0.8281`.
  This is the clearest multiple-choice loss for token-wise; the paired accuracy delta versus both bridges is `-0.0365`.
- ARC-Challenge: token-wise accuracy `0.4427`, bridge_only `0.4323`, bridge_param `0.4375`.
  Token-wise is the best point estimate among the reported models, but the bootstrap intervals are still wide and include zero.
- LAMBADA OpenAI: token-wise KL/NLL `0.2514 / 3.4240`, bridge_only `0.2550 / 3.4334`, bridge_param `0.2661 / 3.4464`, static `0.2587 / 3.4193`.
  Token-wise beats both bridge controls on KL. It also beats the parameter-matched bridge on NLL with the paired NLL CI fully below zero, but versus `bridge_only` the NLL CI still barely touches zero.

## Direct Answers

1. Does `v0.6.0` still beat the bridge controls outside the original Wikitext-style probes?
Partially, not broadly. The strongest external win is on the held-out LAMBADA slice, and the multiple-choice point estimates are favorable on ARC-Challenge plus HellaSwag, but ARC-Easy and PIQA go the other way and WinoGrande is mixed.

2. On which tasks does the token-wise gain remain strongest?
The gain is strongest on the new LM family (`lambada_openai`) and then, more weakly, on ARC-Challenge and HellaSwag. Those are the tasks where token-wise has the cleanest positive point estimates over the bridge baselines.

3. Is the gain mainly visible on language-modeling style metrics, or does it carry over to multiple-choice task accuracy?
It is mainly visible on LM-style metrics. Multiple-choice carryover exists, but only selectively: ARC-Challenge and HellaSwag are positive by point estimate, WinoGrande is split, and ARC-Easy plus PIQA are negative.

4. Does static mixture remain a meaningful intermediate baseline under broader evaluation?
Yes. Static mixture remains informative rather than redundant. It ties token-wise on HellaSwag accuracy, is essentially tied on PIQA except for a tiny margin edge, trails token-wise on ARC-Challenge, and on LAMBADA it trades lower NLL for worse KL. That means the static/token-wise distinction still matters under broader evaluation.

5. Is token-wise still clearly better than the no-small control under broader evaluation?
No, not clearly. Token-wise is better on ARC-Challenge and slightly better on WinoGrande accuracy, but it ties or loses elsewhere, and on LAMBADA the comparison splits: token-wise is better on NLL but worse on KL.

6. Are there any tasks where the bridge baselines recover or surpass the token-wise model?
Yes. ARC-Easy and PIQA are clear recoveries for the bridge baselines. WinoGrande is also not a clean token-wise win because the parameter-matched bridge stays slightly ahead on accuracy.

7. What is the strongest defensible generalization claim after this task?
The strongest defensible claim is that the frozen `v0.6.0` token-wise model retains a real held-out language-modeling advantage and can stay competitive on some multiple-choice tasks, but it does not generalize strongly enough across the bounded external benchmark set to support a broad “better than bridge baselines” claim outside the original Wikitext-style regime.

## Recommendation

Stop and write the paper around `v0.6.0` plus benchmark generalization
