# Generalization Summary For Paper

- `v0.6.0` remains the frozen best branch.
- `v0_7` and `v0_8` remain analysis-only and do not replace the main claim.
- Recommendation after bounded generalization: stop and write the paper around `v0.6.0` plus benchmark generalization.

## Main Takeaway

The frozen token-wise model does generalize beyond the original Wikitext-style probe regime, but not broadly enough to justify a stronger replication branch yet. Its cleanest external strength is on the held-out LAMBADA LM slice, where it beats both bridge controls on KL and beats the parameter-matched bridge on NLL. The multiple-choice picture is mixed: token-wise is best by point estimate on ARC-Challenge and HellaSwag, roughly tied or split on WinoGrande, and worse than the bridge controls on ARC-Easy and PIQA.

## Compact Benchmark Readout

- HellaSwag: token-wise `0.6719`, bridge_only `0.6562`, bridge_param `0.6562`.
- PIQA: token-wise `0.7240`, bridge_only `0.7344`, bridge_param `0.7448`.
- WinoGrande: token-wise `0.6458`, bridge_only `0.6354`, bridge_param `0.6562`.
- ARC-Easy: token-wise `0.7917`, bridge_only `0.8281`, bridge_param `0.8281`.
- ARC-Challenge: token-wise `0.4427`, bridge_only `0.4323`, bridge_param `0.4375`.
- LAMBADA OpenAI KL/NLL: token-wise `0.2514 / 3.4240`, bridge_only `0.2550 / 3.4334`, bridge_param `0.2661 / 3.4464`.

## Claim Boundary

- Supported: the `v0.6.0` token-wise hybrid is not just a Wikitext artifact. It keeps a held-out LM advantage and stays competitive on part of a bounded commonsense benchmark set.
- Not supported: a broad claim that token-wise delegation beats strong bridge controls across external downstream-style tasks. The bridge baselines recover clearly on ARC-Easy and PIQA, and the positive multiple-choice task deltas are still modest with wide paired-bootstrap intervals.
