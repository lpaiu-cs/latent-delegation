# Conclusion

The strongest result in this project is a bounded positive one. Same-family latent delegation does not work because a fixed hard layer match happens to be adequate. It works only after the fixed contiguous substitution is rejected, a local asymmetric shortlist is identified, and the delegated computation is reformulated as a low-capacity two-path routing problem.

The resulting final token-wise model beats both strong bridge controls on the development holdout and again on an untouched confirmation holdout in the frozen Gemma-2 9B -> 2B setting. That is stronger than the original fixed-window feasibility result and is the correct main claim for the paper.

At the same time, the paper should remain disciplined. The monotone-corridor and sublayer-attribution branches explain the result better, but they do not replace the final model. The bounded generalization suite shows real external carryover, especially on LM-style evaluation, but the broader multiple-choice picture is mixed rather than broad.

The correct conclusion is therefore narrow and strong at the same time: one-way same-family latent delegation can beat strong bridge baselines inside this frozen-backbone Gemma-2 setting, but the current evidence does not justify a broad downstream-superiority claim outside that regime.
