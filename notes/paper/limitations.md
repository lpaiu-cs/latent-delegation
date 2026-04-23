# Limitations

This paper should be read as a bounded systems-and-mechanism result, not as a broad benchmark result.

## Scope Limits

- only one model family is carried through the full continuation path
- only one default Gemma-2 9B -> 2B pair receives the full training-and-evaluation treatment
- the strongest positive claim is intentionally same-family and same-tokenizer by design
- all work stays inside a one-GPU frozen-backbone budget

## Architectural Limits

- all backbones remain frozen
- interface capacity is intentionally small
- the final model still depends on a hand-bounded local two-path shortlist rather than a fully learned network-wide alignment
- Stage C is intentionally not started, so the paper does not test whether later logit-level distillation would preserve or extend the bridge win

## Evaluation Limits

- the strongest evidence remains on LM-style scoring
- the untouched confirmation holdout is stronger than the development holdout, but both remain within the same bounded Gemma-2 evaluation regime
- the external benchmark suite is intentionally bounded rather than comprehensive
- the multiple-choice evaluation is scoring-based rather than open-ended generation evaluation
- broader external generalization is mixed rather than broad
- the no-small comparison is weaker on the untouched confirmation holdout than the bridge comparison

## Interpretation Limits

- this work does not establish that delegated small-model computation is universally better than strong large-space bridges
- it does not establish broad downstream superiority
- it does not establish cross-family robustness
- it does not support a full thought-transfer framing

These limits are part of the paper's value rather than an afterthought. The positive claim is defensible because it remains narrow.
