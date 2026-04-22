# Limitations

This repo should be read as a bounded systems-and-mechanism paper, not as a benchmark paper.

## Scope Limits

- only one model family was tested in earnest
- only one default pair was carried through the full continuation path
- the strongest result is in-family and same-tokenizer by design
- all work stayed inside a one-GPU feasibility budget

## Architectural Limits

- all backbones remain frozen
- interface capacity is intentionally small
- the final best result still uses a hand-bounded local two-path shortlist rather than a general learned alignment over the whole network
- Stage C was not started, so the repo does not test whether later logit-level distillation would preserve or extend the `v0.6.0` win

## Evaluation Limits

- the strongest evidence remains on the repo’s internal held-out LM-style probes
- the external benchmark set is intentionally bounded rather than comprehensive
- the multiple-choice evaluation is scoring-based rather than full open-ended reasoning evaluation
- broader generalization is mixed, not broad

## Interpretation Limits

- this work does not establish that delegated small-model computation is universally better than strong large-space bridges
- it does not establish broad downstream superiority
- it does not establish cross-family robustness
- it does not support a full thought-transfer framing

These limits are not incidental. They are part of the project design. The value of the repo is that the positive claim stays narrow enough to be defensible.
