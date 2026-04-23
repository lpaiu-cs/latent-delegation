# Experiments

## Hardware And Workflow

All experiments were run in a native Windows workflow on a single RTX 5090-class GPU using same-family Gemma-2 9B and 2B backbones. The bring-up artifacts record successful CUDA, Hugging Face auth, Gemma access, and real Gemma smoke execution on the target machine. The project remained strictly single-GPU throughout.

## Seeds And Reproducibility

The core confirmed result families use seeds `42, 43, 44`. Deterministic evaluation subsets and saved sample IDs are used throughout. The paper-facing tables are generated directly from frozen artifacts, and the reproducibility package records artifact roots, exact slice-definition files, benchmark sample-ID files, environment summary, and Windows-native commands.

## Primary Internal Evaluation

The primary internal evaluation family is a teacher-style next-token probe. The output-first ranking is:

1. KL to the frozen full-large teacher
2. NLL
3. perplexity
4. top-1 agreement
5. top-5 overlap

KL is ranked first because the core research question is whether delegated computation reproduces the functional role of the removed large-model block relative to the frozen large teacher. Hidden-space MSE and cosine remain diagnostic only.

## Development Holdout And Untouched Confirmation Holdout

The paper distinguishes two LM-style holdout policies:

- development holdout: the original held-out slice reused during model development and screening
- untouched confirmation holdout: a fresh Wikitext test-split slice sampled only after the winning continuation structure was fixed

The untouched confirmation holdout uses the saved policy:

- dataset: `wikitext`, config `wikitext-103-v1`, split `test`
- sample count: `32`
- sampling seed: `7606`

The main claim should be read through the untouched confirmation holdout first, because it is the stricter check against repeated reuse of the development slice.

## Bounded Generalization

The bounded external generalization suite evaluates frozen baselines on:

- HellaSwag
- PIQA
- WinoGrande
- ARC-Easy
- ARC-Challenge
- LAMBADA OpenAI held-out LM slice

Multiple-choice tasks are scored by conditional answer log-likelihood with a fixed normalization policy. The LM-style external slice uses next-token scoring with KL, NLL, and PPL. Uncertainty is reported with paired bootstrap estimates for token-wise routing versus static mixture, `bridge_only`, and the parameter-matched bridge.
