# Experiments

## Hardware And Workflow

All experiments were run in a native Windows workflow on a single RTX 5090-class GPU. The bring-up artifacts record successful CUDA, Hugging Face auth, Gemma access, and real Gemma smoke execution on the target machine. The repo remained explicitly single-GPU throughout: no multi-GPU training, no FSDP, no DeepSpeed, and no distributed execution.

## Seeds And Reproducibility

The core multi-seed result families use seeds `42, 43, 44`. Lightweight pilot phases use the saved seed policy from their artifact roots, and all continuation decisions are traceable to saved JSON/CSV outputs under `artifacts/`. The paper-facing tables in [tables.md](</E:/lab/latent-delegation/notes/paper/tables.md>) are generated directly from those frozen artifacts.

## Evaluation Families

### Main LM-Style Probe Family

The main internal evaluation family is a teacher-style held-out next-token probe with:

- KL to the frozen full-large teacher
- NLL
- perplexity
- top-1 agreement
- top-5 overlap

This family is the primary decision surface for the repo. The continuation work added a fresh untouched holdout slice to avoid reusing the same held-out slice indefinitely for both selection and evaluation.

### Bounded Generalization

The bounded external generalization branch evaluates frozen `v0.6.0` baselines on:

- HellaSwag
- PIQA
- WinoGrande
- ARC-Easy
- ARC-Challenge
- LAMBADA OpenAI held-out LM slice

Multiple-choice tasks are scored by conditional answer log-likelihood with a documented normalization policy. The LM-style external slice uses next-token scoring with KL, NLL, and PPL where meaningful.

## Model Comparison Sets

The paper should keep the comparison sets explicit:

### v0.5.x

- `skip_only`
- `hybrid_no_small`
- `hybrid`
- `bridge_only`
- `bridge_only_param_matched`

### v0.6.0 Continuation

Phase 1:

- legacy fixed split `24..29 -> 14..19`
- candidate `24..27 -> 14..19`
- candidate `24..27 -> 16..18`
- one conservative replacement candidate after the real stage-signature pass

Idea 4:

- best single-path candidates
- static mixture
- static mixture no-small
- token-wise mixture
- token-wise no-small
- `bridge_only`
- parameter-matched bridge

### Analysis Branches

Idea 5 compares proxy-discovered alignments and one bounded derived candidate against the saved `v0.6.0` references. Idea 2 compares selective suppression conditions against the frozen token-wise baseline and controls.

## Statistical Reporting

The main internal `v0.6.0` result is reported as a 3-seed aggregate on both the original and fresh holdout policies. The `v0_9` bounded generalization branch adds paired bootstrap uncertainty estimates for:

- token-wise vs static mixture
- token-wise vs `bridge_only`
- token-wise vs parameter-matched bridge

The paper should rely on those paired estimates rather than on raw point estimates alone for external benchmark claims.
