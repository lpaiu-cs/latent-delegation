# Experiments

## Hardware And Workflow

All experiments were run in a native Windows workflow on a single RTX 5090-class GPU using same-family Gemma-2 9B and 2B backbones. The core confirmed result families use seeds `42, 43, 44`. Deterministic evaluation subsets and saved sample IDs are used throughout.

## Adapter Training Data And Optimization

Adapter training uses fixed lightweight corpus slices rather than a broad training mixture. For each seed, the training pool contains `128` non-empty Wikitext-103-v1 train snippets sampled with the experiment seed and `64` GSM8K train question-answer records sampled with seed offset `+17`; all sequences are tokenized to `seq_len = 256`. Stage A uses the first `144` examples of that pool, corresponding to the configured 75% Stage A cutoff, and Stage B uses the full `192`-example pool.

The confirmed single-path shortlist runs train Stage A for `200` optimizer steps and Stage B for `200` optimizer steps. The static mixture warm-starts both paths from those confirmed single-path checkpoints and trains only Stage B for `200` steps. The token-wise model warm-starts from the static mixture checkpoint, freezes the entry projectors, and trains only the return adapters plus gate network for `200` Stage B steps.

All confirmed runs use AdamW, weight decay `0`, gradient clipping at `1.0`, micro-batch size `1`, gradient accumulation `8`, and final fixed-budget checkpoint selection with no validation-based early stopping. Phase 1 and static mixture Stage B use learning rate `3e-4`; the final token-wise Stage B uses return-adapter LR `1.5e-4` and gate LR `3e-4`.

## Development Holdout And Untouched Confirmation Holdout

We distinguish two LM-style holdout policies. The first is a development holdout: the original held-out slice reused during model development and model-selection decisions. The second is an untouched confirmation holdout: a fresh `wikitext-103-v1` test-split slice sampled only after the winning continuation structure had been fixed. The untouched confirmation holdout contains `32` sequences at `seq_len = 256` sampled with seed `7606`. We treat the untouched confirmation holdout as the primary basis for the strongest internal claim because it is the stricter safeguard against repeated reuse of the development slice.

## Primary Metrics

Primary ranking is output-first: teacher KL, then NLL, then perplexity, then top-1 agreement, then top-5 overlap. KL is ranked first because the central question is whether delegated computation reproduces the functional role of the removed large-model block relative to the frozen full-large teacher. Hidden-space MSE and cosine are reported only as diagnostics.

## Bounded Generalization

For bounded external generalization, we evaluate the frozen final model and key controls on HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, and a held-out LAMBADA slice. Multiple-choice tasks are scored by normalized conditional answer log-likelihood; the LM-style slice is scored by KL, NLL, and perplexity. Uncertainty is reported with paired bootstrap estimates against the main internal baselines.

All six external tasks use deterministic bounded subsets rather than full benchmark sweeps. HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, and LAMBADA each use `64` examples, with fixed sampling seeds `9001`, `9002`, `9003`, `9004`, `9005`, and `9010` respectively; exact sample IDs are saved in the supplementary materials.
