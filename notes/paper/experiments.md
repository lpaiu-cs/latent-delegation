# Experiments

## Hardware And Workflow

All experiments were run in a native Windows workflow on a single RTX 5090-class GPU using same-family Gemma-2 9B and 2B backbones. The core confirmed result families use seeds `42, 43, 44`. Deterministic evaluation subsets and saved sample IDs are used throughout.

## Development Holdout And Untouched Confirmation Holdout

We distinguish two LM-style holdout policies. The first is a development holdout: the original held-out slice reused during model development and model-selection decisions. The second is an untouched confirmation holdout: a fresh `wikitext-103-v1` test-split slice sampled only after the winning continuation structure had been fixed. The untouched confirmation holdout contains `32` sequences at `seq_len = 256` sampled with seed `7606`. We treat the untouched confirmation holdout as the primary basis for the strongest internal claim because it is the stricter safeguard against repeated reuse of the development slice.

## Primary Metrics

Primary ranking is output-first: teacher KL, then NLL, then perplexity, then top-1 agreement, then top-5 overlap. KL is ranked first because the central question is whether delegated computation reproduces the functional role of the removed large-model block relative to the frozen full-large teacher. Hidden-space MSE and cosine are reported only as diagnostics.

## Bounded Generalization

For bounded external generalization, we evaluate the frozen final model and key controls on HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, and a held-out LAMBADA slice. Multiple-choice tasks are scored by normalized conditional answer log-likelihood; the LM-style slice is scored by KL, NLL, and perplexity. Uncertainty is reported with paired bootstrap estimates against the main internal baselines.
