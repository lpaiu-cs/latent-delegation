# Latent Delegation

Single-GPU proof-of-concept for one-way latent delegation between same-family open models.

Default pair:

- large model: `google/gemma-2-9b`
- small model: `google/gemma-2-2b`

## Frozen State

`v0.6.0` is the frozen current best model/result in this repo.

- `v0_7` is an analysis-only Idea 5 discovery branch.
- `v0_8` is an analysis-only Idea 2 attribution branch.
- `v0_9` is a bounded external generalization branch.
- Stage C was not started.
- The current repo task is paperization, not new model-building.

## Main Result

The strongest supported result is a frozen two-path token-wise delegated hybrid inside the bounded Gemma-2 9B -> 2B setting.

That model fixes the removed large window to `24..27` and mixes two delegated small-model paths:

- path B: `24..27 -> 14..19`
- path A: `24..27 -> 16..18`

It beats:

- the best single-path shortlisted candidates
- the token-wise no-small control
- `bridge_only`
- the parameter-matched bridge

Those wins hold on both:

- the original held-out Wikitext-style probe
- a fresh untouched holdout slice

Key `v0.6.0` metrics:

- original holdout, `tokenwise_mixture`: KL `0.255739`, NLL `2.980182`
- original holdout, `bridge_only`: KL `0.288448`, NLL `3.072051`
- fresh holdout, `tokenwise_mixture`: KL `0.248886`, NLL `3.185004`
- fresh holdout, `bridge_only`: KL `0.289564`, NLL `3.295081`

## Claim Boundary

Supported:

- same-family one-way latent delegation is feasible on a single RTX 5090-class GPU
- delegated small-model computation can beat skip/no-small controls
- in this bounded Gemma-2 setting, the frozen `v0.6.0` token-wise hybrid beats strong bridge controls on the main held-out LM-style probes

Not supported:

- a broad claim of downstream superiority over bridge baselines
- a claim that the result already generalizes cleanly across broader task families
- a claim of full thought transfer, general reasoning superiority, or cross-family robustness

## Broader Evaluation

`v0_9` tested bounded external generalization without training new models.

Benchmarks:

- HellaSwag
- PIQA
- WinoGrande
- ARC-Easy
- ARC-Challenge
- LAMBADA OpenAI held-out LM slice

Outcome:

- strongest generalization remains on LM-style evaluation
- multiple-choice transfer is mixed rather than broad
- recommendation: stop and write the paper around `v0.6.0` plus bounded generalization

Representative `v0_9` readout:

- LAMBADA OpenAI, token-wise: KL `0.251354`, NLL `3.423984`
- LAMBADA OpenAI, `bridge_only`: KL `0.254975`, NLL `3.433371`
- ARC-Challenge accuracy: token-wise `0.442708`, `bridge_only` `0.432292`
- ARC-Easy accuracy: token-wise `0.791667`, `bridge_only` `0.828125`

## Repo Progression

- `v0.5.1`: qualified feasibility result; hybrid beat skip/no-small controls but not bridge controls
- Phase 1 (`v0_6`): rejected the legacy fixed contiguous `24..29 -> 14..19` split as the best default and shortlisted `{24..27 -> 14..19, 24..27 -> 16..18}`
- Idea 4 static mixture: first pilot result that beat both bridge controls on KL/NLL
- `v0.6.0` token-wise Idea 4: improved further and became the frozen best branch
- `v0_7`: monotone corridor discovery strengthened the alignment story, but its bounded empirical candidate did not beat `v0.6.0`
- `v0_8`: attribution showed both delegated attention and delegated MLP matter, with larger degradation when MLP is suppressed
- `v0_9`: bounded generalization found real but mixed external validity

## Paper-Facing Files

Top-level narrative:

- [final_report.md](</E:/lab/latent-delegation/notes/final_report.md>)
- [one_page_summary.md](</E:/lab/latent-delegation/notes/one_page_summary.md>)

Paper prose sources:

- [abstract.md](</E:/lab/latent-delegation/notes/paper/abstract.md>)
- [introduction.md](</E:/lab/latent-delegation/notes/paper/introduction.md>)
- [method.md](</E:/lab/latent-delegation/notes/paper/method.md>)
- [experiments.md](</E:/lab/latent-delegation/notes/paper/experiments.md>)
- [results.md](</E:/lab/latent-delegation/notes/paper/results.md>)
- [limitations.md](</E:/lab/latent-delegation/notes/paper/limitations.md>)
- [conclusion.md](</E:/lab/latent-delegation/notes/paper/conclusion.md>)
- [claim_boundary.md](</E:/lab/latent-delegation/notes/paper/claim_boundary.md>)
- [tables.md](</E:/lab/latent-delegation/notes/paper/tables.md>)
- [figures.md](</E:/lab/latent-delegation/notes/paper/figures.md>)

Canonical generated assets:

- [artifacts/paper_tables](</E:/lab/latent-delegation/artifacts/paper_tables>)
- [artifacts/paper_figures](</E:/lab/latent-delegation/artifacts/paper_figures>)

## Repro Commands

Install dependencies:

```powershell
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
```

Environment and auth sanity:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\env_sanity.ps1
```

Real Gemma smoke matrix:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\real_gemma_smoke.ps1
```

Bounded generalization evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\v0_9\run_generalization_eval.ps1
```

Regenerate paper tables and figure specs from frozen artifacts:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_paper_assets.ps1
```

## Current Result Files

Frozen `v0.6.0` decision notes:

- [idea4_tokenwise_combined_decision.md](</E:/lab/latent-delegation/notes/v0_6/idea4_tokenwise_combined_decision.md>)
- [idea4_tokenwise_report.md](</E:/lab/latent-delegation/notes/v0_6/idea4_tokenwise_report.md>)
- [idea4_tokenwise_output_probe.md](</E:/lab/latent-delegation/notes/v0_6/idea4_tokenwise_output_probe.md>)

Analysis branches:

- [idea5_combined_decision.md](</E:/lab/latent-delegation/notes/v0_7/idea5_combined_decision.md>)
- [idea2_combined_decision.md](</E:/lab/latent-delegation/notes/v0_8/idea2_combined_decision.md>)

Generalization:

- [generalization_results.md](</E:/lab/latent-delegation/notes/v0_9/generalization_results.md>)
- [generalization_summary_for_paper.md](</E:/lab/latent-delegation/notes/v0_9/generalization_summary_for_paper.md>)

## Repo Layout

- `configs/`: YAML experiment configs
- `scripts/`: Windows-native PowerShell runners
- `src/models/`: frozen-backbone hybrid and baseline implementations
- `src/train/`: Stage A / B / C training CLIs
- `src/v0_6/`: continuation and Idea 4 code
- `src/v0_7/`: Idea 5 discovery tooling
- `src/v0_8/`: Idea 2 attribution tooling
- `src/v0_9/`: bounded generalization evaluation tooling
- `src/tools/`: reporting and paper-asset generation utilities
- `tests/`: unit coverage for shapes, frozen params, and analysis utilities
- `artifacts/`: frozen experiment outputs
- `notes/`: reports, decision notes, and paper prose sources

## Tests

```powershell
py -3.12 -m pytest -q
```
