# Reproducibility Package

## Freeze

- Canonical result: `v0.6.0`
- Analysis-only branches: `v0_7, v0_8`
- Bounded generalization branch: `v0_9`
- Stage C started: `False`

## Commit And Artifact Roots

- Git commit hash: `8029a082c3011c696d01c4491373bb68570a26fb`
- Git worktree dirty: `True`
- best result: `artifacts/v0_6/idea4_tokenwise`
- phase1: `artifacts/v0_6/phase1_real`
- idea5 analysis: `artifacts/v0_7/idea5_discovery`
- idea2 analysis: `artifacts/v0_8/idea2_attribution`
- generalization: `artifacts/v0_9/generalization`
- paper tables: `artifacts/paper_tables`
- paper figures: `artifacts/paper_figures`

## Seeds

- Core multi-seed runs: `[42, 43, 44]`
- Generalization bootstrap seed: `9090`

## Environment

- overall pass: `True`
- platform: `Windows-11-10.0.26200-SP0`
- device name: `NVIDIA GeForce RTX 5090`
- total vram gb: `31.842`
- python version: `3.12.9`
- torch version: `2.10.0.dev20251104+cu128`
- transformers version: `4.57.3`
- datasets version: `4.4.0`
- bitsandbytes version: `0.49.2`
- cuda available: `True`
- bf16 supported: `True`
- hf auth token present: `True`
- gemma access success: `True`

## Exact Slice ID Files

- Main holdout sample IDs: `artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/seed_42/sample_ids.json`
- Main holdout slice definition: `artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/slice_definition.json`
- Fresh holdout sample IDs: `artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/sample_ids.json`
- Fresh holdout slice definition: `artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/slice_definition.json`
- hellaswag sample IDs: `artifacts/v0_9/generalization/raw/multichoice/hellaswag/sample_ids.json`
- hellaswag slice definition: `artifacts/v0_9/generalization/raw/multichoice/hellaswag/slice_definition.json`
- piqa sample IDs: `artifacts/v0_9/generalization/raw/multichoice/piqa/sample_ids.json`
- piqa slice definition: `artifacts/v0_9/generalization/raw/multichoice/piqa/slice_definition.json`
- winogrande sample IDs: `artifacts/v0_9/generalization/raw/multichoice/winogrande/sample_ids.json`
- winogrande slice definition: `artifacts/v0_9/generalization/raw/multichoice/winogrande/slice_definition.json`
- arc_easy sample IDs: `artifacts/v0_9/generalization/raw/multichoice/arc_easy/sample_ids.json`
- arc_easy slice definition: `artifacts/v0_9/generalization/raw/multichoice/arc_easy/slice_definition.json`
- arc_challenge sample IDs: `artifacts/v0_9/generalization/raw/multichoice/arc_challenge/sample_ids.json`
- arc_challenge slice definition: `artifacts/v0_9/generalization/raw/multichoice/arc_challenge/slice_definition.json`
- lambada_openai sample IDs: `artifacts/v0_9/generalization/raw/lm/lambada_openai/sample_ids.json`
- lambada_openai slice definition: `artifacts/v0_9/generalization/raw/lm/lambada_openai/slice_definition.json`

## Windows-Native Commands

### Install

- `py -3.12 -m pip install --upgrade pip`
- `py -3.12 -m pip install -r requirements.txt`

### Sanity

- `powershell -ExecutionPolicy Bypass -File .\scripts\env_sanity.ps1`
- `powershell -ExecutionPolicy Bypass -File .\scripts\real_gemma_smoke.ps1`

### Paper Assets

- `powershell -ExecutionPolicy Bypass -File .\scripts\run_paper_assets.ps1`

### Generalization

- `powershell -ExecutionPolicy Bypass -File .\scripts\v0_9\run_generalization_eval.ps1`

### Tests

- `py -3.12 -m pytest -q`
