# latent-delegation-adaptive-bridge

Post-paper research fork from the frozen `paper-v0.6.0-final` tag.

This repo is for adaptive-bridge follow-up work only. The original paper repo/result remains frozen, and this fork does not redefine that result.

## Python

This fork is pinned to Python `3.12`.

- PowerShell runners use `py -3.12`
- shell defaults use `python3.12`
- `.python-version` is set to `3.12`

## Objective

Test a bridge-aware residual mixture-of-experts that combines exactly three experts over the same frozen Gemma-2 splice:

- strong large-space bridge expert
- delegated path B: `24..27 -> 14..19`
- delegated path A: `24..27 -> 16..18`

The question is whether bridge + delegation can beat the frozen `v0.6.0` token-wise model on mixed generalization weaknesses without losing the internal LM-style strengths that made `v0.6.0` the paper result.

## Scope And Constraints

- same-family `google/gemma-2-9b` / `google/gemma-2-2b` only
- frozen backbones only
- single GPU only
- no Stage C in this fork
- Windows-native workflow only
- bounded eval only:
  - development holdout
  - untouched confirmation holdout
  - LAMBADA
  - PIQA
  - ARC-Easy

## Implemented In This Fork

- `src/adaptive_bridge/models.py`
  - `BridgeAwareResidualMoE`
  - `BridgeAwareResidualMoENoSmall`
- `src/adaptive_bridge/train.py`
  - Stage-B-style training for:
    - `bridge_only_strong`
    - `bridge_only_param_matched`
    - `adaptive_bridge_no_small`
    - `adaptive_bridge_moe`
- `src/adaptive_bridge/evaluate.py`
  - bounded internal + external evaluation
  - optional comparison against frozen `v0.6.0` token-wise checkpoints
  - binary recommendation logic when the frozen reference is available
- `configs/adaptive_bridge/`
  - `debug_tiny.yaml`
  - `gemma2_first_milestone.yaml`
  - `gemma2_three_seed_replication.yaml`
- `scripts/adaptive_bridge/`
  - `run_debug_smoke.ps1`
  - `run_train.ps1`
  - `run_eval.ps1`
  - `run_eval_hardening.ps1`
  - `run_route_ablation.ps1`
  - `run_real_three_seed_train.ps1`
  - `run_real_three_seed_eval.ps1`
- `tests/`
  - config, shape, freezing, and end-to-end debug smoke coverage for the new path

## Frozen Reference Policy

The adaptive bridge code can optionally:

- warm-start delegated path A/B modules from frozen `v0.6.0` token-wise checkpoints
- evaluate against the frozen `v0.6.0` token-wise model

The default checkpoint templates are in `configs/adaptive_bridge/gemma2_first_milestone.yaml`.

The active real runs in this fork use the local paths:

- `artifacts/v0_6/idea4_tokenwise/confirm/stage_b/seed_42/tokenwise_mixture_checkpoint.pt`
- `artifacts/v0_6/idea4_tokenwise/confirm/stage_b/seed_43/tokenwise_mixture_checkpoint.pt`
- `artifacts/v0_6/idea4_tokenwise/confirm/stage_b/seed_44/tokenwise_mixture_checkpoint.pt`

Those files were copied into the fork so the original paper repo stays untouched. If a checkpoint is missing on another machine, the real warm-start entrypoint fails fast instead of degrading into a scratch run.

## Windows-Native Commands

Install:

```powershell
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
```

Debug smoke:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_debug_smoke.ps1
```

Train the first milestone:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_train.ps1
```

Start the required 1-seed warm-start real run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_real_warm_start_seed42.ps1
```

Run bounded evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_eval.ps1
```

Reproduce the completed seed-42 bounded evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_eval.ps1 -Config configs/adaptive_bridge/gemma2_first_milestone.yaml -TrainDir outputs/adaptive_bridge/real_seed42_warm_start/train -OutputDir outputs/adaptive_bridge/real_seed42_warm_start/eval
```

Train the completed 3-seed replication (`43` and `44`, reusing the existing `42` train artifacts):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_real_three_seed_train.ps1
```

Run the completed 3-seed bounded evaluation (`42/43/44` aggregate):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_real_three_seed_eval.ps1
```

Run the bounded eval hardening package:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_eval_hardening.ps1
```

Run the inference-only route ablation package:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_route_ablation.ps1
```

Run the focused test slice used for this fork:

```powershell
py -3.12 -m pytest -q tests\test_adaptive_bridge.py tests\test_adaptive_bridge_smoke.py
```

## Outputs

Default train outputs:

- `outputs/adaptive_bridge/train/`

Default eval outputs:

- `outputs/adaptive_bridge/eval/`

Key files:

- `results.json`
- `summary.csv`
- `summary_note.md`

## Current Milestone Status

Completed on `2026-04-24`:

- Python pinned to `3.12`
- frozen `v0.6.0` checkpoints restored under the fork-local artifact path for seeds `42/43/44`
- 1-seed warm-start real train completed:
  - `outputs/adaptive_bridge/real_seed42_warm_start/train/`
- bounded evaluation completed:
  - `outputs/adaptive_bridge/real_seed42_warm_start/eval/`
- 3-seed replication train completed:
  - `outputs/adaptive_bridge/real_seed42_43_44_warm_start/train/`
- 3-seed bounded evaluation completed:
  - `outputs/adaptive_bridge/real_seed42_43_44_warm_start/eval/`
- current binary recommendation:
  - `continue_adaptive_bridge`

## Notes

- fairness audit: `notes/adaptive_bridge_fairness_audit.md`
- summary note: `notes/adaptive_bridge_summary.md`
- eval hardening note: `notes/adaptive_bridge_eval_hardening.md`
- route ablation note: `notes/adaptive_bridge_route_ablation.md`
- phase-2 decision: `notes/adaptive_bridge_phase2_decision.md`
- blockers: `notes/blockers.md`

## Current Status And Next Step

The first bounded milestone and its 3-seed replication are no longer blocked in this workspace.

Recommended next step, without widening scope:

- keep the task suite fixed:
  - development holdout
  - confirmation holdout
  - LAMBADA
  - PIQA
  - ARC-Easy
- treat `ARC-Easy` as the primary unresolved weakness
- if the fork continues, constrain the next pilot to a small gate or bridge calibration that preserves the same three-expert fairness budget
