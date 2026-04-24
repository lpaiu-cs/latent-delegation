# latent-delegation-adaptive-bridge

Post-paper research fork from the frozen `paper-v0.6.0-final` tag.

This repo is for adaptive-bridge follow-up work only. The original paper repo/result remains frozen, and this fork does not redefine that result.

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
- `scripts/adaptive_bridge/`
  - `run_debug_smoke.ps1`
  - `run_train.ps1`
  - `run_eval.ps1`
- `tests/`
  - config, shape, freezing, and end-to-end debug smoke coverage for the new path

## Frozen Reference Policy

The adaptive bridge code can optionally:

- warm-start delegated path A/B modules from frozen `v0.6.0` token-wise checkpoints
- evaluate against the frozen `v0.6.0` token-wise model

The default checkpoint templates are in `configs/adaptive_bridge/gemma2_first_milestone.yaml`.

Those artifacts are not checked into this fork. If they are missing, the repo still runs in debug mode and the adaptive training path still runs, but the main comparison report will correctly mark the `v0.6.0` comparison as blocked.

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

Run bounded evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_eval.ps1
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

## Notes

- fairness audit template: `notes/adaptive_bridge_fairness_audit.md`
- summary note template: `notes/adaptive_bridge_summary.md`
- blockers: `notes/blockers.md`

## Current Stop Conditions

Real Gemma runs are still blocked on machine/auth state until:

- CUDA is available on the target Windows GPU machine
- Gemma access is approved and authenticated
- frozen `v0.6.0` token-wise checkpoints are copied into the configured artifact paths
