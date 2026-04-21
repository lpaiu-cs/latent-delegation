# Latent Delegation

Single-GPU proof-of-concept for one-way latent delegation between same-family open models.

Default pair:

- large model: `google/gemma-2-9b`
- small model: `google/gemma-2-2b`

## Project Goal

Test whether a large same-family model can keep ownership of the master residual stream while delegating a middle block of computation to a smaller same-family model through latent-space transfer.

This repo does **not** claim full thought transfer, full-model equivalence, or benchmark SOTA.

## Current Final Status

The repository is frozen at `v0.5.1` as a qualified research result.

- real-hardware Gemma-2 9B -> 2B bring-up succeeded on the target RTX 5090-class Windows machine
- Stage A representation alignment was stable
- hidden-only Stage B improved hidden recovery over `skip_only` and `hybrid_no_small`
- output-aware Stage B improved the hybrid over `skip_only` and `hybrid_no_small` at the output level
- the hybrid did **not** outperform the strong large-space bridge baselines
- entry-projector finetuning did **not** resolve that gap
- Stage C was intentionally **not** pursued

## Quick Result Snapshot

- Smoke matrix: `14/14` cases passed
- Largest successful smoke context: `seq_len=256` for `full_large`, `skip_only`, `bridge_only`, and `hybrid`
- Stage A: train loss `354 -> 91`, held-out cosine `0.0078 -> 0.8447`
- Output-aware Stage B output probe:
  - `hybrid`: KL `0.6553`, NLL `3.4235`
  - `hybrid_no_small`: KL `0.6730`, NLL `3.5018`
  - `bridge_only`: KL `0.6463`, NLL `3.3939`
  - `bridge_only_param_matched`: KL `0.6471`, NLL `3.3954`
- Entry-tune follow-up:
  - `hybrid_frozen_entry`: KL `0.6553`, NLL `3.4235`
  - `hybrid_train_entry`: KL `0.6686`, NLL `3.4518`

## Strongest Claim

In the same-family Gemma-2 9B -> 2B setting, one-way latent delegation is real, runnable on a single GPU, and improves over `skip_only` and no-small controls. After output-aware Stage B, that improvement is visible at the output level as well.

## Non-Claim

This repo does **not** show that delegated small-model computation is better than strong large-space bridge-based alternatives.

## Default Architecture

Conservative split, 0-indexed:

- large prefix: layers `0..23`
- removed large block: layers `24..29`
- large suffix: layers `30..41`
- small reference hidden: after layer `13`
- delegated small block: layers `14..19`

Hybrid path:

1. Run the large prefix.
2. Project the large hidden into small latent space.
3. Run frozen small layers `14..19`.
4. Map back to large space with a low-rank return adapter.
5. Add the returned delta through a learned scalar gate.
6. Continue through the large suffix and large LM head.

Implemented baselines:

- `FullLargeModel`
- `SkipOnlyLargeModel`
- `BridgeOnlyLargeModel`
- `HybridNoSmallModel`
- `HybridDelegationModel`
- `BridgeOnlyParamMatched`

## Windows-Native Setup

Native Windows PowerShell is the default execution path on this machine.

Install dependencies:

```powershell
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
```

## Reproduce The Smoke Path

Environment and auth sanity:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\env_sanity.ps1
```

Real Gemma smoke matrix:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\real_gemma_smoke.ps1
```

## Reproduce The Key Pilot Runs

Stage A pilot:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_stage_a_pilot.ps1
```

Stage B hidden-only pilot:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_stage_b_pilot.ps1 `
  -StageACheckpoint .\artifacts\stage_a_pilot_ckpt\stage_a_checkpoint.pt
```

Stage B hidden-only ablation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_stage_b_ablation.ps1 `
  -StageACheckpoint .\artifacts\stage_a_pilot_ckpt\stage_a_checkpoint.pt
```

Stage B hidden-only output probe:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_stage_b_output_probe.ps1
```

Stage B output-aware ablation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_stage_b_ablation.ps1 `
  -Config .\configs\gemma2_conservative_pilot_256_stage_b_output_aware.yaml `
  -StageACheckpoint .\artifacts\stage_a_pilot_ckpt\stage_a_checkpoint.pt
```

Stage B entry-tune follow-up:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_stage_b_entry_tune.ps1 `
  -StageACheckpoint .\artifacts\stage_a_pilot_ckpt\stage_a_checkpoint.pt
```

Freeze figures and manifest from existing artifacts only:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\freeze_v051.ps1
```

## Final Reports And Release Notes

Key release documents:

- `notes/final_report.md`
- `notes/release_notes_v0.5.1.md`
- `notes/abstract.md`
- `notes/one_page_summary.md`
- `notes/reproducibility.md`
- `notes/real_hardware_report.md`

Key frozen release artifacts:

- `artifacts/manifest_v0.5.1.json`
- `artifacts/final_summary_table.csv`
- `figures/hidden_metrics_stage_b.png`
- `figures/output_metrics_stage_b.png`
- `figures/milestone_progression.png`
- `figures/entry_tune_effect.png`

## Repo Layout

- `configs/`: YAML experiment configs
- `scripts/`: Windows-native PowerShell wrappers and shell helpers
- `src/models/`: Gemma hybrid and baseline implementations
- `src/train/`: Stage A / B / C training CLIs
- `src/eval/`: lightweight evaluation paths
- `src/analysis/`: Stage B comparison and reporting helpers
- `src/tools/`: milestone/report generation utilities
- `tests/`: config, shape, frozen-param, and helper coverage
- `artifacts/`: frozen JSON/CSV/checkpoint outputs
- `notes/`: reports, release notes, and reproducibility docs

## Stage C Note

Stage C was intentionally not pursued for `v0.5.1`. The gating condition was that the delegated hybrid should first beat or at least match the stronger bridge controls on the lightweight output metrics. That did not happen, and entry-projector finetuning did not fix it, so the repo was frozen and written up as a qualified feasibility result.

## Tests

On Windows:

```powershell
py -3.12 -m pytest -q
```
