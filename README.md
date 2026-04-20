# Latent Delegation

Single-GPU proof-of-concept for one-way latent delegation between same-family open models. The default target is:

- large model: `google/gemma-2-9b`
- small model: `google/gemma-2-2b`

The v1 claim is narrow: replace a middle block of the large model with a delegated small-model block plus learned interface adapters, while the large model keeps the master residual state and final logits. This repo does not claim full thought transfer or full-model equivalence.

## Default architecture

Default conservative split, 0-indexed:

- large prefix: layers `0..23`
- removed large block: layers `24..29`
- large suffix: layers `30..41`
- small reference hidden: after layer `13`
- delegated small block: layers `14..19`

Hybrid path:

1. Run the large prefix.
2. Project the large hidden state into small latent space.
3. Run frozen small layers `14..19`.
4. Map back into large space with a low-rank return adapter.
5. Add the returned delta through a near-zero scalar gate.
6. Continue through the large suffix and large LM head.

Implemented baselines:

- `FullLargeModel`
- `SkipOnlyLargeModel`
- `BridgeOnlyLargeModel`
- `HybridDelegationModel`

## Environment

Recommended:

- Python `3.12`
- single GPU for real Gemma runs

This repo includes a debug smoke path that does not require Gemma weights.

Create the environment:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Or:

```bash
make install
```

## Windows-native quick start

Native Windows PowerShell is the default execution path on this machine.

Create or update the environment:

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

If the smoke matrix clears far enough for pilot work, use the native wrappers:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_stage_a_pilot.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_stage_b_pilot.ps1 -StageACheckpoint .\artifacts\stage_a_pilot_ckpt\stage_a_checkpoint.pt
powershell -ExecutionPolicy Bypass -File .\scripts\run_stage_b_ablation.ps1 -StageACheckpoint .\artifacts\stage_a_pilot_ckpt\stage_a_checkpoint.pt
```

To refresh the milestone snapshot and Stage B parameter audit in the hardware report:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\write_milestone_snapshot.ps1
```

## Important access note

The default Gemma repositories are gated on Hugging Face. If the account is not authorized for Gemma, model downloads can fail with `401` or `403` responses.

Before real Gemma runs, make sure:

1. you have accepted the Gemma terms for the Hugging Face account
2. you have authenticated, for example with `huggingface-cli login` or `HF_TOKEN`

If Gemma access is blocked, the repo remains runnable with `configs/debug_tiny.yaml`.

## Smoke test

End-to-end debug path:

```bash
PYTHON_BIN=.venv/bin/python ./scripts/smoke_test.sh
```

Or:

```bash
make smoke PYTHON=.venv/bin/python
```

This runs:

1. Stage A on `configs/debug_tiny.yaml`
2. Stage B using the Stage A checkpoint
3. Stage C using Stage A and Stage B checkpoints
4. lightweight perplexity, GSM8K, StrategyQA, and speed evals

## Real-hardware bring-up

Environment and auth sanity check:

```bash
PYTHON_BIN=.venv/bin/python ./scripts/env_sanity.sh
```

This writes:

- `artifacts/env_sanity.json`
- `notes/real_hardware_report.md`

Real Gemma smoke matrix on the target single-GPU machine:

```bash
PYTHON_BIN=.venv/bin/python ./scripts/real_gemma_smoke.sh
```

This script stops immediately if CUDA or Gemma auth sanity checks fail. If they pass, it runs:

- small-model load only
- large-model load only
- full-large forward at seq_len `64`, `128`, `256`
- skip-only forward at seq_len `64`, `128`, `256`
- bridge-only forward at seq_len `64`, `128`, `256`
- hybrid forward at seq_len `64`, `128`, `256`

Outputs:

- `artifacts/real_gemma_smoke.json`
- updated `notes/real_hardware_report.md`

Operational notes for native Windows:

- Hugging Face cache symlink warnings are not blockers; the cache falls back to regular file copies.
- The PowerShell wrappers set `USE_TF=0` and `USE_FLAX=0` so `transformers` stays on the PyTorch path.

## Exact training commands

Stage A:

```bash
.venv/bin/python -m src.train.stage_a_align --config configs/gemma2_conservative.yaml
```

Stage B, hybrid:

```bash
.venv/bin/python -m src.train.stage_b_recover \
  --config configs/gemma2_conservative.yaml \
  --stage-a-checkpoint outputs/<run>/stage_a_checkpoint.pt
```

Stage C, hybrid:

```bash
.venv/bin/python -m src.train.stage_c_distill \
  --config configs/gemma2_conservative.yaml \
  --stage-a-checkpoint outputs/<run>/stage_a_checkpoint.pt \
  --stage-b-checkpoint outputs/<run>/stage_b_checkpoint.pt
```

Optional bridge-only baseline training with the same Stage B / C objectives:

```bash
.venv/bin/python -m src.train.stage_b_recover \
  --config configs/gemma2_conservative.yaml \
  --variant bridge_only
```

```bash
.venv/bin/python -m src.train.stage_c_distill \
  --config configs/gemma2_conservative.yaml \
  --variant bridge_only \
  --stage-b-checkpoint outputs/<run>/stage_b_checkpoint.pt
```

Shell wrappers:

```bash
./scripts/run_stage_a.sh configs/gemma2_conservative.yaml
./scripts/run_stage_b.sh configs/gemma2_conservative.yaml outputs/<run>/stage_a_checkpoint.pt
./scripts/run_stage_c.sh configs/gemma2_conservative.yaml outputs/<run>/stage_a_checkpoint.pt outputs/<run>/stage_b_checkpoint.pt
```

## Evaluation commands

Perplexity:

```bash
.venv/bin/python -m src.eval.eval_ppl \
  --config configs/gemma2_conservative.yaml \
  --variant hybrid \
  --stage-a-checkpoint outputs/<run>/stage_a_checkpoint.pt \
  --stage-b-checkpoint outputs/<run>/stage_b_checkpoint.pt
```

GSM8K subset:

```bash
.venv/bin/python -m src.eval.eval_gsm8k \
  --config configs/gemma2_conservative.yaml \
  --variant hybrid \
  --stage-a-checkpoint outputs/<run>/stage_a_checkpoint.pt \
  --stage-b-checkpoint outputs/<run>/stage_b_checkpoint.pt
```

StrategyQA subset:

```bash
.venv/bin/python -m src.eval.eval_strategyqa \
  --config configs/gemma2_conservative.yaml \
  --variant hybrid \
  --stage-a-checkpoint outputs/<run>/stage_a_checkpoint.pt \
  --stage-b-checkpoint outputs/<run>/stage_b_checkpoint.pt
```

Speed / VRAM:

```bash
.venv/bin/python -m src.eval.eval_speed \
  --config configs/gemma2_conservative.yaml \
  --stage-a-checkpoint outputs/<run>/stage_a_checkpoint.pt \
  --stage-b-checkpoint outputs/<run>/stage_b_checkpoint.pt
```

All evals:

```bash
./scripts/run_eval_all.sh configs/gemma2_conservative.yaml outputs/<run>/stage_a_checkpoint.pt outputs/<run>/stage_b_checkpoint.pt
```

## Tests

Run unit tests and the debug smoke test:

```bash
.venv/bin/pytest -q
```

Or:

```bash
make test PYTHON=.venv/bin/python
```

On Windows:

```powershell
py -3.12 -m pytest -q
```

## Expected outputs

Each stage writes a run directory under `outputs/<experiment>/<stage>/<timestamp>/` unless an explicit `--output-dir` is provided.

Typical contents:

- `config_snapshot.yaml`
- `metadata.json`
- `sample_ids.json`
- `history.csv`
- `metrics.json`
- `stage_a_checkpoint.pt` or `stage_b_checkpoint.pt` or `stage_c_checkpoint.pt`

Evaluation writes JSON metrics and prediction files under the chosen output directory.

## Repo layout

Main implementation:

- `src/models/backbone_loader.py`: model and tokenizer loading, including debug tiny backbones
- `src/models/hybrid_gemma.py`: explicit Gemma layer runner and hybrid model
- `src/models/baselines.py`: full / skip / bridge baselines
- `src/train/`: Stage A / B / C CLIs
- `src/eval/`: perplexity, GSM8K, StrategyQA, and speed evaluation
- `tests/`: config, shape, frozen-param, and smoke coverage

Planning and notes:

- `notes/references.md`
- `notes/plan.md`
- `notes/blockers.md`
- `notes/final_report.md`

## Known limitations

- Real Gemma loading is blocked until Hugging Face access is authenticated for the gated repos.
- The code assumes Gemma-family execution for the default path; no silent cross-family fallback is implemented.
- Generation uses simple greedy full-sequence decoding rather than a cache-optimized serving path.
- The debug tokenizer is intentionally minimal and exists only to keep tests and smoke runs independent from Gemma access.
- 4-bit loading is implemented for real GPU-backed runs, but the smoke path uses random-initialized tiny Gemma-like backbones instead.
