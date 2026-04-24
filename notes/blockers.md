# Blockers

## 2026-04-20

### Local base environment ABI mismatch

Observed on this machine:
- `python3` resolves to Python `3.12.4`
- preinstalled packages in that environment fail to import cleanly
- `datasets` and `transformers` import chains hit a NumPy 2.x ABI mismatch through compiled dependencies such as `pandas`, `scipy`, and `sklearn`

Impact:
- The system Python environment is not a reliable base for running this repo as-is.

Resolution path for this repo:
- Use a clean Python `3.12` virtual environment
- Pin compatible package versions in `requirements.txt`
- Run smoke tests and local commands inside that isolated environment

Next concrete step:
- Create `.venv` with `python3.12`, install the pinned requirements, and use that environment for validation.

### Gemma Hugging Face access is gated on this machine

Observed in the clean `.venv`:
- `AutoConfig.from_pretrained("google/gemma-2-2b")` returned `401 Client Error: Unauthorized`
- `AutoTokenizer.from_pretrained("google/gemma-2-2b")` failed with `GatedRepoError`
- the failure is for `https://huggingface.co/google/gemma-2-2b/resolve/main/config.json`

Impact:
- The default Gemma pair cannot be loaded in this environment without authenticated access to the gated repository.
- The repo remains runnable through `configs/debug_tiny.yaml`, but real Gemma runs are blocked until access is granted.

Resolution path for this repo:
- Authenticate with Hugging Face for the Gemma repositories and accept the Gemma terms for the account being used.
- Then rerun the existing stage and eval scripts with `configs/gemma2_conservative.yaml`.

Next concrete step:
- Run `huggingface-cli login` or set `HF_TOKEN`, verify `AutoConfig.from_pretrained("google/gemma-2-2b")` succeeds, and then launch Stage A with the default config.

### Real-hardware bring-up is blocked in the current local environment

Observed via `./scripts/env_sanity.sh`:
- `artifacts/env_sanity.json` reports `cuda_available: false`
- no Hugging Face token was detected
- both `google/gemma-2-9b` and `google/gemma-2-2b` config downloads failed with gated-repo `401` errors

Impact:
- Real Gemma smoke validation and pilot training cannot run on this machine.
- The correct stop condition has been reached before any expensive real-model execution.

Resolution path for this repo:
- move to the actual RTX 5090-class target machine
- ensure CUDA and `bitsandbytes` import cleanly there
- provide Hugging Face authentication with accepted Gemma access
- rerun `./scripts/env_sanity.sh`, then `./scripts/real_gemma_smoke.sh`

Next concrete step:
- On the target GPU machine, activate the repo environment and run `PYTHON_BIN=.venv/bin/python ./scripts/env_sanity.sh`.

## 2026-04-22

### v0.6 Phase 1 continuation is implemented, but real Gemma execution is still blocked here

Observed in this workspace:
- `scripts/v0_6/run_phase1_debug_smoke.ps1` completed successfully
- debug-only continuation artifacts were written under:
  - `artifacts/v0_6/phase1_window_search/`
  - `artifacts/v0_6/phase1_stage_signatures/`
- the new real continuation entrypoint is `configs/v0_6/gemma2_phase1.yaml`

Impact:
- The repo now has runnable Phase 1A and Phase 1B continuation infrastructure.
- However, the current local workspace still cannot produce real Gemma Phase 1 evidence because the earlier blockers remain unchanged:
  - no CUDA target device in this environment
  - gated Gemma access is still unavailable here

Resolution path for this repo:
- move to the target RTX 5090-class machine
- authenticate for `google/gemma-2-9b` and `google/gemma-2-2b`
- run:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\v0_6\run_phase1_window_search.ps1`
  - `powershell -ExecutionPolicy Bypass -File .\scripts\v0_6\run_phase1_stage_signatures.ps1`

Next concrete step:
- Re-run Phase 1 on the real target machine with `configs/v0_6/gemma2_phase1.yaml`, then update `notes/v0_6/phase1_combined_decision.md` from debug-smoke status to real-candidate status.

## 2026-04-24

### Historical: this fork initially lacked the frozen `v0.6.0` token-wise checkpoints needed for the main adaptive-bridge comparison

Observed in this workspace:
- the post-paper fork contains the code and configs from the frozen tag
- no `artifacts/v0_6/idea4_tokenwise/confirm/stage_b/seed_*/tokenwise_mixture_checkpoint.pt` files are present here
- the new adaptive-bridge config points to those paths for:
  - warm-start from frozen `v0.6.0`
  - bounded evaluation against frozen `v0.6.0`

Impact at that time:
- the adaptive-bridge code could run in debug mode and could train from scratch in this fork
- the main research question for this fork could not be answered fairly until the frozen `v0.6.0` reference checkpoints were available
- the continue/stop recommendation remained blocked while those checkpoints were absent

Resolution path for this fork:
- copy the frozen `v0.6.0` token-wise checkpoints into the configured artifact paths, or update the adaptive-bridge checkpoint templates to the real storage location
- rerun:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_train.ps1`
  - `powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_eval.ps1`

Historical next concrete step:
- make the frozen `v0.6.0` token-wise checkpoints visible at `artifacts/v0_6/idea4_tokenwise/confirm/stage_b/seed_{seed}/tokenwise_mixture_checkpoint.pt` or edit `configs/adaptive_bridge/gemma2_first_milestone.yaml` to the correct checkpoint root.

### Historical validation: the 1-seed warm-start real run failed fast before the checkpoint was restored

Observed via:
- `powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_real_warm_start_seed42.ps1`

Exact failure:
- `FileNotFoundError: Required warm-start checkpoints are missing for this real run: artifacts\v0_6\idea4_tokenwise\confirm\stage_b\seed_42\tokenwise_mixture_checkpoint.pt`

Impact at that time:
- the requested 1-seed real run had been wired and attempted
- the current blocker was explicit and reproducible
- the run did not silently degrade into a non-warm-start scratch run

Historical next concrete step:
- place the frozen seed-42 token-wise checkpoint at `artifacts\v0_6\idea4_tokenwise\confirm\stage_b\seed_42\tokenwise_mixture_checkpoint.pt`, then rerun the same PowerShell entrypoint

### Resolved: seed-42 frozen checkpoint path is now live in this fork

Observed in this workspace:
- the frozen reference checkpoint now exists at:
  - `artifacts/v0_6/idea4_tokenwise/confirm/stage_b/seed_42/tokenwise_mixture_checkpoint.pt`
- the source paper-repo checkpoint was read from:
  - `E:/lab/latent-delegation/artifacts/v0_6/idea4_tokenwise/confirm/stage_b/seed_42/tokenwise_mixture_checkpoint.pt`
- the original paper repo was not modified

Current impact:
- the first adaptive-bridge milestone is no longer blocked on the frozen `v0.6.0` checkpoint path
- the seed-42 warm-start run and bounded evaluation are reproducible from this fork-local path

Next concrete step:
- keep using the fork-local checkpoint path for future replications so the paper repo remains frozen

### Active repo standard is now Python 3.12

Current repo state:
- active PowerShell runners are standardized on `py -3.12`
- active shell defaults are standardized on `python3.12`
- `.python-version` is set to `3.12`

Interpretation:
- older 3.11 wording in historical notes should be treated as superseded
- current repo bring-up and validation should use Python 3.12 unless a new blocker is recorded

### Legacy `piqa` dataset-script entry is incompatible with the current `datasets` stack on this machine

Observed via the first bounded eval attempt:
- `load_dataset("piqa", split="validation")` failed with:
  - `RuntimeError: Dataset scripts are no longer supported, but found piqa.py`

Resolution used for the completed seed-42 bounded eval:
- switch the adaptive-bridge config to the mirror dataset id `nthngdy/piqa`
- keep task formatting the same:
  - fields `goal`, `sol1`, `sol2`, `label`

Impact:
- bounded eval was completed without changing the benchmark family
- reproducibility now depends on the mirror dataset id rather than the legacy script-backed `piqa` entry

### Current first-milestone status

Observed in this workspace:
- `outputs/adaptive_bridge/real_seed42_warm_start/train/results.json` exists
- `outputs/adaptive_bridge/real_seed42_warm_start/eval/results.json` exists
- `outputs/adaptive_bridge/real_seed42_43_44_warm_start/train/results.json` exists
- `outputs/adaptive_bridge/real_seed42_43_44_warm_start/eval/results.json` exists
- the evaluation recommendation is:
  - `continue_adaptive_bridge`

Interpretation:
- there is no active blocker for the first seed-42 milestone deliverables in this workspace
- there is also no active blocker for the current 3-seed replication deliverables in this workspace
- remaining issues are research risks, not execution blockers:
  - `ARC-Easy` not recovered
