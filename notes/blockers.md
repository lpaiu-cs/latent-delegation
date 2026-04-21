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
- Use a clean Python `3.11` virtual environment
- Pin compatible package versions in `requirements.txt`
- Run smoke tests and local commands inside that isolated environment

Next concrete step:
- Create `.venv` with `python3.11`, install the pinned requirements, and use that environment for validation.

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
