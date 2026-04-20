# AGENTS.md

## Project objective

Implement a single-GPU proof-of-concept for one-way latent delegation between same-family open models.

Default target:
- large model: Gemma 2 9B
- small model: Gemma 2 2B

Research question:
- Can a large model keep the master state while delegating part of the middle computation to a smaller same-family model through latent-space transfer?
- This is a feasibility project, not a SOTA project.

Core framing:
- Do NOT claim full “thought transfer”.
- The correct framing is: replace a middle block of the large model with a delegated small-model computation plus learned interface adapters, while the large model keeps ownership of the master state and final logits.

## Hard constraints

- Single GPU only. Assume one RTX 5090-class GPU.
- No multi-GPU, FSDP, DeepSpeed, tensor parallelism, pipeline parallelism, or distributed training.
- Total project scope must fit a one-month research cycle.
- Optimize for clarity, failure analysis, and feasibility.
- Do not expand scope toward general-purpose agent systems, RL, or full model distillation.
- Backbones must remain frozen in v1.
- Use 4-bit quantized frozen backbones when practical.
- Train only small interface modules in v1.
- No full finetuning.
- No LoRA in v1 by default.
- Optional fallback experiment: LoRA only on the last 2 delegated small-model layers, disabled by default.

## Model-family constraint

Use same-family open models only.

Default pair:
- `google/gemma-2-9b`
- `google/gemma-2-2b`

Do not silently substitute another family.

If Gemma access is blocked by license/authentication:
1. Scaffold the code to remain model-family-agnostic.
2. Record the exact blocker in `notes/blockers.md`.
3. Only add a fallback config for Qwen2.5 7B/1.5B if explicitly enabled by a config flag.
4. Do not switch defaults without an explicit config change.

## Default experiment scope

Only implement and target the conservative split first.

Conservative split, 0-indexed:
- large prefix: layers `0..23`
- removed large middle block: layers `24..29`
- large suffix: layers `30..41`

Small delegated window, 0-indexed:
- small pre-entry reference layer output: layer `13`
- delegated small layers: `14..19`

Interpretation:
- feed the large model hidden state after layer 23 into an entry adapter
- map it into the small model latent space expected before small layer 14
- run frozen small layers 14..19
- map the result back to the large space through a return adapter
- add it as a correction to the large master state
- continue through large layers 30..41

Moderate split may be implemented as a config for later, but it is not the default goal.

## Default architecture

Large model owns:
- input embedding path
- master hidden state
- final suffix
- final logits

Small model owns:
- delegated latent computation only

Required modules:
1. Entry projector:
   - affine map from large hidden size to small hidden size
   - optional RMSNorm after projection
2. Frozen delegated small block
3. Return adapter:
   - low-rank map from small hidden size back to large hidden size
4. Scalar gate on returned delta:
   - initialize near zero
   - hybrid state should be:
     `h_large_after = h_large_prefix + gate * delta_small_to_large`

Default return adapter rank:
- 64

Backbones:
- frozen
- quantized when practical
- no gradient updates allowed

## Training stages

Implement exactly three stages.

### Stage A: representation alignment

Goal:
- learn the entry projector so that the projected large hidden state matches the small-family latent space

Default target:
- large hidden after layer 23
- small hidden after layer 13

Recommended losses:
- MSE
- cosine similarity loss

### Stage B: hidden recovery

Goal:
- learn the return adapter so that delegated small computation helps recover the role of the removed large middle block

Default target:
- teacher target is the full large-model hidden after layer 29

Recommended losses:
- hidden-state MSE
- cosine similarity loss

### Stage C: output distillation

Goal:
- align the hybrid model outputs to the full large model

Recommended losses:
- KL divergence to full large-model logits
- next-token cross entropy
- regularization penalty on delta magnitude

## Required baselines

Implement these baselines:

1. Full large model
2. Skip-only large model
   - remove the target middle block and continue directly to suffix
   - no learned bridge
3. Bridge-only large model
   - no small model
   - learned low-rank bridge in large space
4. Proposed hybrid
   - large prefix -> small delegated block -> large suffix

Optional:
5. small-only model, for reference only

## Evaluation priorities

Primary:
- hidden-state cosine and MSE
- logit KL
- validation perplexity

Secondary:
- small benchmark subsets for reasoning tasks
- latency and peak VRAM

Do not make the benchmark suite large.
Default evaluation should be lightweight and reproducible.

## Benchmark scope

Keep evaluation small.

Suggested defaults:
- validation perplexity on a held-out text subset
- GSM8K test subset: 200 examples
- StrategyQA validation subset: 200 examples

Use fixed seeds and save the sampled evaluation IDs.

Generation:
- greedy decoding only in v1
- fixed prompt templates
- robust answer parsers
- do not spend excessive time on prompt optimization

## Success criteria

This project succeeds if at least one conservative-split hybrid run shows all of the following:

- clear improvement over skip-only on hidden/logit recovery
- measurable recovery of large-model performance relative to skip-only
- runnable on one GPU
- reproducible scripts and saved configs
- clean reporting of failure modes if end-task gains are weak

Do not overclaim.
A negative result with clear diagnostics is acceptable.

## Engineering rules

- Python only.
- Prefer simple, explicit code over framework-heavy abstractions.
- Use type hints.
- Add docstrings to public functions/classes.
- Keep configs in YAML.
- Keep all experiment hyperparameters in config files, not hardcoded.
- Make every stage runnable from CLI.
- Add unit tests for shape contracts and frozen-parameter guarantees.
- Add a tiny debug config that does not require Gemma weights.
- Ensure deterministic seeds where practical.
- Save metrics as JSON/CSV.
- Save run metadata, config snapshot, and git commit hash if available.

## Non-goals

Do NOT:
- implement multi-agent orchestration
- implement token-free production serving
- implement full KV-cache research infrastructure in v1
- optimize for polished UI
- chase benchmark SOTA
- silently widen the scope

## Required repo outputs

At minimum create:

- `README.md`
- `requirements.txt`
- `Makefile`
- `configs/`
- `src/`
- `scripts/`
- `tests/`
- `notes/references.md`
- `notes/plan.md`
- `notes/final_report.md`

## Done when

The task is done only when all of the following exist:

1. A runnable repo scaffold
2. Model-loading code for the default pair
3. Hybrid model implementation for the conservative split
4. All four required baselines
5. Stage A / B / C training scripts
6. Lightweight eval scripts
7. Unit tests for shapes and frozen params
8. A debug smoke test
9. A README with exact commands
10. A short final report template with success/failure interpretation

## Stop conditions

If blocked by missing model access, broken upstream package behavior, or memory failure:
- do not thrash
- document the blocker precisely
- implement the remaining reusable infrastructure
- leave the repo in a runnable partial state
- record the next concrete step in `notes/blockers.md`