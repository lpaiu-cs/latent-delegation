# Plan

## Fork status as of `2026-04-24`

The post-paper adaptive-bridge fork has completed its first bounded milestone and a 3-seed replication.

- seed-42 warm-start real training completed from the frozen `v0.6.0` token-wise checkpoint family
- 3-seed replication completed for seeds `42/43/44`
- bounded evaluation completed on:
  - development holdout
  - confirmation holdout
  - LAMBADA
  - PIQA
  - ARC-Easy
- current decision:
  - `continue_adaptive_bridge`

Decision basis:

- internal KL/NLL preserved relative to frozen `v0.6.0` in the 3-seed aggregate
- LAMBADA KL/NLL preserved relative to frozen `v0.6.0` in the 3-seed aggregate
- `PIQA` recovered over both bridge baselines in the 3-seed aggregate
- `ARC-Easy` did not recover

Immediate next step, without widening scope:

- keep the same three-expert budget and no-small adaptive control
- keep the evaluation suite fixed
- treat `ARC-Easy` as the primary unresolved weakness
- if the fork continues, spend the next pilot on a small gate or bridge calibration rather than adding experts, adding Stage C, or widening the benchmark suite

## Default architecture

Default v1 target:
- large backbone: `google/gemma-2-9b`
- small backbone: `google/gemma-2-2b`
- frozen backbones only
- 4-bit loading when practical
- trainable modules only: entry projector, return adapter, scalar gate

Hybrid path:
1. Tokenize input once.
2. Run large model embeddings and frozen large layers `0..23`.
3. Take the large hidden state after layer `23`.
4. Map large hidden `3584 -> 2304` with an affine entry projector.
5. Optionally apply RMSNorm in small latent space.
6. Feed the projected hidden state into frozen small layers `14..19`.
7. Map the delegated small output back `2304 -> rank -> 3584` with a low-rank return adapter.
8. Form the hybrid master state with a learned near-zero gate:
   `h_hybrid = h_large_prefix + gate * delta_large`
9. Continue with frozen large layers `30..41`.
10. Apply final large-model norm / LM head for logits.

## Exact default split

Large model, 0-indexed:
- prefix: `0..23`
- removed block: `24..29`
- suffix: `30..41`

Small model, 0-indexed:
- alignment reference target: hidden after layer `13`
- delegated block: `14..19`

Training targets:
- Stage A target: projected large hidden after large layer `23` aligned to small hidden after layer `13`
- Stage B target: teacher large hidden after layer `29`
- Stage C target: full-large logits

## Exact file tree to create

```text
.
├── README.md
├── requirements.txt
├── Makefile
├── configs/
│   ├── gemma2_conservative.yaml
│   ├── gemma2_moderate.yaml
│   └── debug_tiny.yaml
├── scripts/
│   ├── run_stage_a.sh
│   ├── run_stage_b.sh
│   ├── run_stage_c.sh
│   ├── run_eval_all.sh
│   └── smoke_test.sh
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── build_corpus.py
│   │   └── collators.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone_loader.py
│   │   ├── adapters.py
│   │   ├── hooks.py
│   │   ├── hybrid_gemma.py
│   │   └── baselines.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── trainer_utils.py
│   │   ├── stage_a_align.py
│   │   ├── stage_b_recover.py
│   │   └── stage_c_distill.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── eval_ppl.py
│   │   ├── eval_gsm8k.py
│   │   ├── eval_strategyqa.py
│   │   └── eval_speed.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── logging_utils.py
│       └── seed.py
├── tests/
│   ├── test_config_loading.py
│   ├── test_shapes.py
│   ├── test_frozen_params.py
│   └── test_smoke_debug_tiny.py
└── notes/
    ├── references.md
    ├── plan.md
    ├── blockers.md
    └── final_report.md
```

## Minimal runnable debug path

The smallest end-to-end path will use `configs/debug_tiny.yaml` with a local tiny random backend:
- tiny backbone pair with Gemma-like hidden sizes and layer windows, but no external model download
- tiny synthetic or lightweight text batch
- Stage A for a few steps
- Stage B for a few steps
- Stage C for a few steps
- perplexity / speed sanity evaluation

Expected smoke command:

```bash
./scripts/smoke_test.sh
```

The smoke script will:
1. create a local output directory
2. run Stage A on `debug_tiny`
3. run Stage B using the saved Stage A adapter
4. run Stage C using the saved Stage B adapter
5. run lightweight evals on the debug model

## Implementation choices

- Use a simple YAML config loader instead of a large experiment framework.
- Keep model code family-agnostic at the loader boundary, but default configs stay Gemma 2 only.
- Implement partial execution through direct access to transformer block lists, not through opaque hooks alone.
- Keep training loops explicit and small, with JSON summary and CSV history outputs for each stage.
- Use fixed evaluation subsets and save sampled IDs.

## Likely blockers and fallback behavior

1. Gemma access / authentication blocker
- Do not switch defaults.
- Keep Gemma configs as default.
- Document the exact failure in `notes/blockers.md`.
- Continue with debug backend and all reusable infrastructure.

2. 4-bit partial-layer execution quirks
- Keep a bf16 / fp32 debug path.
- Allow `load_in_4bit: false` in debug or fallback runs.
- Do not rewrite the architecture to depend on unsupported generation-cache internals.

3. Local environment / dependency ABI issues
- Pin a Python 3.12 environment in the repo entrypoints and workspace marker.
- Run tests and smoke in an isolated `.venv`.
- Record the exact mismatch in `notes/blockers.md` if it affects the current machine.

4. Memory pressure on full Gemma runs
- Keep sequence length small by default (`256`).
- Keep micro-batch size at `1` with gradient accumulation.
- Prioritize forward-only and lightweight eval before expanding.

5. Dataset access failures
- Keep a tiny synthetic fallback for tests.
- Support `wikitext-103-v1` plus GSM8K by default, but make local cached JSONL or synthetic debug data acceptable for smoke tests.

## Immediate execution order

1. Scaffold files and configs.
2. Implement config loading, loader utilities, and the tiny debug backend.
3. Implement adapters, partial execution, hybrid model, and baselines.
4. Implement Stage A/B/C CLIs.
5. Implement minimal data pipeline and eval scripts.
6. Add tests and smoke script.
7. Run the debug smoke path in a clean Python 3.12 environment.

## Memory Topology Audit

### Which model objects are instantiated in each stage

- Stage A:
  one frozen Gemma 2 9B object and one frozen Gemma 2 2B object, shared by the hybrid wrapper and reused for large-prefix and small-reference passes
- Stage B hybrid:
  one frozen 9B object and one frozen 2B object; the large teacher hidden after layer 29 is produced by a second serial pass through the same loaded 9B
- Stage B bridge-only:
  one frozen 9B object only; the 2B is no longer loaded on that path
- Stage C hybrid:
  one frozen 9B object and one frozen 2B object; teacher logits come from a serial no-grad full-large pass on the same 9B used by the student hybrid
- Stage C bridge-only:
  one frozen 9B object only
- Single-variant eval:
  `full_large`, `skip_only`, and `bridge_only` now load only the 9B; `hybrid` loads the 9B and 2B
- Speed comparison eval:
  one 9B and one 2B stay resident for the full comparison because the hybrid case needs both; wrapper modules share those same backbone objects and do not clone weights

### Whether they share weights or not

- Wrapper modules do not create new backbone copies.
- Teacher and student logic reuse the same loaded large backbone object in separate serial forwards.
- The small backbone is loaded only on code paths that truly need it.
- Large-only eval and bridge-only training paths were refactored to avoid loading the 2B unnecessarily.

### Expected VRAM pressure points

- Largest static allocation:
  loading the 9B in 4-bit on the target GPU
- Hybrid static allocation:
  keeping both 9B and 2B resident simultaneously for hybrid bring-up or training
- Largest activation pressure:
  Stage C hybrid, because gradients must flow through the frozen small delegated block and the frozen large suffix back into the trainable adapters
- Bring-up pressure:
  hybrid forward at `seq_len=256`, batch size `1`, which is the mandatory smoke target before any pilot training
