# Final Report Template

## Objective

- Research question:
- Default model pair:
- Why this is a feasibility test rather than a SOTA claim:

## Architecture

- Large backbone:
- Small backbone:
- Entry projector:
- Delegated small block:
- Return adapter:
- Gate:

## Exact Split Used

- Large prefix:
- Large removed block:
- Large suffix:
- Small reference layer:
- Small delegated block:

## Trainable Parameter Count

- Entry projector:
- Return adapter:
- Gate:
- Total trainable params:

## Datasets Used

- Stage A:
- Stage B:
- Stage C:
- Validation perplexity subset:
- GSM8K subset:
- StrategyQA subset:
- Saved sample IDs location:

## Hidden / Logit Recovery Results

- Stage A alignment metrics:
- Stage B hidden MSE:
- Stage B hidden cosine:
- Stage C logit KL:
- Delta norm statistics:

## Benchmark Subset Results

- Full large:
- Skip-only:
- Bridge-only:
- Hybrid:
- Small-only if run:

## Speed / VRAM Results

- Prefill latency:
- Decode throughput:
- Peak VRAM:
- Notes on quantization mode:

## Failure Modes

- Hidden mismatch patterns:
- Tasks that degraded most:
- Quantization or runtime issues:
- Data or evaluation caveats:

## Interpretation

- Did hybrid beat skip-only on hidden/logit recovery?
- Did hybrid recover measurable task performance relative to skip-only?
- Is the result positive, mixed, or negative?
- What cannot be claimed from this run:

## Next Experiments

- Conservative split follow-up:
- Moderate split follow-up:
- Adapter changes:
- Optional LoRA fallback on last 2 delegated small layers:
