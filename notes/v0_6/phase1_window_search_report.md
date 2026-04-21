# Phase 1A Window Search Report

- Config: `configs/v0_6/debug_tiny_phase1.yaml`
- Mode: `pilot`
- Candidate count: `16`
- Seeds: `42`
- Backbones: `debug_tiny`
- Ranking rule: KL, then NLL, then PPL, then hidden MSE, then hidden cosine.

## Top Candidates

1. `L24-27__S16-18` | KL=0.002973 | NLL=5.786417 | PPL=325.843485 | hidden_mse=7.346996 | hidden_cos=0.932718 | vs_skip_nll=0.000002 | vs_no_small_nll=0.000002
2. `L24-27__S14-19` | KL=0.002973 | NLL=5.786417 | PPL=325.843485 | hidden_mse=7.347072 | hidden_cos=0.932717 | vs_skip_nll=0.000002 | vs_no_small_nll=0.000001
3. `L24-27__S16-21` | KL=0.002973 | NLL=5.786417 | PPL=325.843406 | hidden_mse=7.347024 | hidden_cos=0.932717 | vs_skip_nll=0.000001 | vs_no_small_nll=0.000001
4. `L24-27__S14-16` | KL=0.002973 | NLL=5.786416 | PPL=325.843248 | hidden_mse=7.347400 | hidden_cos=0.932714 | vs_skip_nll=0.000001 | vs_no_small_nll=0.000001
5. `L26-29__S14-19` | KL=0.003450 | NLL=5.787068 | PPL=326.055534 | hidden_mse=8.659423 | hidden_cos=0.928159 | vs_skip_nll=0.000000 | vs_no_small_nll=0.000001

## Interpretation

- This artifact is a candidate-ranking harness, not a final scientific claim by itself.
- Negative KL/NLL deltas against `skip_only` or `hybrid_no_small` indicate the delegated path recovered output behavior beyond those controls for that candidate.
- This run used the debug-tiny path only. It validates the continuation pipeline and artifact flow, but it is not evidence about Gemma phase ordering.
- Real Gemma pilot execution remains blocked by the existing environment notes in `notes/blockers.md`.