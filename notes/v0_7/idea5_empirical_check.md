# Phase 1A Window Search Report

- Config: `configs\v0_7\idea5_candidate_24_27_to_15_18.yaml`
- Mode: `pilot`
- Candidate count: `1`
- Seeds: `42`
- Backbones: `real_model`
- Ranking rule: KL, then NLL, then PPL, then hidden MSE, then hidden cosine.

## Top Candidates

1. `L24-27__S15-18` | KL=0.424089 | NLL=2.944444 | PPL=19.000104 | hidden_mse=9.922135 | hidden_cos=0.794385 | vs_skip_nll=-0.232639 | vs_no_small_nll=-0.213194

## Interpretation

- This artifact is a candidate-ranking harness, not a final scientific claim by itself.
- Negative KL/NLL deltas against `skip_only` or `hybrid_no_small` indicate the delegated path recovered output behavior beyond those controls for that candidate.